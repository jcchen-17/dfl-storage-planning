"""Measure whether a scenario pool leaves any room for a learned selector.

Before rebuilding a dataset it is worth knowing whether the current one can
demonstrate anything. Three numbers decide that, and all three are measurable:

  ceiling   -- the perfect-information design's storage value on the test subset.
               Nothing any selection rule does can beat it.
  spread    -- how much the naive rules vary. If random already lands on the
               ceiling every time, there is nothing to learn; if it varies
               wildly, a reliable selector is worth something even when it never
               beats random's best draw.
  headroom  -- ceiling minus the random MEAN. That, not the gap to random's best
               draw, is what a learned selector can actually capture.

The whole pool by default. ``--regime`` narrows it by scenario-name suffix,
which only selects anything on the _price_regimes expansion: there,
``price_1p0x`` picks the windows as they were before augment_price_regimes.py
scaled each one's price spread into five synthetic tariff regimes.

    python scripts/headroom_probe.py
    python scripts/headroom_probe.py --skip-ceiling --time-limit 600
    python scripts/headroom_probe.py --carbon-cap 0.18 --tag cap018
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("STORAGE_DFL_SOLVER_WORKER", "1")  # one in-process solve at a time

import torch  # noqa: F401  (its import is what would arm subprocess isolation)

from storage_dfl.config import load_config
from storage_dfl.data import ScenarioPool
from storage_dfl.dfl import select_scenarios
from storage_dfl.planning import StoragePlanningOracle
from storage_dfl.data import ScenarioCodec
from storage_dfl.stages import (
    ArtifactPaths,
    _experiment_data,
    _load_codec,
    _scenario_set_summary,
)

REPO = Path(__file__).resolve().parent.parent


def _codec(config, feeder) -> ScenarioCodec:
    """The saved normalization when it fits the data, otherwise a fresh one.

    This probe answers whether a pool leaves room for a learned selector, which
    is a question about the data and should not require a trained generator --
    and after a dataset change the saved normalization is usually stale (the
    price-regime expansion carried a fifth context dimension the un-augmented
    windows do not have, so loading it raises a broadcast error). Fitting on the
    train split is what train_cvae_stage does, so the feature space is the same
    one the rest of the pipeline will use.
    """
    _, train_pool = _experiment_data(config, config.data.train_split)
    want = int(train_pool.scenarios[0].context.size)
    try:
        saved = _load_codec(ArtifactPaths(config.output_dir), feeder)
        # Loading a stale normalization succeeds -- the mismatch only surfaces
        # later, inside encode_pool, as a broadcast error. Check the width here
        # so the fallback actually fires.
        if int(saved.context_dim) == want:
            return saved
        print(f"saved normalization has context_dim {saved.context_dim} but the "
              f"dataset has {want}; fitting a codec on the train split instead",
              flush=True)
    except (FileNotFoundError, ValueError) as exc:
        print(f"saved normalization unusable ({exc}); fitting a codec on the "
              "train split instead", flush=True)
    return ScenarioCodec.fit(train_pool, feeder)


def filtered(pool: ScenarioPool, regime: str) -> ScenarioPool:
    if not regime:
        return pool
    keep = [i for i, s in enumerate(pool.scenarios) if s.name.endswith(regime)]
    if not keep:
        raise SystemExit(f"no scenarios end with {regime!r}")
    return ScenarioPool(pool.subset(keep))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/generator_compare_k1t2.yaml")
    # Defaults to the whole pool. The suffix filter only means anything against
    # the _price_regimes expansion, whose names end in price_1p0x and friends;
    # the un-augmented windows carry no suffix, so filtering them is a no-op at
    # best and an empty pool at worst. Leaving the default as a suffix also made
    # the common invocation require passing an empty string, which PowerShell
    # drops before argparse ever sees it.
    ap.add_argument("--regime", default="",
                    help="scenario-name suffix to keep, e.g. price_1p0x on the "
                         "augmented dataset; default keeps the whole pool")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory-limit", type=float, default=9000.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--time-limit", type=float, default=3600.0,
                    help="per-solve seconds")
    # The ceiling comes from a free-design solve over the whole test subset and
    # is by far the most expensive thing here -- minutes at a loose carbon cap,
    # far longer once the cap binds. It is what turns values into capture
    # ratios, but it is not needed to decide whether a pool is worth working on:
    # that follows from whether the rules build storage at all and whether the
    # choice of scenario moves the answer, which the rule rows show on their own.
    ap.add_argument("--skip-ceiling", action="store_true",
                    help="skip the perfect-information solve and report the "
                         "rule rows only")
    # Comparing operating points means solving the same pool under several caps,
    # and editing the config between runs would leave whichever one finished last
    # standing as the committed value. Override it here instead and commit only
    # once the comparison is settled.
    ap.add_argument("--carbon-cap", type=float, default=None,
                    help="override planning.dc_carbon_cap for this probe")
    args = ap.parse_args()

    tag = args.tag or (args.regime or "all").replace("price_", "")
    config = load_config(str(REPO / args.config))

    feeder, val_all = _experiment_data(config, config.data.validation_split)
    _, test_all = _experiment_data(config, config.data.test_split)
    codec = _codec(config, feeder)

    val = filtered(val_all, args.regime)
    test = filtered(test_all, args.regime)
    print(f"regime {args.regime or '(all)'}: validation {len(val.scenarios)}, "
          f"test {len(test.scenarios)}", flush=True)

    # Same subset rule the sweeps and the DFL evaluation use, applied to the
    # filtered pool so the reference, ceiling and rules all land on one set.
    indices = codec.support_indices(
        test, min(config.dfl.final_validation_size, len(test.scenarios))
    )
    test_subset = ScenarioPool(test.subset(indices.tolist())).scenarios
    print(f"test subset: {len(test_subset)} scenarios", flush=True)

    cap = (
        config.planning.dc_carbon_cap
        if args.carbon_cap is None
        else args.carbon_cap
    )
    planning = replace(
        config.planning,
        dc_carbon_cap=cap,
        solver_threads=args.threads,
        solver_max_parallel_workers=1,
        solver_memory_limit_mb=args.memory_limit,
        solver_time_limit_seconds=args.time_limit,
        verbose_solver=False,
    )
    print(f"dc_carbon_cap {cap}, carbon_price "
          f"{config.costs.carbon_price_dollars_per_t:g} $/tCO2", flush=True)
    oracle = StoragePlanningOracle(
        feeder, planning, config.costs, config.data, config.data_center
    )

    print("no-storage reference...", flush=True)
    reference = StoragePlanningOracle(
        feeder, replace(planning, max_storage_sites=0),
        config.costs, config.data, config.data_center,
    ).solve(test_subset, allow_carbon_slack=True)
    ref = float(reference.objective)
    print(f"  {reference.status}  {ref:,.2f}", flush=True)

    pi = None
    pi_bound = None
    if args.skip_ceiling:
        print("perfect-information solve skipped (--skip-ceiling)", flush=True)
    else:
        print("perfect-information design (free design on the test subset)...",
              flush=True)
        started = time.perf_counter()
        pi = oracle.solve(test_subset, allow_carbon_slack=True)
        # The incumbent may not be optimal at the configured gap, so bracket it:
        # nothing beats the dual bound, and the incumbent itself is achievable.
        pi_bound = float(pi.objective) * (1.0 - float(pi.relative_gap))
        print(f"  {pi.status}  {float(pi.objective):,.2f}  "
              f"gap {float(pi.relative_gap):.2e}"
              f"  ({time.perf_counter() - started:.0f}s)", flush=True)

    rows = []
    jobs = [("random", s) for s in range(args.seeds)]
    jobs += [("kmeans", 0), ("farthest", 0), ("aggregate", 0)]
    for rule, seed in jobs:
        scenarios, weights, labels = select_scenarios(rule, val, codec, args.k, seed=seed)
        design = oracle.solve(scenarios, weights=weights)
        if not design.feasible:
            print(f"  {rule} seed={seed}: planning {design.status}", flush=True)
            continue
        oos = oracle.solve(test_subset, fixed_design=design.design, allow_carbon_slack=True)
        value = ref - float(oos.objective)
        installed = list(design.design.installed_buses)
        summary = _scenario_set_summary(scenarios, weights)
        rows.append({
            "rule": rule, "seed": seed, "storage_value": value,
            "installed": installed,
            "energy_mwh": sum(float(design.design.energy_mwh[b]) for b in installed),
            "support": list(labels),
            "peak_net_mw": summary["weighted_peak_net_load_mw"],
            "price_spread": summary["weighted_price_spread_per_mwh"],
        })
        print(f"  {rule:<10} seed={seed}  value {value:>9,.0f}  "
              f"E={rows[-1]['energy_mwh']:.3f}  {labels[0] if labels else ''}", flush=True)

    values = [r["storage_value"] for r in rows if r["rule"] == "random"]
    best_seen = max((r["storage_value"] for r in rows), default=0.0)
    if pi is None:
        ceiling_lo = ceiling_hi = None
    else:
        ceiling_lo = ref - float(pi.objective)
        ceiling_hi = ref - pi_bound
        # A rule that beat the PI incumbent proves the incumbent was not optimal.
        ceiling_lo = max(ceiling_lo, best_seen)

    payload = {
        "regime": args.regime, "k": args.k,
        "no_storage_reference": ref,
        "ceiling_interval": (
            None if ceiling_lo is None else [ceiling_lo, ceiling_hi]
        ),
        "rows": rows,
    }
    out = REPO / "outputs" / f"headroom_{tag}_k{args.k}.json"
    out.write_text(json.dumps(payload, indent=2))

    print(f"\n{'=' * 62}")
    print(f"regime {args.regime or '(all)'}  k={args.k}")
    print(f"{'=' * 62}")
    if ceiling_lo is None:
        print("storage-value ceiling : not computed (--skip-ceiling)")
        print(f"best rule seen        : {best_seen:,.0f}")
    else:
        print(f"storage-value ceiling : [{ceiling_lo:,.0f}, {ceiling_hi:,.0f}]")
    if values and ceiling_lo is None:
        mean = st.mean(values)
        sd = st.stdev(values) if len(values) > 1 else 0.0
        print(f"random  n={len(values)}  mean {mean:,.0f}  sd {sd:,.0f}  "
              f"min {min(values):,.0f}  max {max(values):,.0f}")
        if mean > 0.0:
            print(f"relative spread (sd / mean)      : {100 * sd / mean:.0f}%")
        else:
            print(f"spread (sd)                      : {sd:,.0f} "
                  "(mean is not positive)")
    elif values:
        mean = st.mean(values)
        sd = st.stdev(values) if len(values) > 1 else 0.0
        print(f"random  n={len(values)}  mean {mean:,.0f}  sd {sd:,.0f}  "
              f"min {min(values):,.0f}  max {max(values):,.0f}")
        print(f"\nHEADROOM (ceiling - random mean) : "
              f"{ceiling_lo - mean:,.0f} to {ceiling_hi - mean:,.0f}")
        # A ceiling at zero is not a degenerate print case, it is the answer:
        # the best available design installs nothing, so there is no value for
        # any selection rule to capture and capture ratios are undefined.
        if ceiling_lo <= 0.0:
            print("\n  THE CEILING IS ZERO: the perfect-information design installs no")
            print("  storage, so no selection rule can capture anything. There is no")
            print("  storage-planning problem in this pool -- the correct decision is")
            print("  'do not build', and every rule that builds loses money.")
        else:
            print(f"random mean captures             : "
                  f"{100 * mean / ceiling_hi:.0f}% - {100 * mean / ceiling_lo:.0f}%")
            print(f"random best captures             : "
                  f"{100 * max(values) / ceiling_hi:.0f}% - {100 * max(values) / ceiling_lo:.0f}%")
        if mean > 0.0:
            print(f"relative spread (sd / mean)      : {100 * sd / mean:.0f}%")
        else:
            print(f"spread (sd)                      : {sd:,.0f} "
                  "(mean is not positive; sd/mean is meaningless)")
    for rule in ("kmeans", "farthest", "aggregate"):
        r = next((x for x in rows if x["rule"] == rule), None)
        if r:
            print(f"{rule:<10} {r['storage_value']:>9,.0f}")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
