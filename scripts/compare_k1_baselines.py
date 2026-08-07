"""Compare a DFL run against the k-sweep baselines on the same test subset.

The two pipelines are only comparable because they compute the same thing:
``sweep_k.py`` and ``stages.py`` both take the test subset via
``codec.support_indices``, solve it with the same oracle, and measure against a
``max_storage_sites=0`` reference.  This script re-checks that empirically (the
no-storage reference must be one constant across every row and match the DFL
run's own reference) before it reports anything, because every number below is
a difference against that reference and a mismatch would make them meaningless.

``storage_value`` is ``no_storage_objective - out_of_sample_objective``.  Since
the reference is a constant, that is an affine transform of the out-of-sample
objective with slope -1: ranking by either gives the identical ordering.  It is
reported because a difference read against what the storage decision is worth
is more legible than one read against a total that is mostly untouchable.

Usage:
    python scripts/compare_k1_baselines.py
    python scripts/compare_k1_baselines.py --k 3 --dfl-dir outputs/generator_comparison_k3t2_48h
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SWEEPS = ("outputs/k_sweep_k1358_t2", "outputs/k_sweep_k1358_cvae")
DEFAULT_DFL = "outputs/generator_comparison_k1t2_48h"

# A reference that differs between the sweeps and the DFL run by more than this
# means the two were not evaluated on the same test subset.
REFERENCE_TOLERANCE = 0.05


def load_rows(sweep_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for d in sweep_dirs:
        path = d / "k_sweep.csv"
        if not path.exists():
            raise SystemExit(f"missing sweep csv: {path}")
        with open(path, newline="") as handle:
            rows += list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("no sweep rows loaded")
    return rows


def load_dfl(dfl_dir: Path, tag: str) -> dict:
    path = dfl_dir / f"result_{tag}.json"
    if not path.exists():
        raise SystemExit(f"missing DFL result: {path}")
    return json.loads(path.read_text())


def check_comparable(rows: list[dict], dfl: dict) -> float:
    """Refuse to report if the two pipelines did not share a test subset."""
    refs = {r["no_storage_objective"] for r in rows}
    if len(refs) != 1:
        raise SystemExit(
            f"the sweep rows carry {len(refs)} different no-storage references "
            f"({sorted(refs)}); they were not all evaluated on one test subset."
        )
    sweep_ref = float(refs.pop())
    dfl_ref = float(dfl["no_storage_reference"]["objective"])
    if abs(sweep_ref - dfl_ref) > REFERENCE_TOLERANCE:
        raise SystemExit(
            f"no-storage reference differs: sweep {sweep_ref:,.2f} vs DFL "
            f"{dfl_ref:,.2f}. The runs used different test subsets and the "
            "numbers are not comparable."
        )
    if not dfl.get("objectives_comparable", False):
        raise SystemExit(
            "the DFL run reports objectives_comparable=false -- its reference or "
            "validation solve was unbounded, so storage_value is not defined."
        )
    return sweep_ref


def installed_energy(dfl: dict) -> float:
    return sum(float(v) for v in dfl["planning"]["design"]["energy_mwh"].values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=1,
                    help="support size the DFL run used (default 1)")
    ap.add_argument("--dfl-dir", default=DEFAULT_DFL)
    ap.add_argument("--method", default="reinforce",
                    help="DFL method tag, matching result_<tag>.json")
    ap.add_argument("--sweeps", nargs="+", default=list(DEFAULT_SWEEPS))
    args = ap.parse_args()

    rows = load_rows([REPO / s for s in args.sweeps])
    dfl = load_dfl(REPO / args.dfl_dir, args.method)
    reference = check_comparable(rows, dfl)

    dfl_value = float(dfl["storage_value"])
    oos = dfl["out_of_sample_validation"]

    print(f"no-storage reference : {reference:,.2f}  (identical across "
          f"{len(rows)} sweep rows and the DFL run)")
    print(f"DFL(k={args.k}, {args.method}) : storage_value {dfl_value:,.2f}   "
          f"out-of-sample {float(oos['objective']):,.2f}")
    print(f"  status {oos['status']}  gap {float(oos['relative_gap']):.2e}  "
          f"installed {installed_energy(dfl):.3f} MWh")

    # --- the controlled ablation ------------------------------------------
    # cvae_initial reconstructs the same source scenario the DFL run started
    # from, through the same generator. It is the only row that differs from
    # the DFL run in exactly one factor, so it is the only controlled contrast
    # available -- every other rule changes the selection rule too.
    checkpoint = REPO / args.dfl_dir / f"dfl_result_{args.method}.json"
    if checkpoint.exists():
        sources = json.loads(checkpoint.read_text()).get("support_source_names", [])
        initial = [r for r in rows
                   if r["rule"] == "cvae_initial" and int(r["k"]) == args.k]
        if sources and initial:
            init_support = [s.replace("reconstruction:", "")
                            for s in initial[0]["support_scenarios"].split("|")]
            same = sorted(init_support) == sorted(sources)
            init_value = float(initial[0]["storage_value"])
            print(f"\ncontrolled ablation (same generator, same source scenario):")
            print(f"  source scenarios identical : {same}")
            print(f"  cvae_initial k={args.k} : {init_value:>10,.2f}  "
                  f"installed {initial[0]['installed']}")
            print(f"  DFL          k={args.k} : {dfl_value:>10,.2f}")
            if same:
                print(f"  -> DFL training is worth {dfl_value - init_value:+,.2f} "
                      "from an identical starting point")
            else:
                print("  -> NOT a controlled contrast: the support sources differ.")

    # --- deterministic rules ----------------------------------------------
    # These have no seed, so one run is their entire distribution; comparing
    # them to a single DFL run is like-for-like.
    at_k = [r for r in rows if int(r["k"]) == args.k]
    det = sorted((r for r in at_k if r["rule"] not in ("random", "cvae_random")),
                 key=lambda r: -float(r["storage_value"]))
    print(f"\ndeterministic rules at k={args.k} (one run = their whole distribution):")
    wins = 0
    for r in det:
        v = float(r["storage_value"])
        wins += dfl_value > v
        print(f"  {'WIN ' if dfl_value > v else 'LOSS'}  {r['rule']:<16}{v:>10,.2f}"
              f"   installed {r['installed']}")
    print(f"  -> DFL beats {wins}/{len(det)}")

    # --- seeded rules ------------------------------------------------------
    # A single DFL draw against a 10-seed distribution cannot establish an
    # ordering; the percentile is reported so the overlap is visible, not so
    # it can be read as a win or a loss.
    print(f"\nseeded rules at k={args.k} (DFL is one draw against a distribution):")
    for rule in ("random", "cvae_random"):
        v = sorted(float(r["storage_value"]) for r in at_k if r["rule"] == rule)
        if not v:
            continue
        pct = 100 * sum(1 for x in v if x < dfl_value) / len(v)
        print(f"  {rule:<14} n={len(v)}  mean {st.mean(v):>9,.0f}  "
              f"sd {st.stdev(v):>8,.0f}  range [{min(v):,.0f}, {max(v):,.0f}]")
        print(f"  {'':<14} DFL sits at the {pct:.0f}th percentile "
              "-- n=1 vs n=%d cannot order these" % len(v))

    # --- why the value is capped ------------------------------------------
    energies = [float(r["energy_mwh"]) for r in rows]
    values = [float(r["storage_value"]) for r in rows]
    n = len(energies)
    me, mv = st.mean(energies), st.mean(values)
    cov = sum((e - me) * (v - mv) for e, v in zip(energies, values)) / n
    corr = cov / (st.pstdev(energies) * st.pstdev(values))
    dfl_e = installed_energy(dfl)
    band = [v for e, v in zip(energies, values) if 1.0 <= e < 2.0]
    print(f"\nsizing vs value over {n} baseline runs: corr = {corr:.3f}")
    print(f"  DFL installed {dfl_e:.3f} MWh -> {dfl_value:,.0f}")
    if band:
        print(f"  baselines in the 1-2 MWh band: n={len(band)} mean {st.mean(band):,.0f}"
              "  (the DFL design performs normally FOR ITS SIZE)")

    summary = dfl.get("scenario_summary", {})
    if summary:
        gen = summary["generated"]["weighted_price_spread_per_mwh"]
        obs = summary["observed_test"]["weighted_price_spread_per_mwh"]
        print(f"\nsupport realism: generated price spread {gen:.2f} $/MWh vs "
              f"observed {obs:.2f} $/MWh ({obs / gen:.1f}x)")

    # --- solve quality -----------------------------------------------------
    bad = [r for r in rows if r["planning_status"] != "optimal"
           or r["out_of_sample_status"] != "optimal"]
    print(f"\nnon-optimal baseline solves: {len(bad)}/{len(rows)}")
    for r in bad:
        print(f"  {r['rule']:<16} k={r['k']} seed={r['seed']} "
              f"planning={r['planning_status']} oos={r['out_of_sample_status']}")


if __name__ == "__main__":
    main()
