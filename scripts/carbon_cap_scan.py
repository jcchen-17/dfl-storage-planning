"""Find the dc_carbon_cap range in which storage earns its cost honestly.

The configured cap (0.62) sits above the maximum grid carbon intensity in the
data (0.409 tCO2/MWh), so it has never bound: carbon_slack_cost is 0.0 in every
recorded run. Tightening it is the one lever that makes storage valuable without
touching any data -- charging on low-carbon hours and discharging on high-carbon
ones is what the vintage-tracking (mccormick) formulation exists to represent.

There is a trap, and this script reports the number that detects it.
validation_carbon_slack_dollars is 300,000, multiplied by annual_blocks (182.5)
per unit of slack -- roughly $55M. Once a cap is tight enough that the no-storage
case must buy slack, "storage value" becomes "avoided penalty" and scales with
that arbitrary price rather than with physics. A cap is only useful evidence when
BOTH the reference and the storage design clear it with slack == 0: then the
difference is real operating cost, not a penalty artifact.

    python scripts/carbon_cap_scan.py --caps 0.35 0.30 0.26 0.22
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("STORAGE_DFL_SOLVER_WORKER", "1")

import torch  # noqa: F401

from storage_dfl.config import load_config
from storage_dfl.data import ScenarioPool
from storage_dfl.planning import StoragePlanningOracle
from storage_dfl.stages import ArtifactPaths, _experiment_data, _load_codec

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/generator_compare_k1t2.yaml")
    ap.add_argument("--regime", default="price_1p0x",
                    help="scenario suffix to keep; 1p0x is the un-augmented data")
    ap.add_argument("--caps", type=float, nargs="+",
                    default=[0.40, 0.35, 0.30, 0.26, 0.22])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory-limit", type=float, default=9000.0)
    # The mccormick relaxation lower-bounds nodal_carbon by
    # incoming_carbon / incoming_power_upper, and incoming_power_upper is built
    # from line thermal ratings, which far exceed real flows. That makes the
    # bound far weaker than the true intensity, which is why no cap binds --
    # 0.05 stays feasible although the cleanest grid hour is 0.171. Use 'exact'
    # to enforce the bilinear identity itself (Gurobi/SCIP only; nonconvex, so
    # start with few scenarios).
    ap.add_argument("--formulation", default=None,
                    choices=("exact", "mccormick", "system_average"),
                    help="override planning.carbon_formulation")
    ap.add_argument("--scenarios", type=int, default=None,
                    help="test-subset size; default dfl.final_validation_size")
    ap.add_argument("--envelope", default=None,
                    choices=("line_rating", "demand"),
                    help="override planning.carbon_envelope_bound. Validate a "
                         "switch to 'demand' by comparing objectives against "
                         "'line_rating' at a NON-binding cap: they must match.")
    # The envelope's LOWER bound is w >= xU*y + yU*x - xU*yU, so an xU far above
    # the intensities that actually occur drives it negative whenever the flow
    # sits below its own bound -- then w >= 0 dominates and a line may carry zero
    # carbon. At 1.20 against a real maximum of 0.72 (the generator; the grid
    # peaks at 0.409) that gap is the leak. Lowering it to just above the
    # dirtiest real source is a statement about physics, not a tuning knob, but
    # setting it BELOW 0.72 would cut off feasible dispatches.
    ap.add_argument("--carbon-intensity-max", type=float, default=None,
                    help="override planning.carbon_intensity_max")
    # A binding carbon cap makes these solves far harder: under system_average a
    # 4-scenario instance ran an hour without converging while the same instance
    # at a non-binding cap took 7 seconds. Cap the wait so a scan reports
    # incumbents instead of hanging, and read the status column.
    ap.add_argument("--time-limit", type=float, default=3600.0,
                    help="per-solve seconds")
    # With a price set, exceeding the cap costs dollars per tonne instead of a
    # big-M penalty, so a nonzero carbon cost is an ordinary operating expense
    # rather than the sign of an unusable run. The slack_free test below is
    # skipped accordingly.
    ap.add_argument("--carbon-price", type=float, default=None,
                    help="dollars per tCO2 above dc_carbon_cap; enables the "
                         "priced formulation in place of big-M slack")
    args = ap.parse_args()

    config = load_config(str(REPO / args.config))
    feeder, test_all = _experiment_data(config, config.data.test_split)
    codec = _load_codec(ArtifactPaths(config.output_dir), feeder)

    pool = test_all
    if args.regime:
        keep = [i for i, s in enumerate(pool.scenarios) if s.name.endswith(args.regime)]
        pool = ScenarioPool(pool.subset(keep))
    count = args.scenarios or config.dfl.final_validation_size
    idx = codec.support_indices(pool, min(count, len(pool.scenarios)))
    subset = ScenarioPool(pool.subset(idx.tolist())).scenarios
    formulation = args.formulation or config.planning.carbon_formulation
    envelope = args.envelope or config.planning.carbon_envelope_bound
    intensity_max = args.carbon_intensity_max or config.planning.carbon_intensity_max
    carbon_price = (
        config.costs.carbon_price_dollars_per_t
        if args.carbon_price is None
        else args.carbon_price
    )
    costs = replace(config.costs, carbon_price_dollars_per_t=carbon_price)
    priced = carbon_price > 0.0
    print(f"regime {args.regime or '(all)'}, test subset {len(subset)} scenarios", flush=True)
    print(f"carbon_formulation {formulation}, carbon_envelope_bound {envelope}, "
          f"carbon_intensity_max {intensity_max}, "
          f"carbon_price {carbon_price:g} $/tCO2, "
          f"baseline dc_carbon_cap {config.planning.dc_carbon_cap}\n", flush=True)

    rows = []
    for cap in args.caps:
        base = replace(
            config.planning,
            dc_carbon_cap=cap,
            carbon_formulation=formulation,
            carbon_envelope_bound=envelope,
            carbon_intensity_max=intensity_max,
            solver_threads=args.threads,
            solver_max_parallel_workers=1,
            solver_memory_limit_mb=args.memory_limit,
            solver_time_limit_seconds=args.time_limit,
            verbose_solver=False,
        )
        started = time.perf_counter()
        reference = StoragePlanningOracle(
            feeder, replace(base, max_storage_sites=0),
            costs, config.data, config.data_center,
        ).solve(subset, allow_carbon_slack=True)
        oracle = StoragePlanningOracle(
            feeder, base, costs, config.data, config.data_center
        )
        best = oracle.solve(subset, allow_carbon_slack=True)
        elapsed = time.perf_counter() - started

        installed = list(best.design.installed_buses)
        value = float(reference.objective) - float(best.objective)
        ref_slack = float(reference.carbon_slack_cost)
        pi_slack = float(best.carbon_slack_cost)
        # Under a price, carbon cost is a real operating expense and storage
        # earns its keep by avoiding it, so a nonzero value is the point rather
        # than a disqualification. Only the big-M penalty needs the clean test.
        clean = priced or (ref_slack == 0.0 and pi_slack == 0.0)
        row = {
            "cap": cap,
            "reference_objective": float(reference.objective),
            "reference_carbon_slack": ref_slack,
            "reference_status": reference.status,
            "pi_objective": float(best.objective),
            "pi_carbon_slack": pi_slack,
            "pi_status": best.status,
            "pi_gap": float(best.relative_gap),
            "storage_value": value,
            "installed": installed,
            "power_mw": sum(float(best.design.power_mw[b]) for b in installed),
            "energy_mwh": sum(float(best.design.energy_mwh[b]) for b in installed),
            "slack_free": clean,
            "seconds": round(elapsed, 1),
        }
        rows.append(row)
        flag = "PRICE" if priced else ("OK  " if clean else "SLACK")
        # A time-limited incumbent is not a proven optimum, so a storage_value
        # read off one is not comparable with a converged row. Print the statuses
        # rather than leaving them in the json only.
        converged = reference.status == "optimal" and best.status == "optimal"
        mark = "" if converged else f"  <-- NOT CONVERGED ref={reference.status} pi={best.status}"
        print(f"cap {cap:<5} [{flag}] storage_value {value:>10,.0f}  "
              f"E={row['energy_mwh']:.3f} MWh  P={row['power_mw']:.3f} MW  "
              f"ref_slack={ref_slack:,.0f} pi_slack={pi_slack:,.0f}  ({elapsed:.0f}s){mark}",
              flush=True)
        # Name the file after the settings: two runs that differ only in
        # formulation or envelope are exactly the runs being compared, and a
        # shared filename would leave only the last one on disk.
        out_json = (
            REPO / "outputs"
            / f"carbon_cap_scan_{formulation}_{envelope}"
              f"_imax{intensity_max:g}_price{carbon_price:g}.json"
        )
        out_json.write_text(json.dumps(rows, indent=2))

    print("\n" + "=" * 74)
    # A design that installs nothing scores exactly zero up to solver noise, and
    # rounding decides its sign. Require a value large enough to be a decision,
    # not a rounding artifact, before calling a cap usable.
    usable = [
        r for r in rows
        if r["slack_free"] and r["storage_value"] > 1.0 and r["energy_mwh"] > 0.0
    ]
    if usable:
        print("Caps where storage pays for itself WITHOUT any carbon slack:")
        for r in usable:
            print(f"  cap {r['cap']:<5} value {r['storage_value']:>10,.0f}  "
                  f"E={r['energy_mwh']:.3f} MWh")
        print("\nThese are the defensible operating points: the value is avoided")
        print("operating cost under a binding carbon target, not avoided penalty.")
    else:
        print("No cap in this range makes storage pay with slack == 0.")
        print("Either the caps are still loose (storage_value <= 0) or already so")
        print("tight that compliance needs slack, in which case the value is set by")
        print("validation_carbon_slack_dollars rather than by the physics.")
    print(f"\nwritten outputs/carbon_cap_scan.json")


if __name__ == "__main__":
    main()
