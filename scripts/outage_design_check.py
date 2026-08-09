"""Does an outage change the optimal design, or only the cost?

Adding outages is only worth anything if scenarios containing one lead the
planner somewhere different. If outage and normal supports produce the same
battery, the channel adds cost variance without adding decision diversity, and
selecting among the scenarios still cannot matter.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

from storage_dfl.config import load_config
from storage_dfl.planning import StorageDesign, StoragePlanningOracle
from storage_dfl.stages import _experiment_data, _solve_fixed_design_scenario_wise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset_v2_outage.yaml")
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--evaluation-scenarios", type=int, default=2)
    # Planning may stay loose: it only has to produce a design, and its own
    # objective never enters a comparison. Evaluation may not -- at 3% the same
    # design's measured storage value moved by $20,113 and changed sign.
    parser.add_argument("--planning-gap", type=float, default=0.03)
    parser.add_argument("--evaluation-gap", type=float, default=0.01)
    parser.add_argument("--planning-time-limit", type=float, default=300.0)
    parser.add_argument("--evaluation-time-limit", type=float, default=180.0)
    parser.add_argument("--out", default="outputs/outage_design_check.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    feeder, support_pool = _experiment_data(config, config.data.validation_split)
    _, test_pool = _experiment_data(config, config.data.test_split)

    outage_hours = np.asarray(
        [int((scenario.grid_available < 0.5).sum()) for scenario in support_pool.scenarios]
    )
    with_outage = np.flatnonzero(outage_hours > 0)
    without = np.flatnonzero(outage_hours == 0)
    if with_outage.size == 0:
        raise SystemExit(f"{config.data.dataset_path} carries no outages.")
    # Around the median duration, not the longest. Picking the longest reported
    # the tail as if it were the outage case and drove the battery to
    # max_duration_hours; what matters is whether a typical outage moves the
    # design.
    order = with_outage[np.argsort(outage_hours[with_outage])]
    middle = len(order) // 2
    start = max(0, middle - args.per_group // 2)
    ranked = order[start : start + args.per_group]
    normal = without[: args.per_group]

    # Every design is scored on the same all-available test scenarios, so any
    # difference reflects the design, not an outage in the evaluation set.
    evaluation_hours = np.asarray(
        [int((s.grid_available < 0.5).sum()) for s in test_pool.scenarios]
    )
    clean = [s for s, h in zip(test_pool.scenarios, evaluation_hours) if h == 0]
    evaluation = tuple(clean[: args.evaluation_scenarios])
    weights = tuple(1.0 / len(evaluation) for _ in evaluation)

    planning_config = replace(
        config.planning,
        solver_relative_gap=args.planning_gap,
        solver_time_limit_seconds=args.planning_time_limit,
        solver_max_parallel_workers=1,
    )
    evaluation_config = replace(
        config.planning,
        solver_relative_gap=args.evaluation_gap,
        solver_time_limit_seconds=args.evaluation_time_limit,
        solver_max_parallel_workers=4,
    )
    planning_oracle = StoragePlanningOracle(
        feeder, planning_config, config.costs, config.data, config.data_center
    )
    evaluation_oracle = StoragePlanningOracle(
        feeder, evaluation_config, config.costs, config.data, config.data_center
    )

    no_storage_design = StorageDesign(
        site={bus: 0 for bus in feeder.storage_candidates},
        power_mw={bus: 0.0 for bus in feeder.storage_candidates},
        energy_mwh={bus: 0.0 for bus in feeder.storage_candidates},
    )
    reference_oracle = StoragePlanningOracle(
        feeder,
        replace(evaluation_config, max_storage_sites=0),
        config.costs,
        config.data,
        config.data_center,
    )
    print("no-storage reference...", flush=True)
    reference, _, _ = _solve_fixed_design_scenario_wise(
        reference_oracle,
        evaluation,
        weights,
        no_storage_design,
        config.costs.demand_dollars_per_mw_year,
        args.evaluation_gap,
    )
    print(f"  {reference.objective:,.2f} ({reference.status})", flush=True)

    rows = []
    for group, indices in (("outage", ranked), ("normal", normal)):
        for index in indices:
            scenario = support_pool.scenarios[int(index)]
            hours = int(outage_hours[int(index)])
            print(f"{group} {scenario.name} (outage {hours}h): planning...", flush=True)
            planned = planning_oracle.solve(
                (scenario,),
                weights=(1.0,),
                allow_carbon_slack=config.dfl.training_allow_carbon_slack,
            )
            row = {
                "group": group,
                "name": scenario.name,
                "outage_hours": hours,
                "planning_status": planned.status,
                "installed": list(planned.design.installed_buses),
                "power_mw": sum(
                    float(planned.design.power_mw[b])
                    for b in planned.design.installed_buses
                ),
                "energy_mwh": sum(
                    float(planned.design.energy_mwh[b])
                    for b in planned.design.installed_buses
                ),
            }
            if planned.feasible:
                aggregate, _, _ = _solve_fixed_design_scenario_wise(
                    evaluation_oracle,
                    evaluation,
                    weights,
                    planned.design,
                    config.costs.demand_dollars_per_mw_year,
                    args.evaluation_gap,
                )
                row["storage_value"] = float(reference.objective - aggregate.objective)
            else:
                row["storage_value"] = None
            rows.append(row)
            print(
                f"  {row['installed']} {row['power_mw']:.3f} MW / "
                f"{row['energy_mwh']:.3f} MWh   value "
                + (
                    f"{row['storage_value']:,.0f}"
                    if row["storage_value"] is not None
                    else "n/a"
                ),
                flush=True,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"reference": reference.objective, "rows": rows}, indent=2
        ),
        encoding="utf-8",
    )

    print()
    print(f"{'group':<9}{'outage h':>10}{'MW':>9}{'MWh':>9}{'value':>13}")
    print("-" * 50)
    for row in rows:
        value = row["storage_value"]
        print(
            f"{row['group']:<9}{row['outage_hours']:>10}{row['power_mw']:>9.3f}"
            f"{row['energy_mwh']:>9.3f}"
            + (f"{value:>13,.0f}" if value is not None else f"{'n/a':>13}")
        )
    for group in ("outage", "normal"):
        sized = [r["energy_mwh"] for r in rows if r["group"] == group]
        print(f"mean {group} energy: {np.mean(sized):.3f} MWh")
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
