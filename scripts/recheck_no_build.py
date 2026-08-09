"""Was 'install nothing' the optimum, or the solver meeting its tolerance early?

Storage is worth about 1.86% of the planning objective, and the control ran at a
2% relative gap, so an empty design already satisfied the stopping rule. Three
of the five no-build draws came back 'optimal' in 34-41 s against a median of
911 s for draws that did install something.

This re-solves subsets already scored in the control -- the no-build ones and,
as a control on the control, some that did build -- with the relative tolerance
set to zero so it can never be met, leaving a small absolute tolerance and the
time limit as the only stopping rules. The solver then keeps improving its
incumbent instead of stopping, which is the behaviour the comparison needs; it
is not expected to prove anything.

Read the result one way only: a subset that now installs storage proves the
earlier no-build was an artifact. A subset that still installs nothing after a
full-length search is evidence, not proof, that no-build was genuine.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")

from storage_dfl.config import load_config
from storage_dfl.dfl import select_scenarios
from storage_dfl.planning import StorageDesign, StoragePlanningOracle
from storage_dfl.stages import (
    ArtifactPaths,
    _experiment_data,
    _load_codec,
    _solve_fixed_design_scenario_wise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset_v2_dfl_hourly_layered.yaml")
    parser.add_argument("--control", default="outputs/random_subset_control_k3.json")
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument(
        "--absolute-gap",
        type=float,
        default=2000.0,
        help="Dollars. Below the ~30,000 storage is worth, so the decision is "
        "inside the tolerance rather than swallowed by it.",
    )
    parser.add_argument("--planning-time-limit", type=float, default=900.0)
    parser.add_argument("--evaluation-gap", type=float, default=0.01)
    parser.add_argument("--evaluation-time-limit", type=float, default=300.0)
    parser.add_argument("--memory-limit-mb", type=float, default=5500.0)
    parser.add_argument("--out", default="outputs/recheck_no_build.json")
    return parser.parse_args()


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    control = json.loads(Path(args.control).read_text(encoding="utf-8"))
    reference_objective = control["baselines"]["no_storage"]["objective"]

    paths = ArtifactPaths(config.output_dir)
    feeder, support_pool = _experiment_data(config, config.data.validation_split)
    _, test_pool = _experiment_data(config, config.data.test_split)
    codec = _load_codec(paths, feeder)
    by_name = {s.name: s for s in support_pool.scenarios}

    # Reproduce the control's evaluation set exactly, so the values printed here
    # sit on the same scale as the ones being rechecked.
    evaluation_names = control["fingerprint"]["evaluation_names"]
    evaluation, evaluation_weights, produced = select_scenarios(
        config.dfl.evaluation_selection_rule,
        test_pool,
        codec,
        len(evaluation_names),
        seed=config.seed,
    )
    if list(produced) != list(evaluation_names):
        raise SystemExit(
            "The evaluation set no longer reproduces the control's; values would "
            f"not be comparable. control={evaluation_names} now={list(produced)}"
        )

    draws = [d for d in control["draws"] if d.get("decision_loss") is not None]
    empty = [d for d in draws if not d["design"]["installed_buses"]]
    built = sorted(
        (d for d in draws if d["design"]["installed_buses"]),
        key=lambda d: reference_objective - d["decision_loss"],
        reverse=True,
    )
    groups = [
        ("no_build", empty[: args.per_group]),
        ("built", built[: args.per_group]),
    ]

    planning_config = replace(
        config.planning,
        # Zero, so the relative rule can never fire. All three backends stop on
        # whichever criterion is met first, so leaving it at 2% would keep the
        # absolute tolerance decorative -- which is exactly the bug being tested.
        solver_relative_gap=0.0,
        solver_absolute_gap_dollars=args.absolute_gap,
        solver_time_limit_seconds=args.planning_time_limit,
        solver_memory_limit_mb=args.memory_limit_mb,
        solver_max_parallel_workers=1,
    )
    evaluation_config = replace(
        config.planning,
        solver_relative_gap=args.evaluation_gap,
        solver_time_limit_seconds=args.evaluation_time_limit,
        solver_memory_limit_mb=args.memory_limit_mb,
        solver_max_parallel_workers=4,
    )
    planning_oracle = StoragePlanningOracle(
        feeder, planning_config, config.costs, config.data, config.data_center
    )
    evaluation_oracle = StoragePlanningOracle(
        feeder, evaluation_config, config.costs, config.data, config.data_center
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "settings": {
            "relative_gap": 0.0,
            "absolute_gap_dollars": args.absolute_gap,
            "planning_time_limit": args.planning_time_limit,
            "control": str(Path(args.control).resolve()),
        },
        "no_storage_reference": reference_objective,
        "rows": [],
    }

    print(
        f"relative gap 0, absolute gap ${args.absolute_gap:,.0f}, "
        f"time limit {args.planning_time_limit:.0f}s",
        flush=True,
    )
    print(f"no-storage reference {reference_objective:,.2f}", flush=True)

    for group, entries in groups:
        for entry in entries:
            names = entry["names"]
            supports = tuple(by_name[n] for n in names)
            was_value = reference_objective - entry["decision_loss"]
            print(f"\n{group}: {names}", flush=True)
            print(
                f"  before: {entry['design']['installed_buses']} "
                f"value {was_value:,.0f} gap {entry['planning_gap']:.2%} "
                f"{entry['planning_status']} {entry['seconds']:.0f}s",
                flush=True,
            )
            started = time.perf_counter()
            planned = planning_oracle.solve(
                supports,
                weights=tuple(1.0 / len(supports) for _ in supports),
                allow_carbon_slack=config.dfl.training_allow_carbon_slack,
            )
            seconds = time.perf_counter() - started
            row = {
                "group": group,
                "names": names,
                "before": {
                    "installed": entry["design"]["installed_buses"],
                    "storage_value": was_value,
                    "planning_gap": entry["planning_gap"],
                    "planning_status": entry["planning_status"],
                    "seconds": entry["seconds"],
                },
                "after": {
                    "installed": list(planned.design.installed_buses),
                    "energy_mwh": sum(
                        float(planned.design.energy_mwh[b])
                        for b in planned.design.installed_buses
                    ),
                    "power_mw": sum(
                        float(planned.design.power_mw[b])
                        for b in planned.design.installed_buses
                    ),
                    "planning_gap": _finite(planned.relative_gap),
                    "planning_status": planned.status,
                    "seconds": seconds,
                },
            }
            if planned.feasible:
                aggregate, _, _ = _solve_fixed_design_scenario_wise(
                    evaluation_oracle,
                    evaluation,
                    evaluation_weights,
                    planned.design,
                    config.costs.demand_dollars_per_mw_year,
                    args.evaluation_gap,
                )
                row["after"]["storage_value"] = (
                    None
                    if not math.isfinite(aggregate.objective)
                    else reference_objective - aggregate.objective
                )
            else:
                row["after"]["storage_value"] = None
            report["rows"].append(row)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            after = row["after"]
            print(
                f"  after : {after['installed']} "
                + (
                    f"value {after['storage_value']:,.0f} "
                    if after["storage_value"] is not None
                    else "value n/a "
                )
                + f"gap {after['planning_gap']:.2%} {after['planning_status']} "
                f"{after['seconds']:.0f}s",
                flush=True,
            )

    print()
    print(f"{'group':<10}{'before':<22}{'after':<22}{'verdict'}")
    print("-" * 78)
    flipped = 0
    for row in report["rows"]:
        before_empty = not row["before"]["installed"]
        after_empty = not row["after"]["installed"]
        if before_empty and not after_empty:
            verdict = "ARTIFACT: now builds"
            flipped += 1
        elif before_empty and after_empty:
            verdict = "no-build survives"
        elif not before_empty and after_empty:
            verdict = "now declines to build"
        else:
            verdict = "builds both times"
        print(
            f"{row['group']:<10}"
            f"{str(row['before']['installed']):<22}"
            f"{str(row['after']['installed']):<22}{verdict}"
        )
    print()
    print(
        f"{flipped} of the no-build subsets installed storage once the relative\n"
        "tolerance was removed. Any number above zero means the control's\n"
        "no-build rows, and the bimodal shape built on them, are not safe."
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
