"""Does tightening the carbon cap sharpen the top of the objective?

The K=3 control found the objective flat where it matters: 20 distinct designs
spanning 8x in energy, whose best five out-of-sample values sat within 10% of
each other. A learned selector cannot win a contest decided in that band.

Flatness is what a smooth objective does near an interior optimum, so the fix
has to come from a constraint that bites, and this project already has one. The
same subsets are scored at several caps -- paired, so the comparison is not
reading subset-sampling noise -- and the question is whether the spread among
the good designs grows as the cap tightens.

Each cap gets its own no-storage reference, because the reference moves with the
cap and values measured against different references cannot be compared.
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

import numpy as np

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
    parser.add_argument(
        "--config", default="configs/dataset_v2_dfl_hourly_layered.yaml"
    )
    parser.add_argument(
        "--caps",
        type=float,
        nargs="+",
        default=[0.28, 0.26, 0.24, 0.22],
        help="dc_carbon_cap values. The validation pool's mean carbon intensity "
        "is 0.283, so 0.28 barely binds and is the current setting.",
    )
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--evaluation-scenarios", type=int, default=3)
    parser.add_argument("--planning-gap", type=float, default=0.03)
    parser.add_argument("--evaluation-gap", type=float, default=0.01)
    parser.add_argument("--planning-time-limit", type=float, default=300.0)
    parser.add_argument("--evaluation-time-limit", type=float, default=240.0)
    parser.add_argument("--memory-limit-mb", type=float, default=5500.0)
    parser.add_argument("--out", default="outputs/carbon_cap_sharpness.json")
    return parser.parse_args()


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ArtifactPaths(config.output_dir)
    feeder, support_pool = _experiment_data(config, config.data.validation_split)
    _, test_pool = _experiment_data(config, config.data.test_split)
    codec = _load_codec(paths, feeder)

    count = min(args.evaluation_scenarios, len(test_pool.scenarios))
    evaluation, evaluation_weights, evaluation_names = select_scenarios(
        config.dfl.evaluation_selection_rule, test_pool, codec, count, seed=config.seed
    )

    # One subset list, scored at every cap. Paired comparison: any change in
    # spread is the cap's doing, not a different draw of supports.
    rng = np.random.default_rng(config.seed)
    subsets: list[tuple[int, ...]] = []
    seen: set[tuple[str, ...]] = set()
    while len(subsets) < args.draws:
        picks = tuple(
            int(i)
            for i in rng.choice(len(support_pool.scenarios), size=args.k, replace=False)
        )
        names = tuple(sorted(support_pool.scenarios[i].name for i in picks))
        if names in seen:
            continue
        seen.add(names)
        subsets.append(picks)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict = (
        json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    )
    # Time limits belong here as much as the gaps do: most solves stop on the
    # limit rather than on the gap, so the limit is what actually decides how
    # resolved a row is. Leaving them out let a resumed run append rows from a
    # different budget with nothing marking which was which.
    fingerprint = {
        "config": str(Path(args.config).resolve()),
        "k": args.k,
        "draws": args.draws,
        "seed": config.seed,
        "evaluation_names": list(evaluation_names),
        "planning_gap": args.planning_gap,
        "evaluation_gap": args.evaluation_gap,
        "planning_time_limit": args.planning_time_limit,
        "evaluation_time_limit": args.evaluation_time_limit,
        "evaluation_scenarios": count,
    }
    if report and report.get("fingerprint") != fingerprint:
        raise SystemExit(
            f"{out_path} was written under different settings; move it aside."
        )
    report.setdefault("fingerprint", fingerprint)
    report.setdefault("caps", {})

    def flush() -> None:
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    no_storage_design = StorageDesign(
        site={bus: 0 for bus in feeder.storage_candidates},
        power_mw={bus: 0.0 for bus in feeder.storage_candidates},
        energy_mwh={bus: 0.0 for bus in feeder.storage_candidates},
    )

    for cap in args.caps:
        key = f"{cap:.3f}"
        entry = report["caps"].setdefault(key, {"draws": []})
        planning_config = replace(
            config.planning,
            dc_carbon_cap=cap,
            solver_relative_gap=args.planning_gap,
            solver_time_limit_seconds=args.planning_time_limit,
            solver_memory_limit_mb=args.memory_limit_mb,
            solver_max_parallel_workers=1,
        )
        evaluation_config = replace(
            config.planning,
            dc_carbon_cap=cap,
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

        if "no_storage" not in entry:
            print(f"cap {cap:.2f}: no-storage reference...", flush=True)
            reference_oracle = StoragePlanningOracle(
                feeder,
                replace(evaluation_config, max_storage_sites=0),
                config.costs,
                config.data,
                config.data_center,
            )
            reference, reference_rows, _ = _solve_fixed_design_scenario_wise(
                reference_oracle,
                evaluation,
                evaluation_weights,
                no_storage_design,
                config.costs.demand_dollars_per_mw_year,
                args.evaluation_gap,
            )
            entry["no_storage"] = {
                "objective": _finite(reference.objective),
                "status": reference.status,
                "gap": _finite(max(r.relative_gap for r in reference_rows)),
            }
            flush()
        reference_objective = entry["no_storage"]["objective"]
        print(f"cap {cap:.2f}: reference {reference_objective:,.2f}", flush=True)

        for index, picks in enumerate(subsets):
            if index < len(entry["draws"]):
                continue
            supports = tuple(support_pool.scenarios[i] for i in picks)
            names = [s.name for s in supports]
            started = time.perf_counter()
            planned = planning_oracle.solve(
                supports,
                weights=tuple(1.0 / args.k for _ in picks),
                allow_carbon_slack=config.dfl.training_allow_carbon_slack,
            )
            record: dict = {
                "names": names,
                "planning_status": planned.status,
                "installed": list(planned.design.installed_buses) if planned.feasible else [],
            }
            if planned.feasible:
                aggregate, per_scenario, _ = _solve_fixed_design_scenario_wise(
                    evaluation_oracle,
                    evaluation,
                    evaluation_weights,
                    planned.design,
                    config.costs.demand_dollars_per_mw_year,
                    args.evaluation_gap,
                )
                record["energy_mwh"] = sum(
                    float(planned.design.energy_mwh[b])
                    for b in planned.design.installed_buses
                )
                record["decision_loss"] = _finite(aggregate.objective)
                record["evaluation_gap"] = _finite(
                    max(r.relative_gap for r in per_scenario)
                )
                record["storage_value"] = (
                    None
                    if record["decision_loss"] is None or reference_objective is None
                    else reference_objective - record["decision_loss"]
                )
            else:
                record["storage_value"] = None
            record["seconds"] = time.perf_counter() - started
            entry["draws"].append(record)
            flush()
            print(
                f"  cap {cap:.2f} draw {index + 1}/{len(subsets)}: "
                + (
                    f"{record['storage_value']:,.0f}"
                    if record.get("storage_value") is not None
                    else f"infeasible ({record['planning_status']})"
                )
                + f"   {record['seconds']:.0f}s",
                flush=True,
            )

    print()
    print("sharpness of the top, per cap")
    print(
        f"{'cap':>6}{'built':>8}{'best':>12}{'5th':>12}"
        f"{'best-5th':>11}{'as % best':>11}{'sd/mean':>10}"
    )
    print("-" * 70)
    for cap in args.caps:
        entry = report["caps"][f"{cap:.3f}"]
        values = np.asarray(
            [
                d["storage_value"]
                for d in entry["draws"]
                if d.get("storage_value") is not None
            ]
        )
        built = values[values > 1.0]
        if built.size == 0:
            print(f"{cap:>6.2f}{0:>8}{'-':>12}")
            continue
        top = np.sort(built)[::-1]
        fifth = top[min(4, top.size - 1)]
        spread = top[0] - fifth
        print(
            f"{cap:>6.2f}{built.size:>8}{top[0]:>12,.0f}{fifth:>12,.0f}"
            f"{spread:>11,.0f}{100.0 * spread / top[0]:>10.1f}%"
            f"{built.std(ddof=1) / built.mean() if built.size > 1 else 0.0:>10.2f}"
        )
    print()
    print(
        "A cap that sharpens the top widens 'best-5th'. If it stays near the\n"
        "10% the current setting shows, tightening the cap does not create the\n"
        "contest a learned selector would need."
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
