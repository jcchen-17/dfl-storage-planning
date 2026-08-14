"""Run fair real-scenario baselines for the single-PCC planning experiment.

Every method sees only the training split and receives the same support budget
K.  Its resulting storage design is fixed and re-dispatched on one shared,
held-out test set.  Random repetitions are written independently, so an
interrupted suite can be resumed without repeating completed solves.

Examples
--------
Quick smoke run::

    python scripts/run_scenario_selection_baselines.py --rules random --random-seeds 0

Main K=8 baseline suite::

    python scripts/run_scenario_selection_baselines.py \
      --config configs/dataset_v2_dfl_hourly_layered_t1.yaml \
      --rules random kmeans aggregate farthest \
      --random-seeds 0 1 2 3 4 5 6 7 8 9

Add the expensive all-training-scenarios reference::

    python scripts/run_scenario_selection_baselines.py --include-full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from storage_dfl.config import load_config
from storage_dfl.dfl import normalized_decision_regret, select_scenarios
from storage_dfl.planning import StorageDesign, make_planning_oracle
from storage_dfl.stages import _experiment_data, _fit_codec


DEFAULT_RULES = ("random", "kmeans", "aggregate", "farthest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/dataset_v2_dfl_hourly_layered_t1.yaml"
    )
    parser.add_argument("--k", type=int, default=None, help="support budget; config default is 8")
    parser.add_argument(
        "--rules", nargs="+", choices=DEFAULT_RULES, default=list(DEFAULT_RULES)
    )
    parser.add_argument(
        "--random-seeds", nargs="+", type=int, default=list(range(10))
    )
    parser.add_argument(
        "--cluster-seed",
        type=int,
        default=None,
        help="shared K-means/aggregate initialization seed; defaults to config.seed",
    )
    parser.add_argument(
        "--test-scenarios",
        default=None,
        help="number of shared test representatives, or 'all'; config default is 16",
    )
    parser.add_argument(
        "--test-rule",
        choices=DEFAULT_RULES,
        default=None,
        help="held-out test reduction rule; defaults to config evaluation_selection_rule",
    )
    parser.add_argument(
        "--include-full",
        action="store_true",
        help="also plan on all training scenarios (expensive computational reference)",
    )
    parser.add_argument(
        "--include-perfect-information",
        action="store_true",
        help="optimize directly on the shared test set (evaluation-only lower bound)",
    )
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--relative-gap", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--solver-backend",
        choices=("gurobi", "scip", "highs"),
        default=None,
        help="override the config backend; use one backend for the entire comparison",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="repeat completed jobs")
    return parser.parse_args()


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _bounded(result: Any, accepted_gap: float) -> bool:
    return bool(
        result.feasible
        and math.isfinite(float(result.objective))
        and (
            result.status in {"optimal", "gaplimit"}
            or (
                accepted_gap > 0.0
                and math.isfinite(float(result.relative_gap))
                and float(result.relative_gap) <= accepted_gap
            )
        )
    )


def _zero_design(feeder: Any) -> StorageDesign:
    return StorageDesign(
        site={bus: 0 for bus in feeder.storage_candidates},
        power_mw={bus: 0.0 for bus in feeder.storage_candidates},
        energy_mwh={bus: 0.0 for bus in feeder.storage_candidates},
    )


def _evaluate_design(
    oracle: Any,
    scenarios: tuple,
    weights: tuple[float, ...],
    design: StorageDesign,
) -> tuple[Any, list[Any], float]:
    from storage_dfl.planning import evaluate_fixed_design_recourse

    started = time.perf_counter()
    aggregate, individual = evaluate_fixed_design_recourse(
        oracle,
        scenarios,
        weights,
        design,
        allow_carbon_slack=True,
        use_cache=False,
        accepted_relative_gap=oracle.planning.solver_relative_gap,
    )
    return aggregate, individual, time.perf_counter() - started


def _row(record: dict[str, Any]) -> dict[str, Any]:
    planning = record["planning"]
    evaluation = record["test_evaluation"]
    diagnostics = evaluation.get("recourse_diagnostics") or {}
    design = evaluation["design"]
    bus = next(iter(design["site"]))
    return {
        "method": record["method"],
        "seed": record.get("seed"),
        "k": record["k"],
        "planning_objective": planning["objective"],
        "planning_status": planning["status"],
        "planning_gap": planning["relative_gap"],
        "planning_seconds": planning["solve_time_seconds"],
        "power_mw": design["power_mw"][bus],
        "energy_mwh": design["energy_mwh"][bus],
        "test_objective": evaluation["objective"],
        "test_status": evaluation["status"],
        "test_gap": evaluation["relative_gap"],
        "test_wall_seconds": record["test_wall_seconds"],
        "storage_value": record["storage_value"],
        "decision_regret": record.get("decision_regret"),
        "load_shedding_mwh": diagnostics.get("load_shedding_mwh"),
        "carbon_excess_t": diagnostics.get("carbon_excess_t"),
        "pv_curtailment_mwh": diagnostics.get("pv_curtailment_mwh"),
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = [_row(record) for record in records]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    planning = config.planning
    if args.time_limit is not None:
        planning = replace(planning, solver_time_limit_seconds=args.time_limit)
    if args.relative_gap is not None:
        planning = replace(planning, solver_relative_gap=args.relative_gap)
    if args.workers is not None:
        planning = replace(planning, solver_max_parallel_workers=max(1, args.workers))
    if args.solver_backend is not None:
        planning = replace(planning, solver_backend=args.solver_backend)
    config = replace(config, planning=planning)

    k = int(args.k or config.dfl.num_support_scenarios)
    if k <= 0:
        raise SystemExit("--k must be positive")
    cluster_seed = int(config.seed if args.cluster_seed is None else args.cluster_seed)
    test_rule = args.test_rule or config.dfl.evaluation_selection_rule
    if args.test_scenarios is None:
        test_count = int(config.dfl.final_validation_size)
    elif args.test_scenarios.lower() == "all":
        test_count = None
    else:
        test_count = int(args.test_scenarios)
        if test_count <= 0:
            raise SystemExit("--test-scenarios must be positive or 'all'")

    label = config_path.stem.removeprefix("dataset_v2_dfl_")
    output_dir = args.output_dir or Path("outputs/baselines/scenario_selection") / label
    output_dir.mkdir(parents=True, exist_ok=True)

    feeder, train_pool = _experiment_data(config, config.data.train_split)
    _, test_pool = _experiment_data(config, config.data.test_split)
    codec = _fit_codec(config, train_pool, feeder)
    if test_count is None or test_count >= len(test_pool.scenarios):
        test_scenarios = test_pool.scenarios
        test_weights = test_pool.normalized_weights()
        test_names = tuple(scenario.name for scenario in test_scenarios)
    else:
        test_scenarios, test_weights, test_names = select_scenarios(
            test_rule, test_pool, codec, test_count, seed=cluster_seed
        )

    settings = {
        "config": str(config_path),
        "k": k,
        "test_rule": test_rule,
        "test_scenario_names": list(test_names),
        "solver_backend": config.planning.solver_backend,
        "solver_time_limit_seconds": config.planning.solver_time_limit_seconds,
        "solver_relative_gap": config.planning.solver_relative_gap,
    }

    oracle = make_planning_oracle(
        feeder, config.planning, config.costs, config.data, config.data_center
    )
    print(
        f"Shared test set: {len(test_scenarios)} {test_rule} scenarios from "
        f"split={config.data.test_split}",
        flush=True,
    )
    reference_path = output_dir / "shared_no_storage.json"
    if reference_path.exists() and not args.force:
        reference_payload = json.loads(reference_path.read_text(encoding="utf-8"))
        # Recompute if the shared evaluation set changed.
        if reference_payload.get("settings") != settings:
            reference_payload = None
    else:
        reference_payload = None
    if reference_payload is None:
        print("Computing shared no-storage reference...", flush=True)
        reference, reference_results, reference_wall = _evaluate_design(
            oracle, test_scenarios, test_weights, _zero_design(feeder)
        )
        reference_payload = {
            "settings": settings,
            "scenario_names": list(test_names),
            "scenario_weights": list(test_weights),
            "result": reference.to_dict(),
            "scenario_results": [item.to_dict() for item in reference_results],
            "wall_seconds": reference_wall,
        }
        _write_json(reference_path, reference_payload)
    reference_objective = reference_payload["result"]["objective"]

    perfect_information = None
    pi_path = output_dir / "perfect_information.json"
    if args.include_perfect_information:
        if pi_path.exists() and not args.force:
            perfect_information = json.loads(pi_path.read_text(encoding="utf-8"))["result"]
        else:
            print("Computing perfect-information reference...", flush=True)
            pi = oracle.solve(
                test_scenarios,
                weights=test_weights,
                allow_carbon_slack=True,
                use_cache=False,
            )
            perfect_information = pi.to_dict()
            _write_json(
                pi_path,
                {"scenario_names": list(test_names), "result": perfect_information},
            )

    jobs: list[tuple[str, int | None]] = []
    for rule in args.rules:
        if rule == "random":
            jobs.extend((rule, seed) for seed in dict.fromkeys(args.random_seeds))
        else:
            jobs.append((rule, cluster_seed))
    if args.include_full:
        jobs.append(("full", None))

    for rule, seed in jobs:
        stem = rule if seed is None else f"{rule}_seed_{seed}"
        result_path = output_dir / "runs" / f"{stem}.json"
        if result_path.exists() and not args.force:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("settings") == settings and "test_evaluation" in existing:
                print(f"SKIP completed {stem}", flush=True)
                continue
            raise SystemExit(
                f"{result_path} was produced with different settings or is incomplete. "
                "Use a different --output-dir, or pass --force to replace it."
            )
        if rule == "full":
            supports = train_pool.scenarios
            support_weights = train_pool.normalized_weights()
            support_names = tuple(scenario.name for scenario in supports)
        else:
            supports, support_weights, support_names = select_scenarios(
                rule, train_pool, codec, k, seed=int(seed or 0)
            )
        print(f"RUN {stem}: planning with {len(supports)} supports...", flush=True)
        planning_started = time.perf_counter()
        planned = oracle.solve(
            supports,
            weights=support_weights,
            allow_carbon_slack=config.dfl.training_allow_carbon_slack,
            use_cache=False,
        )
        planning_wall = time.perf_counter() - planning_started
        if not planned.feasible:
            payload = {
                "settings": settings,
                "method": rule,
                "seed": seed,
                "k": len(supports),
                "support_names": list(support_names),
                "support_weights": list(support_weights),
                "planning": planned.to_dict(),
                "error": "planning did not return a feasible incumbent",
            }
            _write_json(result_path, payload)
            print(f"FAILED {stem}: planning status={planned.status}", flush=True)
            continue
        print(f"RUN {stem}: evaluating fixed design on held-out test...", flush=True)
        evaluated, scenario_results, evaluation_wall = _evaluate_design(
            oracle, test_scenarios, test_weights, planned.design
        )
        comparable = _bounded(planned, config.planning.solver_relative_gap) and _bounded(
            evaluated, config.planning.solver_relative_gap
        )
        storage_value = (
            float(reference_objective) - float(evaluated.objective)
            if _bounded(evaluated, config.planning.solver_relative_gap)
            else None
        )
        regret = None
        if perfect_information is not None and evaluated.feasible:
            class _Reference:
                feasible = True
                objective = float(perfect_information["objective"])

            regret = normalized_decision_regret(evaluated, _Reference())
        payload = {
            "settings": settings,
            "method": rule,
            "seed": seed,
            "k": len(supports),
            "config": str(config_path),
            "support_split": config.data.train_split,
            "support_names": list(support_names),
            "support_weights": list(support_weights),
            "test_split": config.data.test_split,
            "test_rule": test_rule,
            "test_scenario_names": list(test_names),
            "test_scenario_weights": list(test_weights),
            "planning": planned.to_dict(),
            "planning_wall_seconds": planning_wall,
            "test_evaluation": evaluated.to_dict(),
            "test_scenario_results": [item.to_dict() for item in scenario_results],
            "test_wall_seconds": evaluation_wall,
            "no_storage_objective": reference_objective,
            "storage_value": storage_value,
            "objectives_comparable": comparable,
            "decision_regret": regret,
        }
        _write_json(result_path, payload)
        print(
            f"DONE {stem}: test={evaluated.objective:.2f}, "
            f"storage_value={storage_value}, gap={evaluated.relative_gap:.4g}",
            flush=True,
        )

    records = []
    for path in sorted((output_dir / "runs").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "test_evaluation" in record:
            records.append(record)
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "settings": settings,
        "config": str(config_path),
        "k": k,
        "training_scenarios": len(train_pool.scenarios),
        "test_rule": test_rule,
        "test_scenarios": len(test_scenarios),
        "test_scenario_names": list(test_names),
        "solver_time_limit_seconds": config.planning.solver_time_limit_seconds,
        "solver_relative_gap": config.planning.solver_relative_gap,
        "no_storage_reference": reference_payload["result"],
        "perfect_information_reference": perfect_information,
        "runs": records,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "summary.csv", records)
    print(f"Summary: {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
