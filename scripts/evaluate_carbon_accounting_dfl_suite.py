"""Cross-evaluate independently trained carbon-accounting DFL runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from storage_dfl.config import load_config
from storage_dfl.data import ScenarioCodec
from storage_dfl.planning import StorageDesign, StoragePlanningOracle, make_planning_oracle
from storage_dfl.stages import (
    _experiment_data,
    _read_json,
    _require_current_codec,
    _solve_fixed_design_scenario_wise,
    _swanlab_writer,
    _write_json,
)

from run_carbon_accounting_ablation import _result_metrics, _test_set


def _design(payload: dict) -> StorageDesign:
    return StorageDesign(
        site={key: int(value) for key, value in payload["site"].items()},
        power_mw={key: float(value) for key, value in payload["power_mw"].items()},
        energy_mwh={key: float(value) for key, value in payload["energy_mwh"].items()},
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--no-swanlab", action="store_true")
    args = parser.parse_args()

    suite = args.suite.resolve()
    summary_path = suite / "summary.json"
    summary = _read_json(summary_path)
    runs = summary["runs"]
    if not runs or any(run.get("status") != "completed" for run in runs):
        raise RuntimeError("Every configured DFL run must complete before evaluation")

    config = load_config(runs[0]["config"])
    feeder, test_pool = _experiment_data(config, config.data.test_split)
    normalization = Path(runs[0]["output_dir"]) / "normalization.json"
    codec = ScenarioCodec.from_normalization_dict(_read_json(normalization), feeder)
    _require_current_codec(config, codec)
    evaluation = summary["evaluation"]
    test_scenarios, test_weights, test_names = _test_set(
        config, test_pool, codec, int(evaluation["test_scenarios"])
    )
    common_planning = replace(
        config.planning,
        carbon_formulation=evaluation["carbon_formulation"],
        carbon_cap_scope=evaluation["carbon_cap_scope"],
    )
    oracle = make_planning_oracle(
        feeder, common_planning, config.costs, config.data, config.data_center
    )
    output_dir = suite / "common_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    swanlab_writer = _swanlab_writer(
        output_dir / "swanlab",
        not args.no_swanlab,
        config,
        experiment_name=f"dfl-common-eval-{suite.name}"[:40],
        group="carbon-accounting-dfl-evaluation",
    )
    tolerance = float(evaluation["violation_tolerance_t_per_hour"])

    records = []
    run_count = len(runs)
    for index, run in enumerate(runs, start=1):
        result_path = output_dir / f"{run['id']}.json"
        if result_path.exists():
            record = _read_json(result_path)
            print(f"[{index}/{run_count}] Resume evaluated {run['label']}", flush=True)
        else:
            latest = _read_json(Path(run["output_dir"]) / "latest.json")
            training = _read_json(Path(latest["result"]))
            design = _design(training["planning"]["design"])
            print(
                f"[{index}/{run_count}] Evaluate independently trained {run['label']} "
                f"on {len(test_scenarios)} strict common scenarios...",
                flush=True,
            )
            aggregate, details, wall = _solve_fixed_design_scenario_wise(
                oracle,
                test_scenarios,
                test_weights,
                design,
                config.costs.demand_dollars_per_mw_year,
                common_planning.solver_relative_gap,
            )
            record = {
                "variant": run["id"],
                "label": run["label"],
                "training_carbon_formulation": run["carbon_formulation"],
                "training_carbon_cap_scope": run["carbon_cap_scope"],
                "checkpoint": run["checkpoint"],
                "training_best_epoch": training.get("best_epoch"),
                "training_decision_regret": training.get("decision_regret"),
                "training_planning": training["planning"],
                "evaluation_wall_seconds": wall,
                "common_test_evaluation": _result_metrics(
                    aggregate,
                    details,
                    test_weights,
                    config.data.horizon,
                    tolerance,
                ),
                "test_scenario_results": [item.to_dict() for item in details],
            }
            _write_json(result_path, record)
        records.append(record)
        metrics = record["common_test_evaluation"]
        for name in (
            "objective_dollars_per_year",
            "operating_dollars_per_year",
            "carbon_slack_dollars_per_year",
            "storage_power_mw",
            "storage_energy_mwh",
            "load_shedding_mwh_per_year",
            "carbon_excess_t_per_year",
            "equivalent_violation_hours_per_year",
        ):
            if swanlab_writer is not None and metrics[name] is not None:
                swanlab_writer.add_scalar(f"common_evaluation/{name}", metrics[name], index)

    reference = None
    reference_path = output_dir / "no_storage_reference.json"
    if bool(evaluation.get("include_no_storage_reference", True)):
        if reference_path.exists():
            reference = _read_json(reference_path)
        else:
            print("Evaluate common no-storage reference...", flush=True)
            reference_oracle = StoragePlanningOracle(
                feeder,
                replace(common_planning, max_storage_sites=0),
                config.costs,
                config.data,
                config.data_center,
            )
            empty = StorageDesign(
                site={bus: 0 for bus in feeder.storage_candidates},
                power_mw={bus: 0.0 for bus in feeder.storage_candidates},
                energy_mwh={bus: 0.0 for bus in feeder.storage_candidates},
            )
            aggregate, details, wall = _solve_fixed_design_scenario_wise(
                reference_oracle,
                test_scenarios,
                test_weights,
                empty,
                config.costs.demand_dollars_per_mw_year,
                common_planning.solver_relative_gap,
            )
            reference = {
                "evaluation_wall_seconds": wall,
                "common_test_evaluation": _result_metrics(
                    aggregate, details, test_weights, config.data.horizon, tolerance
                ),
                "test_scenario_results": [item.to_dict() for item in details],
            }
            _write_json(reference_path, reference)

    reference_objective = (
        None
        if reference is None
        else reference["common_test_evaluation"]["objective_dollars_per_year"]
    )
    rows = []
    for record in records:
        metrics = record["common_test_evaluation"]
        training = record["training_planning"]
        rows.append(
            {
                "variant": record["variant"],
                "label": record["label"],
                "training_carbon_formulation": record["training_carbon_formulation"],
                "training_carbon_cap_scope": record["training_carbon_cap_scope"],
                "best_epoch": record["training_best_epoch"],
                "training_decision_regret": record["training_decision_regret"],
                "training_objective_dollars_per_year": training["objective"],
                "power_mw": metrics["storage_power_mw"],
                "energy_mwh": metrics["storage_energy_mwh"],
                "investment_dollars_per_year": metrics["investment_dollars_per_year"],
                "operating_dollars_per_year": metrics["operating_dollars_per_year"],
                "carbon_slack_dollars_per_year": metrics["carbon_slack_dollars_per_year"],
                "total_dollars_per_year": metrics["objective_dollars_per_year"],
                "load_shedding_mwh_per_year": metrics["load_shedding_mwh_per_year"],
                "carbon_excess_t_per_year": metrics["carbon_excess_t_per_year"],
                "carbon_violation_hours_per_year": metrics["equivalent_violation_hours_per_year"],
                "savings_vs_no_storage_dollars_per_year": (
                    None
                    if reference_objective is None
                    else reference_objective - metrics["objective_dollars_per_year"]
                ),
            }
        )
    rows.sort(key=lambda row: row["total_dollars_per_year"])
    _write_csv(output_dir / "comparison.csv", rows)
    _write_json(
        output_dir / "summary.json",
        {
            "suite": str(suite),
            "common_evaluation": evaluation,
            "test_scenario_names": test_names,
            "comparison": rows,
            "no_storage_reference": reference,
        },
    )
    if swanlab_writer is not None:
        swanlab_writer.close()
    print(f"Completed common evaluation: {output_dir / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
