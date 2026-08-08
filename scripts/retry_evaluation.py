"""Retry only unconverged per-scenario exact evaluation solves."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")

from storage_dfl.config import load_config
from storage_dfl.planning import PlanningResult, StorageDesign, StoragePlanningOracle
from storage_dfl.stages import (
    _aggregate_scenario_wise_results,
    _bounded_result,
    _experiment_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/dataset_v2_dfl_hourly_layered.yaml"
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path(
            "outputs/dataset_v2_dfl_hourly_layered/result_reinforce.json"
        ),
    )
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--memory-limit-mb", type=float, default=8000.0)
    return parser.parse_args()


def result_from_dict(raw: dict) -> PlanningResult:
    design_raw = raw["design"]
    return PlanningResult(
        status=str(raw["status"]),
        objective=float(raw["objective"]),
        investment_cost=float(raw["investment_cost"]),
        operating_cost=float(raw["operating_cost"]),
        carbon_slack_cost=float(raw["carbon_slack_cost"]),
        peak_grid_mw=float(raw["peak_grid_mw"]),
        design=StorageDesign(
            site={key: int(value) for key, value in design_raw["site"].items()},
            power_mw={
                key: float(value) for key, value in design_raw["power_mw"].items()
            },
            energy_mwh={
                key: float(value) for key, value in design_raw["energy_mwh"].items()
            },
        ),
        scenario_names=tuple(raw["scenario_names"]),
        solve_time_seconds=float(raw["solve_time_seconds"]),
        relative_gap=float(raw["relative_gap"]),
    )


def retry_group(
    *,
    label: str,
    raw_results: list[dict],
    scenario_by_name: dict,
    oracle: StoragePlanningOracle,
    accepted_gap: float,
) -> tuple[list[PlanningResult], float, list[dict]]:
    results = [result_from_dict(raw) for raw in raw_results]
    retry_indices = [
        index
        for index, result in enumerate(results)
        if not _bounded_result(result, accepted_gap)
    ]
    history: list[dict] = []
    if not retry_indices:
        print(f"{label}: every scenario already satisfies the gap.", flush=True)
        return results, 0.0, history

    print(
        f"{label}: retrying {len(retry_indices)}/{len(results)} scenarios...",
        flush=True,
    )
    started = time.perf_counter()
    for ordinal, index in enumerate(retry_indices, start=1):
        previous = results[index]
        if len(previous.scenario_names) != 1:
            raise ValueError("Cached scenario result must contain exactly one name.")
        name = previous.scenario_names[0]
        scenario = scenario_by_name.get(name)
        if scenario is None:
            raise ValueError(f"Cached scenario {name!r} is not in the configured test split.")
        print(
            f"  {ordinal}/{len(retry_indices)} {name}: old "
            f"{previous.status}, gap={previous.relative_gap:.4%}",
            flush=True,
        )
        replacement = oracle.solve(
            (scenario,),
            fixed_design=previous.design,
            allow_carbon_slack=True,
        )
        results[index] = replacement
        history.append(
            {
                "scenario": name,
                "old_status": previous.status,
                "old_gap": previous.relative_gap,
                "new_status": replacement.status,
                "new_gap": replacement.relative_gap,
                "new_solve_seconds": replacement.solve_time_seconds,
            }
        )
        print(
            f"      new {replacement.status}, gap={replacement.relative_gap:.4%}, "
            f"objective={replacement.objective:.2f}",
            flush=True,
        )
    return results, time.perf_counter() - started, history


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    accepted_gap = float(
        payload.get("evaluation_relative_gap", config.planning.solver_relative_gap)
    )
    feeder, test_pool = _experiment_data(config, config.data.test_split)
    scenario_by_name = {scenario.name: scenario for scenario in test_pool.scenarios}
    names = tuple(payload["evaluation_scenario_names"])
    weights = tuple(float(value) for value in payload["evaluation_scenario_weights"])
    if len(names) != len(weights):
        raise ValueError("Cached evaluation names and weights have different lengths.")

    retry_planning = replace(
        config.planning,
        solver_time_limit_seconds=args.time_limit,
        solver_memory_limit_mb=args.memory_limit_mb,
        solver_max_parallel_workers=1,
    )
    storage_oracle = StoragePlanningOracle(
        feeder, retry_planning, config.costs, config.data, config.data_center
    )
    storage_results, storage_wall, storage_history = retry_group(
        label="storage design",
        raw_results=payload["out_of_sample_scenario_results"],
        scenario_by_name=scenario_by_name,
        oracle=storage_oracle,
        accepted_gap=accepted_gap,
    )
    storage_design = storage_results[0].design
    validation = _aggregate_scenario_wise_results(
        storage_results,
        weights,
        storage_design,
        config.costs.demand_dollars_per_mw_year,
        accepted_gap,
    )

    reference_oracle = StoragePlanningOracle(
        feeder,
        replace(retry_planning, max_storage_sites=0),
        config.costs,
        config.data,
        config.data_center,
    )
    reference_results, reference_wall, reference_history = retry_group(
        label="no-storage reference",
        raw_results=payload["no_storage_scenario_results"],
        scenario_by_name=scenario_by_name,
        oracle=reference_oracle,
        accepted_gap=accepted_gap,
    )
    no_storage_design = reference_results[0].design
    reference = _aggregate_scenario_wise_results(
        reference_results,
        weights,
        no_storage_design,
        config.costs.demand_dollars_per_mw_year,
        accepted_gap,
    )

    comparable = _bounded_result(validation, accepted_gap) and _bounded_result(
        reference, accepted_gap
    )
    payload["out_of_sample_validation"] = validation.to_dict()
    payload["out_of_sample_scenario_results"] = [
        result.to_dict() for result in storage_results
    ]
    payload["no_storage_reference"] = reference.to_dict()
    payload["no_storage_scenario_results"] = [
        result.to_dict() for result in reference_results
    ]
    payload["out_of_sample_wall_seconds"] = float(
        payload.get("out_of_sample_wall_seconds", 0.0)
    ) + storage_wall
    payload["no_storage_wall_seconds"] = float(
        payload.get("no_storage_wall_seconds", 0.0)
    ) + reference_wall
    payload["objectives_comparable"] = comparable
    payload["storage_value"] = (
        float(reference.objective) - float(validation.objective)
        if comparable
        else None
    )
    payload.setdefault("retry_history", []).append(
        {
            "time_limit_seconds": args.time_limit,
            "memory_limit_mb": args.memory_limit_mb,
            "storage": storage_history,
            "no_storage": reference_history,
        }
    )
    args.result.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"comparable      : {comparable}", flush=True)
    print(f"storage objective: {validation.objective:.2f}", flush=True)
    print(f"reference objective: {reference.objective:.2f}", flush=True)
    print(f"storage value   : {payload['storage_value']}", flush=True)
    print(f"updated         : {args.result.resolve()}", flush=True)


if __name__ == "__main__":
    main()
