"""Plan and evaluate exact K=1 baselines against the trained DFL design."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import torch

from storage_dfl.config import load_config
from storage_dfl.data import Scenario
from storage_dfl.dfl import resolve_device, select_scenarios, support_init_indices
from storage_dfl.planning import StorageDesign, StoragePlanningOracle
from storage_dfl.stages import (
    ArtifactPaths,
    _bounded_result,
    _experiment_data,
    _load_codec,
    _method_tag,
    _solve_fixed_design_scenario_wise,
    load_generator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/dataset_v2_dfl_hourly_layered.yaml"
    )
    parser.add_argument("--scenarios", type=int, default=8)
    parser.add_argument("--planning-gap", type=float, default=0.02)
    parser.add_argument("--evaluation-gap", type=float, default=0.01)
    parser.add_argument("--planning-time-limit", type=float, default=180.0)
    parser.add_argument("--evaluation-time-limit", type=float, default=180.0)
    parser.add_argument("--memory-limit-mb", type=float, default=5500.0)
    parser.add_argument("--capex-scale", type=float, default=1.0)
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def design_from_dict(raw: dict) -> StorageDesign:
    return StorageDesign(
        site={bus: int(value) for bus, value in raw["site"].items()},
        power_mw={bus: float(value) for bus, value in raw["power_mw"].items()},
        energy_mwh={bus: float(value) for bus, value in raw["energy_mwh"].items()},
    )


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def design_size(design: StorageDesign) -> tuple[float, float]:
    return (
        sum(float(design.power_mw[bus]) for bus in design.installed_buses),
        sum(float(design.energy_mwh[bus]) for bus in design.installed_buses),
    )


def mean_scenario(scenarios: tuple[Scenario, ...]) -> Scenario:
    """Componentwise validation-set mean used as a transparent K=1 baseline."""

    def mean(field: str) -> np.ndarray:
        return np.mean(
            np.stack([np.asarray(getattr(scenario, field)) for scenario in scenarios]),
            axis=0,
        )

    # Availability is binary rather than a divisible physical quantity. A grid
    # interval is available in the mean scenario when it is available for the
    # majority of historical scenarios.
    availability = (mean("grid_available") >= 0.5).astype(float)
    return Scenario(
        name="validation_componentwise_mean_k1",
        context=mean("context"),
        active_load_mw=mean("active_load_mw"),
        reactive_load_mvar=mean("reactive_load_mvar"),
        pv_available_mw=mean("pv_available_mw"),
        workload_arrival=mean("workload_arrival"),
        pue=mean("pue"),
        grid_price_per_mwh=mean("grid_price_per_mwh"),
        grid_carbon_t_per_mwh=mean("grid_carbon_t_per_mwh"),
        grid_available=availability,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.capex_scale <= 0.0:
        raise ValueError("capex-scale must be positive.")
    config = replace(
        config,
        costs=replace(config.costs, battery_capex_scale=args.capex_scale),
    )
    paths = ArtifactPaths(config.output_dir)
    feeder, validation_pool = _experiment_data(config, config.data.validation_split)
    _, test_pool = _experiment_data(config, config.data.test_split)
    codec = _load_codec(paths, feeder)

    count = min(args.scenarios, len(test_pool.scenarios))
    evaluation, evaluation_weights, evaluation_names = select_scenarios(
        config.dfl.evaluation_selection_rule,
        test_pool,
        codec,
        count,
        seed=config.seed,
    )
    planning_config = replace(
        config.planning,
        solver_relative_gap=args.planning_gap,
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

    print(
        f"exact baseline comparison: {count} test scenarios; "
        f"planning gap={args.planning_gap:.2%}; evaluation gap={args.evaluation_gap:.2%}",
        flush=True,
    )

    supports: list[tuple[str, tuple[Scenario, ...], tuple[float, ...], list[str]]] = []
    real_support, real_weights, real_names = select_scenarios(
        "kmeans", validation_pool, codec, 1, seed=config.seed
    )
    supports.append(("real_kmeans_k1", real_support, real_weights, real_names))

    trajectories, contexts = codec.encode_pool(validation_pool)
    initial_index = support_init_indices(
        config.dfl.support_init_rule, validation_pool, codec, 1, seed=config.seed
    )
    device = resolve_device(config.dfl.device)
    generator = load_generator(paths, config.generator.kind, device)
    condition = torch.as_tensor(
        contexts[initial_index], dtype=torch.float32, device=device
    )
    with torch.no_grad():
        latent, _ = generator.encode(
            torch.as_tensor(
                trajectories[initial_index], dtype=torch.float32, device=device
            ),
            condition,
        )
        decoded = generator.decode(latent, condition).cpu().numpy()
    initial_generated = codec.decode_batch(
        decoded, contexts[initial_index], name_prefix="cvae_initial"
    )
    supports.append(
        (
            "cvae_initial_k1",
            initial_generated,
            (1.0,),
            [validation_pool.scenarios[int(initial_index[0])].name],
        )
    )
    supports.append(
        (
            "mean_scenario_k1",
            (mean_scenario(validation_pool.scenarios),),
            (1.0,),
            ["all_validation_scenarios"],
        )
    )

    designs: list[tuple[str, StorageDesign, dict | None, list[str]]] = []
    dfl_tag = _method_tag(config)
    dfl_result_path = paths.dfl_json_for("dfl_result", dfl_tag)
    if not dfl_result_path.exists():
        raise FileNotFoundError(
            f"Missing trained DFL result: {dfl_result_path}. Train DFL first."
        )
    dfl_result = json.loads(dfl_result_path.read_text(encoding="utf-8"))
    dfl_design = design_from_dict(dfl_result["planning"]["design"])
    designs.append(
        (
            "dfl_cvae_k1",
            dfl_design,
            dfl_result["planning"],
            list(dfl_result.get("support_source_names", [])),
        )
    )

    for label, scenarios, weights, source_names in supports:
        print(f"{label}: exact planning...", flush=True)
        planned = planning_oracle.solve(
            scenarios,
            weights=weights,
            allow_carbon_slack=config.dfl.training_allow_carbon_slack,
        )
        if not planned.feasible:
            print(
                f"  WARNING: no incumbent ({planned.status}, "
                f"gap={planned.relative_gap})",
                flush=True,
            )
        designs.append((label, planned.design, planned.to_dict(), source_names))

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
    print("no_storage: scenario-wise exact evaluation...", flush=True)
    no_storage, no_storage_results, no_storage_wall = (
        _solve_fixed_design_scenario_wise(
            reference_oracle,
            evaluation,
            evaluation_weights,
            no_storage_design,
            config.costs.demand_dollars_per_mw_year,
            args.evaluation_gap,
        )
    )

    rows: list[dict] = []
    for label, design, planned, source_names in designs:
        print(f"{label}: scenario-wise exact evaluation...", flush=True)
        aggregate, scenario_results, wall_seconds = (
            _solve_fixed_design_scenario_wise(
                evaluation_oracle,
                evaluation,
                evaluation_weights,
                design,
                config.costs.demand_dollars_per_mw_year,
                args.evaluation_gap,
            )
        )
        power, energy = design_size(design)
        comparable = _bounded_result(
            aggregate, args.evaluation_gap
        ) and _bounded_result(no_storage, args.evaluation_gap)
        value = (
            float(no_storage.objective) - float(aggregate.objective)
            if comparable
            else None
        )
        rows.append(
            {
                "method": label,
                "support_source_names": source_names,
                "planning": planned,
                "design": {
                    "site": design.site,
                    "power_mw": design.power_mw,
                    "energy_mwh": design.energy_mwh,
                },
                "power_mw": power,
                "energy_mwh": energy,
                "out_of_sample": aggregate.to_dict(),
                "scenario_results": [result.to_dict() for result in scenario_results],
                "wall_seconds": wall_seconds,
                "objectives_comparable": comparable,
                "storage_value_vs_no_storage": value,
            }
        )
        print(
            f"  buses={design.installed_buses}; P={power:.4f} MW; E={energy:.4f} MWh; "
            f"OOS={aggregate.objective:,.2f}; gap={aggregate.relative_gap:.2%}; "
            f"value={value}",
            flush=True,
        )

    comparable_rows = [row for row in rows if row["objectives_comparable"]]
    best_objective = (
        min(float(row["out_of_sample"]["objective"]) for row in comparable_rows)
        if comparable_rows
        else None
    )
    for row in rows:
        row["regret_vs_best_comparable"] = (
            float(row["out_of_sample"]["objective"]) - best_objective
            if best_objective is not None and row["objectives_comparable"]
            else None
        )

    payload = {
        "config": str(Path(args.config)),
        "carbon_formulation": evaluation_config.carbon_formulation,
        "storage_service_mode": evaluation_config.storage_service_mode,
        "planning_relative_gap": args.planning_gap,
        "evaluation_relative_gap": args.evaluation_gap,
        "planning_time_limit_seconds": args.planning_time_limit,
        "evaluation_time_limit_seconds": args.evaluation_time_limit,
        "battery_capex_scale": args.capex_scale,
        "evaluation_scenario_names": evaluation_names,
        "evaluation_scenario_weights": list(evaluation_weights),
        "no_storage": no_storage.to_dict(),
        "no_storage_scenario_results": [
            result.to_dict() for result in no_storage_results
        ],
        "no_storage_wall_seconds": no_storage_wall,
        "rows": rows,
    }
    scale_tag = f"{100.0 * args.capex_scale:g}".replace(".", "p")
    tag = args.tag or f"exact_{count}scenarios_capex{scale_tag}"
    json_path = paths.root / f"baselines_v2_{tag}.json"
    csv_path = paths.root / f"baselines_v2_{tag}.csv"
    json_path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "method",
                "installed_buses",
                "power_mw",
                "energy_mwh",
                "objective",
                "investment_cost",
                "operating_cost",
                "carbon_slack_cost",
                "peak_grid_mw",
                "relative_gap",
                "objectives_comparable",
                "storage_value_vs_no_storage",
                "regret_vs_best_comparable",
                "wall_seconds",
            ),
        )
        writer.writeheader()
        for row in rows:
            result = row["out_of_sample"]
            writer.writerow(
                {
                    "method": row["method"],
                    "installed_buses": ";".join(row["design"]["site"] and [
                        bus for bus, value in row["design"]["site"].items() if value
                    ]),
                    "power_mw": row["power_mw"],
                    "energy_mwh": row["energy_mwh"],
                    "objective": result["objective"],
                    "investment_cost": result["investment_cost"],
                    "operating_cost": result["operating_cost"],
                    "carbon_slack_cost": result["carbon_slack_cost"],
                    "peak_grid_mw": result["peak_grid_mw"],
                    "relative_gap": result["relative_gap"],
                    "objectives_comparable": row["objectives_comparable"],
                    "storage_value_vs_no_storage": row[
                        "storage_value_vs_no_storage"
                    ],
                    "regret_vs_best_comparable": row[
                        "regret_vs_best_comparable"
                    ],
                    "wall_seconds": row["wall_seconds"],
                }
            )
    print(f"saved JSON: {json_path}", flush=True)
    print(f"saved CSV : {csv_path}", flush=True)


if __name__ == "__main__":
    main()
