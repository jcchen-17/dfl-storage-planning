"""Run the K=8 carbon-accounting/cap-scope planning ablation.

The script deliberately does not call ``evaluate_stage``: that entry point
evaluates the design selected during DFL training.  This experiment instead
decodes the checkpoint's support once, replans a design for each accounting/
cap-scope pair, and evaluates every design under one common strict model.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from storage_dfl.config import load_config
from storage_dfl.data import ScenarioPool, ScenarioCodec
from storage_dfl.dfl import resolve_device, select_scenarios
from storage_dfl.models import build_generator, generator_from_checkpoint
from storage_dfl.planning import StorageDesign, StoragePlanningOracle, make_planning_oracle
from storage_dfl.stages import (
    _experiment_data,
    _load_torch,
    _read_json,
    _require_current_codec,
    _solve_fixed_design_scenario_wise,
    _swanlab_writer,
    _write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    required = {"base_config", "output_dir", "compute", "common_evaluation", "variants"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Experiment config is missing: {sorted(missing)}")
    pairs = {
        (str(item["carbon_formulation"]), str(item["carbon_cap_scope"]))
        for item in payload["variants"]
    }
    allowed = {
        ("layered_pcc", "hourly"),
        ("layered_pcc", "horizon"),
        ("average_pcc", "hourly"),
        ("average_pcc", "horizon"),
    }
    if not pairs or not pairs.issubset(allowed) or len(pairs) != len(payload["variants"]):
        raise ValueError("variants must be unique supported accounting/scope pairs")
    return payload


def _resolve_from_suite(suite: Path, seed: int | None) -> tuple[Path, Path | None]:
    summary_path = suite / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Suite summary not found: {summary_path}")
    summary = _read_json(summary_path)
    matches = [
        run
        for run in summary.get("runs", [])
        if run.get("variant") == "dfl_full"
        and run.get("status") == "completed"
        and (seed is None or int(run.get("seed")) == seed)
    ]
    if not matches:
        raise RuntimeError("No completed dfl_full run matches the requested suite/seed")
    if len(matches) > 1 and seed is None:
        seeds = sorted(int(run["seed"]) for run in matches)
        raise RuntimeError(f"Suite has multiple completed dfl_full seeds {seeds}; pass --seed")
    run = matches[0]
    latest = _read_json(Path(run["output_dir"]) / "latest.json")
    checkpoint = _resolve_project_path(latest["checkpoint"])
    effective_config = _resolve_project_path(run["effective_config"])
    return checkpoint, effective_config


def _find_normalization(checkpoint: Path) -> Path:
    for parent in checkpoint.parents:
        candidate = parent / "normalization.json"
        if candidate.exists():
            return candidate.resolve()
        if parent == PROJECT_ROOT.parent:
            break
    raise FileNotFoundError(
        "Could not find normalization.json above the checkpoint; pass --normalization"
    )


def _decode_support(checkpoint: dict, config, codec: ScenarioCodec, device):
    if "generator_checkpoint" in checkpoint:
        generator = generator_from_checkpoint(checkpoint["generator_checkpoint"], device)
    elif "fine_tuned_generator_state_dict" in checkpoint:
        generator = build_generator(config, codec.trajectory_dim, codec.context_dim).to(device)
        generator.load_state_dict(checkpoint["fine_tuned_generator_state_dict"])
        generator.freeze()
    else:
        raise RuntimeError("Checkpoint does not embed the generator needed to decode support")

    latent = checkpoint["support_latent"].to(device=device, dtype=torch.float32)
    conditions = checkpoint["support_conditions"].to(device=device, dtype=torch.float32)
    with torch.no_grad():
        decoded = generator.decode(latent, conditions).cpu().numpy()
    scenarios = codec.decode_batch(
        decoded, conditions.cpu().numpy(), name_prefix="carbon_ablation_k8"
    )
    if "support_grid_available" in checkpoint:
        availability = checkpoint["support_grid_available"].cpu().numpy()
        scenarios = tuple(
            replace(
                scenario,
                grid_available=np.asarray(availability[index], dtype=float),
                annual_occurrences=None,
            )
            for index, scenario in enumerate(scenarios)
        )
    weights = tuple(float(value) for value in checkpoint["scenario_weights"].cpu())
    if len(scenarios) != 8:
        raise RuntimeError(f"This experiment requires K=8 support, found K={len(scenarios)}")
    return scenarios, weights


def _test_set(config, pool: ScenarioPool, codec: ScenarioCodec, count: int):
    count = min(int(count), len(pool.scenarios))
    if count == len(pool.scenarios):
        return pool.scenarios, pool.normalized_weights(), tuple(
            scenario.name for scenario in pool.scenarios
        )
    return select_scenarios(
        config.dfl.evaluation_selection_rule, pool, codec, count, seed=config.seed
    )


def _design_totals(design: StorageDesign) -> tuple[float, float]:
    return sum(design.power_mw.values()), sum(design.energy_mwh.values())


def _result_metrics(result, scenario_results, weights, horizon: int, tolerance: float):
    diagnostics = result.recourse_diagnostics
    violation_hours_block = 0.0 if scenario_results else None
    status_counts: dict[str, int] = {}
    for scenario_result, weight in zip(scenario_results, weights, strict=True):
        status_counts[scenario_result.status] = status_counts.get(scenario_result.status, 0) + 1
        detail = scenario_result.recourse_diagnostics
        if detail is None:
            continue
        values = np.asarray(detail.hourly_carbon_excess_t_per_hour, dtype=float).reshape(-1)
        assert violation_hours_block is not None
        violation_hours_block += float(weight) * float(np.count_nonzero(values > tolerance))
    power, energy = _design_totals(result.design)
    return {
        "status": result.status,
        "relative_gap": float(result.relative_gap),
        "objective_dollars_per_year": float(result.objective),
        "investment_dollars_per_year": float(result.investment_cost),
        "operating_dollars_per_year": float(result.operating_cost),
        "carbon_slack_dollars_per_year": float(result.carbon_slack_cost),
        "storage_power_mw": power,
        "storage_energy_mwh": energy,
        "peak_grid_mw": float(result.peak_grid_mw),
        "load_shedding_mwh_per_year": (
            None if diagnostics is None else float(diagnostics.load_shedding_mwh)
        ),
        "carbon_excess_t_per_year": (
            None if diagnostics is None else float(diagnostics.carbon_excess_t)
        ),
        "expected_violation_hours_per_48h": violation_hours_block,
        "equivalent_violation_hours_per_year": (
            None
            if violation_hours_block is None
            else violation_hours_block * 8760.0 / horizon
        ),
        "scenario_status_counts": status_counts,
    }


def _csv_row(record: dict[str, Any], reference_objective: float | None) -> dict[str, Any]:
    native = record["planning_on_k8_support"]
    common = record["common_test_evaluation"]
    row = {
        "variant": record["variant"],
        "label": record["label"],
        "planning_carbon_formulation": record["planning_carbon_formulation"],
        "planning_carbon_cap_scope": record["planning_carbon_cap_scope"],
        "power_mw": common["storage_power_mw"],
        "energy_mwh": common["storage_energy_mwh"],
        "planning_status": native["status"],
        "planning_gap": native["relative_gap"],
        "test_status": common["status"],
        "test_gap": common["relative_gap"],
        "investment_dollars_per_year": common["investment_dollars_per_year"],
        "operating_dollars_per_year": common["operating_dollars_per_year"],
        "carbon_slack_dollars_per_year": common["carbon_slack_dollars_per_year"],
        "total_dollars_per_year": common["objective_dollars_per_year"],
        "load_shedding_mwh_per_year": common["load_shedding_mwh_per_year"],
        "carbon_excess_t_per_year": common["carbon_excess_t_per_year"],
        "carbon_violation_hours_per_year": common["equivalent_violation_hours_per_year"],
        "savings_vs_no_storage_dollars_per_year": (
            None if reference_objective is None else reference_objective - common["objective_dollars_per_year"]
        ),
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _log_variant(writer, record: dict[str, Any], step: int) -> None:
    if writer is None:
        return
    planned = record["planning_on_k8_support"]
    tested = record["common_test_evaluation"]
    values = {
        "ablation/planning_objective_dollars_per_year": planned["objective_dollars_per_year"],
        "ablation/test_total_dollars_per_year": tested["objective_dollars_per_year"],
        "ablation/test_operating_dollars_per_year": tested["operating_dollars_per_year"],
        "ablation/test_carbon_slack_dollars_per_year": tested["carbon_slack_dollars_per_year"],
        "ablation/storage_power_mw": tested["storage_power_mw"],
        "ablation/storage_energy_mwh": tested["storage_energy_mwh"],
        "ablation/load_shedding_mwh_per_year": tested["load_shedding_mwh_per_year"],
        "ablation/carbon_excess_t_per_year": tested["carbon_excess_t_per_year"],
        "ablation/carbon_violation_hours_per_year": tested["equivalent_violation_hours_per_year"],
    }
    for name, value in values.items():
        if value is not None:
            writer.add_scalar(name, float(value), step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Ablation YAML")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--suite", type=Path, help="Learning-objective suite directory")
    parser.add_argument("--seed", type=int, help="Required when a suite has multiple seeds")
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Use the same path to resume")
    parser.add_argument("--test-scenarios", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and decode all inputs, but do not start any optimization",
    )
    parser.add_argument("--no-swanlab", action="store_true")
    args = parser.parse_args()

    manifest_path = args.config.resolve()
    manifest = _load_manifest(manifest_path)
    trained_config_path = None
    if args.suite is not None:
        checkpoint_path, trained_config_path = _resolve_from_suite(args.suite.resolve(), args.seed)
    else:
        checkpoint_path = args.checkpoint.resolve()
    normalization_path = (
        args.normalization.resolve() if args.normalization else _find_normalization(checkpoint_path)
    )
    base_config_path = _resolve_project_path(manifest["base_config"])
    config = load_config(base_config_path)
    compute = manifest["compute"]
    config = replace(
        config,
        planning=replace(
            config.planning,
            solver_max_parallel_workers=int(compute["solver_workers"]),
            solver_threads=int(compute["solver_threads"]),
            solver_memory_limit_mb=float(compute["solver_memory_limit_mb"]),
        ),
    )
    cost_overrides = manifest.get("costs", {})
    if cost_overrides:
        unknown = set(cost_overrides).difference(config.costs.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown cost overrides: {sorted(unknown)}")
        config = replace(config, costs=replace(config.costs, **cost_overrides))
    planning_overrides = manifest.get("planning", {})
    if planning_overrides:
        unknown = set(planning_overrides).difference(
            config.planning.__dataclass_fields__
        )
        if unknown:
            raise ValueError(f"Unknown planning overrides: {sorted(unknown)}")
        config = replace(
            config, planning=replace(config.planning, **planning_overrides)
        )

    output_root = (
        args.output_dir.resolve()
        if args.output_dir
        else _resolve_project_path(manifest["output_dir"])
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    feeder, test_pool = _experiment_data(config, config.data.test_split)
    codec = ScenarioCodec.from_normalization_dict(_read_json(normalization_path), feeder)
    _require_current_codec(config, codec)
    device = resolve_device(config.dfl.device)
    checkpoint = _load_torch(checkpoint_path, device)
    support, support_weights = _decode_support(checkpoint, config, codec, device)
    requested_test = args.test_scenarios or int(manifest.get("test_scenarios", 182))
    test_scenarios, test_weights, test_names = _test_set(
        config, test_pool, codec, requested_test
    )
    common_spec = manifest["common_evaluation"]
    common_planning = replace(
        config.planning,
        carbon_formulation=str(common_spec["carbon_formulation"]),
        carbon_cap_scope=str(common_spec["carbon_cap_scope"]),
    )
    common_oracle = make_planning_oracle(
        feeder, common_planning, config.costs, config.data, config.data_center
    )
    tolerance = float(common_spec["violation_tolerance_t_per_hour"])

    metadata = {
        "experiment_config": str(manifest_path),
        "base_config": str(base_config_path),
        "trained_effective_config": None if trained_config_path is None else str(trained_config_path),
        "checkpoint": str(checkpoint_path),
        "normalization": str(normalization_path),
        "training_run_id": checkpoint.get("run_id"),
        "support_size": len(support),
        "test_scenarios": len(test_scenarios),
        "test_scenario_names": test_names,
        "common_evaluation": common_spec,
        "compute": compute,
        "cost_overrides": cost_overrides,
        "planning_overrides": planning_overrides,
    }
    if args.validate_only:
        print(
            f"Validation passed: checkpoint K={len(support)}, "
            f"test scenarios={len(test_scenarios)}, device={device}, "
            f"normalization={normalization_path}",
            flush=True,
        )
        return
    _write_json(output_root / "experiment.json", metadata)
    swanlab_writer = _swanlab_writer(
        output_root / "swanlab",
        not args.no_swanlab,
        config,
        experiment_name=f"carbon-ablation-{output_root.name}"[:40],
        group="carbon-accounting-ablation",
    )
    print(f"Output: {output_root}", flush=True)
    print(f"Fixed support: K={len(support)}; common test set: n={len(test_scenarios)}", flush=True)

    records: list[dict[str, Any]] = []
    variant_count = len(manifest["variants"])
    for index, variant in enumerate(manifest["variants"], start=1):
        result_path = output_root / f"{variant['id']}.json"
        if result_path.exists():
            print(
                f"[{index}/{variant_count}] Resume {variant['id']} from "
                f"{result_path.name}",
                flush=True,
            )
            record = _read_json(result_path)
            records.append(record)
            _log_variant(swanlab_writer, record, index)
            continue
        print(
            f"[{index}/{variant_count}] Plan {variant['label']} on the fixed K=8 support...",
            flush=True,
        )
        native_planning = replace(
            config.planning,
            carbon_formulation=str(variant["carbon_formulation"]),
            carbon_cap_scope=str(variant["carbon_cap_scope"]),
        )
        native_oracle = make_planning_oracle(
            feeder, native_planning, config.costs, config.data, config.data_center
        )
        started = time.perf_counter()
        planned = native_oracle.solve(
            support,
            weights=support_weights,
            allow_carbon_slack=config.dfl.training_allow_carbon_slack,
        )
        planning_wall = time.perf_counter() - started
        if not planned.feasible:
            raise RuntimeError(f"{variant['id']} planning failed: {planned.status}")
        print(
            f"[{index}/{variant_count}] Evaluate its fixed design under Layered-Hourly on "
            f"{len(test_scenarios)} common scenarios...",
            flush=True,
        )
        evaluated, scenario_results, evaluation_wall = _solve_fixed_design_scenario_wise(
            common_oracle,
            test_scenarios,
            test_weights,
            planned.design,
            config.costs.demand_dollars_per_mw_year,
            common_planning.solver_relative_gap,
        )
        record = {
            "variant": str(variant["id"]),
            "label": str(variant["label"]),
            "planning_carbon_formulation": str(variant["carbon_formulation"]),
            "planning_carbon_cap_scope": str(variant["carbon_cap_scope"]),
            "evaluated_carbon_formulation": common_planning.carbon_formulation,
            "evaluated_carbon_cap_scope": common_planning.carbon_cap_scope,
            "planning_wall_seconds": planning_wall,
            "evaluation_wall_seconds": evaluation_wall,
            "planning_on_k8_support": _result_metrics(
                planned, [], [], config.data.horizon, tolerance
            ),
            "common_test_evaluation": _result_metrics(
                evaluated, scenario_results, test_weights, config.data.horizon, tolerance
            ),
            "planning_result": planned.to_dict(),
            "test_scenario_results": [item.to_dict() for item in scenario_results],
        }
        _write_json(result_path, record)
        records.append(record)
        _log_variant(swanlab_writer, record, index)

    reference_record = None
    reference_path = output_root / "no_storage_reference.json"
    if bool(common_spec.get("include_no_storage_reference", True)):
        if reference_path.exists():
            reference_record = _read_json(reference_path)
        else:
            print("Evaluate the common no-storage reference...", flush=True)
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
            reference_record = {
                "evaluation_wall_seconds": wall,
                "common_test_evaluation": _result_metrics(
                    aggregate, details, test_weights, config.data.horizon, tolerance
                ),
                "test_scenario_results": [item.to_dict() for item in details],
            }
            _write_json(reference_path, reference_record)

    reference_objective = (
        None
        if reference_record is None
        else reference_record["common_test_evaluation"]["objective_dollars_per_year"]
    )
    rows = [_csv_row(record, reference_objective) for record in records]
    rows.sort(key=lambda row: row["total_dollars_per_year"])
    _write_csv(output_root / "comparison.csv", rows)
    _write_json(
        output_root / "summary.json",
        {**metadata, "no_storage_reference": reference_record, "comparison": rows},
    )
    print("Completed. Ranked common-evaluation results:", flush=True)
    for row in rows:
        print(
            f"  {row['label']:<16} total=${row['total_dollars_per_year']:,.0f}/yr, "
            f"P={row['power_mw']:.3f} MW, E={row['energy_mwh']:.3f} MWh, "
            f"shed={row['load_shedding_mwh_per_year']:.4g} MWh/yr, "
            f"carbon excess={row['carbon_excess_t_per_year']:.4g} t/yr",
            flush=True,
        )
    if swanlab_writer is not None:
        swanlab_writer.close()


if __name__ == "__main__":
    main()
