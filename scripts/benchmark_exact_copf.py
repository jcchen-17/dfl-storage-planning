"""Time one exact COPF + exact storage-vintage planning solve.

This is deliberately a single-process benchmark.  It fixes the storage site at
the data-center bus (675 in the current feeder), while power and energy capacity
remain planning decisions.  Storage stays inside the feeder COPF and can
therefore help absorb feeder PV; only the siting binary is removed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("STORAGE_DFL_SOLVER_WORKER", "1")
sys.path.insert(0, "src")

from storage_dfl.config import load_config
from storage_dfl.planning import StoragePlanningOracle
from storage_dfl.stages import _experiment_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/dataset_v2_dfl_hourly_layered.yaml"
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--scenario-name", default=None)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--bootstrap-time-limit", type=float, default=90.0)
    parser.add_argument("--relative-gap", type=float, default=1.0e-3)
    parser.add_argument("--max-power-mw", type=float, default=4.0)
    parser.add_argument("--max-energy-mwh", type=float, default=24.0)
    parser.add_argument("--carbon-price", type=float, default=185.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--memory-limit-mb",
        type=float,
        default=8000.0,
        help="single-process solver memory budget (current training config uses 1400)",
    )
    parser.add_argument("--hard-cap", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--no-storage", action="store_true")
    parser.add_argument(
        "--co-optimize-siting",
        action="store_true",
        help="jointly choose one candidate bus and its P/E capacity",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def safe_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split_name = getattr(config.data, f"{args.split}_split")
    feeder, pool = _experiment_data(config, split_name)

    if args.scenario_name is not None:
        matches = [s for s in pool.scenarios if s.name == args.scenario_name]
        if not matches:
            raise ValueError(f"Unknown scenario name: {args.scenario_name}")
        scenario = matches[0]
    else:
        if not 0 <= args.scenario_index < len(pool.scenarios):
            raise IndexError(
                f"scenario-index must be in [0, {len(pool.scenarios) - 1}]"
            )
        scenario = pool.scenarios[args.scenario_index]

    planning = replace(
        config.planning,
        carbon_formulation="layered_dc_exact",
        storage_service_mode=(
            "shared_feeder" if args.co_optimize_siting else "fixed_dc_siting"
        ),
        max_storage_sites=0 if args.no_storage else 1,
        max_power_mw=args.max_power_mw,
        max_energy_mwh=args.max_energy_mwh,
        solver_time_limit_seconds=args.time_limit,
        warm_start_time_limit_seconds=args.bootstrap_time_limit,
        solver_relative_gap=args.relative_gap,
        solver_threads=(config.planning.solver_threads if args.threads is None else args.threads),
        solver_max_parallel_workers=1,
        solver_memory_limit_mb=args.memory_limit_mb,
        verbose_solver=args.verbose_solver,
    )
    costs = replace(
        config.costs, carbon_price_dollars_per_t=args.carbon_price
    )
    oracle = StoragePlanningOracle(
        feeder, planning, costs, config.data, config.data_center
    )
    scenarios, weights = oracle._normalize_job((scenario,), None)
    allow_slack = not args.hard_cap

    print("Exact COPF + exact vintage benchmark", flush=True)
    print(f"scenario       : {scenario.name} ({scenario.horizon} h)", flush=True)
    site_label = (
        "disabled"
        if args.no_storage
        else f"co-optimize among {list(feeder.storage_candidates)}"
        if args.co_optimize_siting
        else feeder.data_center_bus
    )
    print(f"storage site   : {site_label}", flush=True)
    print(f"main limit     : {args.time_limit:g} s", flush=True)
    print(f"bootstrap limit: {args.bootstrap_time_limit:g} s", flush=True)
    print(f"threads/memory : {planning.solver_threads} / {args.memory_limit_mb:g} MB", flush=True)
    print(f"carbon cap     : {'soft' if allow_slack else 'hard'}", flush=True)
    print(f"capacity bounds: {args.max_power_mw:g} MW / {args.max_energy_mwh:g} MWh", flush=True)
    print(f"carbon price   : {args.carbon_price:g} $/tCO2", flush=True)

    # Probe construction separately.  This model is never optimized and is not
    # counted in the actual solve-pipeline wall time below.
    start = time.perf_counter()
    probe = oracle._build_model(
        scenarios, weights, fixed_design=None, allow_carbon_slack=allow_slack
    )
    probe.model.getNVars()
    build_seconds = time.perf_counter() - start
    variables = int(probe.model.getNVars())
    constraints = int(probe.model.getNConss())
    probe.model.freeProb()
    print(
        f"build probe     : {build_seconds:.2f} s "
        f"({variables:,} vars, {constraints:,} constraints)",
        flush=True,
    )

    print("bootstrap solve : running...", flush=True)
    start = time.perf_counter()
    warm_values = oracle._no_storage_bootstrap_values(
        scenarios, weights, allow_carbon_slack=allow_slack
    )
    bootstrap_seconds = time.perf_counter() - start
    warm_key = oracle._warm_start_key(scenarios, weights, allow_slack)
    oracle._warm_start_cache[warm_key] = warm_values
    print(
        f"bootstrap solve : {bootstrap_seconds:.2f} s; "
        f"warm start={'available' if warm_values is not None else 'unavailable'}",
        flush=True,
    )

    print("main solve      : running...", flush=True)
    start = time.perf_counter()
    result = oracle.solve(
        scenarios, weights=weights, allow_carbon_slack=allow_slack, use_cache=False
    )
    main_wall_seconds = time.perf_counter() - start
    pipeline_seconds = bootstrap_seconds + main_wall_seconds
    installed = result.design.installed_buses
    power = sum(result.design.power_mw[bus] for bus in installed)
    energy = sum(result.design.energy_mwh[bus] for bus in installed)

    print("", flush=True)
    print(f"status          : {result.status}", flush=True)
    print(f"objective       : {result.objective:.6g}", flush=True)
    print(f"relative gap    : {result.relative_gap:.6g}", flush=True)
    print(f"main solver time: {result.solve_time_seconds:.2f} s", flush=True)
    print(f"main wall time  : {main_wall_seconds:.2f} s", flush=True)
    print(f"pipeline wall   : {pipeline_seconds:.2f} s (bootstrap + main)", flush=True)
    print(f"design          : buses={installed}, P={power:.4f} MW, E={energy:.4f} MWh", flush=True)

    output = args.output or (
        config.output_dir / f"benchmark_exact_copf_{scenario.name}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": scenario.name,
        "horizon_hours": scenario.horizon,
        "formulation": planning.carbon_formulation,
        "storage_service_mode": planning.storage_service_mode,
        "storage_bus": (
            None
            if args.no_storage
            else "co-optimized"
            if args.co_optimize_siting
            else feeder.data_center_bus
        ),
        "carbon_cap_is_soft": allow_slack,
        "limits": {
            "main_seconds": args.time_limit,
            "bootstrap_seconds": args.bootstrap_time_limit,
            "relative_gap": args.relative_gap,
            "threads": planning.solver_threads,
            "memory_mb": args.memory_limit_mb,
            "max_power_mw": args.max_power_mw,
            "max_energy_mwh": args.max_energy_mwh,
            "carbon_price_dollars_per_t": args.carbon_price,
        },
        "model_size": {"variables": variables, "constraints": constraints},
        "timing_seconds": {
            "build_probe": build_seconds,
            "bootstrap": bootstrap_seconds,
            "main_solver": result.solve_time_seconds,
            "main_wall": main_wall_seconds,
            "pipeline_wall": pipeline_seconds,
        },
        "result": asdict(result),
    }
    output.write_text(
        json.dumps(safe_json(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved           : {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
