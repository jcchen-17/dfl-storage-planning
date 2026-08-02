"""Diagnose whether DFL failures come from hourly physics or long-horizon scale."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from storage_dfl.config import load_config
from storage_dfl.data import Scenario
from storage_dfl.dfl import DirectSupportPolicy, resolve_device
from storage_dfl.planning import StorageDesign, StoragePlanningOracle
from storage_dfl.stages import ArtifactPaths, _experiment_data, _load_codec, _load_cvae


TIME_FIELDS = (
    "active_load_mw",
    "reactive_load_mvar",
    "pv_available_mw",
    "workload_arrival",
    "pue",
    "grid_price_per_mwh",
    "grid_carbon_t_per_mwh",
    "grid_available",
)


def slice_scenario(scenario: Scenario, start: int, stop: int) -> Scenario:
    values = {
        field: np.asarray(getattr(scenario, field)[start:stop]).copy()
        for field in TIME_FIELDS
    }
    return replace(
        scenario,
        name=f"{scenario.name}_{start:03d}_{stop:03d}",
        **values,
    )


def generated_first_epoch(config_path: Path) -> tuple[Scenario, ...]:
    config = load_config(config_path)
    paths = ArtifactPaths(config.output_dir)
    feeder, _ = _experiment_data(config, config.data.validation_split)
    codec = _load_codec(paths, feeder)
    device = resolve_device(config.dfl.device)
    cvae = _load_cvae(paths, device)
    policy = DirectSupportPolicy(
        support_count=config.dfl.num_support_scenarios,
        latent_dim=cvae.latent_dim,
        seed=config.seed,
    ).to(device)
    torch.manual_seed(config.seed + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + 1)
    conditions = torch.as_tensor(
        codec.support_conditions(config.dfl.num_support_scenarios),
        dtype=torch.float32,
        device=device,
    )
    sample = policy.sample(config.dfl.initial_exploration_std)
    with torch.no_grad():
        decoded = cvae.decode(sample.latent, conditions).cpu().numpy()
    return codec.decode_batch(decoded, conditions.cpu().numpy(), name_prefix="diagnostic")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--block-hours", type=int, default=12)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument(
        "--full-planning",
        action="store_true",
        help="Solve the generated scenarios together without updating DFL.",
    )
    parser.add_argument("--fixed-zero", action="store_true")
    parser.add_argument("--allow-carbon-slack", action="store_true")
    parser.add_argument("--iis", action="store_true")
    parser.add_argument("--scenario-index", type=int)
    parser.add_argument("--full-hours", type=int)
    parser.add_argument(
        "--carbon-formulation",
        choices=("exact", "mccormick", "aggregate_mccormick", "system_average"),
    )
    parser.add_argument(
        "--flow-limit-formulation",
        choices=("quadratic", "polygon"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.carbon_formulation or args.flow_limit_formulation:
        config = replace(
            config,
            planning=replace(
                config.planning,
                carbon_formulation=(
                    args.carbon_formulation or config.planning.carbon_formulation
                ),
                flow_limit_formulation=(
                    args.flow_limit_formulation
                    or config.planning.flow_limit_formulation
                ),
            ),
        )
    feeder, _ = _experiment_data(config, config.data.validation_split)
    scenarios = generated_first_epoch(args.config)
    if args.scenario_index is not None:
        scenarios = (scenarios[args.scenario_index],)
    if args.full_hours is not None:
        scenarios = tuple(
            slice_scenario(scenario, 0, args.full_hours) for scenario in scenarios
        )
    diagnostic_planning = replace(
        config.planning,
        max_storage_sites=0,
        solver_time_limit_seconds=args.time_limit,
        solver_relative_gap=max(config.planning.solver_relative_gap, 0.20),
        verbose_solver=False,
    )
    oracle = StoragePlanningOracle(feeder, diagnostic_planning, config.costs, config.data)
    zero_design = StorageDesign(
        site={bus: 0 for bus in feeder.storage_candidates},
        power_mw={bus: 0.0 for bus in feeder.storage_candidates},
        energy_mwh={bus: 0.0 for bus in feeder.storage_candidates},
    )

    if args.full_planning:
        full_planning = replace(
            config.planning,
            solver_time_limit_seconds=args.time_limit,
        )
        full_oracle = StoragePlanningOracle(
            feeder,
            full_planning,
            config.costs,
            config.data,
        )
        fixed_design = zero_design if args.fixed_zero else None
        if args.iis:
            weights = tuple(1.0 / len(scenarios) for _ in scenarios)
            artifacts = full_oracle._build_model(
                scenarios,
                weights,
                fixed_design=fixed_design,
                allow_carbon_slack=args.allow_carbon_slack,
            )
            artifacts.model.optimize()
            if str(artifacts.model.getStatus()) != "infeasible":
                raise SystemExit(
                    f"IIS requires infeasible status, got {artifacts.model.getStatus()}"
                )
            artifacts.model.generateIIS()
            subproblem = artifacts.model.getIIS().getSubscip()
            names = sorted(constraint.name for constraint in subproblem.getConss())
            print(f"IIS constraints: {len(names)}")
            for name in names:
                print(name)
            artifacts.model.freeProb()
            return
        result = full_oracle.solve(
            scenarios,
            fixed_design=fixed_design,
            allow_carbon_slack=args.allow_carbon_slack,
        )
        print(
            f"full planning: status={result.status}, objective={result.objective:.6g}, "
            f"seconds={result.solve_time_seconds:.2f}, gap={result.relative_gap:.6g}",
            flush=True,
        )
        if not result.feasible:
            raise SystemExit("full planning produced no feasible solution")
        return

    failures = 0
    for scenario in scenarios:
        total_load = scenario.active_load_mw.sum(axis=(1, 2))
        total_pv = scenario.pv_available_mw.sum(axis=(1, 2))
        print(
            f"{scenario.name}: load=[{total_load.min():.3f}, {total_load.max():.3f}] MW, "
            f"pv=[{total_pv.min():.3f}, {total_pv.max():.3f}] MW, "
            f"carbon=[{scenario.grid_carbon_t_per_mwh.min():.3f}, "
            f"{scenario.grid_carbon_t_per_mwh.max():.3f}] t/MWh",
            flush=True,
        )
        for start in range(0, scenario.horizon, args.block_hours):
            stop = min(start + args.block_hours, scenario.horizon)
            block = slice_scenario(scenario, start, stop)
            result = oracle.solve(
                (block,),
                fixed_design=zero_design,
                allow_carbon_slack=True,
            )
            print(
                f"  hours {start:03d}-{stop:03d}: status={result.status}, "
                f"objective={result.objective:.6g}, seconds={result.solve_time_seconds:.2f}",
                flush=True,
            )
            failures += int(not result.feasible)

    if failures:
        raise SystemExit(f"diagnostic failed: {failures} block(s) had no feasible solution")
    print("diagnostic passed: every short block has a finite feasible solution")


if __name__ == "__main__":
    main()
