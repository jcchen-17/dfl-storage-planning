"""Diagnose whether real historical windows create economic value for storage."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from storage_dfl.config import ExperimentConfig, load_config
from storage_dfl.data import Scenario, ScenarioCodec, load_historical_scenarios
from storage_dfl.network import ieee13_unbalanced_microgrid
from storage_dfl.planning import PlanningResult, StoragePlanningOracle


def _price_spread_scenario(scenario: Scenario, factor: float) -> Scenario:
    price = np.asarray(scenario.grid_price_per_mwh, dtype=float)
    mean = float(price.mean())
    adjusted = np.maximum(0.0, mean + factor * (price - mean))
    return replace(scenario, name=f"{scenario.name}_price_{factor:g}x", grid_price_per_mwh=adjusted)


def _case_config(config: ExperimentConfig, case: str) -> tuple[ExperimentConfig, float]:
    """Return a modified configuration and the price-spread multiplier."""

    if case == "base":
        return config, 1.0
    if case == "capex_50pct":
        costs = replace(
            config.costs,
            site_dollars=0.5 * config.costs.site_dollars,
            power_dollars_per_mw=0.5 * config.costs.power_dollars_per_mw,
            energy_dollars_per_mwh=0.5 * config.costs.energy_dollars_per_mwh,
        )
        return replace(config, costs=costs), 1.0
    if case == "capex_25pct":
        costs = replace(
            config.costs,
            site_dollars=0.25 * config.costs.site_dollars,
            power_dollars_per_mw=0.25 * config.costs.power_dollars_per_mw,
            energy_dollars_per_mwh=0.25 * config.costs.energy_dollars_per_mwh,
        )
        return replace(config, costs=costs), 1.0
    if case == "demand_3x":
        return replace(
            config,
            costs=replace(
                config.costs,
                demand_dollars_per_mw_year=3.0 * config.costs.demand_dollars_per_mw_year,
            ),
        ), 1.0
    if case == "demand_6x":
        return replace(
            config,
            costs=replace(
                config.costs,
                demand_dollars_per_mw_year=6.0 * config.costs.demand_dollars_per_mw_year,
            ),
        ), 1.0
    if case == "price_spread_2x":
        return config, 2.0
    if case == "price_spread_3x":
        return config, 3.0
    if case == "grid_1mw_per_phase":
        return replace(
            config,
            planning=replace(config.planning, grid_limit_mw=1.0),
        ), 1.0
    if case == "grid_1_2mw_per_phase":
        return replace(
            config,
            planning=replace(config.planning, grid_limit_mw=1.2),
        ), 1.0
    raise ValueError(f"Unknown scan case: {case}")


def _result_payload(result: PlanningResult) -> dict[str, object]:
    return {
        "status": result.status,
        "feasible": result.feasible,
        "objective": result.objective if result.feasible else None,
        "investment_cost": result.investment_cost if result.feasible else None,
        "operating_cost": result.operating_cost if result.feasible else None,
        "carbon_slack_cost": result.carbon_slack_cost if result.feasible else None,
        "peak_grid_mw": result.peak_grid_mw if result.feasible else None,
        "relative_gap": result.relative_gap if result.feasible else None,
        "solve_time_seconds": result.solve_time_seconds,
        "installed_buses": result.design.installed_buses,
        "power_mw": result.design.power_mw,
        "energy_mwh": result.design.energy_mwh,
    }


def _scenario_summary(scenarios: tuple[Scenario, ...]) -> list[dict[str, object]]:
    rows = []
    for scenario in scenarios:
        load = scenario.active_load_mw.sum(axis=(1, 2))
        pv = scenario.pv_available_mw.sum(axis=(1, 2))
        rows.append(
            {
                "name": scenario.name,
                "peak_load_mw": float(load.max()),
                "peak_net_load_mw": float((load - pv).max()),
                "price_spread_per_mwh": float(np.ptp(scenario.grid_price_per_mwh)),
                "peak_carbon_t_per_mwh": float(scenario.grid_carbon_t_per_mwh.max()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare storage-enabled and storage-disabled plans on representative real windows."
    )
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--num-scenarios", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--relative-gap", type=float, default=0.10)
    parser.add_argument(
        "--cases",
        default="base,capex_50pct,demand_3x,price_spread_2x,grid_1mw_per_phase",
        help="Comma-separated scan cases; see _case_config for supported names.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    feeder = ieee13_unbalanced_microgrid()
    pool = load_historical_scenarios(
        config.data.dataset_path,
        split=args.split,
        horizon=config.data.horizon,
    )
    codec = ScenarioCodec.fit(pool, feeder)
    count = min(max(1, args.num_scenarios), len(pool.scenarios))
    indices = codec.support_indices(pool, count)
    selected = pool.subset(indices.tolist())
    cases = tuple(item.strip() for item in args.cases.split(",") if item.strip())
    if not cases:
        raise ValueError("At least one scan case is required.")

    print(f"Selected {count} representative real {args.split} windows:", flush=True)
    for row in _scenario_summary(selected):
        print(
            f"  {row['name']}: peak_net={row['peak_net_load_mw']:.3f} MW, "
            f"price_spread={row['price_spread_per_mwh']:.2f} $/MWh",
            flush=True,
        )

    output: dict[str, object] = {
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "selected_indices": indices.tolist(),
        "selected_scenarios": _scenario_summary(selected),
        "time_limit_seconds": args.time_limit,
        "solver_relative_gap": args.relative_gap,
        "cases": {},
    }

    for case in cases:
        case_config, price_factor = _case_config(config, case)
        planning = replace(
            case_config.planning,
            solver_time_limit_seconds=args.time_limit,
            solver_relative_gap=args.relative_gap,
        )
        case_config = replace(case_config, planning=planning)
        case_scenarios = tuple(
            _price_spread_scenario(scenario, price_factor)
            if price_factor != 1.0
            else scenario
            for scenario in selected
        )
        print(f"\n[{case}] solving without storage...", flush=True)
        disabled_oracle = StoragePlanningOracle(
            feeder,
            replace(planning, max_storage_sites=0),
            case_config.costs,
            case_config.data,
        )
        without_storage = disabled_oracle.solve(case_scenarios)
        print(
            f"  no storage: {without_storage.status}, objective={without_storage.objective:.6g}",
            flush=True,
        )

        print(f"[{case}] solving with storage enabled...", flush=True)
        enabled_oracle = StoragePlanningOracle(
            feeder,
            planning,
            case_config.costs,
            case_config.data,
        )
        with_storage = enabled_oracle.solve(case_scenarios)
        installed = with_storage.design.installed_buses
        print(
            f"  storage: {with_storage.status}, objective={with_storage.objective:.6g}, "
            f"installed={installed}",
            flush=True,
        )

        savings = None
        savings_percent = None
        optional_best = "unresolved"
        if without_storage.feasible and with_storage.feasible:
            raw_savings = float(without_storage.objective - with_storage.objective)
            # Storage is optional, so the storage-enabled feasible set contains
            # the no-storage solution. A worse enabled incumbent is solver
            # truncation, not negative storage value; retain zero storage as the
            # best known optional decision in that case.
            comparison_tolerance = max(1.0e-4, 1.0e-8 * abs(without_storage.objective))
            savings = max(0.0, raw_savings) if raw_savings > comparison_tolerance else 0.0
            savings_percent = 100.0 * savings / max(abs(without_storage.objective), 1.0)
            optional_best = "storage" if savings > 0.0 else "no_storage"
            print(
                f"  optional best: {optional_best}, proven savings={savings:.2f} "
                f"({savings_percent:.3f}%)",
                flush=True,
            )
            if raw_savings < 0.0:
                print(
                    "  note: enabled incumbent is worse than the feasible no-storage "
                    "solution; this is a solver-limit artifact.",
                    flush=True,
                )
        output["cases"][case] = {
            "price_spread_factor": price_factor,
            "planning": {
                "grid_limit_mw_per_phase": planning.grid_limit_mw,
                "demand_dollars_per_mw_year": case_config.costs.demand_dollars_per_mw_year,
                "site_dollars": case_config.costs.site_dollars,
                "power_dollars_per_mw": case_config.costs.power_dollars_per_mw,
                "energy_dollars_per_mwh": case_config.costs.energy_dollars_per_mwh,
            },
            "without_storage": _result_payload(without_storage),
            "with_storage": _result_payload(with_storage),
            "optional_best_known": optional_best,
            "storage_savings_dollars_per_year": savings,
            "storage_savings_percent": savings_percent,
        }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else config.output_dir / "storage_value_scan.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nSaved scan results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
