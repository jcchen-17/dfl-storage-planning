"""Benchmark full-scenario PCC PV-plus-storage planning.

This is deliberately separate from the feeder COPF oracle.  It answers a
scoping question: after aggregating the microgrid behind one PCC, is the full
182-scenario planning problem already cheap enough that DFL scenario reduction
has little computational value?

Two continuous LP formulations are available:

``aggregate``
    Conventional aggregate SOC. Carbon excess is charged on physical grid
    imports at the time of import.

``vintage``
    Consumption-side carbon attribution. Grid- and PV-charged energy from each
    charge interval remain separate layers, and later discharge carries the
    grid carbon intensity of its formation interval.  All coefficients are
    known data, so the formulation remains linear and uses no McCormick
    relaxation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from pyscipopt import Model, quicksum

from storage_dfl.config import load_config
from storage_dfl.stages import _experiment_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/dataset_v2_dfl_hourly_layered.yaml"
    )
    parser.add_argument("--split", default="train", choices=("train", "validation", "test"))
    parser.add_argument(
        "--scenarios",
        type=int,
        default=None,
        help="use the first N scenarios; default uses the entire split",
    )
    parser.add_argument("--carbon-model", choices=("aggregate", "vintage"), default="vintage")
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--max-pv-mw", type=float, default=5.0)
    parser.add_argument(
        "--pv-capex-per-mw",
        type=float,
        default=1_000_000.0,
        help="declared overnight PV CAPEX sensitivity; annualized by the config CRF",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose-solver", action="store_true")
    return parser.parse_args()


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    feeder, pool = _experiment_data(config, args.split)
    scenarios = pool.scenarios
    if args.scenarios is not None:
        if args.scenarios <= 0:
            raise SystemExit("--scenarios must be positive")
        scenarios = scenarios[: args.scenarios]
    scenario_count = len(scenarios)
    horizon = config.data.horizon
    dt = config.data.delta_t_hours
    pcfg = config.planning
    ccfg = config.costs
    dccfg = config.data_center
    reference_pv_mw = float(feeder.pv_capacity_mw.sum())
    if reference_pv_mw <= 0.0:
        raise RuntimeError("The feeder has no reference PV capacity for profile normalization.")

    model = Model(f"pcc_{args.carbon_model}_{scenario_count}")
    model.hideOutput(not args.verbose_solver)
    model.setRealParam("limits/time", float(args.time_limit))
    model.setIntParam("parallel/maxnthreads", max(1, int(pcfg.solver_threads)))

    build_started = time.perf_counter()
    pv_cap = model.addVar(lb=0.0, ub=args.max_pv_mw, name="pv_cap_mw")
    battery_power = model.addVar(lb=0.0, ub=pcfg.max_power_mw, name="battery_power_mw")
    battery_energy = model.addVar(lb=0.0, ub=pcfg.max_energy_mwh, name="battery_energy_mwh")
    peak_grid = model.addVar(lb=0.0, ub=pcfg.grid_limit_mw, name="peak_grid_mw")
    model.addCons(battery_energy >= pcfg.min_duration_hours * battery_power)
    model.addCons(battery_energy <= pcfg.max_duration_hours * battery_power)

    investment = ccfg.capital_recovery_factor * (
        args.pv_capex_per_mw * pv_cap
        + ccfg.battery_capex_scale
        * (
            ccfg.power_dollars_per_mw * battery_power
            + ccfg.energy_dollars_per_mwh * battery_energy
        )
    )
    operating_terms: list[Any] = []
    carbon_terms: list[Any] = []
    tracked: list[dict[str, Any]] = []
    annual_blocks = 8760.0 / (horizon * dt)
    scenario_weight = 1.0 / scenario_count
    scenario_scale = scenario_weight * annual_blocks

    for sid, scenario in enumerate(scenarios):
        load = scenario.active_load_mw.sum(axis=(1, 2)) + dccfg.power_mw(
            scenario.pue, scenario.workload_arrival
        )
        capacity_factor = scenario.pv_available_mw.sum(axis=(1, 2)) / reference_pv_mw
        grid_direct = {}
        grid_charge = {}
        pv_direct = {}
        pv_charge = {}
        curtail = {}
        carbon_excess = {}

        if args.carbon_model == "aggregate":
            soc = {
                t: model.addVar(lb=0.0, ub=pcfg.max_energy_mwh, name=f"soc[{sid},{t}]")
                for t in range(horizon + 1)
            }
            discharge = {
                t: model.addVar(lb=0.0, ub=pcfg.max_power_mw, name=f"dis[{sid},{t}]")
                for t in range(horizon)
            }
            model.addCons(soc[0] == pcfg.initial_soc * battery_energy)
        else:
            # The initial layer has the configured initial carbon intensity.
            initial_energy = {
                t: model.addVar(
                    lb=0.0, ub=pcfg.max_energy_mwh, name=f"initial_e[{sid},{t}]"
                )
                for t in range(horizon + 1)
            }
            initial_discharge = {
                t: model.addVar(
                    lb=0.0, ub=pcfg.max_power_mw, name=f"initial_dis[{sid},{t}]"
                )
                for t in range(horizon)
            }
            model.addCons(initial_energy[0] == pcfg.initial_soc * battery_energy)
            grid_layer_energy: dict[tuple[int, int], Any] = {}
            pv_layer_energy: dict[tuple[int, int], Any] = {}
            grid_layer_discharge: dict[tuple[int, int], Any] = {}
            pv_layer_discharge: dict[tuple[int, int], Any] = {}
            # Create every layer before interval balance constraints reference
            # its discharge. Dynamics are added after the charge variables have
            # been created in the interval loop below.
            for k in range(horizon):
                for source, energy, layer_discharge in (
                    ("grid", grid_layer_energy, grid_layer_discharge),
                    ("pv", pv_layer_energy, pv_layer_discharge),
                ):
                    for state in range(k + 1, horizon + 1):
                        energy[k, state] = model.addVar(
                            lb=0.0,
                            ub=pcfg.max_energy_mwh,
                            name=f"{source}_layer_e[{sid},{k},{state}]",
                        )
                    for t in range(k + 1, horizon):
                        layer_discharge[k, t] = model.addVar(
                            lb=0.0,
                            ub=pcfg.max_power_mw,
                            name=f"{source}_layer_dis[{sid},{k},{t}]",
                        )

        for t in range(horizon):
            availability = float(scenario.grid_available[t])
            grid_direct[t] = model.addVar(
                lb=0.0,
                ub=pcfg.grid_limit_mw * availability,
                name=f"grid_direct[{sid},{t}]",
            )
            grid_charge[t] = model.addVar(
                lb=0.0,
                ub=pcfg.max_power_mw * availability,
                name=f"grid_charge[{sid},{t}]",
            )
            pv_direct[t] = model.addVar(lb=0.0, name=f"pv_direct[{sid},{t}]")
            pv_charge[t] = model.addVar(lb=0.0, name=f"pv_charge[{sid},{t}]")
            curtail[t] = model.addVar(lb=0.0, name=f"pv_curtail[{sid},{t}]")
            carbon_excess[t] = model.addVar(lb=0.0, name=f"carbon_excess[{sid},{t}]")
            model.addCons(
                pv_direct[t] + pv_charge[t] + curtail[t]
                == float(capacity_factor[t]) * pv_cap
            )
            model.addCons(grid_direct[t] + grid_charge[t] <= peak_grid)
            model.addCons(grid_direct[t] + grid_charge[t] <= pcfg.grid_limit_mw * availability)
            model.addCons(grid_charge[t] + pv_charge[t] <= battery_power)

            if args.carbon_model == "aggregate":
                discharge_expression = discharge[t]
                model.addCons(discharge[t] <= battery_power)
                model.addCons(
                    soc[t + 1]
                    == (1.0 - pcfg.self_discharge) * soc[t]
                    + pcfg.charge_efficiency * (grid_charge[t] + pv_charge[t]) * dt
                    - discharge[t] * dt / pcfg.discharge_efficiency
                )
                # Physical-emissions benchmark: price grid-import emissions
                # above the declared intensity benchmark on total PCC imports.
                total_grid = grid_direct[t] + grid_charge[t]
                model.addCons(
                    float(scenario.grid_carbon_t_per_mwh[t]) * total_grid * dt
                    <= pcfg.dc_carbon_cap * total_grid * dt + carbon_excess[t]
                )
            else:
                available_grid_layers = [
                    grid_layer_discharge[k, t] for k in range(t)
                ]
                available_pv_layers = [pv_layer_discharge[k, t] for k in range(t)]
                discharge_expression = (
                    initial_discharge[t]
                    + quicksum(available_grid_layers)
                    + quicksum(available_pv_layers)
                )
                model.addCons(discharge_expression <= battery_power)
                consumption_carbon = (
                    float(scenario.grid_carbon_t_per_mwh[t]) * grid_direct[t]
                    + pcfg.initial_carbon_intensity * initial_discharge[t]
                    + quicksum(
                        float(scenario.grid_carbon_t_per_mwh[k])
                        * grid_layer_discharge[k, t]
                        for k in range(t)
                    )
                )
                model.addCons(
                    consumption_carbon * dt
                    <= pcfg.dc_carbon_cap * float(load[t]) * dt + carbon_excess[t]
                )

            model.addCons(
                grid_direct[t] + pv_direct[t] + discharge_expression == float(load[t])
            )
            throughput = grid_charge[t] + pv_charge[t] + discharge_expression
            operating_terms.append(
                scenario_scale
                * (
                    float(scenario.grid_price_per_mwh[t])
                    * (grid_direct[t] + grid_charge[t])
                    + ccfg.degradation_dollars_per_mwh * throughput
                    + ccfg.curtailment_dollars_per_mwh * curtail[t]
                )
                * dt
            )
            carbon_terms.append(
                scenario_scale * ccfg.carbon_price_dollars_per_t * carbon_excess[t]
            )

        if args.carbon_model == "aggregate":
            for t in range(horizon + 1):
                model.addCons(soc[t] >= pcfg.min_soc * battery_energy)
                model.addCons(soc[t] <= pcfg.max_soc * battery_energy)
            model.addCons(soc[horizon] >= pcfg.initial_soc * battery_energy)
        else:
            retention = 1.0 - pcfg.self_discharge
            for t in range(horizon):
                model.addCons(
                    initial_energy[t + 1]
                    == retention * initial_energy[t]
                    - initial_discharge[t] * dt / pcfg.discharge_efficiency
                )
            for k in range(horizon):
                for source, charge, energy, layer_discharge in (
                    ("grid", grid_charge, grid_layer_energy, grid_layer_discharge),
                    ("pv", pv_charge, pv_layer_energy, pv_layer_discharge),
                ):
                    model.addCons(
                        energy[k, k + 1] == pcfg.charge_efficiency * charge[k] * dt
                    )
                    for t in range(k + 1, horizon):
                        model.addCons(
                            energy[k, t + 1]
                            == retention * energy[k, t]
                            - layer_discharge[k, t] * dt / pcfg.discharge_efficiency
                        )
            for state in range(horizon + 1):
                stored = initial_energy[state] + quicksum(
                    grid_layer_energy[k, state] + pv_layer_energy[k, state]
                    for k in range(state)
                )
                model.addCons(stored >= pcfg.min_soc * battery_energy)
                model.addCons(stored <= pcfg.max_soc * battery_energy)
            terminal = initial_energy[horizon] + quicksum(
                grid_layer_energy[k, horizon] + pv_layer_energy[k, horizon]
                for k in range(horizon)
            )
            model.addCons(terminal >= pcfg.initial_soc * battery_energy)

        tracked.append(
            {
                "grid_direct": grid_direct,
                "grid_charge": grid_charge,
                "pv_direct": pv_direct,
                "pv_charge": pv_charge,
                "curtail": curtail,
                "carbon_excess": carbon_excess,
            }
        )

    operating = quicksum(operating_terms)
    carbon_cost = quicksum(carbon_terms)
    demand_cost = ccfg.demand_dollars_per_mw_year * peak_grid
    model.setObjective(investment + demand_cost + operating + carbon_cost, "minimize")
    build_seconds = time.perf_counter() - build_started
    variables = model.getNVars()
    constraints = model.getNConss()
    print(
        f"built {args.carbon_model} PCC LP: {scenario_count} scenarios, "
        f"{variables:,} variables, {constraints:,} constraints in {build_seconds:.1f}s",
        flush=True,
    )

    solve_started = time.perf_counter()
    model.optimize()
    solve_seconds = time.perf_counter() - solve_started
    status = str(model.getStatus())
    solution_count = model.getNSols()
    result: dict[str, Any] = {
        "config": str(Path(args.config)),
        "split": args.split,
        "carbon_model": args.carbon_model,
        "scenario_count": scenario_count,
        "horizon": horizon,
        "reference_pv_capacity_mw": reference_pv_mw,
        "pv_capex_per_mw": args.pv_capex_per_mw,
        "max_pv_mw": args.max_pv_mw,
        "variables": variables,
        "constraints": constraints,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "status": status,
        "solution_count": solution_count,
    }
    if solution_count:
        result.update(
            {
                "objective": _finite(model.getObjVal()),
                "best_bound": _finite(model.getDualbound()),
                "relative_gap": _finite(model.getGap()),
                "pv_capacity_mw": model.getVal(pv_cap),
                "battery_power_mw": model.getVal(battery_power),
                "battery_energy_mwh": model.getVal(battery_energy),
                "peak_grid_mw": model.getVal(peak_grid),
                "annualized_investment_cost": model.getVal(investment),
                "annual_demand_cost": model.getVal(demand_cost),
                "annual_operating_cost": model.getVal(operating),
                "annual_carbon_cost": model.getVal(carbon_cost),
            }
        )
    output = args.output or (
        Path("outputs")
        / "pcc_benchmark"
        / f"{args.carbon_model}_{args.split}_{scenario_count}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
