"""Single-PCC storage planning with exact source/vintage carbon accounting."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from storage_dfl.config import CostConfig, DataCenterConfig, DataConfig, PlanningConfig
from storage_dfl.data import Scenario
from storage_dfl.network import Feeder

from .solver import PlanningJob, apply_search_strategy, new_model
from .results import (
    PlanningResult,
    RecourseDiagnostics,
    StorageDesign,
    infeasible_result,
    scenario_content_key,
)


@dataclass(frozen=True)
class _PCCArtifacts:
    model: object
    site: object
    power_capacity: object
    energy_capacity: object
    peak_grid: object
    investment: object
    operating: object
    carbon_cost: object
    carbon_ledger: dict[str, object]
    diagnostic_totals: dict[str, object]
    hourly_load_shedding: tuple[tuple[object, ...], ...]
    hourly_carbon_excess: tuple[tuple[object, ...], ...]


class SinglePCCPlanningOracle:
    """Capacity-planning oracle for a data centre behind one grid connection.

    Grid, PV and diesel flows are split between the data-centre load and battery
    charging. Battery inventory is separated into an initial layer, aggregate
    PV and diesel layers, and one grid-charging vintage per interval. Every
    layer has a fixed carbon intensity, so both storage carbon conservation and
    delivered-load carbon caps remain linear and auditable.
    """

    def __init__(
        self,
        feeder: Feeder,
        planning: PlanningConfig,
        costs: CostConfig,
        data: DataConfig,
        data_center: DataCenterConfig | None = None,
    ) -> None:
        if planning.topology != "single_pcc":
            raise ValueError("SinglePCCPlanningOracle requires topology='single_pcc'.")
        if planning.carbon_formulation != "layered_pcc":
            raise ValueError(
                "single_pcc topology requires carbon_formulation='layered_pcc'."
            )
        if planning.carbon_cap_scope not in {"hourly", "horizon"}:
            raise ValueError("carbon_cap_scope must be 'hourly' or 'horizon'.")
        if len(feeder.buses) != 1 or feeder.storage_candidates != (feeder.root,):
            raise ValueError("The single-PCC feeder must contain one storage bus.")
        if not 0.0 < planning.charge_efficiency <= 1.0:
            raise ValueError("charge_efficiency must lie in (0, 1].")
        if not 0.0 < planning.discharge_efficiency <= 1.0:
            raise ValueError("discharge_efficiency must lie in (0, 1].")
        if planning.diesel_carbon_t_per_mwh < 0.0:
            raise ValueError("diesel_carbon_t_per_mwh must be nonnegative.")
        if (data_center or DataCenterConfig()).workload_max_delay_hours != 0:
            raise ValueError(
                "single_pcc currently requires workload_max_delay_hours=0; "
                "data-centre demand is represented directly by the scenario's "
                "active-load trajectory."
            )
        self.feeder = feeder
        self.planning = planning
        self.costs = costs
        self.data = data
        self.data_center = data_center or DataCenterConfig()
        self._cache: dict[tuple, PlanningResult] = {}
        # These attributes preserve the worker protocol shared with the feeder
        # oracle. PCC models need no separate no-storage bootstrap.
        self._warm_start_cache: dict[tuple, dict[str, float] | None] = {}
        self._injected_warm_start = None

    def _normalize_job(
        self,
        scenarios: Iterable[Scenario],
        weights: Iterable[float] | None,
    ) -> tuple[tuple[Scenario, ...], tuple[float, ...]]:
        scenario_tuple = tuple(scenarios)
        if not scenario_tuple:
            raise ValueError("At least one scenario is required.")
        if any(s.horizon != scenario_tuple[0].horizon for s in scenario_tuple):
            raise ValueError("All scenarios must have the same horizon.")
        if any(s.num_buses != 1 for s in scenario_tuple):
            raise ValueError("Single-PCC scenarios must contain exactly one bus.")
        weight_tuple = (
            tuple(weights)
            if weights is not None
            else tuple(1.0 / len(scenario_tuple) for _ in scenario_tuple)
        )
        if len(weight_tuple) != len(scenario_tuple):
            raise ValueError("The number of weights must match the scenarios.")
        total = sum(weight_tuple)
        if total <= 0.0 or any(weight < 0.0 for weight in weight_tuple):
            raise ValueError("Scenario weights must be nonnegative with a positive sum.")
        return scenario_tuple, tuple(weight / total for weight in weight_tuple)

    def _cache_key(
        self,
        scenarios: tuple[Scenario, ...],
        weights: tuple[float, ...],
        fixed_design: StorageDesign | None,
        allow_carbon_slack: bool,
    ) -> tuple:
        bus = self.feeder.root
        design_key = None
        if fixed_design is not None:
            design_key = (
                int(fixed_design.site[bus]),
                float(fixed_design.power_mw[bus]),
                float(fixed_design.energy_mwh[bus]),
            )
        return (
            tuple(scenario_content_key(scenario) for scenario in scenarios),
            tuple(float(weight) for weight in weights),
            design_key,
            bool(allow_carbon_slack),
        )

    def _isolation_active(self) -> bool:
        return (
            os.name == "nt"
            and "torch" in sys.modules
            and os.environ.get("STORAGE_DFL_SOLVER_WORKER") != "1"
        )

    def solve(
        self,
        scenarios: Iterable[Scenario],
        *,
        weights: Iterable[float] | None = None,
        fixed_design: StorageDesign | None = None,
        allow_carbon_slack: bool = False,
        use_cache: bool = False,
    ) -> PlanningResult:
        scenario_tuple, weight_tuple = self._normalize_job(scenarios, weights)
        key = self._cache_key(
            scenario_tuple, weight_tuple, fixed_design, allow_carbon_slack
        )
        if use_cache and key in self._cache:
            return self._cache[key]
        if self._isolation_active():
            result = self._run_worker(
                scenario_tuple,
                weight_tuple,
                fixed_design,
                allow_carbon_slack,
            )
            if use_cache:
                self._cache[key] = result
            return result

        artifacts = self._build_model(
            scenario_tuple,
            weight_tuple,
            fixed_design=fixed_design,
            allow_carbon_slack=allow_carbon_slack,
        )
        model = artifacts.model
        model.optimize()
        status = str(model.getStatus())
        solution = model.getBestSol()
        scenario_names = tuple(scenario.name for scenario in scenario_tuple)
        bus = self.feeder.root
        if solution is None:
            result = infeasible_result(
                status, self.feeder.storage_candidates, scenario_names, model.getSolvingTime()
            )
        else:
            def value(expression: object) -> float:
                if isinstance(expression, (int, float)):
                    return float(expression)
                return float(model.getVal(expression))

            installed = int(value(artifacts.site) > 0.5)
            design = StorageDesign(
                site={bus: installed},
                power_mw={
                    bus: max(0.0, value(artifacts.power_capacity)) if installed else 0.0
                },
                energy_mwh={
                    bus: max(0.0, value(artifacts.energy_capacity)) if installed else 0.0
                },
            )
            result = PlanningResult(
                status=status,
                objective=float(model.getObjVal()),
                investment_cost=value(artifacts.investment),
                operating_cost=value(artifacts.operating),
                carbon_slack_cost=max(0.0, value(artifacts.carbon_cost)),
                peak_grid_mw=max(0.0, value(artifacts.peak_grid)),
                design=design,
                scenario_names=scenario_names,
                solve_time_seconds=float(model.getSolvingTime()),
                relative_gap=float(model.getGap()),
                best_bound=float(model.getDualbound()),
                carbon_ledger={
                    name: value(expression)
                    for name, expression in artifacts.carbon_ledger.items()
                },
                recourse_diagnostics=RecourseDiagnostics(
                    load_shedding_mwh=max(
                        0.0, value(artifacts.diagnostic_totals["load_shedding_mwh"])
                    ),
                    carbon_excess_t=max(
                        0.0, value(artifacts.diagnostic_totals["carbon_excess_t"])
                    ),
                    pv_curtailment_mwh=max(
                        0.0, value(artifacts.diagnostic_totals["pv_curtailment_mwh"])
                    ),
                    served_demand_mwh=max(
                        0.0, value(artifacts.diagnostic_totals["served_demand_mwh"])
                    ),
                    operating_cost=value(artifacts.operating),
                    total_planning_cost=float(model.getObjVal()),
                    hourly_load_shedding_mw=tuple(
                        tuple(max(0.0, value(variable)) for variable in row)
                        for row in artifacts.hourly_load_shedding
                    ),
                    hourly_carbon_excess_t_per_hour=tuple(
                        tuple(max(0.0, value(variable)) for variable in row)
                        for row in artifacts.hourly_carbon_excess
                    ),
                ),
            )
        model.freeProb()
        if use_cache:
            self._cache[key] = result
        return result

    def solve_many(
        self,
        jobs: Iterable[PlanningJob | tuple[Iterable[Scenario], Iterable[float] | None]],
        *,
        allow_carbon_slack: bool = False,
        use_cache: bool = False,
        max_workers: int | None = None,
    ) -> list[PlanningResult]:
        requested = [
            job if isinstance(job, PlanningJob) else PlanningJob(*job) for job in jobs
        ]
        workers = max(
            1,
            int(
                max_workers
                if max_workers is not None
                else self.planning.solver_max_parallel_workers
            ),
        )

        def run(job: PlanningJob) -> PlanningResult:
            return self.solve(
                job.scenarios,
                weights=job.weights,
                fixed_design=job.fixed_design,
                allow_carbon_slack=allow_carbon_slack,
                use_cache=use_cache,
            )

        if workers == 1 or len(requested) <= 1 or not self._isolation_active():
            return [run(job) for job in requested]
        with ThreadPoolExecutor(max_workers=min(workers, len(requested))) as pool:
            return list(pool.map(run, requested))

    def _run_worker(
        self,
        scenarios: tuple[Scenario, ...],
        weights: tuple[float, ...],
        fixed_design: StorageDesign | None,
        allow_carbon_slack: bool,
    ) -> PlanningResult:
        payload = {
            "feeder": self.feeder,
            "planning": self.planning,
            "costs": self.costs,
            "data": self.data,
            "data_center": self.data_center,
            "scenarios": scenarios,
            "weights": weights,
            "fixed_design": fixed_design,
            "allow_carbon_slack": allow_carbon_slack,
            "use_cache": False,
            "warm_start_cached": True,
            "warm_start_values": None,
        }
        with tempfile.TemporaryDirectory(prefix="storage_dfl_pcc_") as directory:
            root = Path(directory)
            input_path = root / "input.pkl"
            output_path = root / "output.pkl"
            input_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            environment = dict(os.environ)
            environment["STORAGE_DFL_SOLVER_WORKER"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "storage_dfl.planning.worker",
                    str(input_path),
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )
            if completed.returncode != 0 or not output_path.exists():
                raise RuntimeError(
                    "Isolated single-PCC solver failed:\n"
                    + completed.stdout
                    + completed.stderr
                )
            response = pickle.loads(output_path.read_bytes())
        result = response.get("result") if isinstance(response, dict) else None
        if not isinstance(result, PlanningResult):
            raise TypeError("The isolated single-PCC worker returned an invalid result.")
        return result

    def _configure_model(self, model: object) -> None:
        pcfg = self.planning
        if not pcfg.verbose_solver:
            model.hideOutput()
        model.setParam("limits/time", float(pcfg.solver_time_limit_seconds))
        model.setParam("limits/gap", float(pcfg.solver_relative_gap))
        if pcfg.solver_absolute_gap_dollars > 0.0:
            model.setParam("limits/absgap", float(pcfg.solver_absolute_gap_dollars))
        if pcfg.solver_memory_limit_mb > 0.0:
            try:
                model.setParam("limits/memory", float(pcfg.solver_memory_limit_mb))
            except KeyError:
                pass
        try:
            model.setParam("parallel/maxnthreads", int(pcfg.solver_threads))
        except KeyError:
            pass
        apply_search_strategy(model, pcfg)

    def _build_model(
        self,
        scenarios: tuple[Scenario, ...],
        weights: tuple[float, ...],
        *,
        fixed_design: StorageDesign | None,
        allow_carbon_slack: bool,
    ) -> _PCCArtifacts:
        pcfg = self.planning
        ccfg = self.costs
        horizon = scenarios[0].horizon
        times = range(horizon)
        states = range(horizon + 1)
        dt = float(self.data.delta_t_hours)
        eta_c = float(pcfg.charge_efficiency)
        eta_d = float(pcfg.discharge_efficiency)
        retention = 1.0 - float(pcfg.self_discharge)
        bus = self.feeder.root
        model, quicksum = new_model("single_pcc_storage_planning", pcfg.solver_backend)
        self._configure_model(model)

        site = model.addVar(vtype="B", name=f"site[{bus}]")
        pcap = model.addVar(lb=0.0, ub=pcfg.max_power_mw, name=f"pcap[{bus}]")
        ecap = model.addVar(lb=0.0, ub=pcfg.max_energy_mwh, name=f"ecap[{bus}]")
        peak_grid = model.addVar(lb=0.0, ub=pcfg.grid_limit_mw, name="peak_grid")
        model.addCons(pcap >= pcfg.min_power_mw * site)
        model.addCons(pcap <= pcfg.max_power_mw * site)
        model.addCons(ecap >= pcfg.min_energy_mwh * site)
        model.addCons(ecap <= pcfg.max_energy_mwh * site)
        model.addCons(ecap >= pcfg.min_duration_hours * pcap)
        model.addCons(ecap <= pcfg.max_duration_hours * pcap)
        model.addCons(site <= int(pcfg.max_storage_sites > 0))
        if pcfg.min_storage_sites > 0:
            model.addCons(site == 1)
        if fixed_design is not None:
            installed = int(fixed_design.site[bus] > 0)
            power = float(fixed_design.power_mw[bus]) if installed else 0.0
            energy = float(fixed_design.energy_mwh[bus]) if installed else 0.0
            model.chgVarLb(site, installed)
            model.chgVarUb(site, installed)
            model.chgVarLb(pcap, power)
            model.chgVarUb(pcap, power)
            model.chgVarLb(ecap, energy)
            model.chgVarUb(ecap, energy)

        investment = ccfg.capital_recovery_factor * ccfg.battery_capex_scale * (
            ccfg.site_dollars * site
            + ccfg.power_dollars_per_mw * pcap
            + ccfg.energy_dollars_per_mwh * ecap
        )
        operating_terms: list[object] = []
        carbon_terms: list[object] = []
        ledger_terms: dict[str, list[object]] = {
            "grid_import_mwh": [],
            "diesel_generation_mwh": [],
            "pv_to_dc_mwh": [],
            "storage_charge_mwh": [],
            "storage_discharge_mwh": [],
            "served_dc_energy_mwh": [],
            "source_operational_carbon_t": [],
            "storage_charge_carbon_t": [],
            "storage_discharge_carbon_t": [],
            "delivered_dc_carbon_t": [],
        }
        diagnostic_terms: dict[str, list[object]] = {
            "load_shedding_mwh": [],
            "carbon_excess_t": [],
            "pv_curtailment_mwh": [],
            "served_demand_mwh": [],
        }
        hourly_load_shedding: list[tuple[object, ...]] = []
        hourly_carbon_excess: list[tuple[object, ...]] = []
        annual_blocks = 8760.0 / (horizon * dt)

        for sid, (scenario, weight) in enumerate(zip(scenarios, weights, strict=True)):
            scale = (
                float(scenario.annual_occurrences)
                if scenario.annual_occurrences is not None
                else float(weight) * annual_blocks
            )
            demand = scenario.active_load_mw.sum(axis=(1, 2))
            pv_available = scenario.pv_available_mw.sum(axis=(1, 2))
            grid_load = {}
            grid_charge = {}
            pv_load = {}
            pv_charge = {}
            pv_curtail = {}
            diesel_load = {}
            diesel_charge = {}
            shed = {}
            charge_mode = {}
            discharge_mode = {}

            # Storage layers: initial inventory, two fixed-source aggregates,
            # and one exact grid-carbon vintage for every charging interval.
            layer_names = ("initial", "pv", "diesel") + tuple(
                f"grid_{k}" for k in times
            )
            layer_intensity = {
                "initial": float(pcfg.initial_carbon_intensity),
                "pv": 0.0,
                "diesel": float(pcfg.diesel_carbon_t_per_mwh) / eta_c,
                **{
                    f"grid_{k}": float(scenario.grid_carbon_t_per_mwh[k]) / eta_c
                    for k in times
                },
            }
            layer_energy = {
                (layer, t): model.addVar(
                    lb=0.0,
                    ub=pcfg.max_energy_mwh,
                    name=f"layer_e[{sid},{layer},{t}]",
                )
                for layer in layer_names
                for t in states
            }
            layer_discharge = {
                (layer, t): model.addVar(
                    lb=0.0,
                    ub=pcfg.max_power_mw,
                    name=f"layer_dis[{sid},{layer},{t}]",
                )
                for layer in layer_names
                for t in times
            }
            for layer in layer_names:
                initial = pcfg.initial_soc * ecap if layer == "initial" else 0.0
                model.addCons(layer_energy[layer, 0] == initial)

            horizon_carbon: list[object] = []
            horizon_served: list[object] = []
            scenario_hourly_shed: list[object] = []
            scenario_hourly_carbon_excess: list[object] = []
            for t in times:
                grid_load[t] = model.addVar(
                    lb=0.0, ub=pcfg.grid_limit_mw, name=f"grid_load[{sid},{t}]"
                )
                grid_charge[t] = model.addVar(
                    lb=0.0, ub=pcfg.grid_limit_mw, name=f"grid_charge[{sid},{t}]"
                )
                pv_load[t] = model.addVar(lb=0.0, name=f"pv_load[{sid},{t}]")
                pv_charge[t] = model.addVar(lb=0.0, name=f"pv_charge[{sid},{t}]")
                pv_curtail[t] = model.addVar(lb=0.0, name=f"pv_curtail[{sid},{t}]")
                diesel_load[t] = model.addVar(
                    lb=0.0,
                    ub=pcfg.backup_generator_mw,
                    name=f"diesel_load[{sid},{t}]",
                )
                diesel_charge[t] = model.addVar(
                    lb=0.0,
                    ub=pcfg.backup_generator_mw,
                    name=f"diesel_charge[{sid},{t}]",
                )
                shed[t] = model.addVar(
                    lb=0.0, ub=float(demand[t]), name=f"dc_shed[{sid},{t}]"
                )
                charge_mode[t] = model.addVar(vtype="B", name=f"charge_mode[{sid},{t}]")
                discharge_mode[t] = model.addVar(
                    vtype="B", name=f"discharge_mode[{sid},{t}]"
                )
                total_grid = grid_load[t] + grid_charge[t]
                total_charge = grid_charge[t] + pv_charge[t] + diesel_charge[t]
                total_discharge = quicksum(
                    layer_discharge[layer, t] for layer in layer_names
                )
                served = float(demand[t]) - shed[t]
                scenario_hourly_shed.append(shed[t])
                model.addCons(
                    grid_load[t]
                    + pv_load[t]
                    + diesel_load[t]
                    + total_discharge
                    == served,
                    name=f"pcc_balance[{sid},{t}]",
                )
                model.addCons(
                    pv_load[t] + pv_charge[t] + pv_curtail[t]
                    == float(pv_available[t]),
                    name=f"pv_balance[{sid},{t}]",
                )
                model.addCons(
                    diesel_load[t] + diesel_charge[t] <= pcfg.backup_generator_mw,
                    name=f"diesel_limit[{sid},{t}]",
                )
                model.addCons(
                    total_grid <= pcfg.grid_limit_mw * float(scenario.grid_available[t]),
                    name=f"grid_available[{sid},{t}]",
                )
                model.addCons(peak_grid >= total_grid)
                model.addCons(total_charge <= pcap)
                model.addCons(total_discharge <= pcap)
                model.addCons(total_charge <= pcfg.max_power_mw * charge_mode[t])
                model.addCons(total_discharge <= pcfg.max_power_mw * discharge_mode[t])
                model.addCons(charge_mode[t] + discharge_mode[t] <= site)

                for layer in layer_names:
                    available = retention * layer_energy[layer, t]
                    model.addCons(layer_discharge[layer, t] / eta_d <= available)
                    addition = 0.0
                    if layer == "pv":
                        addition = eta_c * pv_charge[t]
                    elif layer == "diesel":
                        addition = eta_c * diesel_charge[t]
                    elif layer == f"grid_{t}":
                        addition = eta_c * grid_charge[t]
                    model.addCons(
                        layer_energy[layer, t + 1]
                        == available + addition - layer_discharge[layer, t] / eta_d,
                        name=f"layer_balance[{sid},{layer},{t}]",
                    )
                total_energy = quicksum(
                    layer_energy[layer, t + 1] for layer in layer_names
                )
                model.addCons(total_energy >= pcfg.min_soc * ecap)
                model.addCons(total_energy <= pcfg.max_soc * ecap)

                delivered_carbon = (
                    float(scenario.grid_carbon_t_per_mwh[t]) * grid_load[t]
                    + float(pcfg.diesel_carbon_t_per_mwh) * diesel_load[t]
                    + quicksum(
                        layer_intensity[layer]
                        * layer_discharge[layer, t]
                        / eta_d
                        for layer in layer_names
                    )
                )
                storage_discharge_carbon = quicksum(
                    layer_intensity[layer]
                    * layer_discharge[layer, t]
                    / eta_d
                    for layer in layer_names
                )
                source_operational_carbon = (
                    float(scenario.grid_carbon_t_per_mwh[t]) * total_grid
                    + float(pcfg.diesel_carbon_t_per_mwh)
                    * (diesel_load[t] + diesel_charge[t])
                )
                storage_charge_carbon = (
                    float(scenario.grid_carbon_t_per_mwh[t]) * grid_charge[t]
                    + float(pcfg.diesel_carbon_t_per_mwh) * diesel_charge[t]
                )
                ledger_values = {
                    "grid_import_mwh": total_grid,
                    "diesel_generation_mwh": diesel_load[t] + diesel_charge[t],
                    "pv_to_dc_mwh": pv_load[t],
                    "storage_charge_mwh": total_charge,
                    "storage_discharge_mwh": total_discharge,
                    "served_dc_energy_mwh": served,
                    "source_operational_carbon_t": source_operational_carbon,
                    "storage_charge_carbon_t": storage_charge_carbon,
                    "storage_discharge_carbon_t": storage_discharge_carbon,
                    "delivered_dc_carbon_t": delivered_carbon,
                }
                for ledger_name, expression in ledger_values.items():
                    ledger_terms[ledger_name].append(scale * expression * dt)
                if pcfg.carbon_cap_scope == "hourly":
                    carbon_excess = self._add_carbon_cap(
                        model,
                        delivered_carbon,
                        served,
                        scale,
                        dt,
                        sid,
                        t,
                        allow_carbon_slack,
                        carbon_terms,
                    )
                    scenario_hourly_carbon_excess.append(carbon_excess)
                    diagnostic_terms["carbon_excess_t"].append(
                        scale * carbon_excess * dt
                    )
                else:
                    horizon_carbon.append(delivered_carbon * dt)
                    horizon_served.append(served * dt)

                interval_cost = (
                    float(scenario.grid_price_per_mwh[t]) * total_grid
                    + ccfg.generator_dollars_per_mwh
                    * (diesel_load[t] + diesel_charge[t])
                    + ccfg.degradation_dollars_per_mwh
                    * (total_charge + total_discharge)
                    + ccfg.curtailment_dollars_per_mwh * pv_curtail[t]
                    + ccfg.shedding_dollars_per_mwh * shed[t]
                )
                operating_terms.append(scale * interval_cost * dt)
                diagnostic_terms["load_shedding_mwh"].append(scale * shed[t] * dt)
                diagnostic_terms["pv_curtailment_mwh"].append(
                    scale * pv_curtail[t] * dt
                )
                diagnostic_terms["served_demand_mwh"].append(scale * served * dt)

            terminal_energy = quicksum(
                layer_energy[layer, horizon] for layer in layer_names
            )
            model.addCons(terminal_energy == pcfg.initial_soc * ecap)
            terminal_carbon = quicksum(
                layer_intensity[layer] * layer_energy[layer, horizon]
                for layer in layer_names
            )
            model.addCons(
                terminal_carbon
                <= pcfg.initial_carbon_intensity * pcfg.initial_soc * ecap
            )
            if pcfg.carbon_cap_scope == "horizon":
                horizon_excess = self._add_carbon_cap(
                    model,
                    quicksum(horizon_carbon),
                    quicksum(horizon_served),
                    scale,
                    1.0,
                    sid,
                    None,
                    allow_carbon_slack,
                    carbon_terms,
                )
                diagnostic_terms["carbon_excess_t"].append(scale * horizon_excess)
                # A horizon budget has one aggregate excess rather than one
                # value per hour; retain it as a one-element diagnostic row.
                scenario_hourly_carbon_excess.append(horizon_excess)
            hourly_load_shedding.append(tuple(scenario_hourly_shed))
            hourly_carbon_excess.append(tuple(scenario_hourly_carbon_excess))

        operating = ccfg.demand_dollars_per_mw_year * peak_grid + quicksum(
            operating_terms
        )
        carbon_cost = quicksum(carbon_terms) if carbon_terms else 0.0
        carbon_ledger = {
            name: quicksum(terms) if terms else 0.0
            for name, terms in ledger_terms.items()
        }
        diagnostic_totals = {
            name: quicksum(terms) if terms else 0.0
            for name, terms in diagnostic_terms.items()
        }
        model.setObjective(investment + operating + carbon_cost, "minimize")
        return _PCCArtifacts(
            model=model,
            site=site,
            power_capacity=pcap,
            energy_capacity=ecap,
            peak_grid=peak_grid,
            investment=investment,
            operating=operating,
            carbon_cost=carbon_cost,
            carbon_ledger=carbon_ledger,
            diagnostic_totals=diagnostic_totals,
            hourly_load_shedding=tuple(hourly_load_shedding),
            hourly_carbon_excess=tuple(hourly_carbon_excess),
        )

    def _add_carbon_cap(
        self,
        model: object,
        carbon: object,
        served_energy_or_power: object,
        scenario_scale: float,
        dt: float,
        sid: int,
        time: int | None,
        allow_carbon_slack: bool,
        carbon_terms: list[object],
    ) -> object:
        """Apply an hourly intensity cap or a horizon energy-weighted budget."""

        name = f"{sid}" if time is None else f"{sid},{time}"
        if allow_carbon_slack:
            excess = model.addVar(lb=0.0, name=f"carbon_excess[{name}]")
            model.addCons(
                carbon <= self.planning.dc_carbon_cap * served_energy_or_power + excess,
                name=f"dc_carbon_cap[{name}]",
            )
            carbon_terms.append(
                scenario_scale
                * self.costs.carbon_excess_dollars_per_t
                * excess
                * dt
            )
            return excess
        else:
            model.addCons(
                carbon <= self.planning.dc_carbon_cap * served_energy_or_power,
                name=f"dc_carbon_cap[{name}]",
            )
            return 0.0
