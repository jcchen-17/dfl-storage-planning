from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pyscipopt import Model, quicksum

from storage_dfl.config import CostConfig, DataConfig, PlanningConfig
from storage_dfl.data import Scenario
from storage_dfl.network import Feeder
from storage_dfl.planning.results import (
    PlanningResult,
    StorageDesign,
    infeasible_result,
)


@dataclass(frozen=True)
class _ModelArtifacts:
    model: Model
    site: dict[str, object]
    power_capacity: dict[str, object]
    energy_capacity: dict[str, object]
    peak_grid: object
    investment_expression: object
    operating_expression: object
    carbon_slack_expression: object


class StoragePlanningOracle:
    """SCIP implementation of the paper's nonconvex planning formulation.

    The horizon is configurable.  Each interval creates one candidate carbon
    vintage.  Carbon mass dynamics are represented by the exact identity
    ``mass = vintage_intensity * vintage_energy`` plus the vintage energy
    dynamics; writing the algebraically redundant mass recursion separately is
    intentionally avoided for numerical stability.
    """

    def __init__(
        self,
        feeder: Feeder,
        planning: PlanningConfig,
        costs: CostConfig,
        data: DataConfig,
    ) -> None:
        self.feeder = feeder
        self.planning = planning
        self.costs = costs
        self.data = data
        self._cache: dict[tuple, PlanningResult] = {}

    def solve(
        self,
        scenarios: Iterable[Scenario],
        *,
        weights: Iterable[float] | None = None,
        fixed_design: StorageDesign | None = None,
        allow_carbon_slack: bool = False,
        use_cache: bool = False,
    ) -> PlanningResult:
        scenario_tuple = tuple(scenarios)
        if not scenario_tuple:
            raise ValueError("At least one scenario is required.")
        horizon = scenario_tuple[0].horizon
        if any(s.horizon != horizon for s in scenario_tuple):
            raise ValueError("All scenarios must have the same horizon.")
        weight_tuple = tuple(weights) if weights is not None else tuple(
            1.0 / len(scenario_tuple) for _ in scenario_tuple
        )
        if len(weight_tuple) != len(scenario_tuple):
            raise ValueError("The number of weights must match the scenarios.")
        total_weight = sum(weight_tuple)
        if total_weight <= 0:
            raise ValueError("Scenario weights must have a positive sum.")
        weight_tuple = tuple(weight / total_weight for weight in weight_tuple)

        cache_key = self._cache_key(
            scenario_tuple,
            weight_tuple,
            fixed_design,
            allow_carbon_slack,
        )
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        artifacts = self._build_model(
            scenario_tuple,
            weight_tuple,
            fixed_design=fixed_design,
            allow_carbon_slack=allow_carbon_slack,
        )
        model = artifacts.model
        model.optimize()
        status = str(model.getStatus())
        scenario_names = tuple(scenario.name for scenario in scenario_tuple)
        solution = model.getBestSol()
        if solution is None:
            result = infeasible_result(
                status,
                self.feeder.storage_candidates,
                scenario_names,
                model.getSolvingTime(),
            )
        else:
            def expression_value(expression: object) -> float:
                if isinstance(expression, (int, float)):
                    return float(expression)
                return float(model.getVal(expression))

            installed = {
                bus: int(model.getVal(artifacts.site[bus]) > 0.5)
                for bus in self.feeder.storage_candidates
            }
            design = StorageDesign(
                site=installed,
                power_mw={
                    bus: max(0.0, float(model.getVal(artifacts.power_capacity[bus])))
                    if installed[bus]
                    else 0.0
                    for bus in self.feeder.storage_candidates
                },
                energy_mwh={
                    bus: max(0.0, float(model.getVal(artifacts.energy_capacity[bus])))
                    if installed[bus]
                    else 0.0
                    for bus in self.feeder.storage_candidates
                },
            )
            result = PlanningResult(
                status=status,
                objective=float(model.getObjVal()),
                investment_cost=expression_value(artifacts.investment_expression),
                operating_cost=expression_value(artifacts.operating_expression),
                carbon_slack_cost=max(0.0, expression_value(artifacts.carbon_slack_expression)),
                peak_grid_mw=float(model.getVal(artifacts.peak_grid)),
                design=design,
                scenario_names=scenario_names,
                solve_time_seconds=float(model.getSolvingTime()),
                relative_gap=float(model.getGap()),
            )
        if use_cache:
            self._cache[cache_key] = result
        return result

    def _cache_key(
        self,
        scenarios: tuple[Scenario, ...],
        weights: tuple[float, ...],
        fixed_design: StorageDesign | None,
        allow_carbon_slack: bool,
    ) -> tuple:
        design_key = None
        if fixed_design is not None:
            design_key = tuple(
                (
                    bus,
                    fixed_design.site[bus],
                    round(fixed_design.power_mw[bus], 6),
                    round(fixed_design.energy_mwh[bus], 6),
                )
                for bus in self.feeder.storage_candidates
            )
        return (
            tuple(scenario.name for scenario in scenarios),
            tuple(round(weight, 8) for weight in weights),
            design_key,
            allow_carbon_slack,
        )

    def _build_model(
        self,
        scenarios: tuple[Scenario, ...],
        weights: tuple[float, ...],
        *,
        fixed_design: StorageDesign | None,
        allow_carbon_slack: bool,
    ) -> _ModelArtifacts:
        feeder = self.feeder
        pcfg = self.planning
        ccfg = self.costs
        horizon = scenarios[0].horizon
        dt = self.data.delta_t_hours
        buses = feeder.buses
        candidates = feeder.storage_candidates
        bus_index = feeder.bus_index
        root = feeder.root
        line_keys = tuple((line.parent, line.child) for line in feeder.lines)
        line_data = {(line.parent, line.child): line for line in feeder.lines}
        outgoing = {bus: [key for key in line_keys if key[0] == bus] for bus in buses}
        incoming = {bus: [key for key in line_keys if key[1] == bus] for bus in buses}

        model = Model("dfl_batch_resolved_storage_planning")
        if not pcfg.verbose_solver:
            model.hideOutput()
        model.setParam("limits/time", float(pcfg.solver_time_limit_seconds))
        model.setParam("limits/gap", float(pcfg.solver_relative_gap))
        if fixed_design is not None:
            # SCIP can incorrectly cut off the fixed-design problem while
            # eliminating zero-capacity bilinear vintage equations. Keeping the
            # original equations avoids that presolve artifact in validation.
            model.setParam("presolving/maxrounds", 0)
        try:
            model.setParam("parallel/maxnthreads", int(pcfg.solver_threads))
        except KeyError:
            pass

        site = {bus: model.addVar(vtype="B", name=f"site[{bus}]") for bus in candidates}
        pcap = {
            bus: model.addVar(lb=0.0, ub=pcfg.max_power_mw, name=f"pcap[{bus}]")
            for bus in candidates
        }
        ecap = {
            bus: model.addVar(lb=0.0, ub=pcfg.max_energy_mwh, name=f"ecap[{bus}]")
            for bus in candidates
        }
        peak_grid = model.addVar(lb=0.0, ub=pcfg.grid_limit_mw, name="peak_grid")
        for bus in candidates:
            model.addCons(pcap[bus] >= pcfg.min_power_mw * site[bus])
            model.addCons(pcap[bus] <= pcfg.max_power_mw * site[bus])
            model.addCons(ecap[bus] >= pcfg.min_energy_mwh * site[bus])
            model.addCons(ecap[bus] <= pcfg.max_energy_mwh * site[bus])
            model.addCons(ecap[bus] >= pcfg.min_duration_hours * pcap[bus])
            model.addCons(ecap[bus] <= pcfg.max_duration_hours * pcap[bus])
        model.addCons(quicksum(site.values()) <= pcfg.max_storage_sites)

        if fixed_design is not None:
            for bus in candidates:
                site_value = float(int(fixed_design.site[bus]))
                power_value = float(fixed_design.power_mw[bus]) if site_value > 0.5 else 0.0
                energy_value = float(fixed_design.energy_mwh[bus]) if site_value > 0.5 else 0.0
                # Bounds are numerically more reliable than three additional
                # equalities when SCIP presolves the bilinear carbon equations.
                model.chgVarLb(site[bus], site_value)
                model.chgVarUb(site[bus], site_value)
                model.chgVarLb(pcap[bus], power_value)
                model.chgVarUb(pcap[bus], power_value)
                model.chgVarLb(ecap[bus], energy_value)
                model.chgVarUb(ecap[bus], energy_value)

        investment = ccfg.capital_recovery_factor * quicksum(
            ccfg.site_dollars * site[bus]
            + ccfg.power_dollars_per_mw * pcap[bus]
            + ccfg.energy_dollars_per_mwh * ecap[bus]
            for bus in candidates
        )

        operating_terms = []
        carbon_slack_terms = []
        annual_blocks = 8760.0 / (horizon * dt)
        retention = 1.0 - pcfg.self_discharge
        epsilon_charge = 1.0e-4
        big_m_carbon = pcfg.carbon_intensity_max

        for scenario_index, (scenario, scenario_weight) in enumerate(zip(scenarios, weights)):
            sid = scenario_index
            times = range(horizon)
            states = range(horizon + 1)
            layers = range(horizon + 1)

            processed = {t: model.addVar(lb=0.0, ub=1.2, name=f"work[{sid},{t}]") for t in times}
            backlog = {t: model.addVar(lb=0.0, ub=2.5, name=f"backlog[{sid},{t}]") for t in states}
            shed = {t: model.addVar(lb=0.0, name=f"shed[{sid},{t}]") for t in times}
            grid = {t: model.addVar(lb=0.0, ub=pcfg.grid_limit_mw, name=f"grid[{sid},{t}]") for t in times}
            q_grid = {t: model.addVar(lb=-pcfg.grid_limit_mw, ub=pcfg.grid_limit_mw, name=f"qgrid[{sid},{t}]") for t in times}
            generator = {t: model.addVar(lb=0.0, ub=1.0, name=f"gen[{sid},{t}]") for t in times}
            q_generator = {t: model.addVar(lb=-0.8, ub=0.8, name=f"qgen[{sid},{t}]") for t in times}
            pv = {(bus, t): model.addVar(lb=0.0, name=f"pv[{sid},{bus},{t}]") for bus in buses for t in times}
            curtail = {(bus, t): model.addVar(lb=0.0, name=f"curt[{sid},{bus},{t}]") for bus in buses for t in times}

            flow_p = {(edge, t): model.addVar(lb=-line_data[edge].rating_mva, ub=line_data[edge].rating_mva, name=f"p[{sid},{edge[0]},{edge[1]},{t}]") for edge in line_keys for t in times}
            flow_q = {(edge, t): model.addVar(lb=-line_data[edge].rating_mva, ub=line_data[edge].rating_mva, name=f"q[{sid},{edge[0]},{edge[1]},{t}]") for edge in line_keys for t in times}
            flow_plus = {(edge, t): model.addVar(lb=0.0, ub=line_data[edge].rating_mva, name=f"pplus[{sid},{edge[0]},{edge[1]},{t}]") for edge in line_keys for t in times}
            flow_minus = {(edge, t): model.addVar(lb=0.0, ub=line_data[edge].rating_mva, name=f"pminus[{sid},{edge[0]},{edge[1]},{t}]") for edge in line_keys for t in times}
            direction = {(edge, t): model.addVar(vtype="B", name=f"forward[{sid},{edge[0]},{edge[1]},{t}]") for edge in line_keys for t in times}
            voltage = {(bus, t): model.addVar(lb=0.95**2, ub=1.05**2, name=f"v[{sid},{bus},{t}]") for bus in buses for t in times}
            nodal_carbon = {(bus, t): model.addVar(lb=0.0, ub=pcfg.carbon_intensity_max, name=f"ci[{sid},{bus},{t}]") for bus in buses for t in times}
            carbon_slack = {
                (bus, t): model.addVar(lb=0.0, ub=pcfg.carbon_intensity_max, name=f"cslack[{sid},{bus},{t}]")
                for bus in buses
                for t in times
            } if allow_carbon_slack else {}

            charge_status = {(bus, t): model.addVar(vtype="B", name=f"uch[{sid},{bus},{t}]") for bus in candidates for t in times}
            discharge_status = {(bus, t): model.addVar(vtype="B", name=f"udc[{sid},{bus},{t}]") for bus in candidates for t in times}
            layer_active = {(bus, k): model.addVar(vtype="B", name=f"ulayer[{sid},{bus},{k}]") for bus in candidates for k in range(1, horizon + 1)}
            vintage_carbon = {(bus, k): model.addVar(lb=0.0, ub=pcfg.carbon_intensity_max, name=f"vintage_ci[{sid},{bus},{k}]") for bus in candidates for k in layers}
            layer_charge = {(bus, k): model.addVar(lb=0.0, ub=pcfg.max_power_mw, name=f"layer_ch[{sid},{bus},{k}]") for bus in candidates for k in range(1, horizon + 1)}
            layer_discharge = {(bus, k, t): model.addVar(lb=0.0, ub=pcfg.max_power_mw, name=f"layer_dc[{sid},{bus},{k},{t}]") for bus in candidates for k in layers for t in times}
            layer_energy = {(bus, k, t): model.addVar(lb=0.0, ub=pcfg.max_energy_mwh, name=f"layer_e[{sid},{bus},{k},{t}]") for bus in candidates for k in layers for t in states}
            layer_carbon_mass = {(bus, k, t): model.addVar(lb=0.0, ub=pcfg.carbon_intensity_max * pcfg.max_energy_mwh, name=f"layer_m[{sid},{bus},{k},{t}]") for bus in candidates for k in layers for t in states}

            model.addCons(backlog[0] == 0.0)
            model.addCons(backlog[horizon] == 0.0)
            for t in times:
                model.addCons(backlog[t + 1] == backlog[t] + float(scenario.workload_arrival[t]) - processed[t])
                horizon_deadline = min(horizon - 1, t + 2)
                model.addCons(
                    quicksum(processed[tau] for tau in range(horizon_deadline + 1))
                    >= float(scenario.workload_arrival[: t + 1].sum())
                )
                dc_power = 0.03 + float(scenario.pue[t]) * (0.12 + 0.30 * processed[t])
                model.addCons(shed[t] <= dc_power - 0.12)
                model.addCons(grid[t] <= float(scenario.grid_available[t]) * pcfg.grid_limit_mw)
                model.addCons(grid[t] <= peak_grid)
                for bus in buses:
                    available = float(scenario.pv_available_mw[t, bus_index[bus]])
                    model.addCons(pv[bus, t] + curtail[bus, t] == available)

            for t in range(1, horizon):
                model.addCons(generator[t] - generator[t - 1] <= 0.60)
                model.addCons(generator[t - 1] - generator[t] <= 0.60)

            charge_power: dict[tuple[str, int], object] = {}
            discharge_power: dict[tuple[str, int], object] = {}
            storage_energy: dict[tuple[str, int], object] = {}
            for bus in candidates:
                model.addCons(vintage_carbon[bus, 0] == pcfg.initial_carbon_intensity * site[bus])
                model.addCons(layer_energy[bus, 0, 0] == pcfg.initial_soc * ecap[bus])
                model.addCons(layer_carbon_mass[bus, 0, 0] == pcfg.initial_carbon_intensity * layer_energy[bus, 0, 0])
                for k in range(1, horizon + 1):
                    model.addCons(layer_energy[bus, k, 0] == 0.0)
                    model.addCons(layer_carbon_mass[bus, k, 0] == 0.0)
                    model.addCons(layer_charge[bus, k] >= epsilon_charge * layer_active[bus, k])
                    model.addCons(layer_charge[bus, k] <= pcfg.max_power_mw * layer_active[bus, k])
                    formation_t = k - 1
                    model.addCons(
                        vintage_carbon[bus, k] - nodal_carbon[bus, formation_t]
                        <= big_m_carbon * (1 - layer_active[bus, k])
                    )
                    model.addCons(
                        nodal_carbon[bus, formation_t] - vintage_carbon[bus, k]
                        <= big_m_carbon * (1 - layer_active[bus, k])
                    )
                    model.addCons(vintage_carbon[bus, k] <= pcfg.carbon_intensity_max * layer_active[bus, k])

                for t in times:
                    charge_power[bus, t] = layer_charge[bus, t + 1]
                    discharge_power[bus, t] = quicksum(layer_discharge[bus, k, t] for k in layers)
                    model.addCons(charge_status[bus, t] <= site[bus])
                    model.addCons(discharge_status[bus, t] <= site[bus])
                    model.addCons(charge_status[bus, t] + discharge_status[bus, t] <= 1)
                    model.addCons(charge_power[bus, t] <= pcap[bus])
                    model.addCons(discharge_power[bus, t] <= pcap[bus])
                    model.addCons(charge_power[bus, t] <= pcfg.max_power_mw * charge_status[bus, t])
                    model.addCons(discharge_power[bus, t] <= pcfg.max_power_mw * discharge_status[bus, t])

                    for k in layers:
                        activation = site[bus] if k == 0 else layer_active[bus, k]
                        if k > 0 and t < k - 1:
                            model.addCons(layer_discharge[bus, k, t] == 0.0)
                        model.addCons(layer_discharge[bus, k, t] <= pcfg.max_power_mw * activation)
                        model.addCons(
                            layer_discharge[bus, k, t] * dt / pcfg.discharge_efficiency
                            <= retention * layer_energy[bus, k, t]
                        )
                        charging = layer_charge[bus, k] if k > 0 and t == k - 1 else 0.0
                        model.addCons(
                            layer_energy[bus, k, t + 1]
                            == retention * layer_energy[bus, k, t]
                            + pcfg.charge_efficiency * charging * dt
                            - layer_discharge[bus, k, t] * dt / pcfg.discharge_efficiency
                        )
                    storage_energy[bus, t] = quicksum(layer_energy[bus, k, t] for k in layers)
                    model.addCons(storage_energy[bus, t] >= pcfg.min_soc * ecap[bus])
                    model.addCons(storage_energy[bus, t] <= pcfg.max_soc * ecap[bus])

                terminal_energy = quicksum(layer_energy[bus, k, horizon] for k in layers)
                terminal_mass = quicksum(layer_carbon_mass[bus, k, horizon] for k in layers)
                model.addCons(terminal_energy == layer_energy[bus, 0, 0])
                model.addCons(terminal_mass == layer_carbon_mass[bus, 0, 0])
                for k in layers:
                    activation = site[bus] if k == 0 else layer_active[bus, k]
                    for t in states:
                        model.addCons(layer_energy[bus, k, t] <= pcfg.max_energy_mwh * activation)
                        model.addCons(
                            layer_carbon_mass[bus, k, t]
                            <= pcfg.carbon_intensity_max * pcfg.max_energy_mwh * activation
                        )
                        model.addCons(
                            layer_carbon_mass[bus, k, t]
                            == vintage_carbon[bus, k] * layer_energy[bus, k, t]
                        )

            for t in times:
                for edge in line_keys:
                    line = line_data[edge]
                    model.addCons(flow_p[edge, t] == flow_plus[edge, t] - flow_minus[edge, t])
                    model.addCons(flow_plus[edge, t] <= line.rating_mva * direction[edge, t])
                    model.addCons(flow_minus[edge, t] <= line.rating_mva * (1 - direction[edge, t]))
                    model.addCons(flow_p[edge, t] * flow_p[edge, t] + flow_q[edge, t] * flow_q[edge, t] <= line.rating_mva**2)
                    model.addCons(
                        voltage[edge[1], t]
                        == voltage[edge[0], t]
                        - 2.0 * (line.resistance * flow_p[edge, t] + line.reactance * flow_q[edge, t])
                    )
                model.addCons(voltage[root, t] == 1.0)

                for bus in buses:
                    load_p = float(scenario.active_load_mw[t, bus_index[bus]])
                    load_q = float(scenario.reactive_load_mvar[t, bus_index[bus]])
                    if bus == feeder.data_center_bus:
                        dc_power = 0.03 + float(scenario.pue[t]) * (0.12 + 0.30 * processed[t])
                        load_p = load_p + dc_power - shed[t]
                    local_generation = generator[t] if bus == feeder.generator_bus else 0.0
                    local_q_generation = q_generator[t] if bus == feeder.generator_bus else 0.0
                    grid_injection = grid[t] if bus == root else 0.0
                    grid_q_injection = q_grid[t] if bus == root else 0.0
                    storage_charge = charge_power[bus, t] if bus in candidates else 0.0
                    storage_discharge = discharge_power[bus, t] if bus in candidates else 0.0
                    outgoing_p = quicksum(flow_p[edge, t] for edge in outgoing[bus])
                    incoming_p = quicksum(flow_p[edge, t] for edge in incoming[bus])
                    outgoing_q = quicksum(flow_q[edge, t] for edge in outgoing[bus])
                    incoming_q = quicksum(flow_q[edge, t] for edge in incoming[bus])
                    model.addCons(
                        local_generation + pv[bus, t] + grid_injection + storage_discharge
                        - load_p - storage_charge
                        == outgoing_p - incoming_p
                    )
                    model.addCons(
                        local_q_generation + grid_q_injection - load_q
                        == outgoing_q - incoming_q
                    )

                    parent_forward = quicksum(flow_plus[edge, t] for edge in incoming[bus])
                    child_reverse = quicksum(flow_minus[edge, t] for edge in outgoing[bus])
                    total_incoming_power = (
                        local_generation
                        + pv[bus, t]
                        + grid_injection
                        + storage_discharge
                        + parent_forward
                        + child_reverse
                    )
                    storage_carbon = (
                        quicksum(
                            vintage_carbon[bus, k] * layer_discharge[bus, k, t]
                            for k in layers
                        )
                        if bus in candidates
                        else 0.0
                    )
                    network_carbon = quicksum(
                        nodal_carbon[edge[0], t] * flow_plus[edge, t]
                        for edge in incoming[bus]
                    ) + quicksum(
                        nodal_carbon[edge[1], t] * flow_minus[edge, t]
                        for edge in outgoing[bus]
                    )
                    total_incoming_carbon = (
                        0.72 * local_generation
                        + (float(scenario.grid_carbon_t_per_mwh[t]) * grid[t] if bus == root else 0.0)
                        + storage_carbon
                        + network_carbon
                    )
                    model.addCons(nodal_carbon[bus, t] * total_incoming_power == total_incoming_carbon)
                    cap = pcfg.dc_carbon_cap if bus == feeder.data_center_bus else pcfg.other_bus_carbon_cap
                    slack = carbon_slack[bus, t] if allow_carbon_slack else 0.0
                    model.addCons(nodal_carbon[bus, t] <= cap + slack)
                    if allow_carbon_slack:
                        carbon_slack_terms.append(
                            scenario_weight
                            * annual_blocks
                            * ccfg.validation_carbon_slack_dollars
                            * carbon_slack[bus, t]
                        )

                interval_cost = (
                    float(scenario.grid_price_per_mwh[t]) * grid[t]
                    + 125.0 * generator[t]
                    + ccfg.degradation_dollars_per_mwh
                    * quicksum(charge_power[bus, t] + discharge_power[bus, t] for bus in candidates)
                    + ccfg.delay_dollars_per_task_hour * backlog[t]
                    + ccfg.curtailment_dollars_per_mwh * quicksum(curtail[bus, t] for bus in buses)
                    + ccfg.shedding_dollars_per_mwh * shed[t]
                )
                operating_terms.append(scenario_weight * annual_blocks * interval_cost * dt)

        operating = ccfg.demand_dollars_per_mw_year * peak_grid + quicksum(operating_terms)
        carbon_slack_cost = quicksum(carbon_slack_terms) if carbon_slack_terms else 0.0
        model.setObjective(investment + operating + carbon_slack_cost, "minimize")
        return _ModelArtifacts(
            model=model,
            site=site,
            power_capacity=pcap,
            energy_capacity=ecap,
            peak_grid=peak_grid,
            investment_expression=investment,
            operating_expression=operating,
            carbon_slack_expression=carbon_slack_cost,
        )
