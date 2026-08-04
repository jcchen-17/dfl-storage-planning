from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from pyscipopt import Model, quicksum

from storage_dfl.config import CostConfig, DataCenterConfig, DataConfig, PlanningConfig
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


def _add_mccormick_envelope(
    model: Model,
    product: object,
    first: object,
    second: object,
    first_upper: float,
    second_upper: float,
) -> None:
    """Relax product = first * second for nonnegative bounded variables."""
    model.addCons(product >= 0.0)
    model.addCons(
        product
        >= first_upper * second
        + second_upper * first
        - first_upper * second_upper
    )
    model.addCons(product <= first_upper * second)
    model.addCons(product <= second_upper * first)


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
        data_center: DataCenterConfig | None = None,
    ) -> None:
        self.feeder = feeder
        self.planning = planning
        self.costs = costs
        self.data = data
        # Optional so callers written before the facility scale became tunable
        # keep the original edge-scale coefficients.
        self.data_center = data_center or DataCenterConfig()
        if planning.carbon_formulation not in {
            "exact",
            "mccormick",
            "aggregate_mccormick",
            "system_average",
        }:
            raise ValueError(
                "carbon_formulation must be 'exact', 'mccormick', or "
                "'aggregate_mccormick', or 'system_average'"
            )
        if planning.flow_limit_formulation not in {"quadratic", "polygon"}:
            raise ValueError(
                "flow_limit_formulation must be 'quadratic' or 'polygon'"
            )
        self._cache: dict[tuple, PlanningResult] = {}
        # The no-storage bootstrap depends only on the scenario set, the weights
        # and the carbon-slack switch -- never on fixed_design.  Every solve used
        # to repeat it, which on the 48 h case cost up to 90 s per solve and
        # dominated DFL training.  Cache the resulting variable assignment so one
        # bootstrap serves the whole fixed validation set.
        self._warm_start_cache: dict[tuple, dict[str, float] | None] = {}
        # Consumed by the next solve() so the isolated worker can adopt the
        # parent's cached bootstrap without reconstructing its cache key.
        self._injected_warm_start: tuple[bool, dict[str, float] | None] | None = None

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

        warm_start_key = self._warm_start_key(
            scenario_tuple, weight_tuple, allow_carbon_slack
        )
        if self._injected_warm_start is not None:
            injected_cached, injected_values = self._injected_warm_start
            self._injected_warm_start = None
            if injected_cached:
                self._warm_start_cache[warm_start_key] = injected_values
        warm_start_cached = (
            not self._warm_start_supported(fixed_design)
            or warm_start_key in self._warm_start_cache
        )
        warm_start_values = self._warm_start_cache.get(warm_start_key)

        # The user's Windows environment contains different Intel OpenMP DLLs
        # from conda NumPy and the pip PyTorch wheel.  Repeated SCIP solves after
        # torch inference can initialize both runtimes.  Run each planning solve
        # in a clean Python worker that never imports torch; this is deterministic
        # and avoids the unsafe KMP_DUPLICATE_LIB_OK workaround.
        if (
            os.name == "nt"
            and "torch" in sys.modules
            and os.environ.get("STORAGE_DFL_SOLVER_WORKER") != "1"
        ):
            result, worker_warm_start, worker_computed = self._solve_isolated(
                scenario_tuple,
                weight_tuple,
                fixed_design,
                allow_carbon_slack,
                use_cache,
                warm_start_values,
                warm_start_cached,
            )
            if worker_computed:
                self._warm_start_cache[warm_start_key] = worker_warm_start
            if use_cache:
                self._cache[cache_key] = result
            return result

        if not warm_start_cached:
            warm_start_values = self._no_storage_bootstrap_values(
                scenario_tuple,
                weight_tuple,
                allow_carbon_slack,
            )
            self._warm_start_cache[warm_start_key] = warm_start_values

        artifacts = self._build_model(
            scenario_tuple,
            weight_tuple,
            fixed_design=fixed_design,
            allow_carbon_slack=allow_carbon_slack,
        )
        model = artifacts.model
        warm_start_accepted = self._apply_no_storage_warm_start(
            model,
            warm_start_values,
            scenario_tuple,
            fixed_design,
        )
        if fixed_design is None and not warm_start_accepted:
            warm_start = model.createPartialSol()
            for bus in self.feeder.storage_candidates:
                model.setSolVal(warm_start, artifacts.site[bus], 0.0)
                model.setSolVal(warm_start, artifacts.power_capacity[bus], 0.0)
                model.setSolVal(warm_start, artifacts.energy_capacity[bus], 0.0)
            model.addSol(warm_start)
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

            # Rounding at 0.5 hides a fractional binary: the design would be
            # reported as a whole site while investment_cost still charges the
            # fractional one, so the two disagree with no visible sign of it.
            site_values = {
                bus: float(model.getVal(artifacts.site[bus]))
                for bus in self.feeder.storage_candidates
            }
            for bus, value in site_values.items():
                deviation = min(abs(value), abs(value - 1.0))
                if deviation > 1.0e-4:
                    print(
                        f"WARNING site[{bus}] came back at {value:.6f}, which is not "
                        f"integral (off by {deviation:.6f}); the reported design "
                        f"rounds it but investment_cost does not, so the two are "
                        f"inconsistent for this solution.",
                        flush=True,
                    )
            installed = {bus: int(value > 0.5) for bus, value in site_values.items()}
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
        # Release the SCIP problem before another scenario/design validation
        # model is created.  On Windows this also prevents overlapping solver
        # thread pools from initializing a second Intel OpenMP runtime.
        model.freeProb()
        return result

    def _warm_start_supported(self, fixed_design: StorageDesign | None) -> bool:
        """Whether an idle-storage seed is a valid solution of the target model.

        With self-discharge the prescribed inventory decays, so holding every
        storage trajectory at its starting level violates the energy dynamics
        and the seed would be rejected anyway.  Checking first avoids paying for
        a bootstrap solve whose result cannot be used.
        """

        return fixed_design is None or self.planning.self_discharge == 0.0

    def _warm_start_key(
        self,
        scenarios: tuple[Scenario, ...],
        weights: tuple[float, ...],
        allow_carbon_slack: bool,
    ) -> tuple:
        return (
            tuple(scenario.name for scenario in scenarios),
            tuple(round(weight, 8) for weight in weights),
            allow_carbon_slack,
        )

    def _no_storage_bootstrap_values(
        self,
        scenarios: tuple[Scenario, ...],
        weights: tuple[float, ...],
        allow_carbon_slack: bool,
    ) -> dict[str, float] | None:
        """Solve the storage-disabled model and return its variable assignment.

        The result depends only on the arguments, so callers cache it by
        ``_warm_start_key`` and reuse it for every design evaluated against the
        same scenarios.  Returns None when SCIP found no bootstrap incumbent.
        """

        # The bootstrap is itself a nonconvex MINLP: dropping storage removes the
        # vintage layers but keeps the bilinear nodal-carbon equalities and the
        # flow-direction binaries.  Capping it at a constant made the budget
        # independent of solver_time_limit_seconds, so the bootstrap timed out,
        # returned no solution, and the main solve was left without any incumbent.
        # On the 120 h case SCIP typically needs a little over 30 seconds before
        # it discovers the strong zero-storage operating point.  A 30 s/20%
        # bootstrap therefore seeded the main model with a much worse incumbent,
        # which could make an optional-storage solve report a dominated minimum-
        # capacity installation.  Give the bootstrap enough time and use the
        # same requested gap as the parent solve.
        configured_limit = float(self.planning.warm_start_time_limit_seconds)
        bootstrap_limit = (
            configured_limit
            if configured_limit > 0.0
            else max(60.0, 0.3 * self.planning.solver_time_limit_seconds)
        )
        bootstrap_planning = replace(
            self.planning,
            max_storage_sites=0,
            solver_time_limit_seconds=bootstrap_limit,
            solver_relative_gap=self.planning.solver_relative_gap,
        )
        bootstrap_oracle = StoragePlanningOracle(
            self.feeder,
            bootstrap_planning,
            self.costs,
            self.data,
            self.data_center,
        )
        bootstrap = bootstrap_oracle._build_model(
            scenarios,
            weights,
            fixed_design=None,
            allow_carbon_slack=allow_carbon_slack,
        )
        bootstrap.model.optimize()
        bootstrap_solution = bootstrap.model.getBestSol()
        if bootstrap_solution is None:
            bootstrap.model.freeProb()
            return None
        values = {
            variable.name: float(
                bootstrap.model.getSolVal(bootstrap_solution, variable)
            )
            for variable in bootstrap.model.getVars()
        }
        bootstrap.model.freeProb()
        return values

    def _apply_no_storage_warm_start(
        self,
        target_model: Model,
        bootstrap_values: dict[str, float] | None,
        scenarios: tuple[Scenario, ...],
        fixed_design: StorageDesign | None,
    ) -> bool:
        """Seed a full operating point with storage idle.

        For free planning this is the no-storage incumbent. For fixed-design
        validation it retains the prescribed investment but initializes every
        storage trajectory at its starting inventory with zero charge/discharge.
        The latter guarantees that validation cannot return an incumbent worse
        than no-storage operation plus the fixed annual investment merely
        because SCIP struggled to discover an idle battery schedule.
        """

        if bootstrap_values is None or not self._warm_start_supported(fixed_design):
            return False
        target_variables = tuple(target_model.getVars())
        target_by_name = {variable.name: variable for variable in target_variables}
        warm_start = target_model.createSol()
        # The storage-disabled bootstrap omits charge/discharge, inventory and
        # carbon-vintage variables that exist in the storage-enabled target.
        # A full SCIP solution leaves no unspecified entries, so initialize all
        # target-only variables to their valid zero-storage value before copying
        # the common operating point by name.
        for variable in target_variables:
            target_model.setSolVal(warm_start, variable, 0.0)
        for name, value in bootstrap_values.items():
            target = target_by_name.get(name)
            if target is not None:
                target_model.setSolVal(warm_start, target, value)
        if fixed_design is not None:
            for bus in self.feeder.storage_candidates:
                installed = int(fixed_design.site[bus] > 0)
                power = float(fixed_design.power_mw[bus]) if installed else 0.0
                energy = float(fixed_design.energy_mwh[bus]) if installed else 0.0
                inventory = self.planning.initial_soc * energy
                carbon_mass = self.planning.initial_carbon_intensity * inventory
                for name, value in (
                    (f"site[{bus}]", installed),
                    (f"pcap[{bus}]", power),
                    (f"ecap[{bus}]", energy),
                ):
                    target = target_by_name.get(name)
                    if target is not None:
                        target_model.setSolVal(warm_start, target, float(value))
                for sid in range(len(scenarios)):
                    for t in range(scenarios[0].horizon + 1):
                        for name, value in (
                            (f"aggregate_e[{sid},{bus},{t}]", inventory),
                            (f"aggregate_m[{sid},{bus},{t}]", carbon_mass),
                            (
                                f"aggregate_ci[{sid},{bus},{t}]",
                                self.planning.initial_carbon_intensity * installed,
                            ),
                            (f"layer_e[{sid},{bus},0,{t}]", inventory),
                            (f"layer_m[{sid},{bus},0,{t}]", carbon_mass),
                        ):
                            target = target_by_name.get(name)
                            if target is not None:
                                target_model.setSolVal(warm_start, target, float(value))
                    target = target_by_name.get(f"vintage_ci[{sid},{bus},0]")
                    if target is not None:
                        target_model.setSolVal(
                            warm_start,
                            target,
                            self.planning.initial_carbon_intensity * installed,
                        )
        accepted = bool(target_model.addSol(warm_start))
        if not accepted:
            print(
                "WARNING: SCIP rejected the complete no-storage warm start; "
                "the storage-enabled incumbent may be weaker than a separately "
                "solved no-storage plan.",
                flush=True,
            )
        return accepted

    def _solve_isolated(
        self,
        scenarios: tuple[Scenario, ...],
        weights: tuple[float, ...],
        fixed_design: StorageDesign | None,
        allow_carbon_slack: bool,
        use_cache: bool,
        warm_start_values: dict[str, float] | None,
        warm_start_cached: bool,
    ) -> tuple[PlanningResult, dict[str, float] | None, bool]:
        """Run one solve in a clean worker and report any bootstrap it computed.

        The worker is a fresh process, so it cannot see the parent's warm-start
        cache.  Passing the cached assignment in and handing a newly computed one
        back keeps a single bootstrap per scenario set across the whole run.
        """

        payload = {
            "feeder": self.feeder,
            "planning": self.planning,
            "costs": self.costs,
            "data": self.data,
            # Without this the worker would silently rebuild the facility at the
            # default edge scale while the parent believes it was rescaled.
            "data_center": self.data_center,
            "scenarios": scenarios,
            "weights": weights,
            "fixed_design": fixed_design,
            "allow_carbon_slack": allow_carbon_slack,
            "use_cache": use_cache,
            "warm_start_values": warm_start_values,
            "warm_start_cached": warm_start_cached,
        }
        with tempfile.TemporaryDirectory(prefix="storage_dfl_solver_") as directory:
            directory_path = Path(directory)
            input_path = directory_path / "input.pkl"
            output_path = directory_path / "output.pkl"
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
                    "Isolated SCIP worker failed:\n"
                    + completed.stdout
                    + completed.stderr
                )
            response = pickle.loads(output_path.read_bytes())
        if not isinstance(response, dict) or not isinstance(
            response.get("result"), PlanningResult
        ):
            raise TypeError("The isolated SCIP worker returned an invalid result.")
        return (
            response["result"],
            response.get("warm_start_values"),
            bool(response.get("warm_start_computed", False)),
        )

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
        dccfg = self.data_center
        horizon = scenarios[0].horizon
        dt = self.data.delta_t_hours
        buses = feeder.buses
        candidates = feeder.storage_candidates
        bus_index = feeder.bus_index
        phase_index = feeder.phase_index
        bus_phases = feeder.bus_phases
        root = feeder.root
        line_data = {(line.parent, line.child): line for line in feeder.lines}
        phase_edges = tuple(
            (line.parent, line.child, phase)
            for line in feeder.lines
            for phase in line.phases
        )
        node_phases = tuple(
            (bus, phase) for bus in buses for phase in bus_phases[bus]
        )
        outgoing = {
            (bus, phase): [edge for edge in phase_edges if edge[0] == bus and edge[2] == phase]
            for bus, phase in node_phases
        }
        incoming = {
            (bus, phase): [edge for edge in phase_edges if edge[1] == bus and edge[2] == phase]
            for bus, phase in node_phases
        }

        model = Model("dfl_batch_resolved_storage_planning")
        if not pcfg.verbose_solver:
            model.hideOutput()
        model.setParam("limits/time", float(pcfg.solver_time_limit_seconds))
        model.setParam("limits/gap", float(pcfg.solver_relative_gap))
        if fixed_design is not None and pcfg.carbon_formulation == "exact":
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
        use_system_average = pcfg.carbon_formulation == "system_average"
        use_aggregate = pcfg.carbon_formulation in {
            "aggregate_mccormick",
            "system_average",
        }
        use_vintage_mccormick = pcfg.carbon_formulation == "mccormick"
        use_mccormick = pcfg.carbon_formulation != "exact"
        storage_disabled = pcfg.max_storage_sites == 0 or (
            fixed_design is not None
            and not any(fixed_design.site[bus] for bus in candidates)
        )

        for scenario_index, (scenario, scenario_weight) in enumerate(zip(scenarios, weights)):
            sid = scenario_index
            times = range(horizon)
            states = range(horizon + 1)
            layers = range(horizon + 1)

            processed = {t: model.addVar(lb=0.0, ub=1.2, name=f"work[{sid},{t}]") for t in times}
            backlog = {t: model.addVar(lb=0.0, ub=2.5, name=f"backlog[{sid},{t}]") for t in states}
            shed = {t: model.addVar(lb=0.0, name=f"shed[{sid},{t}]") for t in times}
            grid_phase = {
                (phase, t): model.addVar(
                    lb=0.0,
                    ub=pcfg.grid_limit_mw,
                    name=f"grid[{sid},{phase},{t}]",
                )
                for phase in bus_phases[root]
                for t in times
            }
            q_grid_phase = {
                (phase, t): model.addVar(
                    lb=-pcfg.grid_limit_mw,
                    ub=pcfg.grid_limit_mw,
                    name=f"qgrid[{sid},{phase},{t}]",
                )
                for phase in bus_phases[root]
                for t in times
            }
            grid = {
                t: quicksum(grid_phase[phase, t] for phase in bus_phases[root])
                for t in times
            }
            generator = {
                t: model.addVar(
                    lb=0.0, ub=pcfg.backup_generator_mw, name=f"gen[{sid},{t}]"
                )
                for t in times
            }
            q_generator = {
                t: model.addVar(
                    lb=-pcfg.backup_generator_mvar,
                    ub=pcfg.backup_generator_mvar,
                    name=f"qgen[{sid},{t}]",
                )
                for t in times
            }
            pv = {
                (bus, phase, t): model.addVar(
                    lb=0.0, name=f"pv[{sid},{bus},{phase},{t}]"
                )
                for bus, phase in node_phases
                for t in times
            }
            curtail = {
                (bus, phase, t): model.addVar(
                    lb=0.0, name=f"curt[{sid},{bus},{phase},{t}]"
                )
                for bus, phase in node_phases
                for t in times
            }

            flow_p = {}
            flow_q = {}
            flow_plus = {}
            flow_minus = {}
            direction = {}
            for edge in phase_edges:
                line = line_data[edge[:2]]
                rating = line.phase_rating_mva
                for t in times:
                    key = (edge, t)
                    flow_p[key] = model.addVar(
                        lb=-rating, ub=rating,
                        name=f"p[{sid},{edge[0]},{edge[1]},{edge[2]},{t}]",
                    )
                    flow_q[key] = model.addVar(
                        lb=-rating, ub=rating,
                        name=f"q[{sid},{edge[0]},{edge[1]},{edge[2]},{t}]",
                    )
                    if not use_system_average:
                        flow_plus[key] = model.addVar(
                            lb=0.0, ub=rating,
                            name=f"pplus[{sid},{edge[0]},{edge[1]},{edge[2]},{t}]",
                        )
                        flow_minus[key] = model.addVar(
                            lb=0.0, ub=rating,
                            name=f"pminus[{sid},{edge[0]},{edge[1]},{edge[2]},{t}]",
                        )
                        direction[key] = model.addVar(
                            vtype="B",
                            name=f"forward[{sid},{edge[0]},{edge[1]},{edge[2]},{t}]",
                        )
            voltage = {
                (bus, phase, t): model.addVar(
                    lb=0.95**2,
                    ub=1.05**2,
                    name=f"v[{sid},{bus},{phase},{t}]",
                )
                for bus, phase in node_phases
                for t in times
            }
            nodal_carbon = {} if use_system_average else {(bus, t): model.addVar(lb=0.0, ub=pcfg.carbon_intensity_max, name=f"ci[{sid},{bus},{t}]") for bus in buses for t in times}
            if allow_carbon_slack:
                if use_system_average:
                    carbon_slack = {
                        ("system", t): model.addVar(
                            lb=0.0,
                            ub=pcfg.carbon_intensity_max,
                            name=f"system_cslack[{sid},{t}]",
                        )
                        for t in times
                    }
                else:
                    carbon_slack = {
                        (bus, t): model.addVar(lb=0.0, ub=pcfg.carbon_intensity_max, name=f"cslack[{sid},{bus},{t}]")
                        for bus in buses
                        for t in times
                    }
            else:
                carbon_slack = {}

            charge_status = {} if storage_disabled else {(bus, t): model.addVar(vtype="B", name=f"uch[{sid},{bus},{t}]") for bus in candidates for t in times}
            discharge_status = {} if storage_disabled else {(bus, t): model.addVar(vtype="B", name=f"udc[{sid},{bus},{t}]") for bus in candidates for t in times}
            skip_vintages = storage_disabled or use_aggregate
            layer_active = {} if skip_vintages else {(bus, k): model.addVar(vtype="B", name=f"ulayer[{sid},{bus},{k}]") for bus in candidates for k in range(1, horizon + 1)}
            vintage_carbon = {} if skip_vintages else {(bus, k): model.addVar(lb=0.0, ub=pcfg.carbon_intensity_max, name=f"vintage_ci[{sid},{bus},{k}]") for bus in candidates for k in layers}
            layer_charge = {} if skip_vintages else {(bus, k): model.addVar(lb=0.0, ub=pcfg.max_power_mw, name=f"layer_ch[{sid},{bus},{k}]") for bus in candidates for k in range(1, horizon + 1)}
            layer_discharge = {} if skip_vintages else {(bus, k, t): model.addVar(lb=0.0, ub=pcfg.max_power_mw, name=f"layer_dc[{sid},{bus},{k},{t}]") for bus in candidates for k in layers for t in times}
            layer_energy = {} if skip_vintages else {(bus, k, t): model.addVar(lb=0.0, ub=pcfg.max_energy_mwh, name=f"layer_e[{sid},{bus},{k},{t}]") for bus in candidates for k in layers for t in states}
            layer_carbon_mass = {} if skip_vintages else {(bus, k, t): model.addVar(lb=0.0, ub=pcfg.carbon_intensity_max * pcfg.max_energy_mwh, name=f"layer_m[{sid},{bus},{k},{t}]") for bus in candidates for k in layers for t in states}
            layer_discharge_carbon: dict[tuple[str, int, int], object] = {}
            flow_carbon_plus: dict[tuple[tuple[str, str, str], int], object] = {}
            flow_carbon_minus: dict[tuple[tuple[str, str, str], int], object] = {}

            model.addCons(backlog[0] == 0.0, name=f"backlog_start[{sid}]")
            model.addCons(
                backlog[horizon] == 0.0,
                name=f"backlog_terminal[{sid}]",
            )
            for t in times:
                model.addCons(
                    backlog[t + 1]
                    == backlog[t]
                    + float(scenario.workload_arrival[t])
                    - processed[t],
                    name=f"backlog_dynamics[{sid},{t}]",
                )
                horizon_deadline = min(horizon - 1, t + 2)
                model.addCons(
                    quicksum(
                        processed[tau]
                        for tau in range(t + 1, horizon_deadline + 1)
                    )
                    >= backlog[t + 1],
                    name=f"work_deadline[{sid},{t}]",
                )
                dc_power = dccfg.power_mw(float(scenario.pue[t]), processed[t])
                # Shedding can never take the facility below its IT base draw.
                model.addCons(shed[t] <= dc_power - dccfg.it_base_mw)
                model.addCons(
                    grid[t]
                    <= float(scenario.grid_available[t]) * pcfg.grid_limit_mw,
                    name=f"grid_available[{sid},{t}]",
                )
                model.addCons(
                    grid[t] <= peak_grid,
                    name=f"grid_peak[{sid},{t}]",
                )
                for bus, phase in node_phases:
                    available = float(
                        scenario.pv_available_mw[
                            t, bus_index[bus], phase_index[phase]
                        ]
                    )
                    model.addCons(
                        pv[bus, phase, t] + curtail[bus, phase, t] == available,
                        name=f"pv_available[{sid},{bus},{phase},{t}]",
                    )

            for t in range(1, horizon):
                model.addCons(
                    generator[t] - generator[t - 1] <= 0.60,
                    name=f"generator_ramp_up[{sid},{t}]",
                )
                model.addCons(
                    generator[t - 1] - generator[t] <= 0.60,
                    name=f"generator_ramp_down[{sid},{t}]",
                )

            charge_power: dict[tuple[str, int], object] = {}
            discharge_power: dict[tuple[str, int], object] = {}
            storage_energy: dict[tuple[str, int], object] = {}
            aggregate_discharge_carbon: dict[tuple[str, int], object] = {}
            if storage_disabled:
                for bus in candidates:
                    for t in times:
                        charge_power[bus, t] = 0.0
                        discharge_power[bus, t] = 0.0
                        storage_energy[bus, t] = 0.0
                        aggregate_discharge_carbon[bus, t] = 0.0
            if use_aggregate and not storage_disabled:
                aggregate_energy = {
                    (bus, t): model.addVar(
                        lb=0.0,
                        ub=pcfg.max_energy_mwh,
                        name=f"aggregate_e[{sid},{bus},{t}]",
                    )
                    for bus in candidates
                    for t in states
                }
                aggregate_carbon_mass = {
                    (bus, t): model.addVar(
                        lb=0.0,
                        ub=pcfg.carbon_intensity_max * pcfg.max_energy_mwh,
                        name=f"aggregate_m[{sid},{bus},{t}]",
                    )
                    for bus in candidates
                    for t in states
                }
                aggregate_carbon_intensity = {
                    (bus, t): model.addVar(
                        lb=0.0,
                        ub=pcfg.carbon_intensity_max,
                        name=f"aggregate_ci[{sid},{bus},{t}]",
                    )
                    for bus in candidates
                    for t in states
                }
                for bus in candidates:
                    model.addCons(
                        aggregate_energy[bus, 0] == pcfg.initial_soc * ecap[bus]
                    )
                    model.addCons(
                        aggregate_carbon_mass[bus, 0]
                        == pcfg.initial_carbon_intensity * aggregate_energy[bus, 0]
                    )
                    model.addCons(
                        aggregate_carbon_intensity[bus, 0]
                        == pcfg.initial_carbon_intensity * site[bus]
                    )
                    for t in states:
                        model.addCons(
                            aggregate_energy[bus, t]
                            >= pcfg.min_soc * ecap[bus]
                        )
                        model.addCons(
                            aggregate_energy[bus, t]
                            <= pcfg.max_soc * ecap[bus]
                        )
                        model.addCons(
                            aggregate_energy[bus, t]
                            <= pcfg.max_energy_mwh * site[bus]
                        )
                        model.addCons(
                            aggregate_carbon_mass[bus, t]
                            <= pcfg.carbon_intensity_max
                            * pcfg.max_energy_mwh
                            * site[bus]
                        )
                        model.addCons(
                            aggregate_carbon_intensity[bus, t]
                            <= pcfg.carbon_intensity_max * site[bus]
                        )
                        _add_mccormick_envelope(
                            model,
                            aggregate_carbon_mass[bus, t],
                            aggregate_carbon_intensity[bus, t],
                            aggregate_energy[bus, t],
                            pcfg.carbon_intensity_max,
                            pcfg.max_energy_mwh,
                        )
                    for t in times:
                        charge = model.addVar(
                            lb=0.0,
                            ub=pcfg.max_power_mw,
                            name=f"aggregate_charge[{sid},{bus},{t}]",
                        )
                        discharge = model.addVar(
                            lb=0.0,
                            ub=pcfg.max_power_mw,
                            name=f"aggregate_discharge[{sid},{bus},{t}]",
                        )
                        charge_carbon = model.addVar(
                            lb=0.0,
                            ub=pcfg.carbon_intensity_max * pcfg.max_power_mw,
                            name=f"aggregate_charge_carbon[{sid},{bus},{t}]",
                        )
                        discharge_carbon = model.addVar(
                            lb=0.0,
                            ub=pcfg.carbon_intensity_max * pcfg.max_power_mw,
                            name=f"aggregate_discharge_carbon[{sid},{bus},{t}]",
                        )
                        charge_power[bus, t] = charge
                        discharge_power[bus, t] = discharge
                        storage_energy[bus, t] = aggregate_energy[bus, t]
                        aggregate_discharge_carbon[bus, t] = discharge_carbon
                        model.addCons(charge_status[bus, t] <= site[bus])
                        model.addCons(discharge_status[bus, t] <= site[bus])
                        model.addCons(
                            charge_status[bus, t] + discharge_status[bus, t] <= 1
                        )
                        model.addCons(charge <= pcap[bus])
                        model.addCons(discharge <= pcap[bus])
                        model.addCons(
                            charge <= pcfg.max_power_mw * charge_status[bus, t]
                        )
                        model.addCons(
                            discharge <= pcfg.max_power_mw * discharge_status[bus, t]
                        )
                        model.addCons(
                            aggregate_energy[bus, t + 1]
                            == retention * aggregate_energy[bus, t]
                            + pcfg.charge_efficiency * charge * dt
                            - discharge * dt / pcfg.discharge_efficiency
                        )
                        if use_system_average:
                            model.addCons(
                                charge_carbon
                                == float(scenario.grid_carbon_t_per_mwh[t]) * charge
                            )
                        else:
                            _add_mccormick_envelope(
                                model,
                                charge_carbon,
                                nodal_carbon[bus, t],
                                charge,
                                pcfg.carbon_intensity_max,
                                pcfg.max_power_mw,
                            )
                        _add_mccormick_envelope(
                            model,
                            discharge_carbon,
                            aggregate_carbon_intensity[bus, t],
                            discharge,
                            pcfg.carbon_intensity_max,
                            pcfg.max_power_mw,
                        )
                        model.addCons(
                            aggregate_carbon_mass[bus, t + 1]
                            == retention * aggregate_carbon_mass[bus, t]
                            + pcfg.charge_efficiency * charge_carbon * dt
                            - discharge_carbon * dt / pcfg.discharge_efficiency
                        )
                    model.addCons(
                        aggregate_energy[bus, horizon] == aggregate_energy[bus, 0]
                    )
                    # Same relaxation as the layered branch: end no dirtier than
                    # the start.  Under a single blended intensity this matters
                    # even more, because any charge below the initial intensity
                    # dilutes the pool irreversibly, so equality forbids cycling.
                    model.addCons(
                        aggregate_carbon_mass[bus, horizon]
                        <= aggregate_carbon_mass[bus, 0]
                    )

            for bus in (() if storage_disabled or use_aggregate else candidates):
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
                        if use_vintage_mccormick:
                            discharge_carbon = model.addVar(
                                lb=0.0,
                                ub=pcfg.carbon_intensity_max * pcfg.max_power_mw,
                                name=f"layer_dc_carbon[{sid},{bus},{k},{t}]",
                            )
                            layer_discharge_carbon[bus, k, t] = discharge_carbon
                            _add_mccormick_envelope(
                                model,
                                discharge_carbon,
                                vintage_carbon[bus, k],
                                layer_discharge[bus, k, t],
                                pcfg.carbon_intensity_max,
                                pcfg.max_power_mw,
                            )
                    storage_energy[bus, t] = quicksum(layer_energy[bus, k, t] for k in layers)
                    model.addCons(storage_energy[bus, t] >= pcfg.min_soc * ecap[bus])
                    model.addCons(storage_energy[bus, t] <= pcfg.max_soc * ecap[bus])

                terminal_energy = quicksum(layer_energy[bus, k, horizon] for k in layers)
                terminal_mass = quicksum(layer_carbon_mass[bus, k, horizon] for k in layers)
                model.addCons(terminal_energy == layer_energy[bus, 0, 0])
                # Energy must cycle exactly, but carbon only has to end no dirtier
                # than it started.  Requiring equality assumes some charging hour
                # matches initial_carbon_intensity; when every available hour is
                # cleaner than the initial inventory, equality is unreachable and
                # the only feasible schedule is to never cycle the battery at all.
                model.addCons(terminal_mass <= layer_carbon_mass[bus, 0, 0])
                for k in layers:
                    activation = site[bus] if k == 0 else layer_active[bus, k]
                    for t in states:
                        model.addCons(layer_energy[bus, k, t] <= pcfg.max_energy_mwh * activation)
                        model.addCons(
                            layer_carbon_mass[bus, k, t]
                            <= pcfg.carbon_intensity_max * pcfg.max_energy_mwh * activation
                        )
                        if use_vintage_mccormick:
                            _add_mccormick_envelope(
                                model,
                                layer_carbon_mass[bus, k, t],
                                vintage_carbon[bus, k],
                                layer_energy[bus, k, t],
                                pcfg.carbon_intensity_max,
                                pcfg.max_energy_mwh,
                            )
                        else:
                            model.addCons(
                                layer_carbon_mass[bus, k, t]
                                == vintage_carbon[bus, k] * layer_energy[bus, k, t]
                            )

            for t in times:
                for edge in phase_edges:
                    line = line_data[edge[:2]]
                    rating = line.phase_rating_mva
                    if not use_system_average:
                        model.addCons(flow_p[edge, t] == flow_plus[edge, t] - flow_minus[edge, t])
                        model.addCons(flow_plus[edge, t] <= rating * direction[edge, t])
                        model.addCons(flow_minus[edge, t] <= rating * (1 - direction[edge, t]))
                    if pcfg.flow_limit_formulation == "quadratic":
                        model.addCons(
                            flow_p[edge, t] * flow_p[edge, t]
                            + flow_q[edge, t] * flow_q[edge, t]
                            <= rating**2,
                            name=f"line_limit[{sid},{edge[0]},{edge[1]},{edge[2]},{t}]",
                        )
                    else:
                        # Inner octagon: conservative linear approximation of
                        # p^2 + q^2 <= rating^2, exact at eight circle vertices.
                        polygon_rhs = rating * math.cos(math.pi / 8.0)
                        for facet in range(8):
                            angle = math.pi / 8.0 + facet * math.pi / 4.0
                            model.addCons(
                                math.cos(angle) * flow_p[edge, t]
                                + math.sin(angle) * flow_q[edge, t]
                                <= polygon_rhs,
                                name=(
                                    f"line_limit_polygon[{sid},{edge[0]},"
                                    f"{edge[1]},{edge[2]},{t},{facet}]"
                                ),
                            )
                    model.addCons(
                        voltage[edge[1], edge[2], t]
                        == voltage[edge[0], edge[2], t]
                        - 2.0
                        * (
                            line.resistance * flow_p[edge, t]
                            + line.reactance * flow_q[edge, t]
                        ),
                        name=f"voltage_drop[{sid},{edge[0]},{edge[1]},{edge[2]},{t}]",
                    )
                    if use_mccormick and not use_system_average:
                        carbon_plus = model.addVar(
                            lb=0.0,
                            ub=pcfg.carbon_intensity_max * rating,
                            name=f"carbon_plus[{sid},{edge[0]},{edge[1]},{edge[2]},{t}]",
                        )
                        carbon_minus = model.addVar(
                            lb=0.0,
                            ub=pcfg.carbon_intensity_max * rating,
                            name=f"carbon_minus[{sid},{edge[0]},{edge[1]},{edge[2]},{t}]",
                        )
                        flow_carbon_plus[edge, t] = carbon_plus
                        flow_carbon_minus[edge, t] = carbon_minus
                        _add_mccormick_envelope(
                            model,
                            carbon_plus,
                            nodal_carbon[edge[0], t],
                            flow_plus[edge, t],
                            pcfg.carbon_intensity_max,
                            rating,
                        )
                        _add_mccormick_envelope(
                            model,
                            carbon_minus,
                            nodal_carbon[edge[1], t],
                            flow_minus[edge, t],
                            pcfg.carbon_intensity_max,
                            rating,
                        )
                for phase in bus_phases[root]:
                    model.addCons(voltage[root, phase, t] == 1.0)

                dc_power = dccfg.power_mw(float(scenario.pue[t]), processed[t])
                for bus, phase in node_phases:
                    bi = bus_index[bus]
                    pi = phase_index[phase]
                    load_p = float(scenario.active_load_mw[t, bi, pi])
                    load_q = float(scenario.reactive_load_mvar[t, bi, pi])
                    phase_count = len(bus_phases[bus])
                    if bus == feeder.data_center_bus:
                        load_p += (dc_power - shed[t]) / phase_count
                    local_generation_phase = (
                        generator[t] / phase_count if bus == feeder.generator_bus else 0.0
                    )
                    local_q_generation_phase = (
                        q_generator[t] / phase_count if bus == feeder.generator_bus else 0.0
                    )
                    grid_injection = grid_phase[phase, t] if bus == root else 0.0
                    grid_q_injection = q_grid_phase[phase, t] if bus == root else 0.0
                    storage_charge = (
                        charge_power[bus, t] / phase_count if bus in candidates else 0.0
                    )
                    storage_discharge = (
                        discharge_power[bus, t] / phase_count if bus in candidates else 0.0
                    )
                    outgoing_p = quicksum(
                        flow_p[edge, t] for edge in outgoing[bus, phase]
                    )
                    incoming_p = quicksum(
                        flow_p[edge, t] for edge in incoming[bus, phase]
                    )
                    outgoing_q = quicksum(
                        flow_q[edge, t] for edge in outgoing[bus, phase]
                    )
                    incoming_q = quicksum(
                        flow_q[edge, t] for edge in incoming[bus, phase]
                    )
                    model.addCons(
                        local_generation_phase
                        + pv[bus, phase, t]
                        + grid_injection
                        + storage_discharge
                        - load_p
                        - storage_charge
                        == outgoing_p - incoming_p,
                        name=f"active_balance[{sid},{bus},{phase},{t}]",
                    )
                    model.addCons(
                        local_q_generation_phase + grid_q_injection - load_q
                        == outgoing_q - incoming_q,
                        name=f"reactive_balance[{sid},{bus},{phase},{t}]",
                    )

                for bus in (() if use_system_average else buses):
                    local_generation = generator[t] if bus == feeder.generator_bus else 0.0
                    grid_injection = grid[t] if bus == root else 0.0
                    storage_discharge = discharge_power[bus, t] if bus in candidates else 0.0
                    pv_injection = quicksum(
                        pv[bus, phase, t] for phase in bus_phases[bus]
                    )
                    parent_forward = quicksum(
                        flow_plus[edge, t]
                        for phase in bus_phases[bus]
                        for edge in incoming[bus, phase]
                    )
                    child_reverse = quicksum(
                        flow_minus[edge, t]
                        for phase in bus_phases[bus]
                        for edge in outgoing[bus, phase]
                    )
                    total_incoming_power = (
                        local_generation
                        + pv_injection
                        + grid_injection
                        + storage_discharge
                        + parent_forward
                        + child_reverse
                    )
                    storage_carbon = (
                        0.0
                        if storage_disabled
                        else (
                            aggregate_discharge_carbon[bus, t]
                            if use_aggregate
                            else quicksum(
                                (
                                    layer_discharge_carbon[bus, k, t]
                                    if use_vintage_mccormick
                                    else vintage_carbon[bus, k]
                                    * layer_discharge[bus, k, t]
                                )
                                for k in layers
                            )
                        )
                        if bus in candidates
                        else 0.0
                    )
                    network_carbon = quicksum(
                        (
                            flow_carbon_plus[edge, t]
                            if use_mccormick
                            else nodal_carbon[edge[0], t] * flow_plus[edge, t]
                        )
                        for phase in bus_phases[bus]
                        for edge in incoming[bus, phase]
                    ) + quicksum(
                        (
                            flow_carbon_minus[edge, t]
                            if use_mccormick
                            else nodal_carbon[edge[1], t] * flow_minus[edge, t]
                        )
                        for phase in bus_phases[bus]
                        for edge in outgoing[bus, phase]
                    )
                    total_incoming_carbon = (
                        0.72 * local_generation
                        + (float(scenario.grid_carbon_t_per_mwh[t]) * grid[t] if bus == root else 0.0)
                        + storage_carbon
                        + network_carbon
                    )
                    if use_mccormick:
                        pv_upper = float(
                            scenario.pv_available_mw[t, bus_index[bus]].sum()
                        )
                        incoming_power_upper = (
                            (1.0 if bus == feeder.generator_bus else 0.0)
                            + pv_upper
                            + (pcfg.grid_limit_mw if bus == root else 0.0)
                            + (pcfg.max_power_mw if bus in candidates else 0.0)
                            + sum(
                                line_data[edge[:2]].phase_rating_mva
                                for phase in bus_phases[bus]
                                for edge in incoming[bus, phase]
                            )
                            + sum(
                                line_data[edge[:2]].phase_rating_mva
                                for phase in bus_phases[bus]
                                for edge in outgoing[bus, phase]
                            )
                        )
                        incoming_power_var = model.addVar(
                            lb=0.0,
                            ub=incoming_power_upper,
                            name=f"incoming_power[{sid},{bus},{t}]",
                        )
                        incoming_carbon_var = model.addVar(
                            lb=0.0,
                            ub=pcfg.carbon_intensity_max * incoming_power_upper,
                            name=f"incoming_carbon[{sid},{bus},{t}]",
                        )
                        model.addCons(incoming_power_var == total_incoming_power)
                        model.addCons(incoming_carbon_var == total_incoming_carbon)
                        _add_mccormick_envelope(
                            model,
                            incoming_carbon_var,
                            nodal_carbon[bus, t],
                            incoming_power_var,
                            pcfg.carbon_intensity_max,
                            incoming_power_upper,
                        )
                    else:
                        model.addCons(
                            nodal_carbon[bus, t] * total_incoming_power
                            == total_incoming_carbon
                        )
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

                if use_system_average:
                    system_pv = quicksum(
                        pv[bus, phase, t] for bus, phase in node_phases
                    )
                    system_storage_discharge = quicksum(
                        discharge_power[bus, t] for bus in candidates
                    )
                    system_power = (
                        grid[t]
                        + generator[t]
                        + system_pv
                        + system_storage_discharge
                    )
                    system_carbon = (
                        float(scenario.grid_carbon_t_per_mwh[t]) * grid[t]
                        + 0.72 * generator[t]
                        + quicksum(
                            aggregate_discharge_carbon[bus, t]
                            for bus in candidates
                        )
                    )
                    system_power_upper = (
                        pcfg.grid_limit_mw
                        + 1.0
                        + float(scenario.pv_available_mw[t].sum())
                        + len(candidates) * pcfg.max_power_mw
                    )
                    if allow_carbon_slack:
                        slack = carbon_slack["system", t]
                        model.addCons(
                            system_carbon
                            <= pcfg.dc_carbon_cap * system_power
                            + system_power_upper * slack,
                            name=f"system_carbon_cap[{sid},{t}]",
                        )
                        carbon_slack_terms.append(
                            scenario_weight
                            * annual_blocks
                            * ccfg.validation_carbon_slack_dollars
                            * slack
                        )
                    else:
                        model.addCons(
                            system_carbon <= pcfg.dc_carbon_cap * system_power,
                            name=f"system_carbon_cap[{sid},{t}]",
                        )

                interval_cost = (
                    float(scenario.grid_price_per_mwh[t]) * grid[t]
                    + ccfg.generator_dollars_per_mwh * generator[t]
                    + ccfg.degradation_dollars_per_mwh
                    * quicksum(charge_power[bus, t] + discharge_power[bus, t] for bus in candidates)
                    + ccfg.delay_dollars_per_task_hour * backlog[t]
                    + ccfg.curtailment_dollars_per_mwh
                    * quicksum(curtail[bus, phase, t] for bus, phase in node_phases)
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
