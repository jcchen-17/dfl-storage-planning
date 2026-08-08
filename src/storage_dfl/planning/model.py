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


SOLVER_BACKENDS = ("scip", "gurobi", "highs")


def _variable_name(variable) -> str:
    """Variable name across backends.

    PySCIPOpt spells it ``name`` and gurobipy spells it ``VarName``.  Warm starts
    are transferred between models by name, so this has to work for both or the
    bootstrap silently matches nothing and every solve starts cold.
    """

    name = getattr(variable, "name", None)
    if isinstance(name, str):
        return name
    return str(variable.VarName)


def _new_model(name: str, backend: str):
    """Return ``(model, quicksum)`` for the requested backend.

    The non-SCIP paths return facades exposing the same methods the builder calls
    on a PySCIPOpt model, so the builder needs no knowledge of which solver it is
    talking to and every backend is guaranteed to build the same model.
    """

    if backend not in SOLVER_BACKENDS:
        raise ValueError(f"solver_backend must be one of {SOLVER_BACKENDS}, got {backend!r}.")
    if backend == "scip":
        return Model(name), quicksum
    if backend == "gurobi":
        from storage_dfl.planning import gurobi_backend

        if not gurobi_backend.is_available():
            raise RuntimeError(
                "solver_backend is 'gurobi' but gurobipy is not installed. "
                "Run `python -m pip install gurobipy` and activate a license large "
                "enough for this model; the restricted license caps at 2000 variables."
            )
        return gurobi_backend.GurobiModel(name), gurobi_backend.quicksum
    from storage_dfl.planning import highs_backend

    if not highs_backend.is_available():
        raise RuntimeError(
            "solver_backend is 'highs' but highspy is not installed. "
            "Run `python -m pip install highspy`."
        )
    return highs_backend.HighsModel(name), highs_backend.quicksum


def _check_backend_supports(pcfg: PlanningConfig) -> None:
    """Reject combinations a backend cannot represent.

    HiGHS is linear only. Letting it build the nonconvex carbon identity or the
    quadratic flow circles would either fail deep inside the builder or, worse,
    drop the constraints and return a confidently wrong design.
    """

    if pcfg.solver_backend != "highs":
        return
    unsupported = []
    if pcfg.carbon_formulation in {"exact", "layered_dc_exact"}:
        unsupported.append("carbon_formulation: exact (nonconvex bilinear equalities)")
    if pcfg.flow_limit_formulation == "quadratic":
        unsupported.append("flow_limit_formulation: quadratic (second-order cones)")
    if unsupported:
        raise ValueError(
            "The HiGHS backend solves linear models only, but this configuration "
            "needs " + "; ".join(unsupported) + ". Use solver_backend 'scip' or "
            "'gurobi', or switch to the mccormick/polygon formulation."
        )


_EMPHASIS_SETTINGS = ("default", "feasibility", "optimality", "hardlp", "counter")


def _apply_search_strategy(model: Model, pcfg: PlanningConfig) -> None:
    """Point SCIP at good primal solutions without touching the formulation.

    Both settings only reorder the search.  The model, its constraints and its
    objective are unchanged, so a solution found this way is exactly as valid as
    one found with the defaults -- it is only the effort spent proving optimality
    that shifts.  ``relative_gap`` on the result still tells the caller whether
    the answer was proven.
    """

    emphasis = str(pcfg.solver_emphasis).lower()
    if emphasis not in _EMPHASIS_SETTINGS:
        raise ValueError(
            f"solver_emphasis must be one of {_EMPHASIS_SETTINGS}, got {emphasis!r}."
        )
    delegate = getattr(model, "apply_search_strategy", None)
    if delegate is not None:
        delegate(emphasis, bool(pcfg.solver_aggressive_heuristics))
        return
    try:
        from pyscipopt import SCIP_PARAMEMPHASIS, SCIP_PARAMSETTING

        if emphasis != "default":
            model.setEmphasis(getattr(SCIP_PARAMEMPHASIS, emphasis.upper()))
        if pcfg.solver_aggressive_heuristics:
            model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
    except (ImportError, AttributeError):
        # An older PySCIPOpt without these enums simply keeps the default search.
        pass


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


@dataclass(frozen=True)
class PlanningJob:
    """One solve within a ``solve_many`` batch.

    Named rather than positional because a batch of fixed-design validations and
    a batch of free planning solves differ only in the third field, and reading
    ``(scenarios, None, design)`` at a call site gives no hint of that.
    """

    scenarios: Iterable[Scenario]
    weights: Iterable[float] | None = None
    fixed_design: StorageDesign | None = None


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
            "layered_dc",
            "layered_dc_exact",
            "aggregate_mccormick",
            "average_dc",
            "carbon_bins_dc",
            "binned_storage_dc",
            "system_average",
            "layered_system",
        }:
            raise ValueError(
                "carbon_formulation must be 'exact', 'mccormick', "
                "'layered_dc', 'layered_dc_exact', 'aggregate_mccormick', 'average_dc', "
                "'carbon_bins_dc', 'binned_storage_dc', 'system_average', or 'layered_system'"
            )
        if planning.flow_limit_formulation not in {"quadratic", "polygon"}:
            raise ValueError(
                "flow_limit_formulation must be 'quadratic' or 'polygon'"
            )
        # A typo here would silently fall through to the loose rating bound and
        # leave the carbon cap non-binding, which is the failure this option
        # exists to fix -- so reject it rather than ignore it.
        if planning.carbon_envelope_bound not in {"line_rating", "demand"}:
            raise ValueError(
                "carbon_envelope_bound must be 'line_rating' or 'demand', got "
                f"{planning.carbon_envelope_bound!r}"
            )
        if planning.carbon_cap_scope not in {"hourly", "horizon"}:
            raise ValueError(
                "carbon_cap_scope must be 'hourly' or 'horizon', got "
                f"{planning.carbon_cap_scope!r}"
            )
        if planning.storage_service_mode not in {
            "shared_feeder",
            "dedicated_dc",
            "fixed_dc_siting",
        }:
            raise ValueError(
                "storage_service_mode must be 'shared_feeder', 'dedicated_dc', "
                "or 'fixed_dc_siting'."
            )
        if (
            planning.storage_service_mode == "dedicated_dc"
            and planning.carbon_formulation != "carbon_bins_dc"
        ):
            raise ValueError(
                "dedicated_dc currently requires carbon_formulation='carbon_bins_dc'."
            )
        if (
            planning.carbon_formulation == "carbon_bins_dc"
            and planning.carbon_cap_scope != "hourly"
        ):
            raise ValueError("carbon_bins_dc currently supports hourly carbon caps only.")
        if (
            planning.carbon_cap_scope == "horizon"
            and planning.carbon_formulation
            not in {"system_average", "layered_system"}
        ):
            raise ValueError(
                "carbon_cap_scope='horizon' is supported only by "
                "system_average and layered_system"
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
        scenario_tuple, weight_tuple = self._normalize_job(scenarios, weights)
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
        # Gurobi adds variables lazily.  If the no-storage bootstrap is
        # infeasible there is no named warm start to force an update, and
        # hashing a just-created variable for the fallback partial solution
        # raises "Variable has not yet been added to the model".  Querying the
        # count materializes the model for every backend before either path.
        model.getNVars()
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
            _variable_name(variable): float(
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
        target_by_name = {
            _variable_name(variable): variable for variable in target_variables
        }
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

        payload = self._worker_payload(
            scenarios,
            weights,
            fixed_design,
            allow_carbon_slack,
            use_cache,
            warm_start_values,
            warm_start_cached,
        )
        return self._run_worker(payload)

    def _normalize_job(
        self,
        scenarios: Iterable[Scenario],
        weights: Iterable[float] | None,
    ) -> tuple[tuple[Scenario, ...], tuple[float, ...]]:
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
        total = sum(weight_tuple)
        if total <= 0:
            raise ValueError("Scenario weights must have a positive sum.")
        return scenario_tuple, tuple(weight / total for weight in weight_tuple)

    def _isolation_active(self) -> bool:
        """Whether planning solves currently run in a separate process.

        Concurrency is only offered on that path: it is the one that already
        keeps each solve in its own address space, so running several at once
        adds no new shared state.
        """

        return (
            os.name == "nt"
            and "torch" in sys.modules
            and os.environ.get("STORAGE_DFL_SOLVER_WORKER") != "1"
        )

    def solve_many(
        self,
        jobs: Iterable[PlanningJob | tuple[Iterable[Scenario], Iterable[float] | None]],
        *,
        allow_carbon_slack: bool = False,
        use_cache: bool = False,
        max_workers: int | None = None,
    ) -> list[PlanningResult]:
        """Solve several planning problems, concurrently when possible.

        The samples of one decision-focused epoch are independent, and each solve
        already runs in its own process, so the only thing serialising them was
        the loop itself.  Results are returned in the order the jobs were given,
        so the caller's sample indexing is unchanged.

        Jobs may fix a design, which is what lets an epoch's validation solves
        batch the same way its planning solves do: they share one scenario set
        and differ only in the design being validated.

        Falls back to the ordinary serial path whenever process isolation is not
        in use, which keeps behaviour identical on platforms that never needed it.
        """

        requested = [
            job if isinstance(job, PlanningJob) else PlanningJob(*job) for job in jobs
        ]
        normalized = [
            (*self._normalize_job(job.scenarios, job.weights), job.fixed_design)
            for job in requested
        ]
        if not normalized:
            return []

        workers = max_workers if max_workers is not None else self.planning.solver_max_parallel_workers
        workers = max(1, int(workers))
        if workers == 1 or len(normalized) == 1 or not self._isolation_active():
            return [
                self.solve(
                    scenarios,
                    weights=weights,
                    fixed_design=design,
                    allow_carbon_slack=allow_carbon_slack,
                    use_cache=use_cache,
                )
                for scenarios, weights, design in normalized
            ]

        results: list[PlanningResult | None] = [None] * len(normalized)
        cache_keys = [
            self._cache_key(scenarios, weights, design, allow_carbon_slack)
            for scenarios, weights, design in normalized
        ]
        # Serial solving deduplicated repeats through the cache on the way past.
        # A batch has to do it up front instead, or two samples that converged on
        # the same design would each pay for the identical validation solve.
        pending: list[int] = []
        duplicate_of: dict[int, int] = {}
        leaders: dict[tuple, int] = {}
        for index, cache_key in enumerate(cache_keys):
            if use_cache and cache_key in self._cache:
                results[index] = self._cache[cache_key]
                continue
            if use_cache:
                leader = leaders.get(cache_key)
                if leader is not None:
                    duplicate_of[index] = leader
                    continue
                leaders[cache_key] = index
            pending.append(index)

        # None marks a job that needs no bootstrap, so nothing is shared and it
        # can go in any wave.
        warm_start_keys: dict[int, tuple | None] = {}
        for index in pending:
            scenarios, weights, design = normalized[index]
            warm_start_keys[index] = (
                self._warm_start_key(scenarios, weights, allow_carbon_slack)
                if self._warm_start_supported(design)
                else None
            )

        from concurrent.futures import ThreadPoolExecutor

        remaining = pending
        while remaining:
            wave: list[int] = []
            deferred: list[int] = []
            claimed: set[tuple] = set()
            for index in remaining:
                warm_start_key = warm_start_keys[index]
                if warm_start_key is None or warm_start_key in self._warm_start_cache:
                    wave.append(index)
                elif warm_start_key in claimed:
                    # A job already in this wave is computing that bootstrap.
                    # Every validation of an epoch shares one, and it costs up to
                    # warm_start_time_limit_seconds, so waiting a wave to inherit
                    # it beats having each worker recompute the same thing.
                    deferred.append(index)
                else:
                    claimed.add(warm_start_key)
                    wave.append(index)

            payloads = []
            for index in wave:
                scenarios, weights, design = normalized[index]
                warm_start_key = warm_start_keys[index]
                payloads.append(
                    self._worker_payload(
                        scenarios,
                        weights,
                        design,
                        allow_carbon_slack,
                        False,
                        self._warm_start_cache.get(warm_start_key)
                        if warm_start_key is not None
                        else None,
                        warm_start_key is None
                        or warm_start_key in self._warm_start_cache,
                    )
                )

            # Threads, not processes: each worker blocks in subprocess.run, which
            # releases the GIL, and the real parallelism is between the child SCIP
            # processes rather than inside this interpreter.
            with ThreadPoolExecutor(max_workers=min(workers, len(payloads))) as pool:
                responses = list(pool.map(self._run_worker, payloads))

            # Merged here rather than in the workers so the caches are only
            # mutated by this thread once every job of the wave has finished.
            for index, (result, warm_start_values, computed) in zip(
                wave, responses, strict=True
            ):
                warm_start_key = warm_start_keys[index]
                if computed and warm_start_key is not None:
                    self._warm_start_cache[warm_start_key] = warm_start_values
                results[index] = result
                if use_cache:
                    self._cache[cache_keys[index]] = result

            remaining = deferred

        for index, leader in duplicate_of.items():
            results[index] = results[leader]
        completed = [result for result in results if result is not None]
        if len(completed) != len(results):
            # Callers index this against their own sample list, so a short list
            # would silently misattribute results rather than fail.
            raise RuntimeError("A batched planning job returned no result.")
        return completed

    def _worker_payload(
        self,
        scenarios: tuple[Scenario, ...],
        weights: tuple[float, ...],
        fixed_design: StorageDesign | None,
        allow_carbon_slack: bool,
        use_cache: bool,
        warm_start_values: dict[str, float] | None,
        warm_start_cached: bool,
    ) -> dict:
        return {
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

    def _run_worker(
        self, payload: dict
    ) -> tuple[PlanningResult, dict[str, float] | None, bool]:
        """Run one payload in a clean subprocess.

        Holds no oracle state, so several of these may run concurrently.
        """

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
        # Descendants of each bus in the radial tree, used by the "demand"
        # envelope bound to size how much power can flow through a bus toward
        # what lies below it. Built once; the topology does not vary by scenario.
        children: dict[str, list[str]] = {bus: [] for bus in buses}
        for line in feeder.lines:
            children[line.parent].append(line.child)
        descendants: dict[str, tuple[str, ...]] = {}

        def _collect(bus: str) -> tuple[str, ...]:
            if bus not in descendants:
                found: list[str] = []
                for child in children[bus]:
                    found.append(child)
                    found.extend(_collect(child))
                descendants[bus] = tuple(found)
            return descendants[bus]

        for bus in buses:
            _collect(bus)

        # Every backend builds this model through the same 1,500 lines below, so
        # a difference between them can only come from the solver, never from a
        # divergent formulation.
        _check_backend_supports(pcfg)
        model, quicksum = _new_model(
            "dfl_batch_resolved_storage_planning", pcfg.solver_backend
        )
        if pcfg.solver_backend == "gurobi" and pcfg.carbon_formulation in {
            "exact",
            "layered_dc_exact",
        }:
            # The exact carbon identity is a bilinear equality, which Gurobi only
            # accepts once nonconvex quadratics are enabled.
            model.setNonconvex(True)
        if not pcfg.verbose_solver:
            model.hideOutput()
        model.setParam("limits/time", float(pcfg.solver_time_limit_seconds))
        model.setParam("limits/gap", float(pcfg.solver_relative_gap))
        if float(pcfg.solver_memory_limit_mb) > 0.0:
            # Without this SCIP aborts the process when an allocation fails, which
            # takes the whole run with it. Bounded, it stops cleanly and keeps the
            # incumbent it already had.
            model.setParam("limits/memory", float(pcfg.solver_memory_limit_mb))
        if fixed_design is not None and pcfg.carbon_formulation in {
            "exact",
            "layered_dc_exact",
        }:
            # SCIP can incorrectly cut off the fixed-design problem while
            # eliminating zero-capacity bilinear vintage equations. Keeping the
            # original equations avoids that presolve artifact in validation.
            model.setParam("presolving/maxrounds", 0)
        try:
            model.setParam("parallel/maxnthreads", int(pcfg.solver_threads))
        except KeyError:
            pass
        # Emphasis has to be applied before the heuristics setting, because it
        # resets the parameters that setHeuristics then overrides.
        _apply_search_strategy(model, pcfg)

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
        if pcfg.storage_service_mode in {"dedicated_dc", "fixed_dc_siting"}:
            if feeder.data_center_bus not in candidates:
                raise ValueError("The data-center bus must be a storage candidate.")
            for bus in candidates:
                required = int(
                    pcfg.max_storage_sites > 0 and bus == feeder.data_center_bus
                )
                model.addCons(site[bus] == required)

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
            ccfg.battery_capex_scale
            * (
                ccfg.site_dollars * site[bus]
                + ccfg.power_dollars_per_mw * pcap[bus]
                + ccfg.energy_dollars_per_mwh * ecap[bus]
            )
            for bus in candidates
        )

        operating_terms = []
        carbon_slack_terms = []
        annual_blocks = 8760.0 / (horizon * dt)
        retention = 1.0 - pcfg.self_discharge
        epsilon_charge = 1.0e-4
        big_m_carbon = pcfg.carbon_intensity_max
        # Both system-boundary formulations apply exactly the same carbon cap.
        # They differ only inside storage: system_average blends all inventory,
        # while layered_system retains the carbon vintage of every charge.
        use_system_boundary = pcfg.carbon_formulation in {
            "system_average",
            "layered_system",
        }
        use_system_average = pcfg.carbon_formulation == "system_average"
        use_layered_system = pcfg.carbon_formulation == "layered_system"
        use_carbon_bins = pcfg.carbon_formulation == "carbon_bins_dc"
        use_binned_storage = pcfg.carbon_formulation == "binned_storage_dc"
        use_storage_bins = use_carbon_bins or use_binned_storage
        carbon_bin_intensity = (0.0, 0.15, 0.22, 0.28, 0.35, 0.45, 0.72, 0.75)
        use_aggregate = pcfg.carbon_formulation in {
            "aggregate_mccormick",
            "average_dc",
            "system_average",
        }
        use_vintage_mccormick = pcfg.carbon_formulation in {
            "mccormick",
            "layered_dc",
        }
        use_mccormick = pcfg.carbon_formulation not in {
            "exact",
            "layered_dc_exact",
        }
        storage_disabled = pcfg.max_storage_sites == 0 or (
            fixed_design is not None
            and not any(fixed_design.site[bus] for bus in candidates)
        )

        for scenario_index, (scenario, scenario_weight) in enumerate(zip(scenarios, weights)):
            sid = scenario_index
            times = range(horizon)
            states = range(horizon + 1)
            layers = range(horizon + 1)
            horizon_carbon_terms: list[object] = []
            horizon_energy_terms: list[object] = []
            horizon_energy_upper = 0.0

            # Power a bus can absorb / inject in one interval. Used by the
            # "demand" carbon envelope to replace line thermal ratings, which
            # exceed real flows by more than an order of magnitude and leave the
            # relaxation too loose for any carbon cap to bind. Loads are data,
            # so absorption is exact; injection uses each source's own limit.
            def absorbed_at(bus: str, t: int) -> float:
                total = float(scenario.active_load_mw[t, bus_index[bus]].sum())
                if bus == feeder.data_center_bus:
                    total += dccfg.power_mw(
                        float(scenario.pue[t]), dccfg.workload_processing_upper
                    )
                if bus in candidates and not storage_disabled:
                    total += pcfg.max_power_mw
                return total

            def injected_at(bus: str, t: int) -> float:
                total = float(scenario.pv_available_mw[t, bus_index[bus]].sum())
                if bus == feeder.generator_bus:
                    total += pcfg.backup_generator_mw
                if bus == root:
                    total += pcfg.grid_limit_mw
                if bus in candidates and not storage_disabled:
                    total += pcfg.max_power_mw
                return total

            def subtree_absorbed(bus: str, t: int) -> float:
                return absorbed_at(bus, t) + sum(
                    absorbed_at(other, t) for other in descendants[bus]
                )

            def subtree_injected(bus: str, t: int) -> float:
                return injected_at(bus, t) + sum(
                    injected_at(other, t) for other in descendants[bus]
                )

            processed = {
                t: model.addVar(
                    lb=0.0,
                    ub=dccfg.workload_processing_upper,
                    name=f"work[{sid},{t}]",
                )
                for t in times
            }
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
                    if not use_system_boundary:
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
            nodal_carbon = {} if use_system_boundary or use_carbon_bins else {(bus, t): model.addVar(lb=0.0, ub=pcfg.carbon_intensity_max, name=f"ci[{sid},{bus},{t}]") for bus in buses for t in times}
            nodal_bin_active: dict[tuple[str, int, int], object] = {}
            nodal_bin_ci: dict[tuple[str, int, int], object] = {}
            if use_binned_storage:
                for bus in buses:
                    for t in times:
                        for b, upper in enumerate(carbon_bin_intensity):
                            lower = 0.0 if b == 0 else carbon_bin_intensity[b - 1]
                            nodal_bin_active[bus, t, b] = model.addVar(
                                vtype="B", name=f"ci_bin_active[{sid},{bus},{t},{b}]"
                            )
                            nodal_bin_ci[bus, t, b] = model.addVar(
                                lb=0.0, ub=upper,
                                name=f"ci_bin_value[{sid},{bus},{t},{b}]",
                            )
                            model.addCons(
                                nodal_bin_ci[bus, t, b]
                                >= lower * nodal_bin_active[bus, t, b]
                            )
                            model.addCons(
                                nodal_bin_ci[bus, t, b]
                                <= upper * nodal_bin_active[bus, t, b]
                            )
                        model.addCons(
                            quicksum(
                                nodal_bin_active[bus, t, b]
                                for b in range(len(carbon_bin_intensity))
                            ) == 1
                        )
                        model.addCons(
                            quicksum(
                                nodal_bin_ci[bus, t, b]
                                for b in range(len(carbon_bin_intensity))
                            ) == nodal_carbon[bus, t]
                        )

            def add_piecewise_carbon_product(
                product: object,
                bus: str,
                t: int,
                power: object,
                power_upper: float,
                prefix: str,
            ) -> None:
                pieces = []
                for b, upper in enumerate(carbon_bin_intensity):
                    lower = 0.0 if b == 0 else carbon_bin_intensity[b - 1]
                    active = nodal_bin_active[bus, t, b]
                    ci_piece = nodal_bin_ci[bus, t, b]
                    power_piece = model.addVar(
                        lb=0.0, ub=power_upper,
                        name=f"{prefix}_power_piece[{b}]",
                    )
                    carbon_piece = model.addVar(
                        lb=0.0, ub=upper * power_upper,
                        name=f"{prefix}_carbon_piece[{b}]",
                    )
                    model.addCons(power_piece <= power_upper * active)
                    model.addCons(carbon_piece >= lower * power_piece)
                    model.addCons(carbon_piece <= upper * power_piece)
                    model.addCons(
                        carbon_piece
                        >= upper * power_piece + power_upper * ci_piece
                        - upper * power_upper * active
                    )
                    model.addCons(
                        carbon_piece
                        <= lower * power_piece + power_upper * ci_piece
                        - lower * power_upper * active
                    )
                    pieces.append((power_piece, carbon_piece))
                model.addCons(quicksum(piece[0] for piece in pieces) == power)
                model.addCons(quicksum(piece[1] for piece in pieces) == product)
            if allow_carbon_slack:
                if use_system_boundary or use_carbon_bins:
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
            skip_vintages = storage_disabled or use_aggregate or use_storage_bins
            layer_active = {} if skip_vintages else {(bus, k): model.addVar(vtype="B", name=f"ulayer[{sid},{bus},{k}]") for bus in candidates for k in range(1, horizon + 1)}
            vintage_carbon = {} if skip_vintages else {(bus, k): model.addVar(lb=0.0, ub=pcfg.carbon_intensity_max, name=f"vintage_ci[{sid},{bus},{k}]") for bus in candidates for k in layers}
            layer_charge = {} if skip_vintages else {(bus, k): model.addVar(lb=0.0, ub=pcfg.max_power_mw, name=f"layer_ch[{sid},{bus},{k}]") for bus in candidates for k in range(1, horizon + 1)}
            layer_discharge = {} if skip_vintages else {(bus, k, t): model.addVar(lb=0.0, ub=pcfg.max_power_mw, name=f"layer_dc[{sid},{bus},{k},{t}]") for bus in candidates for k in layers for t in times}
            layer_energy = {} if skip_vintages else {(bus, k, t): model.addVar(lb=0.0, ub=pcfg.max_energy_mwh, name=f"layer_e[{sid},{bus},{k},{t}]") for bus in candidates for k in layers for t in states}
            layer_carbon_mass = {} if skip_vintages else {(bus, k, t): model.addVar(lb=0.0, ub=pcfg.carbon_intensity_max * pcfg.max_energy_mwh, name=f"layer_m[{sid},{bus},{k},{t}]") for bus in candidates for k in layers for t in states}
            layer_discharge_carbon: dict[tuple[str, int, int], object] = {}
            flow_carbon_plus: dict[tuple[tuple[str, str, str], int], object] = {}
            flow_carbon_minus: dict[tuple[tuple[str, str, str], int], object] = {}
            bin_charge: dict[tuple[str, int, int], object] = {}
            bin_discharge: dict[tuple[str, int, int], object] = {}
            bin_energy: dict[tuple[str, int, int], object] = {}
            bin_flow_plus: dict[tuple[tuple[str, str, str], int, int], object] = {}
            bin_flow_minus: dict[tuple[tuple[str, str, str], int, int], object] = {}
            bin_active: dict[tuple[str, int, int], object] = {}

            model.addCons(backlog[0] == 0.0, name=f"backlog_start[{sid}]")
            model.addCons(
                backlog[horizon] == 0.0,
                name=f"backlog_terminal[{sid}]",
            )
            if dccfg.workload_max_delay_hours < 0:
                raise ValueError("workload_max_delay_hours must be nonnegative.")
            for t in times:
                model.addCons(
                    backlog[t + 1]
                    == backlog[t]
                    + float(scenario.workload_arrival[t])
                    - processed[t],
                    name=f"backlog_dynamics[{sid},{t}]",
                )
                if dccfg.workload_max_delay_hours == 0:
                    model.addCons(
                        processed[t] == float(scenario.workload_arrival[t]),
                        name=f"work_fixed[{sid},{t}]",
                    )
                else:
                    horizon_deadline = min(
                        horizon - 1, t + dccfg.workload_max_delay_hours
                    )
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
            if use_storage_bins and not storage_disabled:
                initial_bin = min(
                    range(len(carbon_bin_intensity)),
                    key=lambda b: abs(carbon_bin_intensity[b] - pcfg.initial_carbon_intensity),
                )
                for bus in candidates:
                    for b in range(len(carbon_bin_intensity)):
                        for t in states:
                            bin_energy[bus, b, t] = model.addVar(
                                lb=0.0,
                                ub=pcfg.max_energy_mwh,
                                name=f"bin_e[{sid},{bus},{b},{t}]",
                            )
                        model.addCons(
                            bin_energy[bus, b, 0]
                            == (pcfg.initial_soc * ecap[bus] if b == initial_bin else 0.0)
                        )
                        for t in times:
                            bin_charge[bus, b, t] = model.addVar(
                                lb=0.0, ub=pcfg.max_power_mw,
                                name=f"bin_ch[{sid},{bus},{b},{t}]",
                            )
                            bin_discharge[bus, b, t] = model.addVar(
                                lb=0.0, ub=pcfg.max_power_mw,
                                name=f"bin_dc[{sid},{bus},{b},{t}]",
                            )
                            if use_binned_storage:
                                bin_active[bus, b, t] = nodal_bin_active[bus, t, b]
                            model.addCons(
                                bin_energy[bus, b, t + 1]
                                == retention * bin_energy[bus, b, t]
                                + pcfg.charge_efficiency * bin_charge[bus, b, t] * dt
                                - bin_discharge[bus, b, t] * dt / pcfg.discharge_efficiency
                            )
                            model.addCons(
                                bin_discharge[bus, b, t] * dt / pcfg.discharge_efficiency
                                <= retention * bin_energy[bus, b, t]
                            )
                            if use_binned_storage:
                                model.addCons(
                                    bin_charge[bus, b, t]
                                    <= pcfg.max_power_mw * bin_active[bus, b, t]
                                )
                    for t in times:
                        charge_power[bus, t] = quicksum(
                            bin_charge[bus, b, t] for b in range(len(carbon_bin_intensity))
                        )
                        discharge_power[bus, t] = quicksum(
                            bin_discharge[bus, b, t] for b in range(len(carbon_bin_intensity))
                        )
                        model.addCons(charge_status[bus, t] <= site[bus])
                        model.addCons(discharge_status[bus, t] <= site[bus])
                        model.addCons(charge_status[bus, t] + discharge_status[bus, t] <= 1)
                        model.addCons(charge_power[bus, t] <= pcap[bus])
                        model.addCons(discharge_power[bus, t] <= pcap[bus])
                        model.addCons(charge_power[bus, t] <= pcfg.max_power_mw * charge_status[bus, t])
                        model.addCons(discharge_power[bus, t] <= pcfg.max_power_mw * discharge_status[bus, t])
                        storage_energy[bus, t] = quicksum(
                            bin_energy[bus, b, t] for b in range(len(carbon_bin_intensity))
                        )
                        model.addCons(storage_energy[bus, t] >= pcfg.min_soc * ecap[bus])
                        model.addCons(storage_energy[bus, t] <= pcfg.max_soc * ecap[bus])
                    terminal_energy = quicksum(
                        bin_energy[bus, b, horizon] for b in range(len(carbon_bin_intensity))
                    )
                    terminal_carbon = quicksum(
                        carbon_bin_intensity[b] * bin_energy[bus, b, horizon]
                        for b in range(len(carbon_bin_intensity))
                    )
                    model.addCons(terminal_energy == pcfg.initial_soc * ecap[bus])
                    model.addCons(
                        terminal_carbon
                        <= pcfg.initial_carbon_intensity * pcfg.initial_soc * ecap[bus]
                    )
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
                        if use_system_boundary:
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

            for bus in (
                ()
                if storage_disabled or use_aggregate or use_storage_bins
                else candidates
            ):
                model.addCons(vintage_carbon[bus, 0] == pcfg.initial_carbon_intensity * site[bus])
                model.addCons(layer_energy[bus, 0, 0] == pcfg.initial_soc * ecap[bus])
                model.addCons(layer_carbon_mass[bus, 0, 0] == pcfg.initial_carbon_intensity * layer_energy[bus, 0, 0])
                for k in range(1, horizon + 1):
                    model.addCons(layer_energy[bus, k, 0] == 0.0)
                    model.addCons(layer_carbon_mass[bus, k, 0] == 0.0)
                    model.addCons(layer_charge[bus, k] >= epsilon_charge * layer_active[bus, k])
                    model.addCons(layer_charge[bus, k] <= pcfg.max_power_mw * layer_active[bus, k])
                    formation_t = k - 1
                    if use_layered_system:
                        model.addCons(
                            vintage_carbon[bus, k]
                            == float(scenario.grid_carbon_t_per_mwh[formation_t])
                            * layer_active[bus, k]
                        )
                    else:
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
                        elif use_layered_system:
                            fixed_ci = (
                                pcfg.initial_carbon_intensity
                                if k == 0
                                else float(scenario.grid_carbon_t_per_mwh[k - 1])
                            )
                            model.addCons(
                                layer_carbon_mass[bus, k, t]
                                == fixed_ci * layer_energy[bus, k, t]
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
                    if not use_system_boundary:
                        model.addCons(flow_p[edge, t] == flow_plus[edge, t] - flow_minus[edge, t])
                        model.addCons(flow_plus[edge, t] <= rating * direction[edge, t])
                        model.addCons(flow_minus[edge, t] <= rating * (1 - direction[edge, t]))
                        if use_carbon_bins:
                            for b in range(len(carbon_bin_intensity)):
                                bin_flow_plus[edge, t, b] = model.addVar(
                                    lb=0.0, ub=rating,
                                    name=f"bin_pplus[{sid},{edge[0]},{edge[1]},{edge[2]},{t},{b}]",
                                )
                                bin_flow_minus[edge, t, b] = model.addVar(
                                    lb=0.0, ub=rating,
                                    name=f"bin_pminus[{sid},{edge[0]},{edge[1]},{edge[2]},{t},{b}]",
                                )
                            model.addCons(
                                quicksum(bin_flow_plus[edge, t, b] for b in range(len(carbon_bin_intensity)))
                                == flow_plus[edge, t]
                            )
                            model.addCons(
                                quicksum(bin_flow_minus[edge, t, b] for b in range(len(carbon_bin_intensity)))
                                == flow_minus[edge, t]
                            )
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
                    if use_mccormick and not use_system_boundary and not use_carbon_bins:
                        # Bounds on the two directional flows. The rating is a
                        # thermal limit; what actually constrains the flow is
                        # what lies below the edge -- forward flow can only serve
                        # the child subtree's demand, reverse flow can only carry
                        # that subtree's own injection. min() keeps this no looser
                        # than the rating, so it can never admit a new solution.
                        plus_upper = rating
                        minus_upper = rating
                        if pcfg.carbon_envelope_bound == "demand":
                            child = edge[1]
                            plus_upper = min(rating, subtree_absorbed(child, t))
                            minus_upper = min(rating, subtree_injected(child, t))
                        carbon_plus = model.addVar(
                            lb=0.0,
                            ub=pcfg.carbon_intensity_max * plus_upper,
                            name=f"carbon_plus[{sid},{edge[0]},{edge[1]},{edge[2]},{t}]",
                        )
                        carbon_minus = model.addVar(
                            lb=0.0,
                            ub=pcfg.carbon_intensity_max * minus_upper,
                            name=f"carbon_minus[{sid},{edge[0]},{edge[1]},{edge[2]},{t}]",
                        )
                        flow_carbon_plus[edge, t] = carbon_plus
                        flow_carbon_minus[edge, t] = carbon_minus
                        if use_binned_storage:
                            add_piecewise_carbon_product(
                                carbon_plus, edge[0], t, flow_plus[edge, t],
                                plus_upper,
                                f"pw_plus[{sid},{edge[0]},{edge[1]},{edge[2]},{t}]",
                            )
                            add_piecewise_carbon_product(
                                carbon_minus, edge[1], t, flow_minus[edge, t],
                                minus_upper,
                                f"pw_minus[{sid},{edge[0]},{edge[1]},{edge[2]},{t}]",
                            )
                        else:
                            _add_mccormick_envelope(
                                model, carbon_plus, nodal_carbon[edge[0], t],
                                flow_plus[edge, t], pcfg.carbon_intensity_max, plus_upper,
                            )
                            _add_mccormick_envelope(
                                model, carbon_minus, nodal_carbon[edge[1], t],
                                flow_minus[edge, t], pcfg.carbon_intensity_max, minus_upper,
                            )
                for phase in bus_phases[root]:
                    model.addCons(voltage[root, phase, t] == 1.0)

                dc_power = dccfg.power_mw(float(scenario.pue[t]), processed[t])
                dc_bin_load = {}
                if use_carbon_bins:
                    for b in range(len(carbon_bin_intensity)):
                        dc_bin_load[b] = model.addVar(
                            lb=0.0,
                            ub=dccfg.power_mw(float(scenario.pue[t]), dccfg.workload_processing_upper),
                            name=f"dc_bin_load[{sid},{t},{b}]",
                        )
                    model.addCons(
                        quicksum(dc_bin_load[b] for b in range(len(carbon_bin_intensity)))
                        == dc_power - shed[t]
                    )
                    if (
                        pcfg.storage_service_mode == "dedicated_dc"
                        and not storage_disabled
                    ):
                        for b in range(len(carbon_bin_intensity)):
                            # Reserve every unit discharged by the colocated
                            # battery for the data-center sink of the same carbon
                            # attribute. Network/PV supply may cover the rest.
                            model.addCons(
                                dc_bin_load[b]
                                >= bin_discharge[feeder.data_center_bus, b, t]
                            )
                    dc_carbon = quicksum(
                        carbon_bin_intensity[b] * dc_bin_load[b]
                        for b in range(len(carbon_bin_intensity))
                    )
                    if pcfg.carbon_cap_scope == "horizon":
                        horizon_carbon_terms.append(dc_carbon * dt)
                        horizon_energy_terms.append((dc_power - shed[t]) * dt)
                    elif ccfg.carbon_price_dollars_per_t > 0.0:
                        excess = model.addVar(lb=0.0, name=f"dc_bin_excess[{sid},{t}]")
                        model.addCons(
                            dc_carbon <= pcfg.dc_carbon_cap * (dc_power - shed[t]) + excess
                        )
                        carbon_slack_terms.append(
                            scenario_weight * annual_blocks
                            * ccfg.carbon_price_dollars_per_t * excess * dt
                        )
                    else:
                        slack = carbon_slack["system", t] if allow_carbon_slack else 0.0
                        model.addCons(
                            dc_carbon <= pcfg.dc_carbon_cap * (dc_power - shed[t]) + slack
                        )
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
                    if use_carbon_bins:
                        base_bin_load = {
                            b: model.addVar(
                                lb=0.0,
                                ub=float(scenario.active_load_mw[t, bi, pi]),
                                name=f"base_bin_load[{sid},{bus},{phase},{t},{b}]",
                            )
                            for b in range(len(carbon_bin_intensity))
                        }
                        model.addCons(
                            quicksum(base_bin_load.values())
                            == float(scenario.active_load_mw[t, bi, pi])
                        )
                        grid_bin = next(
                            b for b, intensity in enumerate(carbon_bin_intensity)
                            if intensity >= float(scenario.grid_carbon_t_per_mwh[t]) - 1.0e-9
                        )
                        generator_bin = len(carbon_bin_intensity) - 1
                        for b in range(len(carbon_bin_intensity)):
                            commodity_source = (
                                (grid_phase[phase, t] if bus == root and b == grid_bin else 0.0)
                                + (local_generation_phase if bus == feeder.generator_bus and b == generator_bin else 0.0)
                                + (pv[bus, phase, t] if b == 0 else 0.0)
                                + (
                                    bin_discharge[bus, b, t] / phase_count
                                    if bus in candidates and not storage_disabled else 0.0
                                )
                            )
                            commodity_in = quicksum(
                                bin_flow_plus[edge, t, b] for edge in incoming[bus, phase]
                            ) + quicksum(
                                bin_flow_minus[edge, t, b] for edge in outgoing[bus, phase]
                            )
                            commodity_out = quicksum(
                                bin_flow_plus[edge, t, b] for edge in outgoing[bus, phase]
                            ) + quicksum(
                                bin_flow_minus[edge, t, b] for edge in incoming[bus, phase]
                            )
                            commodity_sink = base_bin_load[b]
                            if bus == feeder.data_center_bus:
                                commodity_sink += dc_bin_load[b] / phase_count
                            if bus in candidates and not storage_disabled:
                                commodity_sink += bin_charge[bus, b, t] / phase_count
                            model.addCons(
                                commodity_source + commodity_in
                                == commodity_sink + commodity_out,
                                name=f"bin_balance[{sid},{bus},{phase},{t},{b}]",
                            )

                for bus in (() if use_system_boundary or use_carbon_bins else buses):
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
                                carbon_bin_intensity[b] * bin_discharge[bus, b, t]
                                for b in range(len(carbon_bin_intensity))
                            )
                            if use_binned_storage
                            else quicksum(
                                (
                                    layer_discharge_carbon[bus, k, t]
                                    if use_vintage_mccormick
                                    else (
                                        pcfg.initial_carbon_intensity
                                        if use_layered_system and k == 0
                                        else float(scenario.grid_carbon_t_per_mwh[k - 1])
                                        if use_layered_system
                                        else vintage_carbon[bus, k]
                                    ) * layer_discharge[bus, k, t]
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
                        if pcfg.carbon_envelope_bound == "demand":
                            # Rearranging the per-phase active balance and summing
                            # over phases gives the identity
                            #     incoming = load + charge + reverse_in + forward_out
                            # with every term nonnegative, so bounding each term
                            # separately bounds the sum: this bus absorbs what it
                            # absorbs, forward_out cannot exceed the subtree's
                            # demand, and reverse_in cannot exceed its injection.
                            incoming_power_upper = min(
                                incoming_power_upper,
                                subtree_absorbed(bus, t) + subtree_injected(bus, t),
                            )
                            # At a leaf there are no child reverse-in or forward-
                            # out terms. Summed active balance therefore gives
                            # incoming sources = local demand + storage charge
                            # exactly; including possible local injection in the
                            # product bound only weakens the carbon envelope.
                            if not children[bus]:
                                incoming_power_upper = min(
                                    incoming_power_upper, absorbed_at(bus, t)
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
                        if use_binned_storage:
                            add_piecewise_carbon_product(
                                incoming_carbon_var, bus, t, incoming_power_var,
                                incoming_power_upper,
                                f"pw_incoming[{sid},{bus},{t}]",
                            )
                        else:
                            _add_mccormick_envelope(
                                model, incoming_carbon_var, nodal_carbon[bus, t],
                                incoming_power_var, pcfg.carbon_intensity_max,
                                incoming_power_upper,
                            )
                    else:
                        model.addCons(
                            nodal_carbon[bus, t] * total_incoming_power
                            == total_incoming_carbon
                        )
                    cap = pcfg.dc_carbon_cap if bus == feeder.data_center_bus else pcfg.other_bus_carbon_cap
                    if allow_carbon_slack and ccfg.carbon_price_dollars_per_t > 0.0:
                        # Price physical excess emissions (tCO2 per interval),
                        # not a dimensionless intensity slack.  The legacy
                        # intensity penalty can overwhelm annualized battery
                        # capex merely because its coefficient is tied to a
                        # big-M scale.  This form has a stable economic meaning:
                        # incoming_carbon - cap * incoming_energy is the excess
                        # tonnes attributable to this bus in the interval.
                        excess = model.addVar(
                            lb=0.0, name=f"nodal_carbon_excess[{sid},{bus},{t}]"
                        )
                        model.addCons(
                            total_incoming_carbon
                            <= cap * total_incoming_power + excess
                        )
                        carbon_slack_terms.append(
                            scenario_weight
                            * annual_blocks
                            * ccfg.carbon_price_dollars_per_t
                            * excess
                            * dt
                        )
                    else:
                        slack = carbon_slack[bus, t] if allow_carbon_slack else 0.0
                        model.addCons(nodal_carbon[bus, t] <= cap + slack)
                        if allow_carbon_slack:
                            carbon_slack_terms.append(
                                scenario_weight
                                * annual_blocks
                                * ccfg.validation_carbon_slack_dollars
                                * carbon_slack[bus, t]
                            )

                if use_system_boundary:
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
                        + (0.0 if storage_disabled else (
                            quicksum(
                                aggregate_discharge_carbon[bus, t]
                                for bus in candidates
                            )
                            if use_system_average
                            else quicksum(
                                (
                                    pcfg.initial_carbon_intensity
                                    if k == 0
                                    else float(scenario.grid_carbon_t_per_mwh[k - 1])
                                ) * layer_discharge[bus, k, t]
                                for bus in candidates
                                for k in layers
                            )
                        ))
                    )
                    system_power_upper = (
                        pcfg.grid_limit_mw
                        + 1.0
                        + float(scenario.pv_available_mw[t].sum())
                        + len(candidates) * pcfg.max_power_mw
                    )
                    if pcfg.carbon_cap_scope == "horizon":
                        # Preserve hourly carbon physics, but defer compliance
                        # until all interval carbon mass and delivered energy
                        # have been summed over the scenario horizon.
                        horizon_carbon_terms.append(system_carbon * dt)
                        horizon_energy_terms.append(system_power * dt)
                        horizon_energy_upper += system_power_upper * dt
                    elif ccfg.carbon_price_dollars_per_t > 0.0:
                        # Excess carbon in tonnes per hour: system_carbon is an
                        # intensity times a power, so the difference is already
                        # t/h and needs no big-M to scale it. Priced with dt, as
                        # every other interval cost is, so a dollar here means
                        # the same as a dollar of energy.
                        excess = model.addVar(
                            lb=0.0, name=f"carbon_excess[{sid},{t}]"
                        )
                        model.addCons(
                            system_carbon
                            <= pcfg.dc_carbon_cap * system_power + excess,
                            name=f"system_carbon_cap[{sid},{t}]",
                        )
                        carbon_slack_terms.append(
                            scenario_weight
                            * annual_blocks
                            * ccfg.carbon_price_dollars_per_t
                            * excess
                            * dt
                        )
                    elif allow_carbon_slack:
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

            if use_system_boundary and pcfg.carbon_cap_scope == "horizon":
                horizon_carbon = quicksum(horizon_carbon_terms)
                horizon_energy = quicksum(horizon_energy_terms)
                if ccfg.carbon_price_dollars_per_t > 0.0:
                    # This variable is tonnes of CO2 per scenario block, unlike
                    # the legacy dimensionless intensity slack. Its price is
                    # therefore directly interpretable in dollars per tonne.
                    excess = model.addVar(
                        lb=0.0, name=f"carbon_budget_excess[{sid}]"
                    )
                    model.addCons(
                        horizon_carbon
                        <= pcfg.dc_carbon_cap * horizon_energy + excess,
                        name=f"system_carbon_budget[{sid}]",
                    )
                    carbon_slack_terms.append(
                        scenario_weight
                        * annual_blocks
                        * ccfg.carbon_price_dollars_per_t
                        * excess
                    )
                elif allow_carbon_slack:
                    # Backward-compatible emergency slack. It is dimensionless
                    # intensity slack, scaled by a valid constant upper bound;
                    # priced-excess experiments should use carbon_price instead.
                    slack = model.addVar(
                        lb=0.0,
                        ub=pcfg.carbon_intensity_max,
                        name=f"carbon_budget_slack[{sid}]",
                    )
                    model.addCons(
                        horizon_carbon
                        <= pcfg.dc_carbon_cap * horizon_energy
                        + horizon_energy_upper * slack,
                        name=f"system_carbon_budget[{sid}]",
                    )
                    carbon_slack_terms.append(
                        scenario_weight
                        * annual_blocks
                        * ccfg.validation_carbon_slack_dollars
                        * slack
                    )
                else:
                    model.addCons(
                        horizon_carbon <= pcfg.dc_carbon_cap * horizon_energy,
                        name=f"system_carbon_budget[{sid}]",
                    )

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
