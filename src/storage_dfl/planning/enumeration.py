from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from math import isfinite
from typing import Iterable

from ..config import CostConfig, DataCenterConfig, DataConfig, PlanningConfig
from ..data.schema import Scenario
from ..network.ieee13 import Feeder
from .model import PlanningJob, StoragePlanningOracle
from .results import PlanningResult, StorageDesign


def _expanded_design(design: StorageDesign, buses: tuple[str, ...]) -> StorageDesign:
    return StorageDesign(
        site={bus: int(design.site.get(bus, 0)) for bus in buses},
        power_mw={bus: float(design.power_mw.get(bus, 0.0)) for bus in buses},
        energy_mwh={bus: float(design.energy_mwh.get(bus, 0.0)) for bus in buses},
    )


class EnumeratedStoragePlanningOracle:
    """Exact one-site decomposition with the StoragePlanningOracle interface.

    Free planning is solved over ``F_none union F_bus+``.  Each ``F_bus+`` has
    exactly one candidate and enforces one installation.  Therefore taking the
    best incumbent is exactly the original max-one-site problem, not a proxy or
    relaxation.  Fixed-design jobs are delegated unchanged to the base oracle.
    """

    def __init__(
        self,
        feeder: Feeder,
        planning: PlanningConfig,
        costs: CostConfig,
        data: DataConfig,
        data_center: DataCenterConfig | None = None,
    ) -> None:
        if planning.max_storage_sites != 1 or planning.min_storage_sites != 0:
            raise ValueError(
                "Site enumeration requires max_storage_sites=1 and "
                "min_storage_sites=0."
            )
        self.feeder = feeder
        self.planning = planning
        self.costs = costs
        self.data = data
        self.data_center = data_center or DataCenterConfig()
        disabled = replace(planning, enumerate_storage_sites=False)
        self._base = StoragePlanningOracle(
            feeder, disabled, costs, data, self.data_center
        )
        self._none = StoragePlanningOracle(
            feeder,
            replace(disabled, min_storage_sites=0, max_storage_sites=0),
            costs,
            data,
            self.data_center,
        )
        self._cache: dict[tuple, PlanningResult] = {}

    def _part_planning(self) -> PlanningConfig:
        seconds = float(self.planning.enumeration_site_time_limit_seconds)
        if seconds <= 0.0:
            seconds = float(self.planning.solver_time_limit_seconds)
        return replace(
            self.planning,
            enumerate_storage_sites=False,
            min_storage_sites=1,
            max_storage_sites=1,
            solver_time_limit_seconds=seconds,
            solver_relative_gap=0.0,
            solver_absolute_gap_dollars=float(
                self.planning.enumeration_absolute_gap_dollars
            ),
            # Match the protocol used to select the 60 s budget.  Changing the
            # search policy would invalidate that empirical completeness check.
            solver_emphasis="optimality",
            solver_aggressive_heuristics=False,
            solver_max_parallel_workers=1,
        )

    def _solve_part(
        self,
        bus: str,
        scenarios: tuple[Scenario, ...],
        weights: tuple[float, ...],
        allow_carbon_slack: bool,
        bootstrap: dict[str, float] | None,
    ) -> PlanningResult:
        restricted = replace(self.feeder, storage_candidates=(bus,))
        oracle = StoragePlanningOracle(
            restricted,
            self._part_planning(),
            self.costs,
            self.data,
            self.data_center,
        )
        warm_key = oracle._warm_start_key(scenarios, weights, allow_carbon_slack)
        # A no-storage dispatch can be lifted to the minimum idle forced design.
        # Sharing it means the bootstrap is paid once per scenario set, not once
        # per candidate bus.
        oracle._warm_start_cache[warm_key] = bootstrap
        result = oracle.solve(
            scenarios,
            weights=weights,
            allow_carbon_slack=allow_carbon_slack,
        )
        if result.feasible and result.design.installed_buses != [bus]:
            raise RuntimeError(
                f"Forced-site part {bus!r} returned "
                f"{result.design.installed_buses!r}."
            )
        return result

    def _combine(self, parts: list[PlanningResult]) -> PlanningResult:
        feasible = [part for part in parts if part.feasible]
        if not feasible:
            # Preserve the most informative actual solver result.
            return parts[0]
        winner = min(feasible, key=lambda part: part.objective)
        bounds = []
        for part in parts:
            if part.status == "infeasible":
                bounds.append(float("inf"))
            else:
                bounds.append(float(part.best_bound))
        global_bound = min(bounds, default=float("-inf"))
        gap = (
            max(0.0, winner.objective - global_bound)
            / max(1.0, abs(winner.objective))
            if isfinite(global_bound)
            else float("inf")
        )
        certified = all(
            part.status in {"optimal", "infeasible"} for part in parts
        )
        return replace(
            winner,
            status="optimal" if certified else "enumeration_limit",
            design=_expanded_design(winner.design, self.feeder.storage_candidates),
            solve_time_seconds=sum(part.solve_time_seconds for part in parts),
            relative_gap=gap,
            best_bound=global_bound,
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
        if fixed_design is not None:
            return self._base.solve(
                scenarios,
                weights=weights,
                fixed_design=fixed_design,
                allow_carbon_slack=allow_carbon_slack,
                use_cache=use_cache,
            )
        scenario_tuple, weight_tuple = self._base._normalize_job(scenarios, weights)
        cache_key = self._base._cache_key(
            scenario_tuple, weight_tuple, None, allow_carbon_slack
        )
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        none = self._none.solve(
            scenario_tuple,
            weights=weight_tuple,
            allow_carbon_slack=allow_carbon_slack,
            use_cache=use_cache,
        )
        warm_key = self._none._warm_start_key(
            scenario_tuple, weight_tuple, allow_carbon_slack
        )
        bootstrap = self._none._warm_start_cache.get(warm_key)
        workers = max(1, int(self.planning.solver_max_parallel_workers))
        buses = self.feeder.storage_candidates
        with ThreadPoolExecutor(max_workers=min(workers, len(buses))) as pool:
            site_parts = list(
                pool.map(
                    lambda bus: self._solve_part(
                        bus,
                        scenario_tuple,
                        weight_tuple,
                        allow_carbon_slack,
                        bootstrap,
                    ),
                    buses,
                )
            )
        result = self._combine([none, *site_parts])
        if use_cache:
            self._cache[cache_key] = result
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
        if not requested:
            return []
        # Fixed-design validation has no siting decision and stays on the
        # original batched implementation.
        if all(job.fixed_design is not None for job in requested):
            return self._base.solve_many(
                requested,
                allow_carbon_slack=allow_carbon_slack,
                use_cache=use_cache,
                max_workers=max_workers,
            )
        if any(job.fixed_design is not None for job in requested):
            return [
                self.solve(
                    job.scenarios,
                    weights=job.weights,
                    fixed_design=job.fixed_design,
                    allow_carbon_slack=allow_carbon_slack,
                    use_cache=use_cache,
                )
                for job in requested
            ]

        normalized = [
            self._base._normalize_job(job.scenarios, job.weights)
            for job in requested
        ]
        cache_keys = [
            self._base._cache_key(scenarios, weights, None, allow_carbon_slack)
            for scenarios, weights in normalized
        ]
        results: list[PlanningResult | None] = [None] * len(requested)
        pending = []
        for index, key in enumerate(cache_keys):
            if use_cache and key in self._cache:
                results[index] = self._cache[key]
            else:
                pending.append(index)
        if not pending:
            return [result for result in results if result is not None]

        workers = max_workers
        if workers is None:
            workers = self.planning.solver_max_parallel_workers
        workers = max(1, int(workers))
        none_jobs = [
            PlanningJob(normalized[index][0], normalized[index][1])
            for index in pending
        ]
        none_results = self._none.solve_many(
            none_jobs,
            allow_carbon_slack=allow_carbon_slack,
            use_cache=use_cache,
            max_workers=workers,
        )

        tasks = []
        parts: dict[int, list[PlanningResult]] = {}
        for index, none in zip(pending, none_results, strict=True):
            scenarios, weights = normalized[index]
            warm_key = self._none._warm_start_key(
                scenarios, weights, allow_carbon_slack
            )
            bootstrap = self._none._warm_start_cache.get(warm_key)
            parts[index] = [none]
            for bus in self.feeder.storage_candidates:
                tasks.append((index, bus, scenarios, weights, bootstrap))

        def run_task(task: tuple) -> tuple[int, PlanningResult]:
            index, bus, scenarios, weights, bootstrap = task
            return (
                index,
                self._solve_part(
                    bus,
                    scenarios,
                    weights,
                    allow_carbon_slack,
                    bootstrap,
                ),
            )

        # One global queue enforces the memory/solver limit across both samples
        # and buses.  With four samples and five buses this needs five waves,
        # rather than running two bus waves separately for every sample.
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            for index, part in pool.map(run_task, tasks):
                parts[index].append(part)

        for index in pending:
            result = self._combine(parts[index])
            results[index] = result
            if use_cache:
                self._cache[cache_keys[index]] = result
        completed = [result for result in results if result is not None]
        if len(completed) != len(results):
            raise RuntimeError("An enumerated planning job returned no result.")
        return completed


def make_planning_oracle(
    feeder: Feeder,
    planning: PlanningConfig,
    costs: CostConfig,
    data: DataConfig,
    data_center: DataCenterConfig | None = None,
) -> StoragePlanningOracle | EnumeratedStoragePlanningOracle:
    if planning.enumerate_storage_sites:
        return EnumeratedStoragePlanningOracle(
            feeder, planning, costs, data, data_center
        )
    return StoragePlanningOracle(feeder, planning, costs, data, data_center)
