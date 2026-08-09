"""Fixed-design operational recourse using the existing planning MILP.

This module does not define another optimization model.  It fixes only the
``StorageDesign`` passed to ``StoragePlanningOracle.solve``; the oracle then
rebuilds and reoptimizes every operational variable under the realized data.
"""

from __future__ import annotations

from math import isfinite
from typing import Iterable

import numpy as np

from storage_dfl.data import Scenario

from .solver import PlanningJob
from .results import (
    PlanningResult,
    RecourseDiagnostics,
    StorageDesign,
    weighted_carbon_ledger,
)


def _bounded(result: PlanningResult, accepted_relative_gap: float) -> bool:
    if not result.feasible or not isfinite(result.objective):
        return False
    if result.status in {"optimal", "gaplimit"}:
        return True
    return (
        accepted_relative_gap > 0.0
        and isfinite(result.relative_gap)
        and result.relative_gap <= accepted_relative_gap
    )


def aggregate_fixed_design_results(
    results: list[PlanningResult],
    weights: Iterable[float],
    design: StorageDesign,
    demand_dollars_per_mw_year: float,
    accepted_relative_gap: float,
) -> PlanningResult:
    """Reconstruct the joint fixed-design objective from scenario solves."""

    probabilities = np.asarray(tuple(weights), dtype=float)
    if len(results) != probabilities.size or probabilities.size == 0:
        raise ValueError("Recourse results and weights must be non-empty and aligned.")
    if np.any(probabilities < 0.0) or probabilities.sum() <= 0.0:
        raise ValueError("Recourse weights must be nonnegative and sum positive.")
    probabilities /= probabilities.sum()
    names = tuple(name for result in results for name in result.scenario_names)
    if any(not result.feasible for result in results):
        return PlanningResult(
            status="infeasible",
            objective=float("inf"),
            investment_cost=float("inf"),
            operating_cost=float("inf"),
            carbon_slack_cost=float("inf"),
            peak_grid_mw=float("inf"),
            design=design,
            scenario_names=names,
            solve_time_seconds=sum(result.solve_time_seconds for result in results),
            relative_gap=float("inf"),
        )

    peak = max(float(result.peak_grid_mw) for result in results)
    non_demand_operating = sum(
        float(weight)
        * (
            float(result.operating_cost)
            - demand_dollars_per_mw_year * float(result.peak_grid_mw)
        )
        for weight, result in zip(probabilities, results, strict=True)
    )
    operating = demand_dollars_per_mw_year * peak + non_demand_operating
    carbon_cost = sum(
        float(weight) * float(result.carbon_slack_cost)
        for weight, result in zip(probabilities, results, strict=True)
    )
    investment = float(results[0].investment_cost)
    objective = investment + operating + carbon_cost

    diagnostics = None
    if all(result.recourse_diagnostics is not None for result in results):
        entries = [result.recourse_diagnostics for result in results]

        def weighted(field: str) -> float:
            return sum(
                float(weight) * float(getattr(entry, field))
                for weight, entry in zip(probabilities, entries, strict=True)
                if entry is not None
            )

        diagnostics = RecourseDiagnostics(
            load_shedding_mwh=weighted("load_shedding_mwh"),
            carbon_excess_t=weighted("carbon_excess_t"),
            pv_curtailment_mwh=weighted("pv_curtailment_mwh"),
            served_demand_mwh=weighted("served_demand_mwh"),
            operating_cost=operating,
            total_planning_cost=objective,
            hourly_load_shedding_mw=tuple(
                row
                for entry in entries
                if entry is not None
                for row in entry.hourly_load_shedding_mw
            ),
            hourly_carbon_excess_t_per_hour=tuple(
                row
                for entry in entries
                if entry is not None
                for row in entry.hourly_carbon_excess_t_per_hour
            ),
        )

    bounded = all(_bounded(result, accepted_relative_gap) for result in results)
    return PlanningResult(
        status="optimal" if bounded else "scenario_limit",
        objective=objective,
        investment_cost=investment,
        operating_cost=operating,
        carbon_slack_cost=carbon_cost,
        peak_grid_mw=peak,
        design=design,
        scenario_names=names,
        solve_time_seconds=sum(result.solve_time_seconds for result in results),
        relative_gap=max(float(result.relative_gap) for result in results),
        best_bound=sum(
            float(weight) * float(result.best_bound)
            for weight, result in zip(probabilities, results, strict=True)
        ),
        carbon_ledger=weighted_carbon_ledger(results, probabilities),
        recourse_diagnostics=diagnostics,
    )


def evaluate_fixed_design_recourse(
    oracle,
    scenarios: Iterable[Scenario],
    weights: Iterable[float],
    design: StorageDesign,
    *,
    allow_carbon_slack: bool = True,
    use_cache: bool = True,
    accepted_relative_gap: float | None = None,
) -> tuple[PlanningResult, list[PlanningResult]]:
    """Fix ``site/P/E`` and reoptimize all scenario operations as true MILPs."""

    scenario_tuple = tuple(scenarios)
    weight_tuple = tuple(float(value) for value in weights)
    results = oracle.solve_many(
        [PlanningJob((scenario,), fixed_design=design) for scenario in scenario_tuple],
        allow_carbon_slack=allow_carbon_slack,
        use_cache=use_cache,
    )
    aggregate = aggregate_fixed_design_results(
        results,
        weight_tuple,
        design,
        oracle.costs.demand_dollars_per_mw_year,
        (
            float(accepted_relative_gap)
            if accepted_relative_gap is not None
            else float(oracle.planning.solver_relative_gap)
        ),
    )
    return aggregate, results
