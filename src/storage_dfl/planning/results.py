from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class StorageDesign:
    site: dict[str, int]
    power_mw: dict[str, float]
    energy_mwh: dict[str, float]

    @property
    def installed_buses(self) -> list[str]:
        return [bus for bus, value in self.site.items() if value > 0]


@dataclass(frozen=True)
class RecourseDiagnostics:
    """Physical and economic outputs from a planning or fixed-design solve.

    Energy/carbon totals use the same annualisation as the planning objective.
    Hourly arrays retain unscaled physical units and are ordered like
    ``PlanningResult.scenario_names``.  They are deliberately solver outputs,
    not Torch tensors: every MILP decision is a stop-gradient constant in DFL.
    """

    load_shedding_mwh: float
    carbon_excess_t: float
    pv_curtailment_mwh: float
    served_demand_mwh: float
    operating_cost: float
    total_planning_cost: float
    hourly_load_shedding_mw: tuple[tuple[float, ...], ...] = ()
    hourly_carbon_excess_t_per_hour: tuple[tuple[float, ...], ...] = ()
    # Per scenario, compliance-hour by grid-source-hour derivative of reported
    # carbon with respect to grid intensity. This exposes which charging
    # vintage ultimately supplies each discharge hour without differentiating
    # through the MILP.
    grid_carbon_exposure_mw: tuple[tuple[tuple[float, ...], ...], ...] = ()


@dataclass(frozen=True)
class PlanningResult:
    status: str
    objective: float
    investment_cost: float
    operating_cost: float
    carbon_slack_cost: float
    peak_grid_mw: float
    design: StorageDesign
    scenario_names: tuple[str, ...]
    solve_time_seconds: float
    relative_gap: float
    # Best proven bound. Kept alongside the gap because enumerating the storage
    # site turns one solve into several, and the enumerated problem's global
    # bound is min_i best_bound_i -- the winning subproblem's own gap describes
    # only that subproblem. Defaulted so results written before the field
    # existed still load.
    best_bound: float = float("-inf")
    # Optional auditable carbon/energy totals. The single-PCC oracle reports
    # annualized source, storage and delivered-load layers here; legacy feeder
    # results leave it unset.
    carbon_ledger: dict[str, float] | None = None
    # Optional recourse-oriented metrics. Legacy feeder results and historical
    # artifacts leave this unset; the single-PCC oracle populates it.
    recourse_diagnostics: RecourseDiagnostics | None = None

    @property
    def feasible(self) -> bool:
        return isfinite(self.objective) and self.status not in {"infeasible", "unbounded"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def weighted_carbon_ledger(
    results: list[PlanningResult], weights: Iterable[float]
) -> dict[str, float] | None:
    """Combine scenario-wise PCC ledgers with the evaluation probabilities."""

    if not results or any(result.carbon_ledger is None for result in results):
        return None
    normalized = [float(weight) for weight in weights]
    total = sum(normalized)
    if total <= 0.0 or len(normalized) != len(results):
        raise ValueError("Carbon-ledger weights must match results and sum positive.")
    normalized = [weight / total for weight in normalized]
    keys = set(results[0].carbon_ledger or {})
    if any(set(result.carbon_ledger or {}) != keys for result in results):
        raise ValueError("Scenario carbon ledgers do not expose the same fields.")
    return {
        key: sum(
            weight * float((result.carbon_ledger or {})[key])
            for weight, result in zip(normalized, results, strict=True)
        )
        for key in sorted(keys)
    }


def scenario_content_key(scenario: Any) -> str:
    """Stable content digest for a scenario used by solver caches.

    Scenario names are labels, not identities: generated samples routinely
    reuse a prefix while their trajectories change after every gradient step.
    Hashing every model input prevents a stale MILP result from being returned
    for new load/PV/carbon data.
    """

    digest = hashlib.sha256()
    for field in (
        "context",
        "active_load_mw",
        "reactive_load_mvar",
        "pv_available_mw",
        "workload_arrival",
        "pue",
        "grid_price_per_mwh",
        "grid_carbon_t_per_mwh",
        "grid_available",
    ):
        values = np.ascontiguousarray(
            np.asarray(getattr(scenario, field), dtype=np.float64)
        )
        digest.update(field.encode("utf-8"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    occurrence = getattr(scenario, "annual_occurrences", None)
    digest.update(
        np.asarray(
            [np.nan if occurrence is None else float(occurrence)],
            dtype=np.float64,
        ).tobytes()
    )
    return digest.hexdigest()


def infeasible_result(status: str, buses: tuple[str, ...], scenarios: tuple[str, ...], solve_time: float) -> PlanningResult:
    empty_design = StorageDesign(
        site={bus: 0 for bus in buses},
        power_mw={bus: 0.0 for bus in buses},
        energy_mwh={bus: 0.0 for bus in buses},
    )
    return PlanningResult(
        status=status,
        objective=float("inf"),
        investment_cost=float("inf"),
        operating_cost=float("inf"),
        carbon_slack_cost=float("inf"),
        peak_grid_mw=float("inf"),
        design=empty_design,
        scenario_names=scenarios,
        solve_time_seconds=solve_time,
        relative_gap=float("inf"),
    )
