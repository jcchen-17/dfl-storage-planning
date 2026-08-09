from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class StorageDesign:
    site: dict[str, int]
    power_mw: dict[str, float]
    energy_mwh: dict[str, float]

    @property
    def installed_buses(self) -> list[str]:
        return [bus for bus, value in self.site.items() if value > 0]


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

    @property
    def feasible(self) -> bool:
        return isfinite(self.objective) and self.status not in {"infeasible", "unbounded"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
