"""Small content-addressed caches for expensive decision-loss solves."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from storage_dfl.data import Scenario
from storage_dfl.planning.results import PlanningResult, scenario_content_key


@dataclass
class PerfectInformationCache:
    """Cache reference plans for repeated real scenario bags."""

    enabled: bool = True
    _values: dict[tuple, PlanningResult] = field(default_factory=dict)

    def solve(
        self,
        oracle,
        scenarios: Iterable[Scenario],
        weights: Iterable[float],
        *,
        allow_carbon_slack: bool,
    ) -> PlanningResult:
        scenario_tuple = tuple(scenarios)
        weight_tuple = tuple(float(value) for value in weights)
        key = (
            tuple(scenario_content_key(scenario) for scenario in scenario_tuple),
            weight_tuple,
            bool(allow_carbon_slack),
        )
        if self.enabled and key in self._values:
            return self._values[key]
        result = oracle.solve(
            scenario_tuple,
            weights=weight_tuple,
            allow_carbon_slack=allow_carbon_slack,
            use_cache=self.enabled,
        )
        if self.enabled:
            self._values[key] = result
        return result
