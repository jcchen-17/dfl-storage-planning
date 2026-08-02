from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    context: np.ndarray
    active_load_mw: np.ndarray
    reactive_load_mvar: np.ndarray
    pv_available_mw: np.ndarray
    workload_arrival: np.ndarray
    pue: np.ndarray
    grid_price_per_mwh: np.ndarray
    grid_carbon_t_per_mwh: np.ndarray
    grid_available: np.ndarray

    @property
    def horizon(self) -> int:
        return int(self.workload_arrival.shape[0])

    @property
    def num_buses(self) -> int:
        return int(self.active_load_mw.shape[1])

    @property
    def num_phases(self) -> int:
        return int(self.active_load_mw.shape[2])

    def validate(self) -> None:
        if self.active_load_mw.ndim != 3:
            raise ValueError(f"{self.name}: active load must have shape [time, bus, phase]")
        horizon, buses, phases = self.active_load_mw.shape
        if phases != 3:
            raise ValueError(f"{self.name}: the phase axis must be A/B/C")
        if self.context.ndim != 1 or self.context.size == 0:
            raise ValueError(f"{self.name}: context must be a nonempty vector")
        if self.reactive_load_mvar.shape != (horizon, buses, phases):
            raise ValueError(f"{self.name}: reactive-load shape mismatch")
        if self.pv_available_mw.shape != (horizon, buses, phases):
            raise ValueError(f"{self.name}: PV shape mismatch")
        for vector in (
            self.workload_arrival,
            self.pue,
            self.grid_price_per_mwh,
            self.grid_carbon_t_per_mwh,
            self.grid_available,
        ):
            if vector.shape != (horizon,):
                raise ValueError(f"{self.name}: time-series shape mismatch")
        if np.any(self.active_load_mw < 0) or np.any(self.pv_available_mw < 0):
            raise ValueError(f"{self.name}: power profiles must be nonnegative")


@dataclass(frozen=True)
class ScenarioPool:
    scenarios: tuple[Scenario, ...]

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("A scenario pool cannot be empty.")
        for scenario in self.scenarios:
            scenario.validate()
        shapes = {
            (scenario.horizon, scenario.num_buses, scenario.num_phases)
            for scenario in self.scenarios
        }
        if len(shapes) != 1:
            raise ValueError("All scenarios must use the same horizon and network.")
        context_dims = {scenario.context.shape for scenario in self.scenarios}
        if len(context_dims) != 1:
            raise ValueError("All scenarios must use the same context dimension.")

    @property
    def context_dim(self) -> int:
        return int(self.scenarios[0].context.size)

    def subset(self, indices: tuple[int, ...] | list[int]) -> tuple[Scenario, ...]:
        return tuple(self.scenarios[index] for index in indices)

    def names(self, indices: tuple[int, ...] | list[int]) -> list[str]:
        return [self.scenarios[index].name for index in indices]
