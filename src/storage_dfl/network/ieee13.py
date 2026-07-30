from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Line:
    parent: str
    child: str
    resistance: float
    reactance: float
    rating_mva: float


@dataclass(frozen=True)
class Feeder:
    buses: tuple[str, ...]
    root: str
    lines: tuple[Line, ...]
    base_active_load_mw: np.ndarray
    base_reactive_load_mvar: np.ndarray
    pv_capacity_mw: np.ndarray
    storage_candidates: tuple[str, ...]
    data_center_bus: str
    generator_bus: str

    @property
    def bus_index(self) -> dict[str, int]:
        return {bus: index for index, bus in enumerate(self.buses)}

    def validate(self) -> None:
        if len(self.buses) != 13:
            raise ValueError("The demonstration feeder must contain 13 buses.")
        if len(self.lines) != len(self.buses) - 1:
            raise ValueError("LinDistFlow requires a radial feeder with N-1 lines.")
        if self.root not in self.buses:
            raise ValueError("The root bus is not in the feeder.")
        children = {line.child for line in self.lines}
        if children != set(self.buses) - {self.root}:
            raise ValueError("Every non-root bus must have exactly one parent.")
        for array in (
            self.base_active_load_mw,
            self.base_reactive_load_mvar,
            self.pv_capacity_mw,
        ):
            if array.shape != (len(self.buses),):
                raise ValueError("Nodal parameter arrays must have one entry per bus.")


def ieee13_balanced_microgrid() -> Feeder:
    """Return a balanced radial equivalent of the IEEE 13-node feeder.

    The original feeder is unbalanced and phase-specific.  This compact model keeps
    its named buses and radial topology, but uses single-phase per-unit line data so
    that it is consistent with the paper's balanced LinDistFlow formulation.
    """

    buses = (
        "650",
        "632",
        "633",
        "634",
        "645",
        "646",
        "671",
        "680",
        "684",
        "611",
        "652",
        "692",
        "675",
    )
    lines = (
        Line("650", "632", 0.0030, 0.0060, 5.0),
        Line("632", "633", 0.0024, 0.0048, 3.0),
        Line("633", "634", 0.0020, 0.0040, 2.0),
        Line("632", "645", 0.0028, 0.0055, 2.0),
        Line("645", "646", 0.0020, 0.0040, 1.5),
        Line("632", "671", 0.0032, 0.0062, 4.0),
        Line("671", "680", 0.0016, 0.0032, 2.0),
        Line("671", "684", 0.0022, 0.0043, 2.0),
        Line("684", "611", 0.0018, 0.0035, 1.0),
        Line("684", "652", 0.0025, 0.0048, 1.0),
        Line("671", "692", 0.0010, 0.0020, 2.5),
        Line("692", "675", 0.0022, 0.0042, 2.5),
    )

    # MW/Mvar values are scaled from the standard feeder load pattern.  The data
    # center is added separately at bus 675 by the planning model.
    active = np.array(
        [0.00, 0.10, 0.12, 0.40, 0.17, 0.23, 0.68, 0.20, 0.10, 0.17, 0.13, 0.17, 0.34],
        dtype=float,
    )
    reactive = 0.28 * active
    pv_capacity = np.zeros(len(buses), dtype=float)
    index = {bus: i for i, bus in enumerate(buses)}
    pv_capacity[index["646"]] = 0.55
    pv_capacity[index["680"]] = 0.75
    pv_capacity[index["675"]] = 0.65

    feeder = Feeder(
        buses=buses,
        root="650",
        lines=lines,
        base_active_load_mw=active,
        base_reactive_load_mvar=reactive,
        pv_capacity_mw=pv_capacity,
        storage_candidates=("634", "646", "671", "675"),
        data_center_bus="675",
        generator_bus="671",
    )
    feeder.validate()
    return feeder
