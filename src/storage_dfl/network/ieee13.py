from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PHASES = ("A", "B", "C")


@dataclass(frozen=True)
class Line:
    parent: str
    child: str
    phases: tuple[str, ...]
    resistance: float
    reactance: float
    rating_mva: float

    @property
    def phase_rating_mva(self) -> float:
        return self.rating_mva / len(self.phases)


@dataclass(frozen=True)
class Feeder:
    buses: tuple[str, ...]
    phases: tuple[str, ...]
    root: str
    lines: tuple[Line, ...]
    phase_mask: np.ndarray
    base_active_load_mw: np.ndarray
    base_reactive_load_mvar: np.ndarray
    pv_capacity_mw: np.ndarray
    storage_candidates: tuple[str, ...]
    data_center_bus: str
    generator_bus: str

    @property
    def bus_index(self) -> dict[str, int]:
        return {bus: index for index, bus in enumerate(self.buses)}

    @property
    def phase_index(self) -> dict[str, int]:
        return {phase: index for index, phase in enumerate(self.phases)}

    @property
    def bus_phases(self) -> dict[str, tuple[str, ...]]:
        return {
            bus: tuple(
                phase
                for phase, present in zip(self.phases, self.phase_mask[index], strict=True)
                if present
            )
            for index, bus in enumerate(self.buses)
        }

    @property
    def base_active_load_bus_mw(self) -> np.ndarray:
        return self.base_active_load_mw.sum(axis=1)

    @property
    def base_reactive_load_bus_mvar(self) -> np.ndarray:
        return self.base_reactive_load_mvar.sum(axis=1)

    @property
    def pv_capacity_bus_mw(self) -> np.ndarray:
        return self.pv_capacity_mw.sum(axis=1)

    def validate(self) -> None:
        if not self.buses or self.phases != PHASES:
            raise ValueError("A feeder must contain at least one bus and phases A/B/C.")
        if len(self.lines) != len(self.buses) - 1:
            raise ValueError("LinDistFlow requires a radial feeder with N-1 lines.")
        if self.root not in self.buses:
            raise ValueError("The root bus is not in the feeder.")
        children = {line.child for line in self.lines}
        if children != set(self.buses) - {self.root}:
            raise ValueError("Every non-root bus must have exactly one parent.")
        expected = (len(self.buses), len(self.phases))
        for array in (
            self.phase_mask,
            self.base_active_load_mw,
            self.base_reactive_load_mvar,
            self.pv_capacity_mw,
        ):
            if array.shape != expected:
                raise ValueError(f"Phase-resolved parameter arrays must have shape {expected}.")
        if np.any(self.base_active_load_mw[~self.phase_mask] != 0.0):
            raise ValueError("A load was assigned to a phase absent from its bus.")
        if np.any(self.pv_capacity_mw[~self.phase_mask] != 0.0):
            raise ValueError("PV was assigned to a phase absent from its bus.")
        bus_phases = self.bus_phases
        for line in self.lines:
            allowed = set(bus_phases[line.parent]) & set(bus_phases[line.child])
            if not set(line.phases).issubset(allowed):
                raise ValueError(f"{line.parent}-{line.child}: invalid line phases")


def ieee13_unbalanced_microgrid() -> Feeder:
    """Return the IEEE13 phase topology used by the historical dataset.

    The planning oracle creates one voltage and P/Q flow per available phase.
    Mutual phase impedance is not represented yet, so this is a decoupled
    multi-phase LinDistFlow approximation rather than a full OpenDSS solve.
    """

    buses = (
        "650", "632", "633", "634", "645", "646", "671",
        "680", "684", "611", "652", "692", "675",
    )
    bus_phase_names = {
        "650": "ABC", "632": "ABC", "633": "ABC", "634": "ABC",
        "645": "BC", "646": "BC", "671": "ABC", "680": "ABC",
        "684": "AC", "611": "C", "652": "A", "692": "ABC", "675": "ABC",
    }
    phase_mask = np.asarray(
        [[phase in bus_phase_names[bus] for phase in PHASES] for bus in buses],
        dtype=bool,
    )
    lines = (
        Line("650", "632", tuple("ABC"), 0.0030, 0.0060, 5.0),
        Line("632", "633", tuple("ABC"), 0.0024, 0.0048, 3.0),
        Line("633", "634", tuple("ABC"), 0.0020, 0.0040, 2.0),
        Line("632", "645", tuple("BC"), 0.0028, 0.0055, 2.0),
        Line("645", "646", tuple("BC"), 0.0020, 0.0040, 1.5),
        Line("632", "671", tuple("ABC"), 0.0032, 0.0062, 4.0),
        Line("671", "680", tuple("ABC"), 0.0016, 0.0032, 2.0),
        Line("671", "684", tuple("AC"), 0.0022, 0.0043, 2.0),
        Line("684", "611", tuple("C"), 0.0018, 0.0035, 1.0),
        Line("684", "652", tuple("A"), 0.0025, 0.0048, 1.0),
        # The 671-692-675 branch feeds the data-center bus.  The original 2.5 MVA
        # rating is already marginal for the base facility and cannot host a
        # larger one, so this branch carries the interconnection upgrade a real
        # study would require.  Ratings elsewhere are unchanged.
        Line("671", "692", tuple("ABC"), 0.0010, 0.0020, 4.0),
        Line("692", "675", tuple("ABC"), 0.0022, 0.0042, 4.0),
    )

    active = np.zeros((len(buses), 3), dtype=float)
    reactive = np.zeros_like(active)
    index = {bus: i for i, bus in enumerate(buses)}
    pidx = {phase: i for i, phase in enumerate(PHASES)}
    phase_loads = {
        ("634", "A"): (0.160, 0.110), ("634", "B"): (0.120, 0.090),
        ("634", "C"): (0.120, 0.090), ("645", "B"): (0.170, 0.125),
        ("646", "B"): (0.115, 0.066), ("646", "C"): (0.115, 0.066),
        ("671", "A"): (0.402, 0.230), ("671", "B"): (0.451, 0.258),
        ("671", "C"): (0.502, 0.288), ("611", "C"): (0.170, 0.080),
        ("652", "A"): (0.128, 0.086), ("692", "A"): (0.085, 0.0755),
        ("692", "C"): (0.085, 0.0755), ("675", "A"): (0.485, 0.190),
        ("675", "B"): (0.068, 0.060), ("675", "C"): (0.290, 0.212),
    }
    for (bus, phase), (p_mw, q_mvar) in phase_loads.items():
        active[index[bus], pidx[phase]] = p_mw
        reactive[index[bus], pidx[phase]] = q_mvar

    pv_capacity = np.zeros_like(active)
    for bus, capacity in (("634", 0.25), ("675", 0.65), ("680", 0.75)):
        pv_capacity[index[bus], :] = capacity / 3.0

    feeder = Feeder(
        buses=buses,
        phases=PHASES,
        root="650",
        lines=lines,
        phase_mask=phase_mask,
        base_active_load_mw=active,
        base_reactive_load_mvar=reactive,
        pv_capacity_mw=pv_capacity,
        # Five three-phase buses, each admitted for a distinct reason, so that no
        # two candidates test the same siting argument:
        #   632  feeder head, the only bus carrying the whole feeder's flow;
        #   671  largest load (1.355 MW over ABC) and the backup generator bus;
        #   675  the data-center bus, 0.843 MW of load plus 0.65 MW of PV;
        #   680  largest PV (0.75 MW) with no local load, so the bus where
        #        curtailment is most likely and charging is cheapest;
        #   692  on the 671-692-675 branch that this study had to uprate to
        #        4.0 MVA to host the data center. Storage here is the deferral
        #        alternative to that upgrade, which is the classic non-wires
        #        argument for siting; without it the study never tests it.
        # Excluded, and why:
        #   650  substation root. Storage there is grid-side and produces no
        #        network effect in this formulation.
        #   633, 684  pass-through buses with neither load nor PV.
        #   634  three-phase with 0.40 MW of load and 0.25 MW of PV, but behind
        #        the 3.0/2.0 MVA 632-633-634 lateral, which never binds here.
        #        This is the nearest omission and the one to revisit first if a
        #        solution ever sites storage upstream at 632.
        #   645, 646, 611, 652  one- or two-phase. Storage power is split over
        #        len(bus_phases[bus]), so a candidate here is a single- or
        #        two-phase battery, which this study does not model.
        storage_candidates=("632", "671", "675", "680", "692"),
        data_center_bus="675",
        generator_bus="671",
    )
    feeder.validate()
    return feeder


def ieee13_balanced_microgrid() -> Feeder:
    """Backward-compatible function name; now returns the phase-resolved feeder."""

    return ieee13_unbalanced_microgrid()


def single_pcc_microgrid() -> Feeder:
    """Return the single-bus, behind-the-meter data-centre microgrid.

    The three phase slots are retained in the scenario schema so the existing
    CVAE/DFL pipeline can read both topologies. The PCC planning oracle sums the
    slots and does not model reactive power or phase imbalance.
    """

    phase_mask = np.ones((1, 3), dtype=bool)
    feeder = Feeder(
        buses=("PCC",),
        phases=PHASES,
        root="PCC",
        lines=(),
        phase_mask=phase_mask,
        # These arrays define codec masks and physically reasonable clipping
        # bounds. Actual data-centre demand is built from workload and PUE when
        # historical scenarios are aggregated in stages._experiment_data.
        base_active_load_mw=np.full((1, 3), 1.0 / 3.0, dtype=float),
        base_reactive_load_mvar=np.zeros((1, 3), dtype=float),
        # Aggregate nameplate PV on the original IEEE-13 case: 0.25+0.65+0.75.
        pv_capacity_mw=np.full((1, 3), 1.65 / 3.0, dtype=float),
        storage_candidates=("PCC",),
        data_center_bus="PCC",
        generator_bus="PCC",
    )
    feeder.validate()
    return feeder
