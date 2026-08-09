from .solver import PlanningJob
from .pcc import SinglePCCPlanningOracle
from .recourse import evaluate_fixed_design_recourse
from .results import (
    PlanningResult,
    RecourseDiagnostics,
    StorageDesign,
    weighted_carbon_ledger,
)


class StoragePlanningOracle:
    """Single retained planning-oracle constructor."""

    def __new__(cls, feeder, planning, costs, data, data_center=None):
        if planning.topology != "single_pcc":
            raise ValueError("Only planning.topology='single_pcc' is retained.")
        return SinglePCCPlanningOracle(feeder, planning, costs, data, data_center)


def make_planning_oracle(feeder, planning, costs, data, data_center=None):
    """Build the active single-PCC oracle."""

    if planning.topology != "single_pcc":
        raise ValueError("Only planning.topology='single_pcc' is retained.")
    if planning.enumerate_storage_sites:
        raise ValueError("Site enumeration is not applicable to the single PCC.")
    return SinglePCCPlanningOracle(feeder, planning, costs, data, data_center)

__all__ = [
    "PlanningJob",
    "StoragePlanningOracle",
    "PlanningResult",
    "RecourseDiagnostics",
    "StorageDesign",
    "SinglePCCPlanningOracle",
    "make_planning_oracle",
    "weighted_carbon_ledger",
    "evaluate_fixed_design_recourse",
]
