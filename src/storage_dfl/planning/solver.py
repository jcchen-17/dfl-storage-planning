"""Small solver-adapter surface used by the single-PCC MILP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pyscipopt import Model, quicksum

from storage_dfl.config import PlanningConfig
from storage_dfl.data import Scenario

from .results import StorageDesign

SOLVER_BACKENDS = ("scip", "gurobi", "highs")
_EMPHASIS_SETTINGS = ("default", "feasibility", "optimality", "hardlp", "counter")


def new_model(name: str, backend: str):
    if backend not in SOLVER_BACKENDS:
        raise ValueError(
            f"solver_backend must be one of {SOLVER_BACKENDS}, got {backend!r}."
        )
    if backend == "scip":
        return Model(name), quicksum
    if backend == "gurobi":
        from storage_dfl.planning import gurobi_backend

        if not gurobi_backend.is_available():
            raise RuntimeError("solver_backend='gurobi' requires gurobipy and a license.")
        return gurobi_backend.GurobiModel(name), gurobi_backend.quicksum
    from storage_dfl.planning import highs_backend

    if not highs_backend.is_available():
        raise RuntimeError("solver_backend='highs' requires highspy.")
    return highs_backend.HighsModel(name), highs_backend.quicksum


def apply_search_strategy(model: object, planning: PlanningConfig) -> None:
    emphasis = str(planning.solver_emphasis).lower()
    if emphasis not in _EMPHASIS_SETTINGS:
        raise ValueError(
            f"solver_emphasis must be one of {_EMPHASIS_SETTINGS}, got {emphasis!r}."
        )
    delegate = getattr(model, "apply_search_strategy", None)
    if delegate is not None:
        delegate(emphasis, bool(planning.solver_aggressive_heuristics))
        return
    try:
        from pyscipopt import SCIP_PARAMEMPHASIS, SCIP_PARAMSETTING

        if emphasis != "default":
            model.setEmphasis(getattr(SCIP_PARAMEMPHASIS, emphasis.upper()))
        if planning.solver_aggressive_heuristics:
            model.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)
    except (ImportError, AttributeError):
        pass


@dataclass(frozen=True)
class PlanningJob:
    scenarios: Iterable[Scenario]
    weights: Iterable[float] | None = None
    fixed_design: StorageDesign | None = None
