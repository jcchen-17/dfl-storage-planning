"""A PySCIPOpt-shaped facade over HiGHS.

Same purpose and same shape as the Gurobi facade: the production formulation is
a pure MILP, and MILP is where solver choice matters, so the builder has to be
able to hand its model to more than one solver.  HiGHS earns its place by being
open source and licence-free, which makes it the backend that always works.

As with the Gurobi facade, the 1,500-line builder is untouched -- both backends
run the identical construction code, so the two can only differ in how they
solve, never in what they solve.

HiGHS has no nonlinear support, so the ``exact`` carbon formulation and the
``quadratic`` flow limits cannot use this backend; ``_new_model`` rejects those
combinations rather than letting HiGHS silently drop the constraints.
"""

from __future__ import annotations

from typing import Any

# The pipeline's ``PlanningResult.feasible`` keys off exactly 'infeasible' and
# 'unbounded', so those two names have to survive translation intact.
_STATUS_NAMES = {
    "kOptimal": "optimal",
    "kInfeasible": "infeasible",
    "kUnbounded": "unbounded",
    "kUnboundedOrInfeasible": "infeasible",
    "kTimeLimit": "timelimit",
    "kIterationLimit": "iterlimit",
    "kSolutionLimit": "sollimit",
    "kObjectiveBound": "objectivelimit",
    "kObjectiveTarget": "objectivelimit",
    "kInterrupt": "userinterrupt",
    "kHighsInterrupt": "userinterrupt",
    "kMemoryLimit": "memlimit",
    "kModelEmpty": "modelempty",
    "kNotset": "unknown",
    "kUnknown": "unknown",
    "kLoadError": "error",
    "kModelError": "error",
    "kPresolveError": "error",
    "kSolveError": "error",
    "kPostsolveError": "error",
}

_OPTION_NAMES = {
    "limits/time": "time_limit",
    "limits/gap": "mip_rel_gap",
    "limits/absgap": "mip_abs_gap",
    "parallel/maxnthreads": "threads",
}


class _Solution(dict):
    """Collects a warm start; HiGHS receives it as a dense column vector."""


class HighsModel:
    """The subset of the PySCIPOpt ``Model`` API that the planner uses."""

    def __init__(self, name: str = "") -> None:
        import highspy

        self._highs_module = highspy
        self._model = highspy.Highs()
        self._variables: list[Any] = []
        # highs_var exposes no lb/ub, and changeColBounds needs both at once, so
        # the facade keeps its own record rather than querying the model back.
        self._bounds: list[list[float]] = []
        self._name = name
        self._solved = False

    # -- construction ---------------------------------------------------------

    def addVar(
        self,
        lb: float = 0.0,
        ub: float | None = None,
        vtype: str = "C",
        name: str = "",
        obj: float = 0.0,
    ) -> Any:
        highspy = self._highs_module
        kind = vtype.upper()[:1]
        upper = float("inf") if ub is None else float(ub)
        lower = float("-inf") if lb is None else float(lb)
        if kind == "B":
            # addBinary fixes the bounds at [0, 1]; the builder never asks for a
            # binary with other bounds.
            variable = self._model.addBinary(obj=obj, name=name or None)
            lower, upper = 0.0, 1.0
        else:
            variable_type = (
                highspy.HighsVarType.kInteger
                if kind == "I"
                else highspy.HighsVarType.kContinuous
            )
            variable = self._model.addVariable(
                lb=lower, ub=upper, obj=obj, type=variable_type, name=name or None
            )
        self._variables.append(variable)
        self._bounds.append([lower, upper])
        return variable

    def addCons(self, constraint: Any, name: str = "") -> Any:
        return self._model.addConstr(constraint, name=name or None)

    def setObjective(self, expression: Any, sense: str = "minimize") -> None:
        highspy = self._highs_module
        direction = (
            highspy.ObjSense.kMaximize
            if str(sense).startswith("max")
            else highspy.ObjSense.kMinimize
        )
        # setObjective only records the objective; minimize()/maximize() would
        # also solve immediately, which would bypass the parameters set later.
        self._model.setObjective(expression, sense=direction)

    def chgVarLb(self, variable: Any, value: float) -> None:
        bounds = self._bounds[variable.index]
        bounds[0] = float(value)
        self._model.changeColBounds(variable.index, bounds[0], bounds[1])

    def chgVarUb(self, variable: Any, value: float) -> None:
        bounds = self._bounds[variable.index]
        bounds[1] = float(value)
        self._model.changeColBounds(variable.index, bounds[0], bounds[1])

    def getVars(self) -> tuple:
        return tuple(self._variables)

    def getNVars(self) -> int:
        return len(self._variables)

    def getNConss(self) -> int:
        return int(self._model.getNumRow())

    # -- solver configuration -------------------------------------------------

    def hideOutput(self) -> None:
        self._model.setOptionValue("output_flag", False)

    def setParam(self, name: str, value: Any) -> None:
        if name == "presolving/maxrounds":
            self._model.setOptionValue("presolve", "off" if int(value) == 0 else "choose")
            return
        if name == "limits/memory":
            # HiGHS exposes no memory budget. Saying so is better than silently
            # ignoring a limit the caller believes is in force.
            raise KeyError(
                "HiGHS has no memory limit option; leave solver_memory_limit_mb at 0 "
                "for this backend, or lower solver_max_parallel_workers instead."
            )
        translated = _OPTION_NAMES.get(name)
        if translated is None:
            raise KeyError(name)
        self._model.setOptionValue(translated, value)

    def apply_search_strategy(self, emphasis: str, aggressive_heuristics: bool) -> None:
        """HiGHS exposes no emphasis setting; heuristic effort is the closest lever."""

        if aggressive_heuristics:
            try:
                self._model.setOptionValue("mip_heuristic_effort", 0.2)
            except Exception:  # noqa: BLE001 - older builds lack the option
                pass

    # -- warm starts ----------------------------------------------------------

    def createSol(self) -> _Solution:
        return _Solution()

    def createPartialSol(self) -> _Solution:
        return _Solution()

    def setSolVal(self, solution: _Solution, variable: Any, value: float) -> None:
        solution[variable.index] = float(value)

    def addSol(self, solution: _Solution) -> bool:
        if not solution:
            return False
        # HiGHS needs a value for every column, so unset entries default to zero,
        # which is the same convention the builder already uses when it seeds a
        # storage-disabled operating point.
        values = [0.0] * len(self._variables)
        for index, value in solution.items():
            if 0 <= index < len(values):
                values[index] = value
        try:
            self._model.setSolution(values)
            return True
        except Exception:  # noqa: BLE001 - a rejected start must not fail the solve
            return False

    def trySol(self, solution: _Solution) -> bool:
        return self.addSol(solution)

    # -- solving and results --------------------------------------------------

    def optimize(self) -> None:
        self._model.run()
        self._solved = True

    def getStatus(self) -> str:
        raw = str(self._model.getModelStatus()).split(".")[-1]
        return _STATUS_NAMES.get(raw, raw.lower())

    def getBestSol(self) -> Any:
        info = self._model.getInfo()
        has_solution = bool(getattr(info, "primal_solution_status", 0))
        return _Solution() if has_solution else None

    def getSolVal(self, solution: _Solution, variable: Any) -> float:
        return float(self._model.val(variable))

    def getVal(self, expression: Any) -> float:
        if isinstance(expression, (int, float)):
            return float(expression)
        return float(self._model.val(expression))

    def getObjVal(self) -> float:
        return float(self._model.getObjectiveValue())

    def getGap(self) -> float:
        gap = getattr(self._model.getInfo(), "mip_gap", None)
        return float(gap) if gap is not None else 0.0

    def getDualbound(self) -> float:
        bound = getattr(self._model.getInfo(), "mip_dual_bound", None)
        return float(bound) if bound is not None else self.getObjVal()

    def getSolvingTime(self) -> float:
        return float(self._model.getRunTime())

    def freeProb(self) -> None:
        self._model.clear()
        self._variables.clear()
        self._bounds.clear()

    def freeTransform(self) -> None:
        self._solved = False

    def writeProblem(self, path: str) -> None:
        self._model.writeModel(path)

    @property
    def raw(self):
        return self._model


def quicksum(terms) -> Any:
    """HiGHS expressions accumulate with ``sum``; there is no dedicated helper."""

    total = None
    for term in terms:
        total = term if total is None else total + term
    return 0 if total is None else total


def is_available() -> bool:
    try:
        import highspy  # noqa: F401
    except ImportError:
        return False
    return True
