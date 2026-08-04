"""A PySCIPOpt-shaped facade over gurobipy.

Under the production formulation -- McCormick carbon envelopes and polygon flow
limits -- the planning problem is a pure MILP: every one of its 168k constraints
is linear and 1,972 variables are binary.  That is the regime where solver choice
matters most, so the model has to be solvable by more than one solver.

Rewriting the 1,500-line builder against a second API would risk introducing
modelling differences that are invisible until the two solvers disagree.  Instead
this module presents exactly the handful of PySCIPOpt methods the builder uses,
so the builder itself is untouched and both backends are guaranteed to be
building the same model.

The other formulations remain solvable too: ``quadratic`` flow limits are convex
second-order cones, and the ``exact`` carbon equalities are nonconvex bilinear
terms, which Gurobi accepts once ``NonConvex`` is set.
"""

from __future__ import annotations

from typing import Any

# SCIP reports a status string; the pipeline's ``PlanningResult.feasible`` keys
# off exactly 'infeasible' and 'unbounded', so those two names must survive the
# translation intact.
_STATUS_NAMES = {
    1: "unknown",  # LOADED
    2: "optimal",  # also returned once MIPGap is reached, which SCIP calls gaplimit
    3: "infeasible",
    4: "infeasible",  # INF_OR_UNBD, reported conservatively
    5: "unbounded",
    6: "cutoff",
    7: "iterlimit",
    8: "nodelimit",
    9: "timelimit",
    10: "sollimit",
    11: "userinterrupt",
    12: "numericalfailure",
    13: "suboptimal",
    14: "inprogress",
    15: "userobjlimit",
    16: "worklimit",
    17: "memlimit",
}

# SCIP parameter names used by the builder, mapped to their Gurobi equivalents.
# 'limits/memory' is handled separately because SCIP counts megabytes and Gurobi
# counts gigabytes; passing the number through unchanged would multiply the
# budget by 1024 and defeat the limit entirely.
_PARAMETER_NAMES = {
    "limits/time": "TimeLimit",
    "limits/gap": "MIPGap",
    "parallel/maxnthreads": "Threads",
}

_MIP_FOCUS = {"default": 0, "feasibility": 1, "optimality": 2, "hardlp": 3, "counter": 1}


class _Solution(dict):
    """Stand-in for a SCIP solution handle.

    Gurobi has no solution object to pass around: values are read from the
    variables after ``optimize``.  Warm starts, however, are assembled before
    solving, so they are collected here and written to ``Var.Start`` on
    ``addSol``.
    """


class GurobiModel:
    """The subset of the PySCIPOpt ``Model`` API that the planner uses."""

    def __init__(self, name: str = "") -> None:
        import gurobipy as gp

        self._gp = gp
        self._model = gp.Model(name)
        self._model.setParam("OutputFlag", 1)
        self._nonconvex_needed = False

    # -- construction ---------------------------------------------------------

    def addVar(
        self,
        lb: float = 0.0,
        ub: float | None = None,
        vtype: str = "C",
        name: str = "",
        obj: float = 0.0,
    ) -> Any:
        gp = self._gp
        kind = {"B": gp.GRB.BINARY, "I": gp.GRB.INTEGER, "C": gp.GRB.CONTINUOUS}[
            vtype.upper()[:1]
        ]
        # PySCIPOpt defaults an omitted bound to infinity; gurobipy defaults ub to
        # infinity already but treats None as an error.
        upper = gp.GRB.INFINITY if ub is None else ub
        lower = -gp.GRB.INFINITY if lb is None else lb
        return self._model.addVar(lb=lower, ub=upper, vtype=kind, name=name, obj=obj)

    def addCons(self, constraint: Any, name: str = "") -> Any:
        if name:
            return self._model.addConstr(constraint, name=name)
        return self._model.addConstr(constraint)

    def setObjective(self, expression: Any, sense: str = "minimize") -> None:
        gp = self._gp
        direction = gp.GRB.MAXIMIZE if str(sense).startswith("max") else gp.GRB.MINIMIZE
        self._model.setObjective(expression, direction)

    def chgVarLb(self, variable: Any, value: float) -> None:
        variable.LB = float(value)

    def chgVarUb(self, variable: Any, value: float) -> None:
        variable.UB = float(value)

    def getVars(self) -> tuple:
        # The builder reads ``variable.name``; gurobipy spells it ``VarName`` but
        # also accepts lowercase attribute access, so update() is enough to make
        # names available before optimize().
        self._model.update()
        return tuple(self._model.getVars())

    def getNVars(self) -> int:
        self._model.update()
        return self._model.NumVars

    def getNConss(self) -> int:
        self._model.update()
        return self._model.NumConstrs + self._model.NumQConstrs

    # -- solver configuration -------------------------------------------------

    def hideOutput(self) -> None:
        self._model.setParam("OutputFlag", 0)

    def setParam(self, name: str, value: Any) -> None:
        if name == "presolving/maxrounds":
            # SCIP disables presolve with 0 rounds; Gurobi's Presolve uses -1 for
            # automatic and 0 for off, so only the "off" intent transfers.
            self._model.setParam("Presolve", 0 if int(value) == 0 else -1)
            return
        if name == "limits/memory":
            self._model.setParam("MemLimit", float(value) / 1024.0)
            return
        translated = _PARAMETER_NAMES.get(name)
        if translated is None:
            raise KeyError(name)
        self._model.setParam(translated, value)

    def setNonconvex(self, enabled: bool) -> None:
        """Allow nonconvex quadratic equalities, needed by the 'exact' carbon model."""

        self._nonconvex_needed = bool(enabled)
        self._model.setParam("NonConvex", 2 if enabled else -1)

    def apply_search_strategy(self, emphasis: str, aggressive_heuristics: bool) -> None:
        """Gurobi's counterpart to SCIP's emphasis and heuristic settings.

        MIPFocus 1 prioritises finding good incumbents over proving optimality,
        which is what decision-focused training needs: samples are ranked by
        incumbent objective. Neither setting changes the model.
        """

        focus = _MIP_FOCUS.get(str(emphasis).lower(), 0)
        if focus:
            self._model.setParam("MIPFocus", focus)
        if aggressive_heuristics:
            # The default heuristic effort is 0.05 of runtime.
            self._model.setParam("Heuristics", 0.5)

    # -- warm starts ----------------------------------------------------------

    def createSol(self) -> _Solution:
        return _Solution()

    def createPartialSol(self) -> _Solution:
        return _Solution()

    def setSolVal(self, solution: _Solution, variable: Any, value: float) -> None:
        solution[variable] = float(value)

    def addSol(self, solution: _Solution) -> bool:
        for variable, value in solution.items():
            variable.Start = value
        return True

    def trySol(self, solution: _Solution) -> bool:
        return self.addSol(solution)

    # -- solving and results --------------------------------------------------

    def optimize(self) -> None:
        if self._nonconvex_needed:
            self.setNonconvex(True)
        self._model.optimize()

    def getStatus(self) -> str:
        return _STATUS_NAMES.get(int(self._model.Status), f"status{self._model.Status}")

    def getBestSol(self) -> Any:
        # Only its presence is tested by the caller.
        return _Solution() if self._model.SolCount > 0 else None

    def getSolVal(self, solution: _Solution, variable: Any) -> float:
        # Gurobi keeps the incumbent on the variables themselves, so the solution
        # handle carries no values and the read goes straight to the variable.
        return float(variable.X)

    def freeProb(self) -> None:
        self._model.dispose()

    def getVal(self, expression: Any) -> float:
        if isinstance(expression, (int, float)):
            return float(expression)
        if hasattr(expression, "X"):
            return float(expression.X)
        if hasattr(expression, "getValue"):
            return float(expression.getValue())
        raise TypeError(f"Cannot read a value from {type(expression).__name__}.")

    def getObjVal(self) -> float:
        return float(self._model.ObjVal)

    def getGap(self) -> float:
        try:
            return float(self._model.MIPGap)
        except (AttributeError, self._gp.GurobiError):
            # A continuous relaxation has no MIP gap; it is solved exactly.
            return 0.0

    def getSolvingTime(self) -> float:
        return float(self._model.Runtime)

    def freeTransform(self) -> None:
        """No transformed-problem stage exists in Gurobi; resetting is enough."""

        self._model.reset()

    def writeProblem(self, path: str) -> None:
        self._model.write(path)

    @property
    def raw(self):
        """The underlying gurobipy model, for anything this facade omits."""

        return self._model


def quicksum(terms) -> Any:
    import gurobipy as gp

    return gp.quicksum(terms)


def is_available() -> bool:
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        return False
    return True
