"""Stability statistics for the enumeration budget sweep, computed correctly.

The sweep script's own summary has a flaw: when a budget produced no incumbent
for some buses, it still formed a global incumbent from whatever parts did
finish, including the no-storage part. A partial enumeration is not comparable
with a complete one -- its minimum is taken over fewer options, so it looks
worse than it is, and its winning bus is whichever of the survivors happened to
solve. Those rows have to be identified and excluded rather than silently mixed
in.

This reads the raw per-bus results the sweep recorded and recomputes the
statistics, so no solve has to be repeated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", default="outputs/enumeration_budget_sweep.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--reference-budget",
        type=float,
        default=None,
        help="comparison reference; default is the largest budget",
    )
    parser.add_argument("--out", default="outputs/enumeration_budget_stability.json")
    return parser.parse_args()


# A part that proved infeasibility has answered its question: that branch is
# eliminated and its objective is +inf. Only a part that stopped without
# deciding anything leaves the enumeration incomplete. Gurobi reports
# INF_OR_UNBD as 'infeasible' too, which is safe here because every variable in
# this model is bounded, so unboundedness is not a possibility.
PROVEN_EMPTY = {"infeasible", "unbounded"}


def spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation with average ranks for ties.

    Ties matter here: several buses can reach the same objective, and assigning
    them arbitrary distinct ranks would invent an ordering the solver never
    found.
    """

    if len(a) < 3:
        return None

    def ranks(values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        order = np.argsort(array)
        out = np.empty(len(array), dtype=float)
        out[order] = np.arange(len(array), dtype=float)
        for value in np.unique(array):
            tied = array == value
            if tied.sum() > 1:
                out[tied] = out[tied].mean()
        return out

    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    args = parse_args()
    sweep = json.loads(Path(args.sweep).read_text(encoding="utf-8"))
    candidates = sweep["settings"]["candidates"]
    # Sorted by value, not left in the order they were written: the reference is
    # the largest budget, and taking the last entry would silently pick the
    # wrong one if the sweep were ever run with the budgets in another order.
    budget_values = sorted(float(b) for b in sweep["settings"]["budgets"])
    budgets = [f"{b:g}" for b in budget_values]
    subsets = sweep["subsets"]
    if not subsets:
        raise SystemExit(f"{args.sweep} has no completed subsets yet.")

    def evaluate(subset: dict, budget: str) -> dict:
        """Recompute one cell, marking whether the enumeration was complete."""

        entry = subset["budgets"].get(budget)
        if entry is None:
            return {"complete": False, "reason": "budget missing"}
        parts = entry["parts"]
        none_objective = subset["none"]["objective"]
        usable = {
            b: parts[b]["objective"]
            for b in candidates
            if parts.get(b, {}).get("objective") is not None
        }
        # Undecided, not merely empty: a proven-infeasible bus is resolved at
        # +inf and leaves the enumeration complete.
        undecided = [
            b
            for b in candidates
            if parts.get(b, {}).get("objective") is None
            and parts.get(b, {}).get("status") not in PROVEN_EMPTY
        ]
        eliminated = [
            b for b in candidates if parts.get(b, {}).get("status") in PROVEN_EMPTY
        ]
        if undecided or none_objective is None:
            return {
                "complete": False,
                "reason": (
                    f"undecided {undecided}" if undecided else "no none part"
                ),
                "buses_solved": len(usable),
                "buses_eliminated": len(eliminated),
            }
        if not usable:
            # Every bus proved infeasible, so not building is the only option
            # left and the enumeration is complete.
            return {
                "complete": True,
                "winner": "none",
                "builds": False,
                "global_incumbent": none_objective,
                "winner_energy_mwh": 0.0,
                "winner_power_mw": 0.0,
                "buses_eliminated": len(eliminated),
                "pipeline_seconds": entry["pipeline_seconds"],
            }
        best_bus = min(usable, key=lambda b: usable[b])
        builds = usable[best_bus] < none_objective
        return {
            "complete": True,
            "winner": best_bus if builds else "none",
            "builds": builds,
            "global_incumbent": min(min(usable.values()), none_objective),
            "winner_energy_mwh": parts[best_bus]["energy_mwh"] if builds else 0.0,
            "winner_power_mw": parts[best_bus]["power_mw"] if builds else 0.0,
            "buses_eliminated": len(eliminated),
            "pipeline_seconds": entry["pipeline_seconds"],
        }

    cells = {
        budget: [evaluate(subset, budget) for subset in subsets] for budget in budgets
    }

    print(f"{len(subsets)} subsets, budgets {', '.join(budgets)}s")
    print()
    print("completeness (a partial enumeration cannot be compared with a full one)")
    print(f"{'budget':>8}{'complete':>11}{'undecided':>12}{'infeasible':>12}   reasons")
    print("-" * 74)
    for budget in budgets:
        complete = [c for c in cells[budget] if c["complete"]]
        partial = [c for c in cells[budget] if not c["complete"]]
        eliminated = sum(c.get("buses_eliminated", 0) for c in cells[budget])
        reasons = sorted({c.get("reason", "?") for c in partial})
        print(
            f"{budget:>8}{len(complete):>11}{len(partial):>12}{eliminated:>12}   "
            + (", ".join(reasons)[:50] if reasons else "-")
        )

    reference = (
        budgets[-1]
        if args.reference_budget is None
        else f"{args.reference_budget:g}"
    )
    if reference not in budgets:
        raise SystemExit(
            f"reference budget {reference}s is not present; choices are {budgets}."
        )
    print()
    print(f"stability against {reference}s, on subsets complete at BOTH budgets")
    print(
        f"{'budget':>8}{'n':>5}{'spearman':>11}{'top-k':>9}{'same bus':>11}"
        f"{'|dE| med':>11}{'|dObj| med':>13}{'pipeline':>11}"
    )
    print("-" * 80)
    stability = {}
    for budget in budgets:
        paired = [
            (here, there)
            for here, there in zip(cells[budget], cells[reference])
            if here["complete"] and there["complete"]
        ]
        if len(paired) < 2:
            print(f"{budget:>8}{len(paired):>5}   too few complete pairs")
            stability[budget] = {"pairs": len(paired)}
            continue
        here_values = [p[0]["global_incumbent"] for p in paired]
        there_values = [p[1]["global_incumbent"] for p in paired]
        rho = spearman(here_values, there_values)
        order_here = np.argsort(here_values)[: args.top_k]
        order_there = np.argsort(there_values)[: args.top_k]
        overlap = len(set(order_here.tolist()) & set(order_there.tolist())) / max(
            len(order_there), 1
        )
        same_bus = float(np.mean([p[0]["winner"] == p[1]["winner"] for p in paired]))
        energy = [abs(p[0]["winner_energy_mwh"] - p[1]["winner_energy_mwh"]) for p in paired]
        objective = [abs(p[0]["global_incumbent"] - p[1]["global_incumbent"]) for p in paired]
        pipeline = float(np.mean([p[0]["pipeline_seconds"] for p in paired]))
        stability[budget] = {
            "pairs": len(paired),
            "spearman": rho,
            "top_k_overlap": overlap,
            "same_winning_bus": same_bus,
            "median_energy_shift_mwh": float(np.median(energy)),
            "median_objective_shift": float(np.median(objective)),
            "mean_pipeline_seconds": pipeline,
        }
        print(
            f"{budget:>8}{len(paired):>5}"
            + (f"{rho:>11.3f}" if rho is not None else f"{'-':>11}")
            + f"{overlap:>8.0%}{same_bus:>11.0%}"
            + f"{np.median(energy):>11.3f}{np.median(objective):>13,.0f}"
            + f"{pipeline:>10.0f}s"
        )

    print()
    print("per-subset winner by budget")
    header = f"{'subset':<8}" + "".join(f"{b + 's':>12}" for b in budgets)
    print(header)
    print("-" * len(header))
    for index, subset in enumerate(subsets):
        row = f"{index:<8}"
        for budget in budgets:
            cell = cells[budget][index]
            row += f"{(cell['winner'] if cell['complete'] else '-'):>12}"
        print(row)

    Path(args.out).write_text(
        json.dumps(
            {"reference_budget": reference, "stability": stability, "cells": cells},
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(
        "Read the cheapest budget whose Spearman is near 1, whose winning bus\n"
        "agrees, and whose energy barely moves. A wide optimality gap does not\n"
        "disqualify it; an unstable ranking does."
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
