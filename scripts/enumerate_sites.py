"""Joint siting versus enumerating the storage site, at equal budget.

With ``max_storage_sites: 1`` and five candidate buses, the planning problem is
exactly the minimum over six restricted problems -- build nothing, or build at
one named bus with power and energy still continuous. This is an equivalence,
not an approximation, so any difference in the answer is the solver's search,
not the formulation's.

What enumeration is expected to buy is not speed. Each part is still a nonconvex
MIQCP with the same bilinear carbon equalities. What it fixes is the allocation
of search effort: the joint model decides for itself how long to spend on each
bus, so a scenario can score well simply by being easy to search, and a DFL
reward built on that learns the solver rather than the problem. Enumerated, each
bus gets its own equal budget.

Three arms, so speed and stability are not confounded:

    A  joint siting, one solve, budget T
    B  six subproblems, budget T each          -- best achievable stability
    C  six subproblems, budget T/6 each        -- equal total core-hours as A

The global bound of an enumerated run is min_i best_bound_i, never the winning
subproblem's own gap, which describes only its own restricted region.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")

from storage_dfl.config import load_config
from storage_dfl.planning import StoragePlanningOracle
from storage_dfl.stages import _experiment_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset_v2_dfl_hourly_layered.yaml")
    parser.add_argument("--control", default="outputs/random_subset_control_k3.json")
    parser.add_argument("--subsets", type=int, default=3)
    parser.add_argument("--budget", type=float, default=900.0, help="seconds for arm A and each part of B")
    parser.add_argument("--absolute-gap", type=float, default=100.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--memory-limit-mb", type=float, default=9000.0)
    parser.add_argument(
        "--arms", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"]
    )
    parser.add_argument("--out", default="outputs/enumerate_sites.json")
    return parser.parse_args()


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    feeder, support_pool = _experiment_data(config, config.data.validation_split)
    by_name = {s.name: s for s in support_pool.scenarios}
    candidates = list(feeder.storage_candidates)

    control = json.loads(Path(args.control).read_text(encoding="utf-8"))
    draws = [d for d in control["draws"] if d.get("decision_loss") is not None]
    # A spread of outcomes rather than the top few, so the comparison is not run
    # only on subsets the joint solver already handled well.
    ordered = sorted(
        draws,
        key=lambda d: control["baselines"]["no_storage"]["objective"] - d["decision_loss"],
    )
    picks = [
        ordered[0],
        ordered[len(ordered) // 2],
        ordered[-1],
    ][: args.subsets]

    def make_oracle(seconds: float, sites: tuple[str, ...] | None, allow: int):
        """`sites` None keeps every candidate; a tuple restricts the feeder.

        A restricted part also requires a site, so it answers "given a battery
        here, how large" rather than repeating the build-or-not question that
        the `allow=0` part already covers on its own.
        """

        restricted = feeder if sites is None else replace(feeder, storage_candidates=sites)
        planning = replace(
            config.planning,
            solver_relative_gap=0.0,
            solver_absolute_gap_dollars=args.absolute_gap,
            solver_time_limit_seconds=seconds,
            solver_threads=args.threads,
            solver_memory_limit_mb=args.memory_limit_mb,
            solver_max_parallel_workers=1,
            max_storage_sites=allow,
            min_storage_sites=1 if (sites is not None and allow > 0) else 0,
            solver_emphasis="optimality",
            solver_aggressive_heuristics=False,
        )
        return StoragePlanningOracle(
            restricted, planning, config.costs, config.data, config.data_center
        )

    def solve_one(oracle, supports):
        started = time.perf_counter()
        planned = oracle.solve(
            supports,
            weights=tuple(1.0 / len(supports) for _ in supports),
            allow_carbon_slack=config.dfl.training_allow_carbon_slack,
        )
        return planned, time.perf_counter() - started

    def describe(planned, seconds, label):
        installed = list(planned.design.installed_buses) if planned.feasible else []
        energy = sum(
            float(planned.design.energy_mwh[b]) for b in installed
        )
        power = sum(float(planned.design.power_mw[b]) for b in installed)
        print(
            f"    {label:<12}{planned.status:<11}"
            f"obj {planned.objective:>13,.0f}  bound {planned.best_bound:>13,.0f}  "
            f"{str(installed):<10}{power:>7.3f} MW{energy:>8.3f} MWh{seconds:>7.0f}s",
            flush=True,
        )
        return {
            "label": label,
            "status": planned.status,
            "objective": _finite(planned.objective),
            "best_bound": _finite(planned.best_bound),
            "relative_gap": _finite(planned.relative_gap),
            "installed": installed,
            "power_mw": power,
            "energy_mwh": energy,
            "seconds": seconds,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "settings": {
            "budget_seconds": args.budget,
            "absolute_gap_dollars": args.absolute_gap,
            "relative_gap": 0.0,
            "threads": args.threads,
            "candidates": candidates,
        },
        "subsets": [],
    }

    for entry in picks:
        names = entry["names"]
        supports = tuple(by_name[n] for n in names)
        print(f"\nsubset {names}", flush=True)
        record: dict = {"names": names, "arms": {}}

        if "A" in args.arms:
            print("  A: joint siting", flush=True)
            oracle = make_oracle(args.budget, None, 1)
            planned, seconds = solve_one(oracle, supports)
            record["arms"]["A"] = {
                "wall_seconds": seconds,
                "core_hours": seconds * args.threads / 3600.0,
                "parts": [describe(planned, seconds, "joint")],
            }

        for arm, per_part in (("B", args.budget), ("C", args.budget / (len(candidates) + 1))):
            if arm not in args.arms:
                continue
            print(f"  {arm}: enumerated, {per_part:.0f}s per part", flush=True)
            parts = []
            started = time.perf_counter()
            oracle = make_oracle(per_part, (candidates[0],), 0)
            planned, seconds = solve_one(oracle, supports)
            parts.append(describe(planned, seconds, "none"))
            for bus in candidates:
                oracle = make_oracle(per_part, (bus,), 1)
                planned, seconds = solve_one(oracle, supports)
                parts.append(describe(planned, seconds, f"bus {bus}"))
            wall = time.perf_counter() - started
            usable = [p for p in parts if p["objective"] is not None]
            bounds = [p["best_bound"] for p in parts if p["best_bound"] is not None]
            best = min(usable, key=lambda p: p["objective"]) if usable else None
            # min over parts on both sides: the incumbent because the union of the
            # restricted regions is the original region, the bound because a bound
            # for the whole is only as strong as its weakest part.
            global_bound = min(bounds) if bounds else None
            record["arms"][arm] = {
                "wall_seconds": wall,
                "core_hours": wall * args.threads / 3600.0,
                "parts": parts,
                "winner": best["label"] if best else None,
                "global_incumbent": best["objective"] if best else None,
                "global_bound": global_bound,
                "global_relative_gap": (
                    None
                    if best is None or global_bound is None or best["objective"] == 0
                    else (best["objective"] - global_bound) / abs(best["objective"])
                ),
            }
            if best is not None:
                print(
                    f"    -> winner {best['label']}, incumbent {best['objective']:,.0f}, "
                    f"global bound {global_bound:,.0f}, "
                    f"global gap {record['arms'][arm]['global_relative_gap']:.2%}",
                    flush=True,
                )
        report["subsets"].append(record)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"{'subset':<8}{'arm':<5}{'incumbent':>14}{'global bound':>15}{'gap':>9}{'design':>18}{'wall':>8}{'core-h':>9}")
    print("-" * 88)
    for index, record in enumerate(report["subsets"]):
        for arm, data in record["arms"].items():
            if arm == "A":
                part = data["parts"][0]
                incumbent, bound = part["objective"], part["best_bound"]
                gap = part["relative_gap"]
                design = str(part["installed"])
            else:
                incumbent, bound = data["global_incumbent"], data["global_bound"]
                gap = data["global_relative_gap"]
                design = data["winner"] or "-"
            print(
                f"{index:<8}{arm:<5}"
                + (f"{incumbent:>14,.0f}" if incumbent is not None else f"{'-':>14}")
                + (f"{bound:>15,.0f}" if bound is not None else f"{'-':>15}")
                + (f"{gap:>8.2%}" if gap is not None else f"{'-':>9}")
                + f"{design:>18}{data['wall_seconds']:>8.0f}{data['core_hours']:>9.2f}"
            )
    print()
    print(
        "A vs B is the stability question: does enumeration find a better or a\n"
        "more repeatable design at the same per-solve budget. A vs C is the\n"
        "honest speed question, at equal total core-hours."
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
