"""Re-solve only undecided site parts from an enumeration budget sweep.

The source sweep is never overwritten.  Each targeted result is atomically
checkpointed, then merged into a copy whose 60-second row can be analysed as a
complete oracle candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")

from storage_dfl.config import load_config
from storage_dfl.planning import StoragePlanningOracle
from storage_dfl.stages import _experiment_data


PROVEN_EMPTY = {"infeasible", "unbounded"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset_v2_dfl_hourly_layered.yaml")
    parser.add_argument("--sweep", default="outputs/enumeration_budget_sweep.json")
    parser.add_argument("--budget", type=float, default=60.0)
    parser.add_argument("--out", default="outputs/enumeration_missing_60s.json")
    parser.add_argument(
        "--merged", default="outputs/enumeration_budget_sweep_completed_60s.json"
    )
    return parser.parse_args()


def finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def result_dict(planned, pipeline_seconds: float) -> dict:
    installed = list(planned.design.installed_buses) if planned.feasible else []
    return {
        "status": planned.status,
        "objective": finite(planned.objective),
        "best_bound": finite(planned.best_bound),
        "relative_gap": finite(planned.relative_gap),
        "installed": installed,
        "power_mw": sum(float(planned.design.power_mw[b]) for b in installed),
        "energy_mwh": sum(float(planned.design.energy_mwh[b]) for b in installed),
        "solver_seconds": float(planned.solve_time_seconds),
        "pipeline_seconds": float(pipeline_seconds),
    }


def recompute_budget(subset: dict, budget_key: str, candidates: list[str]) -> None:
    entry = subset["budgets"][budget_key]
    parts = entry["parts"]
    usable = {bus: parts[bus] for bus in candidates if parts[bus]["objective"] is not None}
    none = subset["none"]
    none_objective = none["objective"]
    best_bus = min(usable, key=lambda bus: usable[bus]["objective"]) if usable else None
    entry["winner"] = best_bus
    entry["builds"] = bool(
        best_bus is not None
        and none_objective is not None
        and usable[best_bus]["objective"] < none_objective
    )
    objectives = [part["objective"] for part in usable.values()]
    if none_objective is not None:
        objectives.append(none_objective)
    entry["global_incumbent"] = min(objectives) if objectives else None
    bounds = [part["best_bound"] for part in parts.values() if part["best_bound"] is not None]
    if none["best_bound"] is not None:
        bounds.append(none["best_bound"])
    entry["global_bound"] = min(bounds) if bounds else None
    entry["solver_seconds"] = sum(float(part["solver_seconds"]) for part in parts.values())
    entry["pipeline_seconds"] = sum(
        float(part["pipeline_seconds"]) for part in parts.values()
    )
    entry["wall_seconds"] = entry["pipeline_seconds"]


def main() -> None:
    args = parse_args()
    source_path = Path(args.sweep)
    sweep = json.loads(source_path.read_text(encoding="utf-8"))
    budget_key = f"{args.budget:g}"
    candidates = list(sweep["settings"]["candidates"])

    config = load_config(args.config)
    feeder, pool = _experiment_data(config, config.data.validation_split)
    by_name = {scenario.name: scenario for scenario in pool.scenarios}
    weights_for = lambda supports: tuple(1.0 / len(supports) for _ in supports)

    targets: list[tuple[int, list[str], str]] = []
    for index, subset in enumerate(sweep["subsets"]):
        parts = subset["budgets"][budget_key]["parts"]
        for bus in candidates:
            part = parts[bus]
            if part["objective"] is None and part["status"] not in PROVEN_EMPTY:
                targets.append((index, list(subset["names"]), bus))

    output_path = Path(args.out)
    if output_path.exists():
        report = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        report = {
            "settings": {
                "source": str(source_path.resolve()),
                "config": str(Path(args.config).resolve()),
                "budget": args.budget,
                "threads": sweep["settings"]["threads"],
                "absolute_gap_dollars": sweep["settings"]["absolute_gap_dollars"],
            },
            "results": [],
        }
    done = {(tuple(row["names"]), row["bus"]) for row in report["results"]}
    print(f"targeted undecided parts: {len(targets)}; already complete: {len(done)}", flush=True)

    bootstrap_by_subset: dict[tuple[str, ...], dict[str, float] | None] = {}
    for number, (index, names, bus) in enumerate(targets, start=1):
        key = (tuple(names), bus)
        if key in done:
            print(f"[{number}/{len(targets)}] {bus} {names}: already done", flush=True)
            continue
        supports = tuple(by_name[name] for name in names)
        weights = weights_for(supports)
        restricted = replace(feeder, storage_candidates=(bus,))
        planning = replace(
            config.planning,
            solver_relative_gap=0.0,
            solver_absolute_gap_dollars=float(sweep["settings"]["absolute_gap_dollars"]),
            solver_time_limit_seconds=args.budget,
            solver_threads=int(sweep["settings"]["threads"]),
            solver_max_parallel_workers=1,
            max_storage_sites=1,
            min_storage_sites=1,
            solver_emphasis="optimality",
            solver_aggressive_heuristics=False,
        )
        oracle = StoragePlanningOracle(
            restricted, planning, config.costs, config.data, config.data_center
        )
        subset_key = tuple(names)
        warm_key = oracle._warm_start_key(
            tuple(supports), tuple(weights), config.dfl.training_allow_carbon_slack
        )
        if subset_key in bootstrap_by_subset:
            oracle._warm_start_cache[warm_key] = bootstrap_by_subset[subset_key]

        print(f"[{number}/{len(targets)}] subset {index}, bus {bus}: solving...", flush=True)
        started = time.perf_counter()
        planned = oracle.solve(
            supports,
            weights=weights,
            allow_carbon_slack=config.dfl.training_allow_carbon_slack,
        )
        elapsed = time.perf_counter() - started
        bootstrap_by_subset[subset_key] = oracle._warm_start_cache.get(warm_key)
        result = result_dict(planned, elapsed)
        if planned.feasible and result["installed"] != [bus]:
            raise RuntimeError(
                f"Forced bus {bus} returned {result['installed']}; refusing to merge."
            )
        report["results"].append(
            {"subset_index": index, "names": names, "bus": bus, "result": result}
        )
        atomic_write(output_path, report)
        print(
            f"    {planned.status}, objective={result['objective']}, "
            f"installed={result['installed']}, pipeline={elapsed:.1f}s",
            flush=True,
        )

    merged = json.loads(source_path.read_text(encoding="utf-8"))
    by_subset = {tuple(subset["names"]): subset for subset in merged["subsets"]}
    for row in report["results"]:
        subset = by_subset[tuple(row["names"])]
        subset["budgets"][budget_key]["parts"][row["bus"]] = row["result"]
    for subset in merged["subsets"]:
        recompute_budget(subset, budget_key, candidates)
    merged.setdefault("augmentation", {})[budget_key] = {
        "source": str(output_path.resolve()),
        "targeted_parts": len(report["results"]),
        "warm_start": "minimum forced-build idle storage",
    }
    atomic_write(Path(args.merged), merged)
    print(f"wrote {output_path}", flush=True)
    print(f"wrote {args.merged}", flush=True)


if __name__ == "__main__":
    main()
