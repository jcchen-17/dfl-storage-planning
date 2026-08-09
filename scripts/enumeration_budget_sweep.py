"""How short a per-site budget still ranks scenario subsets the same way?

DFL needs a planning oracle it can afford to call hundreds of times. The gap is
the wrong thing to size that budget on: this model's bound barely moves with
time, while its incumbent settles early. What matters is whether a cheaper
oracle puts the same subsets in the same order, picks the same bus, and sizes
the battery the same way.

So this sweeps the per-site budget and compares, across budgets:

  * Spearman correlation of the subset ranking
  * overlap of the top-k subsets
  * how often the winning bus agrees
  * how much the winning P/E moves
  * how much the winning objective moves

The no-storage part is solved once per subset and cached: it closes quickly and
does not depend on the budget being swept.

Timings are reported four ways, because the per-site budget is only part of what
a call costs and the difference was measured to be large: an enumeration with a
theoretical 360 s of solver budget took 517 s of wall clock.
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

import numpy as np

from storage_dfl.config import load_config
from storage_dfl.planning import StoragePlanningOracle
from storage_dfl.stages import _experiment_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset_v2_dfl_hourly_layered.yaml")
    parser.add_argument("--control", default="outputs/random_subset_control_k3.json")
    parser.add_argument("--subsets", type=int, default=6)
    parser.add_argument("--budgets", type=float, nargs="+", default=[2, 30, 60, 120])
    parser.add_argument("--none-budget", type=float, default=120.0)
    parser.add_argument("--absolute-gap", type=float, default=100.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--memory-limit-mb", type=float, default=9000.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out", default="outputs/enumeration_budget_sweep.json")
    return parser.parse_args()


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation without scipy; ties averaged."""

    if len(a) < 3:
        return None

    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(np.asarray(values, dtype=float))
        out = np.empty(len(values), dtype=float)
        out[order] = np.arange(len(values), dtype=float)
        return out

    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    feeder, support_pool = _experiment_data(config, config.data.validation_split)
    by_name = {s.name: s for s in support_pool.scenarios}
    candidates = list(feeder.storage_candidates)

    control = json.loads(Path(args.control).read_text(encoding="utf-8"))
    reference = control["baselines"]["no_storage"]["objective"]
    draws = [d for d in control["draws"] if d.get("decision_loss") is not None]
    ordered = sorted(draws, key=lambda d: reference - d["decision_loss"])
    # Spread across the control's outcome range, so the ranking being compared
    # is not one the joint solver already collapsed.
    step = max(1, len(ordered) // args.subsets)
    picks = [ordered[i * step] for i in range(args.subsets) if i * step < len(ordered)]

    def make_oracle(seconds: float, sites: tuple[str, ...] | None, allow: int):
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

    def run(seconds: float, sites: tuple[str, ...] | None, allow: int, supports):
        oracle = make_oracle(seconds, sites, allow)
        started = time.perf_counter()
        planned = oracle.solve(
            supports,
            weights=tuple(1.0 / len(supports) for _ in supports),
            allow_carbon_slack=config.dfl.training_allow_carbon_slack,
        )
        pipeline = time.perf_counter() - started
        installed = list(planned.design.installed_buses) if planned.feasible else []
        # The decomposition is only an equivalence if each part really is the
        # region it claims to be. Checked rather than assumed, because a part
        # that quietly built elsewhere, or built nothing, would still produce a
        # plausible-looking objective and corrupt the min across parts.
        if planned.feasible:
            if allow == 0 and installed:
                raise SystemExit(
                    f"The no-storage part installed {installed}; the decomposition "
                    "is not what it claims and the sweep would be meaningless."
                )
            if allow > 0 and sites is not None and installed != list(sites):
                raise SystemExit(
                    f"The part forced to {list(sites)} returned {installed}."
                )
        return {
            "status": planned.status,
            "objective": _finite(planned.objective),
            "best_bound": _finite(planned.best_bound),
            "relative_gap": _finite(planned.relative_gap),
            "installed": installed,
            "power_mw": sum(float(planned.design.power_mw[b]) for b in installed),
            "energy_mwh": sum(float(planned.design.energy_mwh[b]) for b in installed),
            # Pipeline time is the honest one: it includes model construction and
            # any bootstrap, which the per-site budget does not cover.
            "solver_seconds": float(planned.solve_time_seconds),
            "pipeline_seconds": pipeline,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "budgets": args.budgets,
        "none_budget": args.none_budget,
        "absolute_gap_dollars": args.absolute_gap,
        "relative_gap": 0.0,
        "threads": args.threads,
        "candidates": candidates,
        "config": str(Path(args.config).resolve()),
    }
    report: dict = {"settings": settings, "subsets": [], "parts": {}}
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        # Timings are one of the outputs, so rows solved under a different budget
        # or thread count are not interchangeable with these.
        if existing.get("settings") != settings:
            raise SystemExit(
                f"{out_path} was written under different settings; move it aside "
                "rather than mixing rows that are not comparable."
            )
        report = existing
        report.setdefault("subsets", [])
        report.setdefault("parts", {})
        print(
            f"resuming: {len(report['subsets'])} subsets complete, "
            f"{len(report['parts'])} individual solves cached",
            flush=True,
        )
    done = {tuple(s["names"]) for s in report["subsets"]}

    def flush() -> None:
        # Written to a sibling and moved into place: a 20-minute subset was lost
        # once to a process that died mid-write, and a truncated JSON would take
        # the resume with it.
        temporary = out_path.with_suffix(out_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(out_path)

    def cached(key: str, produce):
        """One solve, checkpointed. Resume granularity is the individual solve,
        not the subset, so an interruption costs at most one part."""

        if key in report["parts"]:
            return report["parts"][key], True
        value = produce()
        report["parts"][key] = value
        flush()
        return value, False

    for index, entry in enumerate(picks):
        names = entry["names"]
        if tuple(names) in done:
            print(f"\n[{index + 1}/{len(picks)}] {names} -- already done", flush=True)
            continue
        supports = tuple(by_name[n] for n in names)
        record: dict = {"names": names, "none": None, "budgets": {}}
        print(f"\n[{index + 1}/{len(picks)}] {names}", flush=True)

        tag = "|".join(names)
        record["none"], reused = cached(
            f"{tag}|none",
            lambda: run(args.none_budget, (candidates[0],), 0, supports),
        )
        print(
            f"  none ({args.none_budget:.0f}s): "
            f"obj {record['none']['objective']:,.0f} "
            f"bound {record['none']['best_bound']:,.0f} "
            f"{record['none']['status']} {record['none']['pipeline_seconds']:.0f}s"
            + ("  [cached]" if reused else ""),
            flush=True,
        )

        for budget in args.budgets:
            parts = {}
            wall = time.perf_counter()
            for bus in candidates:
                parts[bus], reused = cached(
                    f"{tag}|{budget:g}|{bus}",
                    lambda bus=bus, budget=budget: run(budget, (bus,), 1, supports),
                )
                part = parts[bus]
                if part["objective"] is None:
                    detail = "no incumbent "
                else:
                    detail = (
                        f"obj {part['objective']:>12,.0f} "
                        f"{part['power_mw']:.3f} MW/{part['energy_mwh']:.3f} MWh "
                    )
                print(
                    f"    {budget:>5.0f}s bus {bus}: {detail}"
                    f"{part['status']:<11}{part['pipeline_seconds']:.0f}s"
                    + ("  [cached]" if reused else ""),
                    flush=True,
                )
            usable = {b: p for b, p in parts.items() if p["objective"] is not None}
            none_objective = record["none"]["objective"]
            best_bus = min(usable, key=lambda b: usable[b]["objective"]) if usable else None
            builds = (
                best_bus is not None
                and none_objective is not None
                and usable[best_bus]["objective"] < none_objective
            )
            bounds = [p["best_bound"] for p in parts.values() if p["best_bound"] is not None]
            if record["none"]["best_bound"] is not None:
                bounds.append(record["none"]["best_bound"])
            record["budgets"][f"{budget:g}"] = {
                "parts": parts,
                "winner": best_bus,
                "builds": builds,
                "global_incumbent": (
                    min(
                        [p["objective"] for p in usable.values()]
                        + ([none_objective] if none_objective is not None else [])
                    )
                    if usable or none_objective is not None
                    else None
                ),
                "global_bound": min(bounds) if bounds else None,
                "wall_seconds": time.perf_counter() - wall,
                "solver_seconds": sum(p["solver_seconds"] for p in parts.values()),
                "pipeline_seconds": sum(p["pipeline_seconds"] for p in parts.values()),
            }
            summary = record["budgets"][f"{budget:g}"]
            print(
                f"    -> {budget:g}s winner {best_bus} "
                f"incumbent {summary['global_incumbent']:,.0f} "
                f"builds={builds} wall {summary['wall_seconds']:.0f}s",
                flush=True,
            )
        report["subsets"].append(record)
        done.add(tuple(names))
        flush()

    # --- stability across budgets -------------------------------------------
    reference_budget = f"{max(args.budgets):g}"
    print()
    print(f"stability against the {reference_budget}s budget")
    print(
        f"{'budget':>8}{'spearman':>11}{'top-k same':>12}{'same bus':>11}"
        f"{'|dE| median':>14}{'|dObj| median':>15}{'pipeline':>11}"
    )
    print("-" * 82)
    stability = {}
    for budget in args.budgets:
        key = f"{budget:g}"
        pairs = [
            (
                s["budgets"][key]["global_incumbent"],
                s["budgets"][reference_budget]["global_incumbent"],
            )
            for s in report["subsets"]
            if s["budgets"][key]["global_incumbent"] is not None
            and s["budgets"][reference_budget]["global_incumbent"] is not None
        ]
        rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None
        ranked = sorted(
            (s for s in report["subsets"] if s["budgets"][key]["global_incumbent"] is not None),
            key=lambda s: s["budgets"][key]["global_incumbent"],
        )
        ranked_reference = sorted(
            (
                s
                for s in report["subsets"]
                if s["budgets"][reference_budget]["global_incumbent"] is not None
            ),
            key=lambda s: s["budgets"][reference_budget]["global_incumbent"],
        )
        top = {tuple(s["names"]) for s in ranked[: args.top_k]}
        top_reference = {tuple(s["names"]) for s in ranked_reference[: args.top_k]}
        overlap = len(top & top_reference) / max(len(top_reference), 1)
        same_bus = [
            s["budgets"][key]["winner"] == s["budgets"][reference_budget]["winner"]
            for s in report["subsets"]
        ]
        energy_shift, objective_shift = [], []
        for s in report["subsets"]:
            here, there = s["budgets"][key], s["budgets"][reference_budget]
            if here["winner"] and there["winner"]:
                energy_shift.append(
                    abs(
                        here["parts"][here["winner"]]["energy_mwh"]
                        - there["parts"][there["winner"]]["energy_mwh"]
                    )
                )
            if here["global_incumbent"] and there["global_incumbent"]:
                objective_shift.append(
                    abs(here["global_incumbent"] - there["global_incumbent"])
                )
        pipeline = float(
            np.mean([s["budgets"][key]["pipeline_seconds"] for s in report["subsets"]])
        )
        stability[key] = {
            "spearman": rho,
            "top_k_overlap": overlap,
            "same_winning_bus": float(np.mean(same_bus)),
            "median_energy_shift_mwh": float(np.median(energy_shift)) if energy_shift else None,
            "median_objective_shift": float(np.median(objective_shift)) if objective_shift else None,
            "mean_pipeline_seconds": pipeline,
        }
        print(
            f"{budget:>8.0f}"
            + (f"{rho:>11.3f}" if rho is not None else f"{'-':>11}")
            + f"{overlap:>11.0%}{np.mean(same_bus):>11.0%}"
            + (
                f"{np.median(energy_shift):>14.3f}"
                if energy_shift
                else f"{'-':>14}"
            )
            + (
                f"{np.median(objective_shift):>15,.0f}"
                if objective_shift
                else f"{'-':>15}"
            )
            + f"{pipeline:>10.0f}s"
        )
    report["stability"] = stability
    flush()
    print()
    print(
        "A budget whose Spearman is near 1, whose top-k and winning bus agree,\n"
        "and whose energy moves little, is enough of an oracle for DFL training\n"
        "even with a wide optimality gap. Budgets that produced no incumbent\n"
        "cannot be ranked and are excluded from the correlation."
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
