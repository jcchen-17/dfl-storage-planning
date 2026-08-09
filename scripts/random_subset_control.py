"""Random subset search over observed scenarios: the control a selector must beat.

The open question is not whether decision-focused selection produces good
designs -- it does -- but whether the *learning* contributes anything over
drawing the same number of candidate supports at random, scoring each on the
same fixed evaluation set, and keeping the best. This measures that control
directly, on real scenarios, so a learned selector has a number to beat at
equal budget.

Every draw is appended to the output JSON as it completes, so an interrupted
run resumes where it stopped rather than repeating hours of solves.
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
from storage_dfl.data import Scenario
from storage_dfl.dfl import select_scenarios
from storage_dfl.planning import StorageDesign, StoragePlanningOracle
from storage_dfl.stages import (
    ArtifactPaths,
    _experiment_data,
    _load_codec,
    _solve_fixed_design_scenario_wise,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/dataset_v2_dfl_hourly_layered.yaml"
    )
    parser.add_argument("--k", type=int, default=3, help="support scenarios per draw")
    parser.add_argument("--budget", type=int, default=24, help="random draws")
    parser.add_argument(
        "--evaluation-scenarios",
        type=int,
        default=4,
        help="fixed test scenarios every design is scored on",
    )
    # Measured on the K=3 run: planning reached 1% on 0 of 24 draws and 2% on
    # 8, with a 2.478% median. 3% is what this model actually delivers at these
    # time limits; asking for less only mislabels time-limit incumbents.
    parser.add_argument("--planning-gap", type=float, default=0.03)
    # Evaluation stays at 1%: it is the only number that enters a comparison,
    # and at 3% one design's measured value moved by $20,113 and changed sign.
    parser.add_argument("--evaluation-gap", type=float, default=0.01)
    parser.add_argument("--planning-time-limit", type=float, default=600.0)
    parser.add_argument("--evaluation-time-limit", type=float, default=300.0)
    parser.add_argument("--memory-limit-mb", type=float, default=5500.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="outputs/random_subset_control.json")
    return parser.parse_args()


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = config.seed if args.seed is None else args.seed
    paths = ArtifactPaths(config.output_dir)

    feeder, support_pool = _experiment_data(config, config.data.validation_split)
    _, test_pool = _experiment_data(config, config.data.test_split)
    codec = _load_codec(paths, feeder)

    count = min(args.evaluation_scenarios, len(test_pool.scenarios))
    evaluation, evaluation_weights, evaluation_names = select_scenarios(
        config.dfl.evaluation_selection_rule, test_pool, codec, count, seed=config.seed
    )

    planning_config = replace(
        config.planning,
        solver_relative_gap=args.planning_gap,
        solver_time_limit_seconds=args.planning_time_limit,
        solver_memory_limit_mb=args.memory_limit_mb,
        solver_max_parallel_workers=1,
    )
    evaluation_config = replace(
        config.planning,
        solver_relative_gap=args.evaluation_gap,
        solver_time_limit_seconds=args.evaluation_time_limit,
        solver_memory_limit_mb=args.memory_limit_mb,
        solver_max_parallel_workers=4,
    )
    planning_oracle = StoragePlanningOracle(
        feeder, planning_config, config.costs, config.data, config.data_center
    )
    evaluation_oracle = StoragePlanningOracle(
        feeder, evaluation_config, config.costs, config.data, config.data_center
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    if out_path.exists():
        report = json.loads(out_path.read_text(encoding="utf-8"))
    # A resumed run must not silently mix settings, which is the failure that
    # once appended tighter-gap rows to a looser sweep with nothing marking them.
    fingerprint = {
        "config": str(Path(args.config).resolve()),
        "k": args.k,
        "seed": seed,
        "evaluation_names": list(evaluation_names),
        "planning_gap": args.planning_gap,
        "evaluation_gap": args.evaluation_gap,
        "planning_time_limit": args.planning_time_limit,
        "evaluation_time_limit": args.evaluation_time_limit,
    }
    if report and report.get("fingerprint") != fingerprint:
        raise SystemExit(
            f"{out_path} was written under different settings. Move it aside or "
            "pass a different --out; mixing the two makes the rows incomparable."
        )
    report.setdefault("fingerprint", fingerprint)
    report.setdefault("draws", [])
    report.setdefault("baselines", {})

    def flush() -> None:
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def score(
        supports: tuple[Scenario, ...],
        weights: tuple[float, ...],
    ) -> dict:
        """Plan on the supports, then score the resulting design out of sample."""

        started = time.perf_counter()
        planned = planning_oracle.solve(
            supports,
            weights=weights,
            allow_carbon_slack=config.dfl.training_allow_carbon_slack,
        )
        if not planned.feasible:
            return {
                "feasible": False,
                "planning_status": planned.status,
                "seconds": time.perf_counter() - started,
            }
        aggregate, per_scenario, _ = _solve_fixed_design_scenario_wise(
            evaluation_oracle,
            evaluation,
            evaluation_weights,
            planned.design,
            config.costs.demand_dollars_per_mw_year,
            args.evaluation_gap,
        )
        return {
            "feasible": bool(aggregate.feasible),
            "planning_status": planned.status,
            "planning_gap": _finite(planned.relative_gap),
            "planning_objective": _finite(planned.objective),
            "decision_loss": _finite(aggregate.objective),
            "evaluation_status": aggregate.status,
            "evaluation_gap": _finite(
                max(result.relative_gap for result in per_scenario)
            ),
            "design": {
                "installed_buses": planned.design.installed_buses,
                "power_mw": {
                    bus: float(planned.design.power_mw[bus])
                    for bus in planned.design.installed_buses
                },
                "energy_mwh": {
                    bus: float(planned.design.energy_mwh[bus])
                    for bus in planned.design.installed_buses
                },
            },
            "seconds": time.perf_counter() - started,
        }

    # One no-storage reference, so every decision loss can be expressed as the
    # storage value it produces against a common constant.
    if "no_storage" not in report["baselines"]:
        print("no-storage reference...", flush=True)
        no_storage_design = StorageDesign(
            site={bus: 0 for bus in feeder.storage_candidates},
            power_mw={bus: 0.0 for bus in feeder.storage_candidates},
            energy_mwh={bus: 0.0 for bus in feeder.storage_candidates},
        )
        reference_oracle = StoragePlanningOracle(
            feeder,
            replace(evaluation_config, max_storage_sites=0),
            config.costs,
            config.data,
            config.data_center,
        )
        reference, reference_rows, _ = _solve_fixed_design_scenario_wise(
            reference_oracle,
            evaluation,
            evaluation_weights,
            no_storage_design,
            config.costs.demand_dollars_per_mw_year,
            args.evaluation_gap,
        )
        report["baselines"]["no_storage"] = {
            "objective": _finite(reference.objective),
            "status": reference.status,
            "gap": _finite(max(r.relative_gap for r in reference_rows)),
        }
        flush()
    reference_objective = report["baselines"]["no_storage"]["objective"]
    print(f"no-storage reference: {reference_objective:,.2f}", flush=True)

    def storage_value(loss: float | None) -> float | None:
        if loss is None or reference_objective is None:
            return None
        return reference_objective - loss

    # Heuristic reference points, scored exactly like every random draw.
    for rule in ("kmeans", "farthest"):
        if rule in report["baselines"]:
            continue
        supports, weights, names = select_scenarios(
            rule, support_pool, codec, args.k, seed=seed
        )
        print(f"baseline {rule} K={args.k}: {list(names)}", flush=True)
        record = score(supports, weights)
        record["names"] = list(names)
        report["baselines"][rule] = record
        flush()
        value = storage_value(record.get("decision_loss"))
        print(
            f"  storage value {value:,.2f}" if value is not None else "  infeasible",
            flush=True,
        )

    rng = np.random.default_rng(seed)
    done = {tuple(sorted(entry["names"])) for entry in report["draws"]}
    # Replaying the stream from the start keeps a resumed run on the same draw
    # sequence as an uninterrupted one, so the budget curve stays reproducible.
    attempt = 0
    while len(report["draws"]) < args.budget:
        attempt += 1
        if attempt > 50 * args.budget:
            raise SystemExit("Could not draw enough distinct subsets.")
        picks = rng.choice(len(support_pool.scenarios), size=args.k, replace=False)
        names = tuple(sorted(support_pool.scenarios[int(i)].name for i in picks))
        if names in done:
            continue
        done.add(names)
        index = len(report["draws"]) + 1
        print(f"draw {index}/{args.budget}: {list(names)}", flush=True)
        supports = tuple(support_pool.scenarios[int(i)] for i in picks)
        weights = tuple(1.0 / args.k for _ in picks)
        record = score(supports, weights)
        record["names"] = list(names)
        report["draws"].append(record)
        flush()
        value = storage_value(record.get("decision_loss"))
        if value is None:
            print(f"  infeasible ({record['planning_status']})", flush=True)
        else:
            best = max(
                filter(
                    None,
                    (storage_value(e.get("decision_loss")) for e in report["draws"]),
                ),
                default=value,
            )
            print(
                f"  storage value {value:,.2f}   best so far {best:,.2f}"
                f"   ({record['seconds']:.0f}s)",
                flush=True,
            )

    values = [storage_value(e.get("decision_loss")) for e in report["draws"]]
    usable = [v for v in values if v is not None]
    print()
    print(f"{len(usable)} usable of {len(values)} draws, K={args.k}")
    if usable:
        running = []
        best = -math.inf
        for v in values:
            if v is not None:
                best = max(best, v)
            running.append(best)
        print()
        print("best-of-budget curve (this is the number a selector must beat)")
        print(f"{'budget':>8}{'best storage value':>22}")
        for budget in (1, 2, 4, 8, 16, 24, 32, 48, 64):
            if budget <= len(running):
                print(f"{budget:>8}{running[budget - 1]:>22,.2f}")
        print()
        print(f"{'mean':>8}{np.mean(usable):>22,.2f}")
        print(f"{'sd':>8}{np.std(usable):>22,.2f}")
        print(f"{'worst':>8}{min(usable):>22,.2f}")
    for rule in ("kmeans", "farthest"):
        record = report["baselines"].get(rule)
        if record:
            value = storage_value(record.get("decision_loss"))
            print(
                f"{rule:>8}{value:>22,.2f}"
                if value is not None
                else f"{rule:>8}{'infeasible':>22}"
            )
    print()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
