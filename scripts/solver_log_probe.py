"""Solve one planning model in-process with the solver log visible.

During a DFL run every solve happens in an isolated worker whose stdout is
captured and discarded unless the worker fails, so ``verbose_solver`` shows
nothing.  This script deliberately imports nothing that pulls in torch, which is
what triggers that isolation, so the model is built and solved right here and
the solver prints its log to the terminal.

Use it to answer, for the configured K and candidate set:

  * is the thread parameter reaching the solver ("Thread count: N of M")
  * how much of the solve is single-threaded presolve
  * how large the branch-and-bound tree gets before the gap is met
  * how long one solve actually takes, against solver_time_limit_seconds

    python scripts/solver_log_probe.py --config configs/generator_compare_k3t2.yaml
    python scripts/solver_log_probe.py --config ... --threads 16 --no-warm-start
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from storage_dfl.config import load_config
from storage_dfl.data import load_historical_scenarios
from storage_dfl.network import ieee13_unbalanced_microgrid
from storage_dfl.planning.model import StoragePlanningOracle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generator_compare_k3t2.yaml")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override solver_threads; the config value is used otherwise.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Override solver_time_limit_seconds, e.g. to cap a probe run.",
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Skip the no-storage bootstrap, isolating the main solve.",
    )
    parser.add_argument("--split", default="validation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    planning = replace(config.planning, verbose_solver=True)
    if args.threads is not None:
        planning = replace(planning, solver_threads=args.threads)
    if args.time_limit is not None:
        planning = replace(planning, solver_time_limit_seconds=args.time_limit)

    feeder = ieee13_unbalanced_microgrid()
    pool = load_historical_scenarios(
        config.data.dataset_path, split=args.split, horizon=config.data.horizon
    )
    k = config.dfl.num_support_scenarios
    scenarios = tuple(pool.scenarios[:k])
    weights = tuple(1.0 / k for _ in scenarios)

    print(
        f"K={k}  candidates={len(feeder.storage_candidates)}  "
        f"horizon={config.data.horizon}  backend={planning.solver_backend}\n"
        f"threads={planning.solver_threads}  gap={planning.solver_relative_gap}  "
        f"time_limit={planning.solver_time_limit_seconds}s  "
        f"memory_limit={planning.solver_memory_limit_mb} MB\n",
        flush=True,
    )

    oracle = StoragePlanningOracle(
        feeder, planning, config.costs, config.data, config.data_center
    )
    if args.no_warm_start:
        # A present key with no values makes solve() treat the bootstrap as
        # already done and seed nothing.
        oracle._warm_start_cache[
            oracle._warm_start_key(scenarios, weights, False)
        ] = None

    started = time.perf_counter()
    result = oracle.solve(scenarios, weights=weights)
    elapsed = time.perf_counter() - started

    print(
        f"\nwall clock {elapsed:.1f}s   solver time {result.solve_time_seconds:.1f}s"
        f"   status {result.status}   gap {result.relative_gap:.6f}"
    )
    print(
        f"objective {result.objective:,.0f}   installed {result.design.installed_buses}"
    )
    print(
        "\nIn the log above: 'Thread count' shows whether solver_threads reached "
        "the solver, 'Presolve time' is single-threaded, and the node counter "
        "shows whether the tree search (which is what actually parallelises) "
        "ever got going."
    )


if __name__ == "__main__":
    main()
