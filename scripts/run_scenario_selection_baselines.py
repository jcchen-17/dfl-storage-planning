"""Batch-run observed-scenario selection baselines.

Examples (run from the repository root in the storage-dfl environment):

  # Fast smoke comparison using the config seed and all 182 test scenarios.
  python scripts/run_scenario_selection_baselines.py

  # Paper comparison: 30 stochastic seeds, two jobs at a time.
  python scripts/run_scenario_selection_baselines.py `
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 `
    --parallel-runs 2

``random``, ``kmeans`` and ``aggregate`` run once per requested seed;
``farthest`` is deterministic and therefore runs once.  Every method plans on
K observed training scenarios, fixes the resulting design, and evaluates it on
the identical held-out test set.  The no-storage reference is computed once.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from storage_dfl.baselines import (
    comparison_metrics,
    run_reference,
    run_selection_baseline,
)
from storage_dfl.config import load_config
from storage_dfl.dfl.selection import SELECTION_RULES
from storage_dfl.stages import _write_json


DEFAULT_RULES = ("random", "kmeans", "farthest", "aggregate")


def _unique(values: list[Any]) -> list[Any]:
    return list(dict.fromkeys(values))


def _config_label(path: Path) -> str:
    prefix = "dataset_v2_dfl_"
    return path.stem[len(prefix) :] if path.stem.startswith(prefix) else path.stem


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    _write_json(temporary, payload)
    temporary.replace(path)


def _worker_command(args: argparse.Namespace, output: Path, **job: Any) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-kind",
        str(job["kind"]),
        "--config",
        str(Path(args.config).resolve()),
        "--output",
        str(output.resolve()),
        "--support-scenarios",
        str(args.support_scenarios),
        "--test-scenarios",
        str(args.test_scenarios),
        "--solver-workers-per-run",
        str(args.solver_workers_per_run),
        "--solver-threads-per-run",
        str(args.solver_threads_per_run),
    ]
    if args.solver_relative_gap is not None:
        command.extend(["--solver-relative-gap", str(args.solver_relative_gap)])
    if args.solver_time_limit_seconds is not None:
        command.extend(
            ["--solver-time-limit-seconds", str(args.solver_time_limit_seconds)]
        )
    if job["kind"] == "reference":
        if args.perfect_information:
            command.append("--perfect-information")
    else:
        command.extend(
            ["--rule", str(job["rule"]), "--worker-seed", str(job["seed"])]
        )
    return command


def _run_subprocess(command: list[str], log_path: Path) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND\n" + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent.parent,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode, str(log_path.resolve())


def _write_comparison_csv(path: Path, records: list[dict[str, Any]]) -> None:
    completed = [record for record in records if record.get("status") == "completed"]
    fields = [
        "rule",
        "seed",
        "support_scenarios",
        "test_scenarios",
        "planning_converged",
        "planning_status",
        "planning_relative_gap",
        "test_status",
        "test_relative_gap",
        "test_objective",
        "storage_value",
        "decision_regret",
        "power_mw",
        "energy_mwh",
        "load_shedding_mwh",
        "load_shedding_probability",
        "carbon_excess_t",
        "carbon_excess_probability",
        "planning_wall_seconds",
        "evaluation_wall_seconds",
        "objectives_comparable",
        "result",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in completed:
            writer.writerow({field: record.get(field) for field in fields})


def _worker(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if args.worker_kind == "reference":
        payload = run_reference(
            args.config,
            test_scenarios=args.test_scenarios,
            perfect_information=args.perfect_information,
            solver_workers=args.solver_workers_per_run,
            solver_threads=args.solver_threads_per_run,
            solver_relative_gap=args.solver_relative_gap,
            solver_time_limit_seconds=args.solver_time_limit_seconds,
        )
    else:
        if args.rule is None or args.worker_seed is None:
            raise ValueError("A baseline worker requires --rule and --worker-seed.")
        payload = run_selection_baseline(
            args.config,
            rule=args.rule,
            seed=args.worker_seed,
            support_scenarios=args.support_scenarios,
            test_scenarios=args.test_scenarios,
            solver_workers=args.solver_workers_per_run,
            solver_threads=args.solver_threads_per_run,
            solver_relative_gap=args.solver_relative_gap,
            solver_time_limit_seconds=args.solver_time_limit_seconds,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    print(f"Wrote {output.resolve()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare observed-scenario Random/K-means/Farthest/Aggregate "
            "storage-planning baselines on one common held-out test set."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/dataset_v2_dfl_hourly_layered_t1.yaml",
    )
    parser.add_argument(
        "--rules", nargs="+", choices=SELECTION_RULES, default=list(DEFAULT_RULES)
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="seeds for random/kmeans/aggregate; default uses the config seed",
    )
    parser.add_argument(
        "--support-scenarios",
        type=int,
        default=None,
        help="K training supports; default uses dfl.num_support_scenarios",
    )
    parser.add_argument(
        "--test-scenarios",
        type=int,
        default=0,
        help="held-out scenarios; 0 (default) evaluates the complete test split",
    )
    parser.add_argument("--parallel-runs", type=int, default=1)
    parser.add_argument("--solver-workers-per-run", type=int, default=None)
    parser.add_argument("--solver-threads-per-run", type=int, default=None)
    parser.add_argument("--solver-relative-gap", type=float, default=None)
    parser.add_argument("--solver-time-limit-seconds", type=float, default=None)
    parser.add_argument(
        "--perfect-information",
        action="store_true",
        help=(
            "also solve one joint planning MILP on the test set; expensive for "
            "the complete 182-scenario split"
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")

    # Internal subprocess mode keeps solver state and failures isolated in the
    # same way as run_learning_objective_baselines.py.
    parser.add_argument(
        "--worker-kind",
        choices=("reference", "baseline"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--rule", choices=SELECTION_RULES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--output", help=argparse.SUPPRESS)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    args.rules = _unique(args.rules)
    args.seeds = _unique(args.seeds or [config.seed])
    args.support_scenarios = (
        int(args.support_scenarios)
        if args.support_scenarios is not None
        else int(config.dfl.num_support_scenarios)
    )
    if args.support_scenarios <= 0:
        parser.error("--support-scenarios must be positive")
    if args.test_scenarios < 0:
        parser.error("--test-scenarios cannot be negative")
    if args.parallel_runs <= 0:
        parser.error("--parallel-runs must be positive")
    if args.solver_workers_per_run is not None and args.solver_workers_per_run <= 0:
        parser.error("--solver-workers-per-run must be positive")
    if args.solver_threads_per_run is not None and args.solver_threads_per_run <= 0:
        parser.error("--solver-threads-per-run must be positive")
    if args.solver_relative_gap is not None and args.solver_relative_gap < 0.0:
        parser.error("--solver-relative-gap cannot be negative")
    if (
        args.solver_time_limit_seconds is not None
        and args.solver_time_limit_seconds <= 0.0
    ):
        parser.error("--solver-time-limit-seconds must be positive")

    base_workers = int(config.planning.solver_max_parallel_workers)
    args.solver_workers_per_run = args.solver_workers_per_run or max(
        1, base_workers // args.parallel_runs
    )
    args.solver_threads_per_run = (
        args.solver_threads_per_run or int(config.planning.solver_threads)
    )
    if args.solver_workers_per_run <= 0 or args.solver_threads_per_run <= 0:
        parser.error("solver workers and threads per run must be positive")

    if args.worker_kind is not None:
        if args.output is None:
            parser.error("worker mode requires --output")
        _worker(args)
        return

    project_root = Path(__file__).resolve().parent.parent
    suite_id = time.strftime("%Y%m%d-%H%M%S")
    suite_dir = (
        project_root
        / "outputs"
        / "baselines"
        / "scenario_selection"
        / _config_label(config_path)
        / "suites"
        / suite_id
    )
    summary_path = suite_dir / "summary.json"
    comparison_path = suite_dir / "comparison.csv"
    reference_path = suite_dir / "reference.json"

    jobs: list[dict[str, Any]] = []
    for rule in args.rules:
        seeds = [args.seeds[0]] if rule == "farthest" else args.seeds
        for seed in seeds:
            result_path = suite_dir / "results" / rule / f"seed_{seed}.json"
            jobs.append(
                {
                    "rule": rule,
                    "seed": seed,
                    "support_scenarios": args.support_scenarios,
                    "test_scenarios": args.test_scenarios,
                    "status": "queued",
                    "result": str(result_path.resolve()),
                    "console_log": str(
                        (suite_dir / "logs" / f"{rule}_seed_{seed}.log").resolve()
                    ),
                }
            )

    summary: dict[str, Any] = {
        "suite_id": suite_id,
        "base_config": str(config_path),
        "rules": args.rules,
        "seeds": args.seeds,
        "support_scenarios": args.support_scenarios,
        "test_scenarios_requested": args.test_scenarios,
        "parallel_runs": args.parallel_runs,
        "solver_workers_per_run": args.solver_workers_per_run,
        "solver_threads_per_run": args.solver_threads_per_run,
        "solver_relative_gap": (
            args.solver_relative_gap
            if args.solver_relative_gap is not None
            else config.planning.solver_relative_gap
        ),
        "solver_time_limit_seconds": (
            args.solver_time_limit_seconds
            if args.solver_time_limit_seconds is not None
            else config.planning.solver_time_limit_seconds
        ),
        "perfect_information_requested": args.perfect_information,
        "reference": {"status": "queued", "result": str(reference_path.resolve())},
        "runs": jobs,
    }
    _write_summary(summary_path, summary)

    print(
        "Computing one shared no-storage reference on "
        f"{'all' if args.test_scenarios == 0 else args.test_scenarios} test scenarios...",
        flush=True,
    )
    reference_command = _worker_command(
        args, reference_path, kind="reference"
    )
    reference_log = suite_dir / "logs" / "reference.log"
    code, log = _run_subprocess(reference_command, reference_log)
    if code != 0:
        summary["reference"] = {
            "status": "failed",
            "result": str(reference_path.resolve()),
            "console_log": log,
            "error": f"reference worker exited with code {code}",
        }
        _write_summary(summary_path, summary)
        raise SystemExit(f"Reference failed; inspect {log}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    summary["reference"] = {
        "status": "completed",
        "result": str(reference_path.resolve()),
        "console_log": log,
        "test_scenarios_evaluated": reference["test_scenarios_evaluated"],
        "no_storage_objective": reference["no_storage_reference"]["objective"],
        "perfect_information_objective": (
            reference["perfect_information_reference"]["objective"]
            if reference.get("perfect_information_reference") is not None
            else None
        ),
    }
    _write_summary(summary_path, summary)

    print(
        f"Running {len(jobs)} baseline job(s), {args.parallel_runs} at a time; "
        f"K={args.support_scenarios}, test N={reference['test_scenarios_evaluated']}",
        flush=True,
    )

    def launch(index: int) -> tuple[int, int, str]:
        job = jobs[index]
        command = _worker_command(
            args,
            Path(job["result"]),
            kind="baseline",
            rule=job["rule"],
            seed=job["seed"],
        )
        code, job_log = _run_subprocess(command, Path(job["console_log"]))
        return index, code, job_log

    failures = 0
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        futures = {executor.submit(launch, index): index for index in range(len(jobs))}
        for future in as_completed(futures):
            index = futures[future]
            job = jobs[index]
            try:
                _, code, job_log = future.result()
                if code != 0:
                    raise RuntimeError(f"worker exited with code {code}")
                payload = json.loads(Path(job["result"]).read_text(encoding="utf-8"))
                metrics = comparison_metrics(payload, reference)
                job.update(metrics)
                job["status"] = "completed"
                job["test_scenarios"] = payload["test_scenarios_evaluated"]
                job["console_log"] = job_log
            except Exception as exc:
                failures += 1
                job["status"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
            _write_summary(summary_path, summary)
            _write_comparison_csv(comparison_path, jobs)
            print(
                f"[{job['status'].upper()}] {job['rule']} seed={job['seed']} | "
                f"storage_value={job.get('storage_value', 'n/a')} | "
                f"log={job['console_log']}",
                flush=True,
            )

    print(f"Suite summary: {summary_path.resolve()}", flush=True)
    print(f"Comparison CSV: {comparison_path.resolve()}", flush=True)
    if failures and not args.continue_on_error:
        raise SystemExit(f"{failures} baseline job(s) failed; inspect summary and logs.")


if __name__ == "__main__":
    main()
