"""Batch-run learning-objective baselines from one shared pretrained CVAE.

USAGE (run from the repository root in the storage-dfl environment)

Core comparison, sequential:
  python scripts/run_learning_objective_baselines.py --config configs/dataset_v2_dfl_hourly_layered_t4.yaml

Core comparison, two experiments in parallel (recommended):
  python scripts/run_learning_objective_baselines.py --config configs/dataset_v2_dfl_hourly_layered_t4.yaml --parallel-runs 2

All learning-objective ablations, at most two in parallel:
  python scripts/run_learning_objective_baselines.py --config configs/dataset_v2_dfl_hourly_layered_t4.yaml --variants all --parallel-runs 2

Selected variants and multiple seeds:
  python scripts/run_learning_objective_baselines.py --config configs/dataset_v2_dfl_hourly_layered_t4.yaml --variants cvae_only dfl_full ipl_only opl_only --seeds 20260808 20260809 20260810 --parallel-runs 2

Training only, without final test-set evaluation:
  python scripts/run_learning_objective_baselines.py --config configs/dataset_v2_dfl_hourly_layered_t4.yaml --parallel-runs 2 --skip-evaluation

For every seed, the script first trains one CVAE for ``cvae.epochs`` complete
passes over the training split. Every variant for that seed starts from that
exact checkpoint. ``cvae_only`` performs no DFL updates and proceeds directly
to planning/evaluation; the remaining variants fine-tune the pretrained CVAE.

The default variants are cvae_only and dfl_full. Parallel jobs run in isolated
Python processes and write separate console.log files. When --parallel-runs is
greater than one, solver_max_parallel_workers is automatically divided among
the concurrent experiments so nested MILP workers do not multiply unchecked.
Override that limit with --solver-workers-per-run when required.

Outputs:
  outputs/baselines/learning_objective/<config>/
    suites/<timestamp>/{summary.json,configs/,logs/,pretrained/}
    <variant>/seed_<seed>/{latest.json,runs/<timestamp>/...}

Variants:
  cvae_only               lambda=0; statistical CVAE loss only
  dfl_full                IPL+OPL; alpha=0.5
  ipl_only                alpha=1.0
  opl_only                alpha=0.0
  no_gradient_balance     full DFL without dynamic gradient balancing
  no_reconstruction_reg   statistical regularization is zero after warm-up
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


DEFAULT_VARIANTS = ("cvae_only", "dfl_full")
VARIANT_OVERRIDES: dict[str, dict[str, float]] = {
    "cvae_only": {
        "lambda_dfl": 0.0,
        "infeasibility_aversion_alpha": 0.5,
        "gradient_balance_ratio": 0.0,
        "dfl_statistical_weight": 1.0,
    },
    "dfl_full": {
        "lambda_dfl": 1.0,
        "infeasibility_aversion_alpha": 0.5,
    },
    "ipl_only": {
        "lambda_dfl": 1.0,
        "infeasibility_aversion_alpha": 1.0,
    },
    "opl_only": {
        "lambda_dfl": 1.0,
        "infeasibility_aversion_alpha": 0.0,
    },
    "no_gradient_balance": {
        "lambda_dfl": 1.0,
        "infeasibility_aversion_alpha": 0.5,
        "gradient_balance_ratio": 0.0,
    },
    "no_reconstruction_reg": {
        "lambda_dfl": 1.0,
        "infeasibility_aversion_alpha": 0.5,
        "dfl_statistical_weight": 0.0,
    },
}


def _config_label(path: Path) -> str:
    prefix = "dataset_v2_dfl_"
    return path.stem[len(prefix):] if path.stem.startswith(prefix) else path.stem


def _unique(values: list) -> list:
    return list(dict.fromkeys(values))


def _parse_variants(values: list[str]) -> list[str]:
    if values == ["all"]:
        return list(VARIANT_OVERRIDES)
    values = _unique(values)
    unknown = sorted(set(values) - set(VARIANT_OVERRIDES))
    if unknown:
        raise ValueError(
            f"Unknown variants: {unknown}. Choose from {sorted(VARIANT_OVERRIDES)} or all."
        )
    return values


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _run_job(
    job: dict,
    *,
    project_root: Path,
    skip_evaluation: bool,
    no_tensorboard: bool,
) -> dict:
    """Run one isolated train/evaluate pair and return its summary record."""

    record = dict(job["record"])
    config_path = Path(job["config"])
    log_path = Path(job["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    train_command = [
        sys.executable,
        str(project_root / "scripts" / "train_dfl.py"),
        "--config",
        str(config_path),
    ]
    if no_tensorboard:
        train_command.append("--no-tensorboard")

    with log_path.open("w", encoding="utf-8") as log:
        log.write("TRAIN COMMAND\n" + subprocess.list2cmdline(train_command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            train_command,
            cwd=project_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode != 0:
            record["status"] = "failed"
            record["error"] = f"training exited with code {completed.returncode}"
            record["console_log"] = str(log_path)
            return record

        latest_path = Path(record["output_dir"]) / "latest.json"
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        record["checkpoint"] = latest["checkpoint"]
        record["training_result"] = latest["result"]

        if not skip_evaluation:
            evaluate_command = [
                sys.executable,
                str(project_root / "scripts" / "evaluate.py"),
                "--config",
                str(config_path),
                "--checkpoint",
                latest["checkpoint"],
            ]
            log.write(
                "\nEVALUATION COMMAND\n"
                + subprocess.list2cmdline(evaluate_command)
                + "\n\n"
            )
            log.flush()
            completed = subprocess.run(
                evaluate_command,
                cwd=project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if completed.returncode != 0:
                record["status"] = "failed"
                record["error"] = f"evaluation exited with code {completed.returncode}"
                record["console_log"] = str(log_path)
                return record
            record["evaluation"] = str(
                Path(latest["checkpoint"]).with_name("evaluation.json")
            )

    record["status"] = "completed"
    record["console_log"] = str(log_path)
    return record


def _run_pretraining_job(
    job: dict,
    *,
    project_root: Path,
    no_tensorboard: bool,
) -> dict:
    """Train the single shared CVAE checkpoint used by one seed."""

    record = dict(job["record"])
    config_path = Path(job["config"])
    log_path = Path(job["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(project_root / "scripts" / "train_generator.py"),
        "--config",
        str(config_path),
    ]
    if no_tensorboard:
        command.append("--no-tensorboard")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("PRETRAIN COMMAND\n" + subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=project_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        record["status"] = "failed"
        record["error"] = f"pretraining exited with code {completed.returncode}"
        record["console_log"] = str(log_path)
        return record

    output_dir = Path(record["output_dir"])
    checkpoint = output_dir / "cvae.pt"
    normalization = output_dir / "normalization.json"
    if not checkpoint.exists() or not normalization.exists():
        record["status"] = "failed"
        record["error"] = "pretraining did not produce cvae.pt and normalization.json"
        record["console_log"] = str(log_path)
        return record
    record.update(
        status="completed",
        checkpoint=str(checkpoint),
        normalization=str(normalization),
        console_log=str(log_path),
    )
    return record


def _install_pretrained_artifacts(pretraining: dict, output_dir: Path) -> None:
    """Copy one immutable starting point into an isolated variant directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pretraining["checkpoint"], output_dir / "cvae.pt")
    shutil.copy2(pretraining["normalization"], output_dir / "normalization.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run CVAE-only and DFL learning-objective baselines."
    )
    parser.add_argument(
        "--config",
        default="configs/dataset_v2_dfl_hourly_layered_t4.yaml",
        help="base YAML; all non-learning-objective settings are inherited",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="variant names, or 'all'",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="random seeds; default uses the seed in the base YAML",
    )
    parser.add_argument(
        "--parallel-runs",
        type=int,
        default=1,
        help="number of independent baseline runs to execute concurrently",
    )
    parser.add_argument(
        "--solver-workers-per-run",
        type=int,
        default=None,
        help="override solver_max_parallel_workers for each concurrent run",
    )
    parser.add_argument(
        "--solver-threads-per-run",
        type=int,
        default=None,
        help="override solver_threads for each concurrent run",
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="finish the suite and return success even when a job fails",
    )
    args = parser.parse_args()
    if args.parallel_runs <= 0:
        parser.error("--parallel-runs must be positive")
    if args.solver_workers_per_run is not None and args.solver_workers_per_run <= 0:
        parser.error("--solver-workers-per-run must be positive")
    if args.solver_threads_per_run is not None and args.solver_threads_per_run <= 0:
        parser.error("--solver-threads-per-run must be positive")

    project_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    # Derived YAMLs live below outputs rather than configs. Freeze inherited
    # filesystem paths before load_config resolves them relative to that YAML.
    base["data"]["dataset_path"] = str(
        (config_path.parent.parent / base["data"]["dataset_path"]).resolve()
    )
    variants = _parse_variants(args.variants)
    seeds = _unique(args.seeds or [int(base["seed"])])
    label = _config_label(config_path)
    suite_root = (
        project_root / "outputs" / "baselines" / "learning_objective" / label
    ).resolve()
    suite_id = time.strftime("%Y%m%d-%H%M%S")
    suite_dir = suite_root / "suites" / suite_id
    summary_path = suite_dir / "summary.json"

    base_workers = int(base["planning"]["solver_max_parallel_workers"])
    workers_per_run = args.solver_workers_per_run or max(
        1, base_workers // args.parallel_runs
    )
    threads_per_run = args.solver_threads_per_run or int(
        base["planning"]["solver_threads"]
    )
    jobs: list[dict] = []
    records: list[dict] = []
    pretraining_jobs: list[dict] = []
    pretraining_records: list[dict] = []
    for seed in seeds:
        pretrained = yaml.safe_load(yaml.safe_dump(base))
        pretrained["seed"] = int(seed)
        pretrained_dir = (suite_dir / "pretrained" / f"seed_{seed}").resolve()
        pretrained["output_dir"] = str(pretrained_dir)
        pretrained_config = suite_dir / "configs" / f"pretrain_seed_{seed}.yaml"
        pretrained_config.parent.mkdir(parents=True, exist_ok=True)
        pretrained_config.write_text(
            yaml.safe_dump(pretrained, sort_keys=False), encoding="utf-8"
        )
        pretraining_record = {
            "seed": seed,
            "epochs": int(pretrained["cvae"]["epochs"]),
            "output_dir": str(pretrained_dir),
            "effective_config": str(pretrained_config),
            "status": "queued",
        }
        pretraining_records.append(pretraining_record)
        pretraining_jobs.append(
            {
                "record": pretraining_record,
                "config": str(pretrained_config),
                "log": str(suite_dir / "logs" / f"pretrain_seed_{seed}.log"),
            }
        )
        for variant in variants:
            effective = yaml.safe_load(yaml.safe_dump(base))
            effective["seed"] = int(seed)
            effective["dfl"]["initialize_from_pretrained"] = True
            effective["dfl"]["dfl_start_epoch"] = 0
            effective["dfl"].update(VARIANT_OVERRIDES[variant])
            effective["planning"]["solver_max_parallel_workers"] = workers_per_run
            effective["planning"]["solver_threads"] = threads_per_run
            output_dir = (suite_root / variant / f"seed_{seed}").resolve()
            effective["output_dir"] = str(output_dir)
            config_out = suite_dir / "configs" / f"{variant}_seed_{seed}.yaml"
            config_out.parent.mkdir(parents=True, exist_ok=True)
            config_out.write_text(
                yaml.safe_dump(effective, sort_keys=False), encoding="utf-8"
            )
            record = {
                "variant": variant,
                "seed": seed,
                "output_dir": str(output_dir),
                "effective_config": str(config_out),
                "solver_workers": workers_per_run,
                "solver_threads": threads_per_run,
                "status": "queued",
                "pretrained_seed": seed,
            }
            records.append(record)
            jobs.append(
                {
                    "record": record,
                    "config": str(config_out),
                    "log": str(suite_dir / "logs" / f"{variant}_seed_{seed}.log"),
                }
            )

    summary: dict = {
        "suite_id": suite_id,
        "base_config": str(config_path),
        "variants": variants,
        "seeds": seeds,
        "parallel_runs": args.parallel_runs,
        "solver_workers_per_run": workers_per_run,
        "solver_threads_per_run": threads_per_run,
        "pretraining": pretraining_records,
        "runs": records,
    }
    _write_summary(summary_path, summary)
    print(
        f"Pretraining {len(pretraining_jobs)} shared CVAE checkpoint(s), "
        f"{args.parallel_runs} at a time; cvae.epochs={base['cvae']['epochs']}",
        flush=True,
    )
    pretraining_by_seed: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        futures = {
            executor.submit(
                _run_pretraining_job,
                job,
                project_root=project_root,
                no_tensorboard=args.no_tensorboard,
            ): index
            for index, job in enumerate(pretraining_jobs)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = dict(pretraining_records[index])
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
            pretraining_records[index] = record
            if record["status"] == "completed":
                pretraining_by_seed[int(record["seed"])] = record
            _write_summary(summary_path, summary)
            print(
                f"[{record['status'].upper()}] pretrain seed={record['seed']} | "
                f"log={record.get('console_log', 'n/a')}",
                flush=True,
            )

    runnable_jobs: list[dict] = []
    failures = 0
    for index, job in enumerate(jobs):
        seed = int(job["record"]["seed"])
        pretraining = pretraining_by_seed.get(seed)
        if pretraining is None:
            records[index]["status"] = "blocked"
            records[index]["error"] = f"shared CVAE pretraining failed for seed {seed}"
            failures += 1
            continue
        _install_pretrained_artifacts(
            pretraining, Path(job["record"]["output_dir"])
        )
        runnable_jobs.append(job)
    failures += sum(record["status"] != "completed" for record in pretraining_records)
    _write_summary(summary_path, summary)

    print(
        f"Running {len(runnable_jobs)} variant job(s), {args.parallel_runs} at a time; "
        f"solver workers/run={workers_per_run}, threads/solver={threads_per_run}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        futures = {
            executor.submit(
                _run_job,
                job,
                project_root=project_root,
                skip_evaluation=args.skip_evaluation,
                no_tensorboard=args.no_tensorboard,
            ): index
            for job in runnable_jobs
            for index in [jobs.index(job)]
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = dict(records[index])
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
            records[index] = record
            failures += int(record["status"] != "completed")
            _write_summary(summary_path, summary)
            print(
                f"[{record['status'].upper()}] {record['variant']} "
                f"seed={record['seed']} | log={record.get('console_log', 'n/a')}",
                flush=True,
            )

    print(f"Suite summary: {summary_path}", flush=True)
    if failures and not args.continue_on_error:
        raise SystemExit(f"{failures} baseline job(s) failed; inspect summary and logs.")


if __name__ == "__main__":
    main()
