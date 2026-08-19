"""Wait for a learning-objective suite and evaluate every variant consistently."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _read(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _evaluate(run: dict, args: argparse.Namespace, project_root: Path) -> dict:
    record = {"variant": run["variant"], "seed": run["seed"], "status": "running"}
    latest = _read(Path(run["output_dir"]) / "latest.json")
    checkpoint = Path(latest["checkpoint"])
    output = checkpoint.with_name(f"evaluation_{args.scenarios}.json")
    log = checkpoint.with_name(f"evaluation_{args.scenarios}.log")
    command = [
        sys.executable,
        str(project_root / "scripts" / "evaluate.py"),
        "--config",
        run["effective_config"],
        "--checkpoint",
        str(checkpoint),
        "--scenarios",
        str(args.scenarios),
        "--solver-workers",
        str(args.solver_workers),
        "--solver-threads",
        str(args.solver_threads),
        "--output",
        str(output),
        "--skip-perfect-information",
    ]
    with log.open("w", encoding="utf-8") as stream:
        stream.write(subprocess.list2cmdline(command) + "\n\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=project_root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record.update(
        status="completed" if completed.returncode == 0 else "failed",
        returncode=completed.returncode,
        checkpoint=str(checkpoint),
        evaluation=str(output),
        log=str(log),
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--scenarios", type=int, default=182)
    parser.add_argument("--parallel-runs", type=int, default=2)
    parser.add_argument("--solver-workers", type=int, default=3)
    parser.add_argument("--solver-threads", type=int, default=4)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    suite = args.suite.resolve()
    summary_path = suite / "summary.json"
    output_path = suite / f"evaluation_{args.scenarios}_summary.json"
    print(f"Waiting for training suite {suite}", flush=True)
    while True:
        summary = _read(summary_path)
        statuses = [run.get("status") for run in summary["runs"]]
        if any(status in {"failed", "blocked"} for status in statuses):
            raise SystemExit(f"Training suite failed: {statuses}")
        if all(status == "completed" for status in statuses):
            break
        time.sleep(args.poll_seconds)

    payload = {
        "suite": str(suite),
        "scenarios": args.scenarios,
        "parallel_runs": args.parallel_runs,
        "solver_workers": args.solver_workers,
        "solver_threads": args.solver_threads,
        "runs": [],
    }
    _write(output_path, payload)
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        futures = {
            executor.submit(_evaluate, run, args, project_root): run
            for run in summary["runs"]
        }
        for future in as_completed(futures):
            record = future.result()
            payload["runs"].append(record)
            _write(output_path, payload)
            print(
                f"[{record['status'].upper()}] {record['variant']} | "
                f"{record['evaluation']}",
                flush=True,
            )
    failures = [run for run in payload["runs"] if run["status"] != "completed"]
    if failures:
        raise SystemExit(f"{len(failures)} evaluation(s) failed")
    print(f"Evaluation summary: {output_path}", flush=True)


if __name__ == "__main__":
    main()
