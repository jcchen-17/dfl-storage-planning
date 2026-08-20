"""Train independent Full-DFL K=8 carbon-accounting variants."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    pairs = {
        (item["carbon_formulation"], item["carbon_cap_scope"])
        for item in manifest["variants"]
    }
    allowed = {
        ("layered_pcc", "hourly"),
        ("layered_pcc", "horizon"),
        ("average_pcc", "hourly"),
        ("average_pcc", "horizon"),
    }
    if not pairs or not pairs.issubset(allowed) or len(pairs) != len(manifest["variants"]):
        raise ValueError("Variants must be unique supported accounting/scope pairs")
    return manifest


def _completed_checkpoint(output_dir: Path) -> Path | None:
    latest_path = output_dir / "latest.json"
    if not latest_path.exists():
        return None
    checkpoint = Path(_read_json(latest_path)["checkpoint"])
    return checkpoint if checkpoint.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="train only; the suite can be evaluated later",
    )
    args = parser.parse_args()

    manifest_path = args.config.resolve()
    manifest = _load_manifest(manifest_path)
    base_path = _resolve(manifest["base_config"])
    with base_path.open("r", encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    dataset_path = Path(base["data"]["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = (base_path.parent.parent / dataset_path).resolve()
    if int(base["dfl"]["num_support_scenarios"]) != 8:
        raise ValueError("The base configuration must use K=8")
    pretrained_dir = _resolve(manifest["pretrained_artifacts"])
    pretrained_checkpoint = pretrained_dir / "cvae.pt"
    pretrained_normalization = pretrained_dir / "normalization.json"
    for required in (pretrained_checkpoint, pretrained_normalization):
        if not required.exists():
            raise FileNotFoundError(f"Shared pretrained artifact not found: {required}")

    suite_root = _resolve(manifest["output_dir"])
    configs_dir = suite_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    compute = manifest["compute"]
    base["costs"].update(manifest.get("costs", {}))
    base["planning"].update(manifest.get("planning", {}))
    records = []
    for variant in manifest["variants"]:
        effective = yaml.safe_load(yaml.safe_dump(base, sort_keys=False))
        effective["data"]["dataset_path"] = str(dataset_path)
        effective["planning"]["carbon_formulation"] = variant["carbon_formulation"]
        effective["planning"]["carbon_cap_scope"] = variant["carbon_cap_scope"]
        effective["planning"]["solver_max_parallel_workers"] = int(
            compute["solver_workers"]
        )
        effective["planning"]["solver_threads"] = int(compute["solver_threads"])
        effective["planning"]["solver_memory_limit_mb"] = float(
            compute["solver_memory_limit_mb"]
        )
        effective["dfl"]["initialize_from_pretrained"] = True
        output_dir = (suite_root / variant["id"]).resolve()
        effective["output_dir"] = str(output_dir)
        config_path = (configs_dir / f"{variant['id']}.yaml").resolve()
        config_path.write_text(
            yaml.safe_dump(effective, sort_keys=False), encoding="utf-8"
        )
        records.append(
            {
                **variant,
                "status": "queued",
                "config": str(config_path),
                "output_dir": str(output_dir),
            }
        )

    summary_path = suite_root / "summary.json"
    summary = {
        "experiment_config": str(manifest_path),
        "base_config": str(base_path),
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "pretrained_normalization": str(pretrained_normalization),
        "compute": compute,
        "evaluation": manifest["evaluation"],
        "runs": records,
    }
    _write_json(summary_path, summary)

    for index, record in enumerate(records, start=1):
        output_dir = Path(record["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = _completed_checkpoint(output_dir)
        if checkpoint is not None:
            record.update(status="completed", checkpoint=str(checkpoint))
            print(
                f"[{index}/{len(records)}] Resume completed {record['label']}",
                flush=True,
            )
            _write_json(summary_path, summary)
            continue
        shutil.copy2(pretrained_checkpoint, output_dir / "cvae.pt")
        shutil.copy2(pretrained_normalization, output_dir / "normalization.json")
        run_id = f"{record['id']}_{time.strftime('%Y%m%d-%H%M%S')}"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_dfl.py"),
            "--config",
            record["config"],
            "--run-id",
            run_id,
        ]
        if args.no_swanlab:
            command.append("--no-swanlab")
        record["status"] = "running"
        record["run_id"] = run_id
        _write_json(summary_path, summary)
        print(
            f"[{index}/{len(records)}] Train {record['label']} with independent "
            "Full DFL K=8...",
            flush=True,
        )
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            record.update(status="failed", returncode=completed.returncode)
            _write_json(summary_path, summary)
            raise SystemExit(
                f"{record['label']} failed with exit code {completed.returncode}"
            )
        checkpoint = _completed_checkpoint(output_dir)
        if checkpoint is None:
            raise RuntimeError(f"{record['label']} completed without latest checkpoint")
        record.update(status="completed", checkpoint=str(checkpoint), returncode=0)
        _write_json(summary_path, summary)

    if args.skip_evaluation:
        print(f"All DFL runs completed. Suite: {suite_root}", flush=True)
        return
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_carbon_accounting_dfl_suite.py"),
        "--suite",
        str(suite_root),
    ]
    if args.no_swanlab:
        command.append("--no-swanlab")
    print("All training completed; starting common strict evaluation...", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"Suite evaluation failed with exit code {completed.returncode}")


if __name__ == "__main__":
    main()
