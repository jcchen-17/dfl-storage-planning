"""Audit train/validation/test scenario splits without changing any split.

This is deliberately diagnostic. It detects leakage and distribution shift;
it never selects, rescales, or overwrites test scenarios based on model results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from storage_dfl.config import load_config
from storage_dfl.data import Scenario, load_historical_scenarios


FEATURE_NAMES = (
    "peak_load_mw",
    "peak_net_load_mw",
    "p95_net_load_mw",
    "pv_energy_mwh",
    "price_spread_per_mwh",
    "price_p95_per_mwh",
    "carbon_p95_t_per_mwh",
    "workload_p95",
    "unavailable_hours",
)


def _features(scenario: Scenario, delta_t_hours: float) -> np.ndarray:
    load = scenario.active_load_mw.sum(axis=(1, 2))
    pv = scenario.pv_available_mw.sum(axis=(1, 2))
    net = load - pv
    return np.asarray(
        (
            load.max(),
            net.max(),
            np.quantile(net, 0.95),
            pv.sum() * delta_t_hours,
            np.ptp(scenario.grid_price_per_mwh),
            np.quantile(scenario.grid_price_per_mwh, 0.95),
            np.quantile(scenario.grid_carbon_t_per_mwh, 0.95),
            np.quantile(scenario.workload_arrival, 0.95),
            np.sum(scenario.grid_available < 0.5) * delta_t_hours,
        ),
        dtype=float,
    )


def _trajectory_hash(scenario: Scenario) -> str:
    arrays = (
        scenario.context,
        scenario.active_load_mw,
        scenario.reactive_load_mvar,
        scenario.pv_available_mw,
        scenario.workload_arrival,
        scenario.pue,
        scenario.grid_price_per_mwh,
        scenario.grid_carbon_t_per_mwh,
        scenario.grid_available,
    )
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.round(np.asarray(array, dtype=np.float64), 6).tobytes())
    return digest.hexdigest()


def _summary(matrix: np.ndarray) -> dict[str, dict[str, float]]:
    statistics = {}
    for index, name in enumerate(FEATURE_NAMES):
        values = matrix[:, index]
        statistics[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p05": float(np.quantile(values, 0.05)),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
        }
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check scenario splits for leakage and distribution shift."
    )
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument(
        "--output",
        help="JSON destination (default: <config output_dir>/split_audit.json).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    pools = {
        split: load_historical_scenarios(
            config.data.dataset_path,
            split=split,
            horizon=config.data.horizon,
        )
        for split in ("train", "validation", "test")
    }
    matrices = {
        split: np.stack(
            [_features(item, config.data.delta_t_hours) for item in pool.scenarios]
        )
        for split, pool in pools.items()
    }
    name_sets = {
        split: {scenario.name for scenario in pool.scenarios}
        for split, pool in pools.items()
    }
    hash_sets = {
        split: {_trajectory_hash(scenario) for scenario in pool.scenarios}
        for split, pool in pools.items()
    }
    leakage = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        key = f"{left}__{right}"
        leakage[key] = {
            "duplicate_names": sorted(name_sets[left] & name_sets[right]),
            "duplicate_trajectories": len(hash_sets[left] & hash_sets[right]),
        }

    train_mean = matrices["train"].mean(axis=0)
    train_std = np.maximum(matrices["train"].std(axis=0), 1.0e-9)
    standardized_mean_shift = {}
    for split in ("validation", "test"):
        shift = (matrices[split].mean(axis=0) - train_mean) / train_std
        standardized_mean_shift[split] = {
            name: float(value) for name, value in zip(FEATURE_NAMES, shift, strict=True)
        }

    report = {
        "config": str(Path(args.config).resolve()),
        "dataset": str(config.data.dataset_path),
        "horizon": config.data.horizon,
        "scenario_counts": {split: len(pool.scenarios) for split, pool in pools.items()},
        "leakage": leakage,
        "feature_summary": {split: _summary(matrix) for split, matrix in matrices.items()},
        "standardized_mean_shift_from_train": standardized_mean_shift,
        "large_shift_threshold": 0.5,
        "large_shifts": {
            split: {
                name: value
                for name, value in shifts.items()
                if abs(value) >= 0.5
            }
            for split, shifts in standardized_mean_shift.items()
        },
    }
    output = Path(args.output) if args.output else config.output_dir / "split_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"scenario counts: {report['scenario_counts']}")
    print(f"leakage: {leakage}")
    print("standardized mean shifts from train (absolute value >= 0.5 shown):")
    for split, shifts in report["large_shifts"].items():
        print(f"  {split}: {shifts or 'none'}")
    print(f"audit: {output.resolve()}")


if __name__ == "__main__":
    main()
