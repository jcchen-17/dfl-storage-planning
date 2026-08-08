"""Audit structural, statistical and physical invariants of dataset v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from storage_dfl.network import ieee13_unbalanced_microgrid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/ieee13_smartds_dfl_v2/dfl_scenarios_v2_48h.npz"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _trajectory_digest(arrays: np.lib.npyio.NpzFile, index: int) -> str:
    digest = hashlib.sha256()
    for field in (
        "active_load_phase_mw",
        "reactive_load_phase_mvar",
        "pv_available_phase_mw",
        "workload_arrival",
        "pue",
        "grid_price_per_mwh",
        "grid_carbon_t_per_mwh",
        "grid_available",
    ):
        digest.update(np.round(arrays[field][index].astype(np.float64), 7).tobytes())
    return digest.hexdigest()


def _effective_rank(matrix: np.ndarray) -> tuple[float, int]:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    power = np.square(singular)
    if power.sum() <= 0.0:
        return 0.0, 0
    effective = float(power.sum() ** 2 / np.square(power).sum())
    components_90 = int(np.searchsorted(np.cumsum(power) / power.sum(), 0.90) + 1)
    return effective, components_90


def _features(arrays: np.lib.npyio.NpzFile, indices: np.ndarray) -> np.ndarray:
    active = arrays["active_load_phase_mw"][indices].sum(axis=(2, 3))
    pv = arrays["pv_available_phase_mw"][indices].sum(axis=(2, 3))
    net = active - pv
    carbon = arrays["grid_carbon_t_per_mwh"][indices]
    price = arrays["grid_price_per_mwh"][indices]
    return np.stack(
        (
            net.max(axis=1),
            np.quantile(net, 0.95, axis=1),
            carbon.mean(axis=1),
            np.ptp(carbon, axis=1),
            np.ptp(price, axis=1),
        ),
        axis=1,
    )


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else dataset.parent / "audit_v2.json"
    )
    arrays = np.load(dataset)
    required = {
        "dataset_version", "scenario_name", "split", "sample_weight", "group_id",
        "start_timestamp", "context", "active_load_phase_mw",
        "reactive_load_phase_mvar", "pv_available_phase_mw", "workload_arrival",
        "pue", "grid_price_per_mwh", "grid_carbon_t_per_mwh", "grid_available",
        "deterministic_fields", "stochastic_fields",
    }
    missing = sorted(required - set(arrays.files))
    if missing:
        raise ValueError(f"Missing required v2 fields: {missing}")

    splits = arrays["split"].astype(str)
    names = arrays["scenario_name"].astype(str)
    starts = arrays["start_timestamp"].astype("datetime64[h]")
    horizon = int(arrays["grid_carbon_t_per_mwh"].shape[1])
    feeder = ieee13_unbalanced_microgrid()

    checks: dict[str, bool] = {}
    checks["unique_names"] = len(set(names.tolist())) == len(names)
    checks["unique_groups"] = len(set(arrays["group_id"].astype(str).tolist())) == len(names)
    checks["finite_numeric_arrays"] = all(
        np.all(np.isfinite(arrays[field]))
        for field in (
            "context", "sample_weight", "active_load_phase_mw",
            "reactive_load_phase_mvar", "pv_available_phase_mw", "workload_arrival",
            "pue", "grid_price_per_mwh", "grid_carbon_t_per_mwh", "grid_available",
        )
    )
    checks["nonnegative_physical_fields"] = all(
        np.all(arrays[field] >= 0.0)
        for field in (
            "active_load_phase_mw", "reactive_load_phase_mvar",
            "pv_available_phase_mw", "workload_arrival", "grid_price_per_mwh",
            "grid_carbon_t_per_mwh", "grid_available",
        )
    )
    checks["pue_bounds"] = bool(np.all((arrays["pue"] >= 1.15) & (arrays["pue"] <= 1.35)))
    checks["availability_binary"] = bool(
        np.all(np.isin(arrays["grid_available"], (0.0, 1.0)))
    )

    overlap_violations = []
    for split in np.unique(splits):
        ordered = np.sort(starts[splits == split])
        if len(ordered) > 1:
            gaps = np.diff(ordered).astype("timedelta64[h]").astype(int)
            overlap_violations.extend(gaps[gaps < horizon].tolist())
    checks["no_window_overlap"] = not overlap_violations

    weight_sums = {
        str(split): float(arrays["sample_weight"][splits == split].sum())
        for split in np.unique(splits)
    }
    checks["weights_sum_to_one_by_split"] = all(
        abs(value - 1.0) <= 1.0e-6 for value in weight_sums.values()
    )

    expected_ratio = np.divide(
        feeder.base_reactive_load_mvar,
        feeder.base_active_load_mw,
        out=np.zeros_like(feeder.base_reactive_load_mvar),
        where=feeder.base_active_load_mw > 0.0,
    )
    active = arrays["active_load_phase_mw"]
    reactive = arrays["reactive_load_phase_mvar"]
    q_error = np.max(np.abs(reactive - active * expected_ratio[None, None, :, :]))
    checks["reactive_power_factor_consistent"] = bool(q_error <= 1.0e-6)

    digests = {
        split: {
            _trajectory_digest(arrays, int(index))
            for index in np.flatnonzero(splits == split)
        }
        for split in np.unique(splits)
    }
    duplicate_trajectories = {}
    split_names = sorted(digests)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            duplicate_trajectories[f"{left}__{right}"] = len(
                digests[left] & digests[right]
            )
    checks["no_exact_cross_split_trajectories"] = all(
        count == 0 for count in duplicate_trajectories.values()
    )

    feature_names = (
        "peak_net_load_mw", "p95_net_load_mw", "carbon_mean_t_per_mwh",
        "carbon_spread_t_per_mwh", "price_spread_dollars_per_mwh",
    )
    summaries = {}
    effective_ranks = {}
    for split in np.unique(splits):
        indices = np.flatnonzero(splits == split)
        values = _features(arrays, indices)
        summaries[str(split)] = {
            name: {
                "mean": float(values[:, column].mean()),
                "std": float(values[:, column].std()),
                "p05": float(np.quantile(values[:, column], 0.05)),
                "p95": float(np.quantile(values[:, column], 0.95)),
            }
            for column, name in enumerate(feature_names)
        }
        stochastic_flat = np.concatenate(
            (
                arrays["active_load_phase_mw"][indices].reshape(len(indices), -1),
                arrays["pv_available_phase_mw"][indices].reshape(len(indices), -1),
                arrays["grid_carbon_t_per_mwh"][indices].reshape(len(indices), -1),
            ),
            axis=1,
        )
        effective, components_90 = _effective_rank(stochastic_flat)
        effective_ranks[str(split)] = {
            "effective_rank": effective,
            "components_for_90_percent_variance": components_90,
            "scenario_count": int(len(indices)),
        }

    report = {
        "dataset": str(dataset),
        "dataset_version": str(arrays["dataset_version"]),
        "scenario_count": int(len(names)),
        "scenario_counts": {
            str(split): int(np.count_nonzero(splits == split))
            for split in np.unique(splits)
        },
        "horizon_hours": horizon,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "overlap_violations_hours": overlap_violations,
        "weight_sums": weight_sums,
        "maximum_q_reconstruction_error": float(q_error),
        "duplicate_trajectories": duplicate_trajectories,
        "deterministic_fields": arrays["deterministic_fields"].astype(str).tolist(),
        "stochastic_fields": arrays["stochastic_fields"].astype(str).tolist(),
        "feature_summary": summaries,
        "effective_rank": effective_ranks,
    }
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    for split, values in effective_ranks.items():
        print(
            f"{split}: n={values['scenario_count']}, "
            f"effective_rank={values['effective_rank']:.1f}, "
            f"pc90={values['components_for_90_percent_variance']}"
        )
    print(f"audit: {output}")
    if not report["all_checks_pass"]:
        raise SystemExit("dataset v2 audit failed")


if __name__ == "__main__":
    main()
