"""Stage one of the generator comparison: generative quality, no SCIP solves.

Trains (or reuses) a CVAE, a conditional WGAN-GP and a conditional diffusion
model on the same training split with the same codec, then scores each of them
against the held-out split.  Run this before spending solver time so that models
which cannot even reproduce the scenario distribution are removed cheaply, and
only the survivors go through the full DFL pipeline.

    python scripts/compare_generators.py --config configs/demo.yaml
    python scripts/compare_generators.py --config configs/demo.yaml --skip-training
    python scripts/compare_generators.py --config configs/demo.yaml --generators cvae diffusion
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from storage_dfl.config import load_config
from storage_dfl.data import ScenarioCodec, ScenarioPool
from storage_dfl.dfl import resolve_device
from storage_dfl.models import GENERATOR_KINDS, ConditionalGenerator
from storage_dfl.models.metrics import (
    DECISION_STATISTICS,
    autocorrelation_error,
    clipping_report,
    decision_statistics,
    energy_distance,
    field_reconstruction_errors,
    precision_recall,
    project_to_real_subspace,
    wasserstein_1d,
)
from storage_dfl.stages import (
    ArtifactPaths,
    _experiment_data,
    _load_codec,
    load_generator,
    train_generator_stage,
)

SUMMARY_COLUMNS = (
    "generator",
    "latent_dim",
    "train_seconds",
    "sample_seconds",
    "reconstruction_all_fields",
    "decision_wasserstein_mean",
    "peak_net_load_mw_wasserstein",
    "price_spread_per_mwh_wasserstein",
    "energy_distance",
    "autocorrelation_error",
    "precision",
    "recall",
    "shape_precision",
    "shape_recall",
    "clipped_mass",
)


def _subsample(matrix: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if limit <= 0 or matrix.shape[0] <= limit:
        return matrix
    return matrix[rng.choice(matrix.shape[0], size=limit, replace=False)]


def _samples_per_condition(
    model: ConditionalGenerator,
    codec: ScenarioCodec,
    contexts: np.ndarray,
    device: torch.device,
    repeats: int,
    seed: int,
):
    """Draw prior samples paired with the held-out contexts.

    Conditioning the generated set on exactly the observed contexts removes
    context mismatch from the comparison: any distance that remains is the
    model's, not the conditioning distribution's.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    tiled = np.tile(contexts, (repeats, 1))
    conditions = torch.as_tensor(tiled, dtype=torch.float32, device=device)
    latent = model.sample_latent(
        tiled.shape[0], generator=generator
    ).to(device=device, dtype=torch.float32)
    started = time.perf_counter()
    with torch.no_grad():
        decoded = model.decode(latent, conditions).cpu().numpy()
    elapsed = time.perf_counter() - started
    scenarios = codec.decode_batch(decoded, tiled, name_prefix=f"{model.kind}_sample")
    return decoded, scenarios, elapsed


def _reconstruction(
    model: ConditionalGenerator,
    trajectories: np.ndarray,
    contexts: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x = torch.as_tensor(trajectories, dtype=torch.float32, device=device)
    c = torch.as_tensor(contexts, dtype=torch.float32, device=device)
    with torch.no_grad():
        latent, _ = model.encode(x, c)
        return model.decode(latent, c).cpu().numpy()


def evaluate_generator(
    kind: str,
    config_path: str,
    *,
    skip_training: bool,
    repeats: int,
    held_out_limit: int,
    metric_subset: int,
    tensorboard: bool,
) -> dict:
    config = load_config(config_path)
    paths = ArtifactPaths(config.output_dir)
    train_seconds = 0.0
    if not skip_training:
        started = time.perf_counter()
        train_generator_stage(
            config_path, tensorboard=tensorboard, generator_override=kind
        )
        train_seconds = time.perf_counter() - started

    device = resolve_device(config.dfl.device)
    feeder, full_held_out = _experiment_data(config, config.data.test_split)
    codec = _load_codec(paths, feeder)
    # The held-out split has thousands of windows and a diffusion decode costs one
    # network pass per DDIM step, so the comparison runs on a diverse subset
    # chosen by the same farthest-point rule the evaluation stage uses.
    if held_out_limit > 0 and held_out_limit < len(full_held_out.scenarios):
        indices = codec.support_indices(full_held_out, held_out_limit)
        held_out = ScenarioPool(full_held_out.subset(indices.tolist()))
    else:
        held_out = full_held_out
    model = load_generator(paths, kind, device)
    trajectories, contexts = codec.encode_pool(held_out)

    reconstruction = _reconstruction(model, trajectories, contexts, device)
    reconstruction_errors = field_reconstruction_errors(
        trajectories, reconstruction, codec, config.data.horizon
    )

    decoded, generated, sample_seconds = _samples_per_condition(
        model, codec, contexts, device, repeats, config.seed
    )
    generated_pool = ScenarioPool(tuple(generated))

    real_statistics = decision_statistics(held_out.scenarios)
    generated_statistics = decision_statistics(generated_pool.scenarios)
    # Each statistic is scaled by the observed spread so distances across
    # quantities with different units can be averaged into one number.
    wasserstein: dict[str, float] = {}
    for name in DECISION_STATISTICS:
        scale = max(float(real_statistics[name].std()), 1.0e-6)
        wasserstein[name] = (
            wasserstein_1d(real_statistics[name], generated_statistics[name]) / scale
        )

    generated_trajectories, _ = codec.encode_pool(generated_pool)
    # Energy distance and precision/recall are quadratic in the sample count and
    # operate on full trajectories, so they run on a capped random subset. The
    # Wasserstein distances above are linearithmic and use everything.
    rng = np.random.default_rng(config.seed)
    real_subset = _subsample(trajectories, metric_subset, rng)
    generated_subset = _subsample(generated_trajectories, metric_subset, rng)
    # Two coverage readings. The decision space is the interpretable one: does the
    # generator span the range of conditions the planner prices? The PCA space is
    # the shape reading. Raw trajectories are unusable here -- see
    # project_to_real_subspace.
    real_decision = np.stack([real_statistics[name] for name in DECISION_STATISTICS], axis=1)
    generated_decision = np.stack(
        [generated_statistics[name] for name in DECISION_STATISTICS], axis=1
    )
    decision_scale = np.maximum(real_decision.std(axis=0, keepdims=True), 1.0e-8)
    precision, recall = precision_recall(
        real_decision / decision_scale,
        _subsample(generated_decision / decision_scale, metric_subset, rng),
    )
    shape_precision, shape_recall = precision_recall(
        *project_to_real_subspace(real_subset, generated_subset)
    )
    clipping = clipping_report(codec, decoded, generated_pool.scenarios)

    return {
        "generator": kind,
        "latent_dim": int(model.latent_dim),
        "train_seconds": train_seconds,
        "sample_seconds": sample_seconds,
        "held_out_scenarios": len(held_out.scenarios),
        "generated_scenarios": len(generated_pool.scenarios),
        "reconstruction": reconstruction_errors,
        "decision_wasserstein": wasserstein,
        "decision_wasserstein_mean": float(np.mean(list(wasserstein.values()))),
        "energy_distance": energy_distance(real_subset, generated_subset),
        "autocorrelation_error": autocorrelation_error(
            held_out.scenarios, generated_pool.scenarios
        ),
        "precision": precision,
        "recall": recall,
        "shape_precision": shape_precision,
        "shape_recall": shape_recall,
        "clipped_mass": clipping.relative_mass,
        "clipped_mass_max": clipping.maximum_relative_mass,
    }


def _summary_row(report: dict) -> dict:
    return {
        "generator": report["generator"],
        "latent_dim": report["latent_dim"],
        "train_seconds": round(report["train_seconds"], 1),
        "sample_seconds": round(report["sample_seconds"], 3),
        "reconstruction_all_fields": round(report["reconstruction"]["all_fields"], 5),
        "decision_wasserstein_mean": round(report["decision_wasserstein_mean"], 5),
        "peak_net_load_mw_wasserstein": round(
            report["decision_wasserstein"]["peak_net_load_mw"], 5
        ),
        "price_spread_per_mwh_wasserstein": round(
            report["decision_wasserstein"]["price_spread_per_mwh"], 5
        ),
        "energy_distance": round(report["energy_distance"], 5),
        "autocorrelation_error": round(report["autocorrelation_error"], 5),
        "precision": round(report["precision"], 4),
        "recall": round(report["recall"], 4),
        "shape_precision": round(report["shape_precision"], 4),
        "shape_recall": round(report["shape_recall"], 4),
        "clipped_mass": round(report["clipped_mass"], 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument(
        "--generators",
        nargs="+",
        choices=GENERATOR_KINDS,
        default=list(GENERATOR_KINDS),
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Score existing checkpoints instead of retraining each generator.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="Prior samples drawn per held-out context.",
    )
    parser.add_argument(
        "--held-out-limit",
        type=int,
        default=400,
        help="Cap on held-out scenarios scored; 0 uses the whole split.",
    )
    parser.add_argument(
        "--metric-subset",
        type=int,
        default=300,
        help="Cap for the quadratic metrics (energy distance, precision/recall).",
    )
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    reports = []
    for kind in args.generators:
        print(f"\n=== {kind} ===", flush=True)
        reports.append(
            evaluate_generator(
                kind,
                args.config,
                skip_training=args.skip_training,
                repeats=args.repeats,
                held_out_limit=args.held_out_limit,
                metric_subset=args.metric_subset,
                tensorboard=not args.no_tensorboard,
            )
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "generator_comparison.json").open("w", encoding="utf-8") as stream:
        json.dump(reports, stream, indent=2, ensure_ascii=False)
    rows = [_summary_row(report) for report in reports]
    with (output_dir / "generator_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(SUMMARY_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + " | ".join(f"{name:>28}" for name in SUMMARY_COLUMNS))
    for row in rows:
        print(" | ".join(f"{str(row[name]):>28}" for name in SUMMARY_COLUMNS))
    print(f"\nWrote {output_dir / 'generator_comparison.csv'}")
    print(
        "Lower is better for every distance; higher is better for precision and recall."
    )


if __name__ == "__main__":
    main()
