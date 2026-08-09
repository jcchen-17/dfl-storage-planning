"""Build, train, save and load conditional generators by name."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from storage_dfl.config import ExperimentConfig
from storage_dfl.models.base import GENERATOR_KINDS, ConditionalGenerator, GeneratorEpoch
from storage_dfl.models.cvae import ConditionalVAE, train_cvae


def _check_kind(kind: str) -> str:
    if kind not in GENERATOR_KINDS:
        raise ValueError(f"generator.kind must be one of {GENERATOR_KINDS}, got {kind!r}.")
    return kind


def build_generator(
    config: ExperimentConfig,
    trajectory_dim: int,
    context_dim: int,
) -> ConditionalGenerator:
    kind = _check_kind(config.generator.kind)
    shared = config.cvae
    return ConditionalVAE(
        trajectory_dim=trajectory_dim,
        context_dim=context_dim,
        latent_dim=shared.latent_dim,
        hidden_dim=shared.hidden_dim,
    )


def train_generator(
    model: ConditionalGenerator,
    trajectories: np.ndarray,
    contexts: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    *,
    trajectory_mean: np.ndarray | None = None,
    trajectory_std: np.ndarray | None = None,
    field_masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    writer: Any | None = None,
    validation_trajectories: np.ndarray | None = None,
    validation_contexts: np.ndarray | None = None,
) -> tuple[GeneratorEpoch, ...]:
    shared = dict(
        trajectories=trajectories,
        contexts=contexts,
        config=config.cvae,
        horizon=config.data.horizon,
        device=device,
        seed=config.seed,
        trajectory_mean=trajectory_mean,
        trajectory_std=trajectory_std,
        field_masks=field_masks,
        writer=writer,
    )
    if not isinstance(model, ConditionalVAE):
        raise TypeError(f"Unsupported generator type: {type(model).__name__}")
    return train_cvae(
        model,
        validation_trajectories=validation_trajectories,
        validation_contexts=validation_contexts,
        **shared,
    )


def save_generator(model: ConditionalGenerator, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": model.kind,
            "state_dict": model.state_dict(),
            **model.checkpoint_payload(),
        },
        path,
    )


def generator_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
) -> ConditionalGenerator:
    # Checkpoints written before generators became pluggable have no ``kind``
    # and are always CVAEs.
    kind = _check_kind(str(checkpoint.get("kind", "cvae")))
    model: ConditionalGenerator = ConditionalVAE(
        trajectory_dim=int(checkpoint["trajectory_dim"]),
        context_dim=int(checkpoint["context_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.freeze()
    return model
