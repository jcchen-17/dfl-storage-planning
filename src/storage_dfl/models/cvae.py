from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from storage_dfl.config import CVAEConfig
from storage_dfl.models.base import (
    ConditionalGenerator,
    GeneratorEpoch,
    ShapeStatistics,
    TrajectoryLayout,
    batched_indices,
    weighted_field_loss,
)


class ConditionalVAE(ConditionalGenerator):
    """Small CVAE for complete multivariate day trajectories."""

    def __init__(
        self,
        trajectory_dim: int,
        context_dim: int,
        latent_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__(trajectory_dim, context_dim, latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(trajectory_dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.encoder_mean = nn.Linear(hidden_dim, latent_dim)
        self.encoder_log_variance = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, trajectory_dim),
        )

    @property
    def kind(self) -> str:
        return "cvae"

    def encode(self, x: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(torch.cat((x, context), dim=-1))
        mean = self.encoder_mean(hidden)
        log_variance = self.encoder_log_variance(hidden).clamp(-8.0, 6.0)
        return mean, log_variance

    def decode(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat((latent, context), dim=-1))

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_variance = self.encode(x, context)
        standard_deviation = torch.exp(0.5 * log_variance)
        latent = mean + standard_deviation * torch.randn_like(standard_deviation)
        return self.decode(latent, context), mean, log_variance

    def checkpoint_payload(self) -> dict:
        return {
            "trajectory_dim": self.trajectory_dim,
            "context_dim": self.context_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
        }


COMPONENT_NAMES = (
    "loss",
    "reconstruction",
    "ramp",
    "kl",
    "net_load",
    "net_peak",
    "price_spread",
    "carbon_spread",
)


def _components(
    model: ConditionalVAE,
    batch_x: torch.Tensor,
    batch_c: torch.Tensor,
    layout: TrajectoryLayout,
    shape: ShapeStatistics,
    field_weights: Any,
    config: CVAEConfig,
    horizon: int,
    effective_beta: float,
    device: torch.device,
    *,
    sample_latent: bool,
) -> dict[str, torch.Tensor]:
    """Every loss component for one batch.

    ``sample_latent`` selects the training estimator (reparameterised draw) or
    the deterministic posterior mean. The mean is what ``encode``/``decode``
    produce downstream, so held-out curves measured with it describe the
    generator as it is actually used rather than one noisy draw from it.
    """

    mean, log_variance = model.encode(batch_x, batch_c)
    if sample_latent:
        latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(log_variance)
    else:
        latent = mean
    reconstruction = model.decode(latent, batch_c)
    reconstructed_time = layout.as_time(reconstruction)
    observed_time = layout.as_time(batch_x)
    reconstruction_loss = weighted_field_loss(
        reconstructed_time, observed_time, layout, field_weights
    )
    if horizon > 1:
        ramp_loss = weighted_field_loss(
            reconstructed_time[:, 1:] - reconstructed_time[:, :-1],
            observed_time[:, 1:] - observed_time[:, :-1],
            layout,
            field_weights,
        )
    else:
        ramp_loss = torch.zeros((), device=device)
    net_load_loss, net_peak_loss, price_spread_loss = shape.paired_losses(
        reconstructed_time, observed_time
    )
    carbon_spread_loss = shape.carbon_spread_loss(reconstructed_time, observed_time)
    kl_loss = -0.5 * torch.mean(
        1.0 + log_variance - mean.square() - log_variance.exp()
    )
    loss = (
        reconstruction_loss
        + config.ramp_weight * ramp_loss
        + config.net_load_weight * net_load_loss
        + config.net_peak_weight * net_peak_loss
        + config.price_spread_weight * price_spread_loss
        + config.carbon_spread_weight * carbon_spread_loss
        + effective_beta * kl_loss
    )
    return {
        "loss": loss,
        "reconstruction": reconstruction_loss,
        "ramp": ramp_loss,
        "kl": kl_loss,
        "net_load": net_load_loss,
        "net_peak": net_peak_loss,
        "price_spread": price_spread_loss,
        "carbon_spread": carbon_spread_loss,
    }


def cvae_loss_components(
    model: ConditionalVAE,
    batch_x: torch.Tensor,
    batch_c: torch.Tensor,
    layout: TrajectoryLayout,
    shape: ShapeStatistics,
    field_weights: Any,
    config: CVAEConfig,
    horizon: int,
    effective_beta: float,
    device: torch.device,
    *,
    sample_latent: bool = True,
) -> dict[str, torch.Tensor]:
    """Public statistical objective reused by recourse-aware fine-tuning."""

    return _components(
        model,
        batch_x,
        batch_c,
        layout,
        shape,
        field_weights,
        config,
        horizon,
        effective_beta,
        device,
        sample_latent=sample_latent,
    )


@torch.no_grad()
def _evaluate(
    model: ConditionalVAE,
    x: torch.Tensor,
    c: torch.Tensor,
    batch_size: int,
    **kwargs: Any,
) -> dict[str, float]:
    """Deterministic pass over a whole split, averaged per scenario."""

    was_training = model.training
    model.eval()
    totals = {name: 0.0 for name in COMPONENT_NAMES}
    try:
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            values = _components(
                model, x[start:stop], c[start:stop], sample_latent=False, **kwargs
            )
            for name, value in values.items():
                totals[name] += (stop - start) * float(value.detach().cpu())
    finally:
        if was_training:
            model.train()
    return {name: value / x.shape[0] for name, value in totals.items()}


def train_cvae(
    model: ConditionalVAE,
    trajectories: np.ndarray,
    contexts: np.ndarray,
    config: CVAEConfig,
    horizon: int,
    device: torch.device,
    seed: int,
    trajectory_mean: np.ndarray | None = None,
    trajectory_std: np.ndarray | None = None,
    field_masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    writer: Any | None = None,
    validation_trajectories: np.ndarray | None = None,
    validation_contexts: np.ndarray | None = None,
    start_epoch: int = 0,
    optimizer_state: dict | None = None,
    torch_rng_state: torch.Tensor | None = None,
    cuda_rng_state_all: list[torch.Tensor] | None = None,
    training_state_out: dict | None = None,
) -> tuple[GeneratorEpoch, ...]:
    """Pretrain the temporal-spatial manifold before decision-focused learning."""

    torch.manual_seed(seed)
    model.to(device)
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state.cpu())
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])
    x = torch.as_tensor(trajectories, dtype=torch.float32, device=device)
    c = torch.as_tensor(contexts, dtype=torch.float32, device=device)
    layout = TrajectoryLayout.build(
        trajectory_dim=x.shape[1],
        horizon=horizon,
        device=device,
        trajectory_mean=trajectory_mean,
        trajectory_std=trajectory_std,
        field_masks=field_masks,
    )
    # Fitted on the training split only. Refitting on held-out data would rescale
    # every summary loss per split and make the two curves incomparable, which is
    # the one thing these metrics exist to support.
    shape = ShapeStatistics.fit(x, layout)

    if (validation_trajectories is None) != (validation_contexts is None):
        raise ValueError(
            "validation_trajectories and validation_contexts must be given together."
        )
    validation_x = (
        torch.as_tensor(validation_trajectories, dtype=torch.float32, device=device)
        if validation_trajectories is not None
        else None
    )
    validation_c = (
        torch.as_tensor(validation_contexts, dtype=torch.float32, device=device)
        if validation_contexts is not None
        else None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    history: list[GeneratorEpoch] = []
    field_weights = config.field_weights()

    total_epochs = start_epoch + config.epochs
    for local_epoch in range(config.epochs):
        epoch = start_epoch + local_epoch
        totals = {name: 0.0 for name in COMPONENT_NAMES}
        effective_beta = config.beta * min(
            1.0,
            (epoch + 1) / max(1, config.kl_warmup_epochs),
        )
        shared = dict(
            layout=layout,
            shape=shape,
            field_weights=field_weights,
            config=config,
            horizon=horizon,
            effective_beta=effective_beta,
            device=device,
        )
        for indices in batched_indices(x.shape[0], config.batch_size, device):
            optimizer.zero_grad(set_to_none=True)
            values = _components(
                model, x[indices], c[indices], sample_latent=True, **shared
            )
            values["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            weight = int(indices.numel())
            for name, value in values.items():
                totals[name] += weight * float(value.detach().cpu())

        totals = {name: value / x.shape[0] for name, value in totals.items()}
        totals["beta"] = effective_beta
        # The running averages above are the minimised estimator, measured while
        # the weights move within the epoch. Held-out comparison needs both sides
        # measured the same way, so the training split is re-evaluated here under
        # the same deterministic pass the validation split gets.
        if validation_x is not None and validation_c is not None:
            for name, value in _evaluate(
                model, x, c, config.batch_size, **shared
            ).items():
                totals[f"train_eval_{name}"] = value
            for name, value in _evaluate(
                model, validation_x, validation_c, config.batch_size, **shared
            ).items():
                totals[f"validation_{name}"] = value
        history.append(
            GeneratorEpoch(epoch=epoch, loss=totals["loss"], metrics=dict(totals))
        )
        if epoch == 0 or epoch + 1 == total_epochs or (epoch + 1) % 10 == 0:
            held_out = (
                f", val_loss={totals['validation_loss']:.6f}"
                f", val_carbon_spread={totals['validation_carbon_spread']:.6f}"
                if "validation_loss" in totals
                else ""
            )
            print(
                f"CVAE epoch {epoch + 1}/{total_epochs}: "
                f"loss={totals['loss']:.6f}, "
                f"reconstruction={totals['reconstruction']:.6f}, "
                f"net_peak={totals['net_peak']:.6f}, "
                f"price_spread={totals['price_spread']:.6f}, "
                f"carbon_spread={totals['carbon_spread']:.6f}, "
                f"beta={effective_beta:.6g}" + held_out,
                flush=True,
            )
        if writer is not None:
            for name, value in totals.items():
                writer.add_scalar(f"loss/{name}", value, epoch)
            if epoch % 10 == 0:
                writer.flush()

    model.freeze()
    if training_state_out is not None:
        training_state_out["optimizer_state_dict"] = optimizer.state_dict()
        training_state_out["completed_epochs"] = total_epochs
        training_state_out["torch_rng_state"] = torch.get_rng_state()
        if torch.cuda.is_available():
            training_state_out["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if writer is not None:
        writer.flush()
    return tuple(history)
