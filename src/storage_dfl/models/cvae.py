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
) -> tuple[GeneratorEpoch, ...]:
    """Pretrain the temporal-spatial manifold before decision-focused learning."""

    torch.manual_seed(seed)
    model.to(device)
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
    shape = ShapeStatistics.fit(x, layout)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[GeneratorEpoch] = []
    field_weights = config.field_weights()

    for epoch in range(config.epochs):
        totals = {
            name: 0.0
            for name in (
                "loss",
                "reconstruction",
                "ramp",
                "kl",
                "net_load",
                "net_peak",
                "price_spread",
            )
        }
        effective_beta = config.beta * min(
            1.0,
            (epoch + 1) / max(1, config.kl_warmup_epochs),
        )
        for indices in batched_indices(x.shape[0], config.batch_size, device):
            batch_x = x[indices]
            batch_c = c[indices]
            optimizer.zero_grad(set_to_none=True)
            reconstruction, latent_mean, log_variance = model(batch_x, batch_c)
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
            kl_loss = -0.5 * torch.mean(
                1.0 + log_variance - latent_mean.square() - log_variance.exp()
            )
            loss = (
                reconstruction_loss
                + config.ramp_weight * ramp_loss
                + config.net_load_weight * net_load_loss
                + config.net_peak_weight * net_peak_loss
                + config.price_spread_weight * price_spread_loss
                + effective_beta * kl_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            weight = int(indices.numel())
            for name, value in (
                ("loss", loss),
                ("reconstruction", reconstruction_loss),
                ("ramp", ramp_loss),
                ("kl", kl_loss),
                ("net_load", net_load_loss),
                ("net_peak", net_peak_loss),
                ("price_spread", price_spread_loss),
            ):
                totals[name] += weight * float(value.detach().cpu())

        totals = {name: value / x.shape[0] for name, value in totals.items()}
        totals["beta"] = effective_beta
        history.append(
            GeneratorEpoch(epoch=epoch, loss=totals["loss"], metrics=dict(totals))
        )
        if epoch == 0 or epoch + 1 == config.epochs or (epoch + 1) % 10 == 0:
            print(
                f"CVAE epoch {epoch + 1}/{config.epochs}: "
                f"loss={totals['loss']:.6f}, "
                f"reconstruction={totals['reconstruction']:.6f}, "
                f"net_peak={totals['net_peak']:.6f}, "
                f"price_spread={totals['price_spread']:.6f}, "
                f"beta={effective_beta:.6g}",
                flush=True,
            )
        if writer is not None:
            for name, value in totals.items():
                writer.add_scalar(f"loss/{name}", value, epoch)
            if epoch % 10 == 0:
                writer.flush()

    model.freeze()
    if writer is not None:
        writer.flush()
    return tuple(history)
