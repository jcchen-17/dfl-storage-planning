from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from storage_dfl.config import CVAEConfig


class ConditionalVAE(nn.Module):
    """Small CVAE for complete multivariate day trajectories."""

    def __init__(
        self,
        trajectory_dim: int,
        context_dim: int,
        latent_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.context_dim = context_dim
        self.latent_dim = latent_dim
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


@dataclass(frozen=True)
class CVAEEpoch:
    epoch: int
    loss: float
    reconstruction: float
    ramp: float
    kl: float


def train_cvae(
    model: ConditionalVAE,
    trajectories: np.ndarray,
    contexts: np.ndarray,
    config: CVAEConfig,
    horizon: int,
    device: torch.device,
    seed: int,
    writer: Any | None = None,
) -> tuple[CVAEEpoch, ...]:
    """Pretrain the temporal-spatial manifold before decision-focused learning."""

    torch.manual_seed(seed)
    model.to(device)
    x = torch.as_tensor(trajectories, dtype=torch.float32, device=device)
    c = torch.as_tensor(contexts, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[CVAEEpoch] = []

    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        reconstruction, mean, log_variance = model(x, c)
        reconstruction_loss = torch.mean((reconstruction - x) ** 2)
        reconstructed_time = reconstruction.reshape(reconstruction.shape[0], horizon, -1)
        observed_time = x.reshape(x.shape[0], horizon, -1)
        ramp_loss = torch.mean(
            (
                (reconstructed_time[:, 1:] - reconstructed_time[:, :-1])
                - (observed_time[:, 1:] - observed_time[:, :-1])
            )
            ** 2
        )
        kl_loss = -0.5 * torch.mean(1.0 + log_variance - mean.square() - log_variance.exp())
        loss = reconstruction_loss + config.ramp_weight * ramp_loss + config.beta * kl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        history.append(
            CVAEEpoch(
                epoch=epoch,
                loss=float(loss.detach().cpu()),
                reconstruction=float(reconstruction_loss.detach().cpu()),
                ramp=float(ramp_loss.detach().cpu()),
                kl=float(kl_loss.detach().cpu()),
            )
        )
        if writer is not None:
            writer.add_scalar("loss/total", float(loss.detach().cpu()), epoch)
            writer.add_scalar("loss/reconstruction", float(reconstruction_loss.detach().cpu()), epoch)
            writer.add_scalar("loss/temporal_ramp", float(ramp_loss.detach().cpu()), epoch)
            writer.add_scalar("loss/kl", float(kl_loss.detach().cpu()), epoch)
            writer.add_scalar("latent/mean_abs", float(mean.detach().abs().mean().cpu()), epoch)
            writer.add_scalar(
                "latent/std_mean",
                float(torch.exp(0.5 * log_variance.detach()).mean().cpu()),
                epoch,
            )
            if epoch % 10 == 0:
                writer.flush()

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if writer is not None:
        writer.flush()
    return tuple(history)
