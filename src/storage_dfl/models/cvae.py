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
    beta: float
    net_load: float
    net_peak: float
    price_spread: float


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        prediction = prediction[..., mask]
        target = target[..., mask]
    return torch.mean((prediction - target) ** 2)


def _field_reconstruction_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    field_size: int,
    masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    active_mask, reactive_mask, pv_mask = masks
    return (
        _masked_mse(prediction[..., :field_size], target[..., :field_size], active_mask),
        _masked_mse(
            prediction[..., field_size : 2 * field_size],
            target[..., field_size : 2 * field_size],
            reactive_mask,
        ),
        _masked_mse(
            prediction[..., 2 * field_size : 3 * field_size],
            target[..., 2 * field_size : 3 * field_size],
            pv_mask,
        ),
        *(
            _masked_mse(
                prediction[..., 3 * field_size + index],
                target[..., 3 * field_size + index],
            )
            for index in range(4)
        ),
    )


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
) -> tuple[CVAEEpoch, ...]:
    """Pretrain the temporal-spatial manifold before decision-focused learning."""

    torch.manual_seed(seed)
    model.to(device)
    x = torch.as_tensor(trajectories, dtype=torch.float32, device=device)
    c = torch.as_tensor(contexts, dtype=torch.float32, device=device)
    if x.shape[1] % horizon != 0:
        raise ValueError("The trajectory dimension must be divisible by the horizon.")
    features_per_hour = x.shape[1] // horizon
    if (features_per_hour - 4) % 3 != 0:
        raise ValueError("The trajectory layout must contain three spatial fields and four scalars.")
    field_size = (features_per_hour - 4) // 3
    if field_masks is None:
        field_masks = tuple(np.ones(field_size, dtype=bool) for _ in range(3))
    masks = tuple(
        torch.as_tensor(mask, dtype=torch.bool, device=device) for mask in field_masks
    )
    if any(mask.numel() != field_size for mask in masks):
        raise ValueError("Every spatial field mask must match the flattened node-phase size.")

    mean = torch.as_tensor(
        np.zeros(x.shape[1], dtype=np.float32) if trajectory_mean is None else trajectory_mean,
        dtype=torch.float32,
        device=device,
    ).reshape(1, horizon, features_per_hour)
    std = torch.as_tensor(
        np.ones(x.shape[1], dtype=np.float32) if trajectory_std is None else trajectory_std,
        dtype=torch.float32,
        device=device,
    ).reshape(1, horizon, features_per_hour)
    observed_raw = x.reshape(x.shape[0], horizon, features_per_hour) * std + mean
    observed_net_load = (
        observed_raw[..., :field_size].sum(dim=-1)
        - observed_raw[..., 2 * field_size : 3 * field_size].sum(dim=-1)
    )
    net_load_scale = torch.maximum(
        observed_net_load.std(),
        0.1 * observed_net_load.abs().mean(),
    ).clamp_min(0.05)
    observed_net_peak = observed_net_load.max(dim=1).values
    net_peak_scale = torch.maximum(
        observed_net_peak.std(),
        0.1 * observed_net_peak.abs().mean(),
    ).clamp_min(0.05)
    observed_price = observed_raw[..., 3 * field_size + 2]
    observed_price_spread = observed_price.max(dim=1).values - observed_price.min(dim=1).values
    price_spread_scale = torch.maximum(
        observed_price_spread.std(),
        0.1 * observed_price_spread.abs().mean(),
    ).clamp_min(1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[CVAEEpoch] = []
    batch_size = min(max(1, config.batch_size), x.shape[0])
    field_weights = (
        config.active_load_weight,
        config.reactive_load_weight,
        config.pv_weight,
        config.workload_weight,
        config.pue_weight,
        config.price_weight,
        config.carbon_weight,
    )

    for epoch in range(config.epochs):
        totals = {name: 0.0 for name in ("loss", "reconstruction", "ramp", "kl", "net_load", "net_peak", "price_spread")}
        permutation = torch.randperm(x.shape[0], device=device)
        effective_beta = config.beta * min(
            1.0,
            (epoch + 1) / max(1, config.kl_warmup_epochs),
        )
        for start in range(0, x.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = x[indices]
            batch_c = c[indices]
            optimizer.zero_grad(set_to_none=True)
            reconstruction, latent_mean, log_variance = model(batch_x, batch_c)
            reconstructed_time = reconstruction.reshape(-1, horizon, features_per_hour)
            observed_time = batch_x.reshape(-1, horizon, features_per_hour)
            field_losses = _field_reconstruction_losses(
                reconstructed_time,
                observed_time,
                field_size,
                masks,
            )
            reconstruction_loss = sum(
                weight * field_loss
                for weight, field_loss in zip(field_weights, field_losses, strict=True)
            )
            if horizon > 1:
                ramp_fields = _field_reconstruction_losses(
                    reconstructed_time[:, 1:] - reconstructed_time[:, :-1],
                    observed_time[:, 1:] - observed_time[:, :-1],
                    field_size,
                    masks,
                )
                ramp_loss = sum(
                    weight * field_loss
                    for weight, field_loss in zip(field_weights, ramp_fields, strict=True)
                )
            else:
                ramp_loss = torch.zeros((), device=device)

            reconstructed_raw = reconstructed_time * std + mean
            batch_observed_raw = observed_time * std + mean
            reconstructed_net_load = (
                reconstructed_raw[..., :field_size].sum(dim=-1)
                - reconstructed_raw[..., 2 * field_size : 3 * field_size].sum(dim=-1)
            )
            batch_observed_net_load = (
                batch_observed_raw[..., :field_size].sum(dim=-1)
                - batch_observed_raw[..., 2 * field_size : 3 * field_size].sum(dim=-1)
            )
            net_load_loss = torch.mean(
                ((reconstructed_net_load - batch_observed_net_load) / net_load_scale) ** 2
            )
            net_peak_loss = torch.mean(
                (
                    (
                        reconstructed_net_load.max(dim=1).values
                        - batch_observed_net_load.max(dim=1).values
                    )
                    / net_peak_scale
                )
                ** 2
            )
            reconstructed_price = reconstructed_raw[..., 3 * field_size + 2]
            batch_observed_price = batch_observed_raw[..., 3 * field_size + 2]
            reconstructed_spread = reconstructed_price.max(dim=1).values - reconstructed_price.min(dim=1).values
            batch_observed_spread = batch_observed_price.max(dim=1).values - batch_observed_price.min(dim=1).values
            price_spread_loss = torch.mean(
                ((reconstructed_spread - batch_observed_spread) / price_spread_scale) ** 2
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
        history.append(
            CVAEEpoch(
                epoch=epoch,
                loss=totals["loss"],
                reconstruction=totals["reconstruction"],
                ramp=totals["ramp"],
                kl=totals["kl"],
                beta=effective_beta,
                net_load=totals["net_load"],
                net_peak=totals["net_peak"],
                price_spread=totals["price_spread"],
            )
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
            writer.add_scalar("loss/effective_beta", effective_beta, epoch)
            if epoch % 10 == 0:
                writer.flush()

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if writer is not None:
        writer.flush()
    return tuple(history)
