"""Conditional denoising diffusion baseline with a DFL-compatible latent.

A diffusion model has no learned latent space: its only free input is the
initial noise ``x_T``, which lives in the full trajectory dimension (5,808 for
the 48-hour IEEE13 case).  ``DirectSupportPolicy`` perturbs the latent with an
isotropic Gaussian and updates it from a handful of REINFORCE samples per epoch,
so a 5,808-dimensional latent would never move.

``latent_mode: projected`` therefore restricts the initial noise to a fixed
``d``-dimensional subspace,

    x_T = sqrt(D / d) * P z,    z ~ N(0, I_d),

where ``P`` has orthonormal columns.  The scale keeps ``E||x_T||^2 = D``, which
is what the reverse process expects, so the sampler is unchanged.  Setting ``d``
to the CVAE's ``latent_dim`` also makes the three generators controllable
through latent spaces of identical size, which is what makes the downstream
comparison fair.  ``latent_mode: full`` keeps the unrestricted noise for pure
generative-quality measurement; it cannot be used by the DFL stage.

``encode`` is deterministic DDIM inversion followed by projection back onto the
subspace, which is exact for latents that were produced by ``decode`` and a
least-squares approximation otherwise.  It is only used to initialise the DFL
policy at a real scenario, so the approximation is acceptable there.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from storage_dfl.config import CVAEConfig, DiffusionConfig
from storage_dfl.models.base import (
    ConditionalGenerator,
    GeneratorEpoch,
    ShapeStatistics,
    TrajectoryLayout,
    batched_indices,
)

LATENT_MODES = ("projected", "full")


def _beta_schedule(name: str, timesteps: int) -> torch.Tensor:
    if name == "linear":
        return torch.linspace(1.0e-4, 0.02, timesteps, dtype=torch.float32)
    if name == "cosine":
        # Nichol & Dhariwal cosine schedule; gentler at the low-noise end, which
        # matters here because the decision-relevant structure is fine detail.
        steps = torch.arange(timesteps + 1, dtype=torch.float32) / timesteps
        alphas_cumulative = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumulative = alphas_cumulative / alphas_cumulative[0]
        betas = 1.0 - alphas_cumulative[1:] / alphas_cumulative[:-1]
        return betas.clamp(1.0e-5, 0.999)
    raise ValueError("beta_schedule must be 'linear' or 'cosine'.")


class _TimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.projection = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.SiLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=timestep.device)
            / max(half - 1, 1)
        )
        angles = timestep.float()[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dimension:
            embedding = torch.nn.functional.pad(
                embedding, (0, self.dimension - embedding.shape[-1])
            )
        return self.projection(embedding)


class _ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, conditioning_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.conditioning = nn.Linear(conditioning_dim, hidden_dim)
        self.body = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, hidden: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        return hidden + self.body(self.norm(hidden) + self.conditioning(conditioning))


class ConditionalDiffusion(ConditionalGenerator):
    """Epsilon-prediction DDPM with deterministic DDIM sampling and inversion."""

    def __init__(
        self,
        trajectory_dim: int,
        context_dim: int,
        latent_dim: int,
        hidden_dim: int,
        timesteps: int = 400,
        sampling_steps: int = 50,
        beta_schedule: str = "cosine",
        latent_mode: str = "projected",
        projection_seed: int = 0,
        blocks: int = 3,
        x_zero_clamp: float = 5.0,
    ) -> None:
        if latent_mode not in LATENT_MODES:
            raise ValueError(f"latent_mode must be one of {LATENT_MODES}.")
        effective_latent_dim = trajectory_dim if latent_mode == "full" else latent_dim
        super().__init__(trajectory_dim, context_dim, effective_latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.timesteps = int(timesteps)
        self.sampling_steps = int(min(sampling_steps, timesteps))
        self.beta_schedule = beta_schedule
        self.latent_mode = latent_mode
        self.projection_seed = int(projection_seed)
        self.blocks = int(blocks)
        self.x_zero_clamp = float(x_zero_clamp)

        embedding_dim = max(64, hidden_dim // 4)
        self.time_embedding = _TimeEmbedding(embedding_dim)
        self.context_embedding = nn.Sequential(
            nn.Linear(context_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.input_projection = nn.Linear(trajectory_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList(
            _ResidualBlock(hidden_dim, embedding_dim) for _ in range(self.blocks)
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, trajectory_dim),
        )

        betas = _beta_schedule(beta_schedule, self.timesteps)
        alphas_cumulative = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumulative", alphas_cumulative)
        self.register_buffer(
            "latent_projection", self._build_projection(), persistent=True
        )

    def _build_projection(self) -> torch.Tensor:
        """A fixed matrix with orthonormal columns, or an empty tensor in full mode."""

        if self.latent_mode == "full":
            return torch.zeros(0)
        generator = torch.Generator().manual_seed(self.projection_seed)
        gaussian = torch.randn(
            self.trajectory_dim, self.latent_dim, generator=generator
        )
        orthonormal, _ = torch.linalg.qr(gaussian)
        return orthonormal[:, : self.latent_dim].contiguous()

    @property
    def kind(self) -> str:
        return "diffusion"

    @property
    def supports_dfl(self) -> bool:
        return self.latent_mode == "projected"

    @property
    def _projection_scale(self) -> float:
        return math.sqrt(self.trajectory_dim / self.latent_dim)

    def latent_to_noise(self, latent: torch.Tensor) -> torch.Tensor:
        if self.latent_mode == "full":
            return latent
        return self._projection_scale * latent @ self.latent_projection.T

    def noise_to_latent(self, noise: torch.Tensor) -> torch.Tensor:
        if self.latent_mode == "full":
            return noise
        return (noise @ self.latent_projection) / self._projection_scale

    def epsilon(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        conditioning = self.time_embedding(timestep) + self.context_embedding(context)
        hidden = self.input_projection(x)
        for block in self.residual_blocks:
            hidden = block(hidden, conditioning)
        return self.output_projection(hidden)

    def _gather(self, buffer: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return buffer.to(timestep.device)[timestep][:, None]

    def alpha_bar(self, timestep: torch.Tensor) -> torch.Tensor:
        """Cumulative alpha as a ``[batch, 1]`` column, for signal-to-noise weights."""

        return self._gather(self.alphas_cumulative, timestep)

    def add_noise(
        self,
        x_zero: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._gather(self.alphas_cumulative, timestep)
        return alpha.sqrt() * x_zero + (1.0 - alpha).sqrt() * noise

    def predict_x_zero(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Recover ``x_0`` and clamp it back into the standardized data range.

        Without the clamp, a division by a near-zero ``sqrt(alpha)`` at high
        noise levels produces enormous values that both destabilise the sampler
        and dominate any auxiliary loss computed on them.  Observed data is
        standardized, so the range is a genuine bound, not a tuning knob.
        """

        alpha = self._gather(self.alphas_cumulative, timestep)
        x_zero = (x_t - (1.0 - alpha).sqrt() * epsilon) / alpha.sqrt().clamp_min(1.0e-8)
        if self.x_zero_clamp > 0.0:
            x_zero = x_zero.clamp(-self.x_zero_clamp, self.x_zero_clamp)
        return x_zero

    def _ddim_timesteps(self, steps: int | None = None) -> torch.Tensor:
        count = int(steps or self.sampling_steps)
        return torch.linspace(
            0, self.timesteps - 1, count, dtype=torch.long
        ).flip(0)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Deterministic DDIM sampling from the latent-induced initial noise.

        ``eta = 0`` makes this a fixed function of ``latent``, which is what the
        DFL stage needs: the same latent must always produce the same scenario
        and therefore the same planning decision.
        """

        x = self.latent_to_noise(latent)
        schedule = self._ddim_timesteps().to(latent.device)
        for index, timestep in enumerate(schedule):
            batched = timestep.expand(x.shape[0])
            epsilon = self.epsilon(x, batched, context)
            x_zero = self.predict_x_zero(x, batched, epsilon)
            if index + 1 == schedule.numel():
                x = x_zero
                break
            previous = schedule[index + 1].expand(x.shape[0])
            alpha_previous = self._gather(self.alphas_cumulative, previous)
            x = alpha_previous.sqrt() * x_zero + (1.0 - alpha_previous).sqrt() * epsilon
        return x

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """DDIM inversion: run the deterministic map from ``x_0`` back to ``x_T``."""

        schedule = self._ddim_timesteps().to(x.device).flip(0)
        current = x
        for index in range(schedule.numel() - 1):
            timestep = schedule[index].expand(x.shape[0])
            following = schedule[index + 1].expand(x.shape[0])
            epsilon = self.epsilon(current, timestep, context)
            x_zero = self.predict_x_zero(current, timestep, epsilon)
            alpha_following = self._gather(self.alphas_cumulative, following)
            current = (
                alpha_following.sqrt() * x_zero + (1.0 - alpha_following).sqrt() * epsilon
            )
        return self.noise_to_latent(current), None

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.epsilon(x, timestep, context)

    def checkpoint_payload(self) -> dict:
        return {
            "trajectory_dim": self.trajectory_dim,
            "context_dim": self.context_dim,
            # ``latent_dim`` equals ``trajectory_dim`` in full mode, so the
            # constructor's own resolution reproduces it either way.
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "timesteps": self.timesteps,
            "sampling_steps": self.sampling_steps,
            "beta_schedule": self.beta_schedule,
            "latent_mode": self.latent_mode,
            "projection_seed": self.projection_seed,
            "blocks": self.blocks,
            "x_zero_clamp": self.x_zero_clamp,
        }


def train_diffusion(
    model: ConditionalDiffusion,
    trajectories: np.ndarray,
    contexts: np.ndarray,
    config: CVAEConfig,
    diffusion_config: DiffusionConfig,
    horizon: int,
    device: torch.device,
    seed: int,
    trajectory_mean: np.ndarray | None = None,
    trajectory_std: np.ndarray | None = None,
    field_masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    writer: Any | None = None,
) -> tuple[GeneratorEpoch, ...]:
    """Field-weighted epsilon-matching with an auxiliary decision-shape term."""

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
    # The same per-field balance the CVAE reconstruction uses, expressed as
    # per-feature weights so it can be applied to the noise objective directly.
    feature_weights = layout.feature_weights(config.field_weights())

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[GeneratorEpoch] = []

    for epoch in range(config.epochs):
        totals = {name: 0.0 for name in ("loss", "epsilon", "net_peak", "price_spread")}
        for indices in batched_indices(x.shape[0], config.batch_size, device):
            batch_x = x[indices]
            batch_c = c[indices]
            count = int(indices.numel())
            optimizer.zero_grad(set_to_none=True)
            timestep = torch.randint(0, model.timesteps, (count,), device=device)
            noise = torch.randn_like(batch_x)
            noisy = model.add_noise(batch_x, timestep, noise)
            predicted = model.epsilon(noisy, timestep, batch_c)
            squared_error = layout.as_time((predicted - noise) ** 2)
            epsilon_loss = (squared_error * feature_weights).sum(dim=-1).mean()

            if diffusion_config.physics_weight > 0.0:
                predicted_x_zero = model.predict_x_zero(noisy, timestep, predicted)
                _, net_peak_loss, price_spread_loss = shape.paired_losses(
                    layout.as_time(predicted_x_zero),
                    layout.as_time(batch_x),
                    robust=True,
                )
                # x0 predictions from near-pure noise are meaningless, so the
                # shape term is faded out with the signal-to-noise ratio.
                weight = model.alpha_bar(timestep).mean()
                physics_loss = weight * (
                    config.net_peak_weight * net_peak_loss
                    + config.price_spread_weight * price_spread_loss
                )
            else:
                net_peak_loss = torch.zeros((), device=device)
                price_spread_loss = torch.zeros((), device=device)
                physics_loss = torch.zeros((), device=device)

            loss = epsilon_loss + diffusion_config.physics_weight * physics_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            for name, value in (
                ("loss", loss),
                ("epsilon", epsilon_loss),
                ("net_peak", net_peak_loss),
                ("price_spread", price_spread_loss),
            ):
                totals[name] += count * float(value.detach().cpu())

        totals = {name: value / x.shape[0] for name, value in totals.items()}
        history.append(
            GeneratorEpoch(epoch=epoch, loss=totals["loss"], metrics=dict(totals))
        )
        if epoch == 0 or epoch + 1 == config.epochs or (epoch + 1) % 10 == 0:
            print(
                f"Diffusion epoch {epoch + 1}/{config.epochs}: "
                f"loss={totals['loss']:.6f}, "
                f"epsilon={totals['epsilon']:.6f}, "
                f"net_peak={totals['net_peak']:.6f}, "
                f"price_spread={totals['price_spread']:.6f}",
                flush=True,
            )
        if writer is not None:
            for name, value in totals.items():
                writer.add_scalar(f"diffusion/{name}", value, epoch)
            if epoch % 10 == 0:
                writer.flush()

    model.freeze()
    if writer is not None:
        writer.flush()
    return tuple(history)
