"""Conditional WGAN-GP baseline generator with a post-hoc inversion encoder.

The adversarial objective is kept pure: the critic and generator play the
standard Wasserstein game with a gradient penalty, plus a distributional moment
term on the same decision-relevant summaries the CVAE is asked to preserve.  A
GAN has no encoder, so the DFL stage's ``encode`` is provided by a separate
inversion network trained afterwards against a frozen generator.  Keeping the
two phases apart means the reconstruction objective never leaks into the
adversarial training and the comparison against the CVAE stays honest.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from storage_dfl.config import CVAEConfig, GANConfig
from storage_dfl.models.base import (
    ConditionalGenerator,
    GeneratorEpoch,
    ShapeStatistics,
    TrajectoryLayout,
    batched_indices,
    weighted_field_loss,
)


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, hidden_dim),
        activation(),
        nn.Linear(hidden_dim, output_dim),
    )


class ConditionalGAN(ConditionalGenerator):
    """Conditional generator, critic and inversion encoder in one module."""

    def __init__(
        self,
        trajectory_dim: int,
        context_dim: int,
        latent_dim: int,
        hidden_dim: int,
        critic_hidden_dim: int = 512,
    ) -> None:
        super().__init__(trajectory_dim, context_dim, latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.critic_hidden_dim = int(critic_hidden_dim)
        self.generator = _mlp(latent_dim + context_dim, hidden_dim, trajectory_dim)
        # LeakyReLU keeps the critic's gradient alive everywhere, which the
        # gradient penalty relies on to estimate a 1-Lipschitz Wasserstein critic.
        self.critic = _mlp(
            trajectory_dim + context_dim,
            critic_hidden_dim,
            1,
            activation=lambda: nn.LeakyReLU(0.2),  # type: ignore[arg-type]
        )
        self.inversion_encoder = _mlp(
            trajectory_dim + context_dim,
            hidden_dim,
            latent_dim,
        )

    @property
    def kind(self) -> str:
        return "gan"

    def decode(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.generator(torch.cat((latent, context), dim=-1))

    def encode(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Deterministic GAN inversion.

        The second slot is ``None`` because the inversion network is a point
        estimate, not a posterior.  ``train_dfl_stage`` only reads the first
        element, so the CVAE's ``(mean, log_variance)`` contract is preserved.
        """

        return self.inversion_encoder(torch.cat((x, context), dim=-1)), None

    def score(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.critic(torch.cat((x, context), dim=-1))

    def forward(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.decode(latent, context)

    def checkpoint_payload(self) -> dict:
        return {
            "trajectory_dim": self.trajectory_dim,
            "context_dim": self.context_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "critic_hidden_dim": self.critic_hidden_dim,
        }


def _gradient_penalty(
    model: ConditionalGAN,
    real: torch.Tensor,
    fake: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    epsilon = torch.rand(real.shape[0], 1, device=real.device)
    interpolated = (epsilon * real + (1.0 - epsilon) * fake).requires_grad_(True)
    score = model.score(interpolated, context)
    gradient = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    norm = gradient.reshape(gradient.shape[0], -1).norm(2, dim=1)
    return torch.mean((norm - 1.0) ** 2)


def train_gan(
    model: ConditionalGAN,
    trajectories: np.ndarray,
    contexts: np.ndarray,
    config: CVAEConfig,
    gan_config: GANConfig,
    horizon: int,
    device: torch.device,
    seed: int,
    trajectory_mean: np.ndarray | None = None,
    trajectory_std: np.ndarray | None = None,
    field_masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    writer: Any | None = None,
) -> tuple[GeneratorEpoch, ...]:
    """WGAN-GP training followed by inversion-encoder fitting."""

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
    field_weights = config.field_weights()

    generator_parameters = list(model.generator.parameters())
    critic_parameters = list(model.critic.parameters())
    generator_optimizer = torch.optim.Adam(
        generator_parameters,
        lr=gan_config.generator_learning_rate,
        betas=(gan_config.adam_beta1, gan_config.adam_beta2),
    )
    critic_optimizer = torch.optim.Adam(
        critic_parameters,
        lr=gan_config.critic_learning_rate,
        betas=(gan_config.adam_beta1, gan_config.adam_beta2),
    )

    history: list[GeneratorEpoch] = []
    for epoch in range(config.epochs):
        totals = {
            name: 0.0
            for name in (
                "critic",
                "generator",
                "gradient_penalty",
                "wasserstein",
                "moment",
                "batches",
            )
        }
        for indices in batched_indices(x.shape[0], config.batch_size, device):
            batch_x = x[indices]
            batch_c = c[indices]
            count = int(indices.numel())

            for _ in range(max(1, gan_config.critic_steps)):
                critic_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake = model.decode(
                        torch.randn(count, model.latent_dim, device=device), batch_c
                    )
                real_score = model.score(batch_x, batch_c).mean()
                fake_score = model.score(fake, batch_c).mean()
                penalty = _gradient_penalty(model, batch_x, fake, batch_c)
                critic_loss = fake_score - real_score + gan_config.gradient_penalty * penalty
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_parameters, max_norm=10.0)
                critic_optimizer.step()

            generator_optimizer.zero_grad(set_to_none=True)
            fake = model.decode(
                torch.randn(count, model.latent_dim, device=device), batch_c
            )
            adversarial_loss = -model.score(fake, batch_c).mean()
            moment_terms = shape.moment_losses(
                layout.as_time(fake), layout.as_time(batch_x)
            )
            moment_loss = (
                config.net_load_weight * moment_terms[0]
                + config.net_peak_weight * moment_terms[1]
                + config.price_spread_weight * moment_terms[2]
            )
            generator_loss = adversarial_loss + gan_config.moment_weight * moment_loss
            generator_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator_parameters, max_norm=10.0)
            generator_optimizer.step()

            totals["critic"] += float(critic_loss.detach().cpu())
            totals["generator"] += float(generator_loss.detach().cpu())
            totals["gradient_penalty"] += float(penalty.detach().cpu())
            totals["wasserstein"] += float((real_score - fake_score).detach().cpu())
            totals["moment"] += float(moment_loss.detach().cpu())
            totals["batches"] += 1.0

        batches = max(totals.pop("batches"), 1.0)
        totals = {name: value / batches for name, value in totals.items()}
        totals["phase"] = 0.0
        history.append(
            GeneratorEpoch(
                epoch=epoch,
                # The critic loss is the model's own training signal; the
                # Wasserstein estimate is the number to watch for convergence.
                loss=totals["critic"],
                metrics=dict(totals),
            )
        )
        if epoch == 0 or epoch + 1 == config.epochs or (epoch + 1) % 10 == 0:
            print(
                f"GAN epoch {epoch + 1}/{config.epochs}: "
                f"critic={totals['critic']:.6f}, "
                f"generator={totals['generator']:.6f}, "
                f"wasserstein={totals['wasserstein']:.6f}, "
                f"moment={totals['moment']:.6f}",
                flush=True,
            )
        if writer is not None:
            for name, value in totals.items():
                writer.add_scalar(f"gan/{name}", value, epoch)
            if epoch % 10 == 0:
                writer.flush()

    # Inversion phase: the generator is frozen, so this only fits the map used to
    # initialise the DFL latent policy and cannot improve the samples themselves.
    for parameter in model.generator.parameters():
        parameter.requires_grad_(False)
    encoder_optimizer = torch.optim.Adam(
        model.inversion_encoder.parameters(),
        lr=gan_config.encoder_learning_rate,
    )
    for epoch in range(gan_config.encoder_epochs):
        total_loss = 0.0
        total_prior = 0.0
        for indices in batched_indices(x.shape[0], config.batch_size, device):
            batch_x = x[indices]
            batch_c = c[indices]
            encoder_optimizer.zero_grad(set_to_none=True)
            latent, _ = model.encode(batch_x, batch_c)
            reconstruction = model.decode(latent, batch_c)
            reconstruction_loss = weighted_field_loss(
                layout.as_time(reconstruction),
                layout.as_time(batch_x),
                layout,
                field_weights,
            )
            prior_penalty = latent.square().mean()
            loss = reconstruction_loss + gan_config.latent_prior_weight * prior_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.inversion_encoder.parameters(), max_norm=10.0
            )
            encoder_optimizer.step()
            weight = int(indices.numel())
            total_loss += weight * float(reconstruction_loss.detach().cpu())
            total_prior += weight * float(prior_penalty.detach().cpu())
        metrics = {
            "inversion_reconstruction": total_loss / x.shape[0],
            "inversion_latent_norm": total_prior / x.shape[0],
            "phase": 1.0,
        }
        history.append(
            GeneratorEpoch(
                epoch=config.epochs + epoch,
                loss=metrics["inversion_reconstruction"],
                metrics=metrics,
            )
        )
        if epoch == 0 or epoch + 1 == gan_config.encoder_epochs:
            print(
                f"GAN inversion epoch {epoch + 1}/{gan_config.encoder_epochs}: "
                f"reconstruction={metrics['inversion_reconstruction']:.6f}",
                flush=True,
            )
        if writer is not None:
            for name, value in metrics.items():
                writer.add_scalar(f"gan_inversion/{name}", value, epoch)

    model.freeze()
    if writer is not None:
        writer.flush()
    return tuple(history)
