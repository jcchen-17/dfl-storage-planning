from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SupportSample:
    latent: torch.Tensor
    weights: torch.Tensor
    log_probability: torch.Tensor


class DirectSupportPolicy(nn.Module):
    """A distribution over generated latent support points and their weights."""

    def __init__(
        self,
        support_count: int,
        latent_dim: int,
        seed: int,
        initial_latent: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if initial_latent is None:
            generator = torch.Generator().manual_seed(seed)
            initial_latent = 0.35 * torch.randn(
                support_count,
                latent_dim,
                generator=generator,
            )
        elif tuple(initial_latent.shape) != (support_count, latent_dim):
            raise ValueError(
                "initial_latent must have shape "
                f"({support_count}, {latent_dim}), got {tuple(initial_latent.shape)}"
            )
        self.latent_location = nn.Parameter(initial_latent.detach().float().clone())
        self.weight_logit_location = nn.Parameter(torch.zeros(support_count))

    def sample(self, exploration_std: float) -> SupportSample:
        latent_distribution = torch.distributions.Normal(
            self.latent_location,
            torch.full_like(self.latent_location, exploration_std),
        )
        # detach() gives a score-function estimator: no gradient is requested
        # through the decoder or the nonconvex planning oracle.
        latent = latent_distribution.sample().detach()
        log_probability = latent_distribution.log_prob(latent).sum()
        if self.weight_logit_location.numel() == 1:
            # A single support always has weight one. Sampling its logit would
            # add action-independent score-function noise and update a parameter
            # that cannot change either the scenario or the planning decision.
            weights = torch.ones_like(self.weight_logit_location)
        else:
            weight_distribution = torch.distributions.Normal(
                self.weight_logit_location,
                torch.full_like(self.weight_logit_location, exploration_std),
            )
            weight_logits = weight_distribution.sample().detach()
            log_probability = log_probability + weight_distribution.log_prob(weight_logits).sum()
            weights = torch.softmax(weight_logits, dim=0)
        return SupportSample(
            latent=latent,
            weights=weights,
            log_probability=log_probability,
        )

    def deterministic(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.latent_location.detach(), torch.softmax(self.weight_logit_location.detach(), dim=0)

    def diversity_penalty(self, margin: float) -> torch.Tensor:
        if self.latent_location.shape[0] < 2:
            return torch.zeros((), device=self.latent_location.device)
        distances = torch.pdist(self.latent_location, p=2)
        return torch.relu(margin - distances).square().mean()

    def weight_entropy(self) -> torch.Tensor:
        weights = torch.softmax(self.weight_logit_location, dim=0)
        return -(weights * torch.log(weights.clamp_min(1.0e-8))).sum()

    def latent_prior_penalty(self) -> torch.Tensor:
        return self.latent_location.square().mean()
