"""Shared contract and physics-aware loss pieces for conditional generators.

The decision-focused stage treats the generator as a frozen black box with a
latent space.  Everything downstream -- ``DirectSupportPolicy``, the REINFORCE
trainer and the Gaussian-process selector -- only needs three things:

* ``latent_dim`` so a policy can be parameterised;
* ``decode(latent, context)`` to turn a latent point into a normalized
  trajectory;
* ``encode(trajectory, context)`` to initialise that policy at a real scenario.

Any generative model that provides those three can be compared against the CVAE
without touching the planner, so this module defines them once and keeps the
field-balanced reconstruction terms in a single place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

GENERATOR_KINDS = ("cvae",)


@dataclass(frozen=True)
class GeneratorEpoch:
    """One training epoch of any conditional generator.

    ``loss`` is the quantity the model actually minimises, so it is only
    comparable across epochs of the same model, never across models.  Everything
    model specific lives in ``metrics`` so the training stage can serialise a
    history without knowing which generator produced it.
    """

    epoch: int
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)


class ConditionalGenerator(nn.Module, ABC):
    """A conditional generator with an explicit, low-dimensional latent space."""

    def __init__(self, trajectory_dim: int, context_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.trajectory_dim = int(trajectory_dim)
        self.context_dim = int(context_dim)
        self.latent_dim = int(latent_dim)

    @property
    @abstractmethod
    def kind(self) -> str:
        """The registry name used in configurations and artifact file names."""

    @abstractmethod
    def encode(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Map an observed trajectory to a latent point.

        Returns ``(latent, auxiliary)``.  The CVAE puts its log variance in the
        second slot; models with a deterministic inverse return ``None``.
        """

    @abstractmethod
    def decode(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Map a latent point to a normalized trajectory."""

    def sample_latent(
        self,
        count: int,
        *,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw from the model's latent prior.

        Every generator here uses a standard normal prior, including the
        diffusion model, whose projected latent is constructed precisely so this
        stays true.
        """

        return torch.randn(
            count,
            self.latent_dim,
            generator=generator,
            device=device,
        )

    @abstractmethod
    def checkpoint_payload(self) -> dict:
        """Constructor arguments needed to rebuild this model from a checkpoint."""

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)


@dataclass(frozen=True)
class TrajectoryLayout:
    """Time-major view of the packed trajectory vector.

    The codec packs each hour as ``[P_bus_phase, Q_bus_phase, PV_bus_phase,
    workload, pue, price, carbon]``.  Every generator needs the same slicing to
    apply field weights, undo normalization and read decision-relevant
    quantities, so it is derived once here.
    """

    horizon: int
    features_per_hour: int
    field_size: int
    masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def build(
        cls,
        trajectory_dim: int,
        horizon: int,
        device: torch.device,
        trajectory_mean: np.ndarray | None = None,
        trajectory_std: np.ndarray | None = None,
        field_masks: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> "TrajectoryLayout":
        if trajectory_dim % horizon != 0:
            raise ValueError("The trajectory dimension must be divisible by the horizon.")
        features_per_hour = trajectory_dim // horizon
        if (features_per_hour - 4) % 3 != 0:
            raise ValueError(
                "The trajectory layout must contain three spatial fields and four scalars."
            )
        field_size = (features_per_hour - 4) // 3
        if field_masks is None:
            field_masks = tuple(np.ones(field_size, dtype=bool) for _ in range(3))
        masks = tuple(
            torch.as_tensor(mask, dtype=torch.bool, device=device) for mask in field_masks
        )
        if any(mask.numel() != field_size for mask in masks):
            raise ValueError("Every spatial field mask must match the flattened node-phase size.")
        mean = torch.as_tensor(
            np.zeros(trajectory_dim, dtype=np.float32)
            if trajectory_mean is None
            else trajectory_mean,
            dtype=torch.float32,
            device=device,
        ).reshape(1, horizon, features_per_hour)
        std = torch.as_tensor(
            np.ones(trajectory_dim, dtype=np.float32)
            if trajectory_std is None
            else trajectory_std,
            dtype=torch.float32,
            device=device,
        ).reshape(1, horizon, features_per_hour)
        return cls(
            horizon=horizon,
            features_per_hour=features_per_hour,
            field_size=field_size,
            masks=masks,  # type: ignore[arg-type]
            mean=mean,
            std=std,
        )

    @property
    def trajectory_dim(self) -> int:
        return self.horizon * self.features_per_hour

    def as_time(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.reshape(-1, self.horizon, self.features_per_hour)

    def as_flat(self, time_major: torch.Tensor) -> torch.Tensor:
        return time_major.reshape(-1, self.trajectory_dim)

    def raw(self, time_major: torch.Tensor) -> torch.Tensor:
        return time_major * self.std + self.mean

    def net_load(self, raw_time_major: torch.Tensor) -> torch.Tensor:
        """Aggregate feeder net load [batch, horizon] in MW."""

        size = self.field_size
        return (
            raw_time_major[..., :size].sum(dim=-1)
            - raw_time_major[..., 2 * size : 3 * size].sum(dim=-1)
        )

    def price(self, raw_time_major: torch.Tensor) -> torch.Tensor:
        return raw_time_major[..., 3 * self.field_size + 2]

    def feature_weights(self, field_weights: tuple[float, ...]) -> torch.Tensor:
        """Per-feature weights reproducing the field-balanced masked means.

        Weighting feature ``i`` of field ``k`` by ``w_k / n_k`` -- where ``n_k``
        counts the physically present channels -- makes a plain weighted squared
        error identical to the sum of ``w_k * masked_mean_k``.  Models that
        cannot express the loss as a list of per-field means (the diffusion
        noise objective) can then still inherit exactly the same field balance.
        """

        if len(field_weights) != 7:
            raise ValueError("Seven field weights are required.")
        weights = torch.zeros(
            self.features_per_hour,
            dtype=torch.float32,
            device=self.mean.device,
        )
        size = self.field_size
        for index, mask in enumerate(self.masks):
            count = int(mask.sum())
            if count == 0:
                continue
            weights[index * size : (index + 1) * size] = (
                mask.to(weights.dtype) * field_weights[index] / count
            )
        for index in range(4):
            weights[3 * size + index] = field_weights[3 + index]
        return weights


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        prediction = prediction[..., mask]
        target = target[..., mask]
        if prediction.shape[-1] == 0:
            return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return torch.mean((prediction - target) ** 2)


def field_reconstruction_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    layout: TrajectoryLayout,
) -> tuple[torch.Tensor, ...]:
    """Per-field squared errors: P, Q, PV, workload, PUE, price, carbon."""

    size = layout.field_size
    active_mask, reactive_mask, pv_mask = layout.masks
    return (
        masked_mse(prediction[..., :size], target[..., :size], active_mask),
        masked_mse(
            prediction[..., size : 2 * size],
            target[..., size : 2 * size],
            reactive_mask,
        ),
        masked_mse(
            prediction[..., 2 * size : 3 * size],
            target[..., 2 * size : 3 * size],
            pv_mask,
        ),
        *(
            masked_mse(prediction[..., 3 * size + index], target[..., 3 * size + index])
            for index in range(4)
        ),
    )


def weighted_field_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    layout: TrajectoryLayout,
    field_weights: tuple[float, ...],
) -> torch.Tensor:
    losses = field_reconstruction_losses(prediction, target, layout)
    return sum(
        weight * value for weight, value in zip(field_weights, losses, strict=True)
    )


@dataclass(frozen=True)
class ShapeStatistics:
    """Scales for the decision-relevant summaries every generator must preserve.

    Peak net load drives the demand charge and price spread drives arbitrage
    value, so these are the two statistics whose distortion the planner notices
    first.  The scales come from the observed training pool and are shared by
    every generator so the auxiliary terms are numerically comparable.
    """

    layout: TrajectoryLayout
    net_load_scale: torch.Tensor
    net_peak_scale: torch.Tensor
    price_spread_scale: torch.Tensor
    carbon_spread_scale: torch.Tensor

    @classmethod
    def fit(cls, observed_normalized: torch.Tensor, layout: TrajectoryLayout) -> "ShapeStatistics":
        raw = layout.raw(layout.as_time(observed_normalized))
        net_load = layout.net_load(raw)
        net_peak = net_load.max(dim=1).values
        price = layout.price(raw)
        spread = price.max(dim=1).values - price.min(dim=1).values
        carbon = layout.raw(layout.as_time(observed_normalized))[
            ..., 3 * layout.field_size + 3
        ]
        carbon_spread = carbon.max(dim=1).values - carbon.min(dim=1).values
        return cls(
            layout=layout,
            net_load_scale=torch.maximum(
                net_load.std(), 0.1 * net_load.abs().mean()
            ).clamp_min(0.05),
            net_peak_scale=torch.maximum(
                net_peak.std(), 0.1 * net_peak.abs().mean()
            ).clamp_min(0.05),
            price_spread_scale=torch.maximum(
                spread.std(), 0.1 * spread.abs().mean()
            ).clamp_min(1.0),
            carbon_spread_scale=torch.maximum(
                carbon_spread.std(), 0.1 * carbon_spread.abs().mean()
            ).clamp_min(0.01),
        )

    def carbon_spread_loss(
        self,
        prediction_time_major: torch.Tensor,
        target_time_major: torch.Tensor,
    ) -> torch.Tensor:
        """Scaled paired error in the within-horizon carbon-intensity swing."""

        index = 3 * self.layout.field_size + 3
        predicted = self.layout.raw(prediction_time_major)[..., index]
        target = self.layout.raw(target_time_major)[..., index]
        predicted_spread = predicted.max(dim=1).values - predicted.min(dim=1).values
        target_spread = target.max(dim=1).values - target.min(dim=1).values
        return torch.mean(
            ((predicted_spread - target_spread) / self.carbon_spread_scale) ** 2
        )

    def summaries(self, normalized_time_major: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return ``(net_load[b, t], net_peak[b], price_spread[b])`` in raw units."""

        raw = self.layout.raw(normalized_time_major)
        net_load = self.layout.net_load(raw)
        price = self.layout.price(raw)
        return (
            net_load,
            net_load.max(dim=1).values,
            price.max(dim=1).values - price.min(dim=1).values,
        )

    def paired_losses(
        self,
        prediction_time_major: torch.Tensor,
        target_time_major: torch.Tensor,
        *,
        robust: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scaled per-sample errors of the three decision-relevant summaries.

        ``robust`` switches the squared error for a Huber loss.  A diffusion
        model evaluates this on ``x_0`` predictions made from heavily noised
        inputs, where a squared error on a summary that aggregates a hundred
        channels reaches four orders of magnitude above the noise objective and
        drowns it.  Reconstruction-based models never see residuals that large
        and keep the squared form.
        """

        predicted_net, predicted_peak, predicted_spread = self.summaries(prediction_time_major)
        target_net, target_peak, target_spread = self.summaries(target_time_major)
        residuals = (
            (predicted_net - target_net) / self.net_load_scale,
            (predicted_peak - target_peak) / self.net_peak_scale,
            (predicted_spread - target_spread) / self.price_spread_scale,
        )
        if robust:
            return tuple(  # type: ignore[return-value]
                torch.nn.functional.huber_loss(
                    residual, torch.zeros_like(residual), delta=1.0
                )
                for residual in residuals
            )
        return tuple(torch.mean(residual**2) for residual in residuals)  # type: ignore[return-value]

    def moment_losses(
        self,
        generated_time_major: torch.Tensor,
        observed_time_major: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch-level moment gaps for models without a paired reconstruction.

        A GAN sample has no matching observation, so its physical prior has to be
        imposed distributionally: match the mean and spread of the same three
        summaries across the batch instead of sample by sample.
        """

        generated = self.summaries(generated_time_major)
        observed = self.summaries(observed_time_major)
        scales = (self.net_load_scale, self.net_peak_scale, self.price_spread_scale)
        losses = []
        for generated_value, observed_value, scale in zip(generated, observed, scales):
            mean_gap = (generated_value.mean() - observed_value.mean()) / scale
            std_gap = (generated_value.std() - observed_value.std()) / scale
            losses.append(mean_gap.square() + std_gap.square())
        return tuple(losses)  # type: ignore[return-value]


def batched_indices(count: int, batch_size: int, device: torch.device) -> list[torch.Tensor]:
    permutation = torch.randperm(count, device=device)
    size = min(max(1, batch_size), count)
    return [permutation[start : start + size] for start in range(0, count, size)]
