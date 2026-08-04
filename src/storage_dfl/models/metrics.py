"""Generative-quality metrics for choosing between conditional generators.

Stage one of the generator comparison: measure how well each model reproduces
the held-out scenario distribution, before paying for any SCIP solves.  The
metrics are grouped so a model can be rejected for the right reason:

* **reconstruction** -- can the model represent a real scenario at all?  A model
  that cannot invert an observation also cannot give the DFL policy a sensible
  starting point.
* **decision statistics** -- 1-Wasserstein distance on the summaries the planner
  actually prices: peak net load (demand charge), price spread (arbitrage),
  carbon and workload extremes.  A generator can have excellent pixel-level
  fidelity and still be useless here.
* **distribution** -- energy distance over whole standardized trajectories,
  catching mode collapse that per-statistic distances miss.
* **coverage / density** -- the generative analogue of recall and precision:
  does the sample set span the observed manifold, and does it stay on it?
* **feasibility** -- how much of each sample the codec has to clip away.  Mass
  removed by clipping is structure the model invented outside the physical box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from storage_dfl.data import ScenarioCodec, ScenarioPool

DECISION_STATISTICS = (
    "peak_net_load_mw",
    "mean_net_load_mw",
    "net_load_ramp_std_mw",
    "price_spread_per_mwh",
    "mean_price_per_mwh",
    "peak_carbon_t_per_mwh",
    "peak_workload",
    "pv_energy_mwh",
)


def decision_statistics(pool_scenarios) -> dict[str, np.ndarray]:
    """Per-scenario summaries of everything the planning objective prices."""

    values: dict[str, list[float]] = {name: [] for name in DECISION_STATISTICS}
    for scenario in pool_scenarios:
        load = scenario.active_load_mw.sum(axis=(1, 2))
        pv = scenario.pv_available_mw.sum(axis=(1, 2))
        net_load = load - pv
        values["peak_net_load_mw"].append(float(net_load.max()))
        values["mean_net_load_mw"].append(float(net_load.mean()))
        values["net_load_ramp_std_mw"].append(float(np.diff(net_load).std()))
        values["price_spread_per_mwh"].append(float(np.ptp(scenario.grid_price_per_mwh)))
        values["mean_price_per_mwh"].append(float(scenario.grid_price_per_mwh.mean()))
        values["peak_carbon_t_per_mwh"].append(float(scenario.grid_carbon_t_per_mwh.max()))
        values["peak_workload"].append(float(scenario.workload_arrival.max()))
        values["pv_energy_mwh"].append(float(pv.sum()))
    return {name: np.asarray(value, dtype=float) for name, value in values.items()}


def wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    """1-Wasserstein distance between two empirical 1-D samples."""

    grid = np.concatenate((left, right))
    grid.sort()
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    left_cdf = np.searchsorted(left_sorted, grid, side="right") / left_sorted.size
    right_cdf = np.searchsorted(right_sorted, grid, side="right") / right_sorted.size
    return float(np.sum(np.abs(left_cdf - right_cdf)[:-1] * np.diff(grid)))


def pairwise_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix without materialising a rank-3 difference.

    Trajectories are ~5,800-dimensional, so the broadcast form
    ``left[:, None, :] - right[None, :, :]`` needs hundreds of gigabytes at the
    sample counts this comparison uses.  ``torch.cdist`` keeps the footprint at
    the size of the output matrix.
    """

    import torch

    return torch.cdist(
        torch.as_tensor(left, dtype=torch.float32),
        torch.as_tensor(right, dtype=torch.float32),
    ).numpy()


def energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Multivariate energy distance; zero exactly when the distributions match."""

    cross = float(pairwise_distances(left, right).mean())
    within_left = float(pairwise_distances(left, left).mean())
    within_right = float(pairwise_distances(right, right).mean())
    return float(2.0 * cross - within_left - within_right)


def _nearest_neighbour_radii(points: np.ndarray, neighbours: int) -> np.ndarray:
    distances = pairwise_distances(points, points)
    np.fill_diagonal(distances, np.inf)
    order = np.sort(distances, axis=1)
    return order[:, min(neighbours, order.shape[1]) - 1]


def project_to_real_subspace(
    real: np.ndarray,
    generated: np.ndarray,
    components: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Whiten both sets onto the leading principal axes of the observed data.

    Nearest-neighbour manifold metrics are degenerate on raw 5,800-dimensional
    trajectories: every k-NN radius is far smaller than every cross-set
    distance, so precision and recall both collapse to zero for every model and
    the metric carries no information.  Fitting the basis on the real data alone
    keeps it independent of which generator is being scored.
    """

    centre = real.mean(axis=0, keepdims=True)
    centred = real - centre
    _, singular_values, right = np.linalg.svd(centred, full_matrices=False)
    rank = min(components, right.shape[0])
    basis = right[:rank].T
    scale = np.maximum(singular_values[:rank] / np.sqrt(max(real.shape[0] - 1, 1)), 1.0e-8)
    return ((real - centre) @ basis) / scale, ((generated - centre) @ basis) / scale


def precision_recall(
    real: np.ndarray,
    generated: np.ndarray,
    neighbours: int = 3,
) -> tuple[float, float]:
    """Kynkaanniemi improved precision and recall.

    Precision is the fraction of generated samples inside the real manifold --
    physical plausibility.  Recall is the fraction of real samples covered by the
    generated manifold -- diversity.  A collapsed GAN scores high precision and
    near-zero recall, which is exactly the failure this comparison must catch.

    Callers must pass a low-dimensional representation; see
    ``project_to_real_subspace``.
    """

    real_radii = _nearest_neighbour_radii(real, neighbours)
    generated_radii = _nearest_neighbour_radii(generated, neighbours)
    cross = pairwise_distances(generated, real)
    precision = float(np.mean(np.any(cross <= real_radii[None, :], axis=1)))
    recall = float(np.mean(np.any(cross.T <= generated_radii[None, :], axis=1)))
    return precision, recall


@dataclass(frozen=True)
class ClippingReport:
    relative_mass: float
    maximum_relative_mass: float


def clipping_report(
    codec: ScenarioCodec,
    decoded_normalized: np.ndarray,
    scenarios,
) -> ClippingReport:
    """How much of each generated trajectory the physical clipping removed.

    ``ScenarioCodec._unpack`` clips loads, PV, price and carbon into the feasible
    box.  Large clipped mass means the raw sample left the physical manifold and
    only survives because the codec repaired it, which flatters the model's
    apparent feasibility.
    """

    raw = decoded_normalized * codec.trajectory_std + codec.trajectory_mean
    repaired = np.stack([codec.pack(scenario) for scenario in scenarios])
    denominator = np.maximum(np.abs(raw).sum(axis=1), 1.0e-8)
    relative = np.abs(repaired - raw).sum(axis=1) / denominator
    return ClippingReport(
        relative_mass=float(relative.mean()),
        maximum_relative_mass=float(relative.max()),
    )


def field_reconstruction_errors(
    reference: np.ndarray,
    reconstruction: np.ndarray,
    codec: ScenarioCodec,
    horizon: int,
) -> dict[str, float]:
    """Normalized per-field reconstruction RMSE of ``decode(encode(x), c)``."""

    features_per_hour = reference.shape[1] // horizon
    field_size = (features_per_hour - 4) // 3
    reference_time = reference.reshape(-1, horizon, features_per_hour)
    reconstruction_time = reconstruction.reshape(-1, horizon, features_per_hour)
    masks = codec.field_masks()
    errors: dict[str, float] = {}
    spatial = (
        ("active_load", 0, masks[0]),
        ("reactive_load", 1, masks[1]),
        ("pv", 2, masks[2]),
    )
    for name, index, mask in spatial:
        start = index * field_size
        difference = (
            reconstruction_time[:, :, start : start + field_size]
            - reference_time[:, :, start : start + field_size]
        )
        errors[name] = float(np.sqrt((difference[:, :, mask] ** 2).mean()))
    for offset, name in enumerate(("workload", "pue", "price", "carbon")):
        column = 3 * field_size + offset
        difference = reconstruction_time[:, :, column] - reference_time[:, :, column]
        errors[name] = float(np.sqrt((difference**2).mean()))
    errors["all_fields"] = float(np.sqrt(((reconstruction - reference) ** 2).mean()))
    return errors


def autocorrelation_error(
    real_scenarios,
    generated_scenarios,
    max_lag: int = 24,
) -> float:
    """Mean absolute gap between the net-load autocorrelation profiles.

    Marginal statistics cannot tell a plausible daily shape from white noise with
    the right mean and variance; the storage schedule depends entirely on shape.
    """

    def profile(scenarios) -> np.ndarray:
        curves = []
        for scenario in scenarios:
            series = (
                scenario.active_load_mw.sum(axis=(1, 2))
                - scenario.pv_available_mw.sum(axis=(1, 2))
            )
            series = series - series.mean()
            variance = float((series**2).mean())
            if variance <= 1.0e-12:
                curves.append(np.zeros(max_lag))
                continue
            lags = min(max_lag, series.size - 1)
            curve = np.array(
                [float((series[lag:] * series[:-lag]).mean() / variance) for lag in range(1, lags + 1)]
            )
            if curve.size < max_lag:
                curve = np.pad(curve, (0, max_lag - curve.size))
            curves.append(curve)
        return np.stack(curves).mean(axis=0)

    return float(np.abs(profile(real_scenarios) - profile(generated_scenarios)).mean())


def standardized_matrix(pool: ScenarioPool, codec: ScenarioCodec) -> np.ndarray:
    trajectories, _ = codec.encode_pool(pool)
    return trajectories
