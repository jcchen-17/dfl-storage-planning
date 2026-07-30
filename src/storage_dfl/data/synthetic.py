from __future__ import annotations

import numpy as np

from storage_dfl.data.schema import Scenario, ScenarioPool
from storage_dfl.network import Feeder


def _smooth_noise(rng: np.random.Generator, length: int, scale: float) -> np.ndarray:
    raw = rng.normal(0.0, scale, length + 2)
    return np.convolve(raw, np.array([0.2, 0.6, 0.2]), mode="valid")


def make_toy_scenarios(
    feeder: Feeder,
    num_scenarios: int,
    horizon: int,
    seed: int,
) -> ScenarioPool:
    """Create ten small, correlated trajectories for an end-to-end smoke test.

    The trajectories are deliberately heterogeneous.  Some are statistically
    ordinary; others contain the low-carbon-charge/high-carbon-discharge ordering
    that matters to the batch-resolved storage model.
    """

    if num_scenarios < 4:
        raise ValueError("At least four scenarios are needed for the DFL demo.")
    if horizon < 4:
        raise ValueError("The horizon must contain at least four intervals.")

    rng = np.random.default_rng(seed)
    t = np.arange(horizon, dtype=float)
    center = 0.47 * (horizon - 1)
    solar_shape = np.exp(-0.5 * ((t - center) / max(1.0, 0.22 * horizon)) ** 2)
    solar_shape[solar_shape < 0.08] = 0.0
    evening_shape = 0.82 + 0.20 * (t / max(1, horizon - 1)) + 0.07 * np.sin(2 * np.pi * t / horizon)

    scenarios: list[Scenario] = []
    for scenario_id in range(num_scenarios):
        archetype = scenario_id % 5
        annual_angle = 2.0 * np.pi * scenario_id / num_scenarios
        context = np.array(
            [np.sin(annual_angle), np.cos(annual_angle), float(scenario_id % 7 >= 5)],
            dtype=float,
        )
        load_scale = [0.90, 1.00, 1.10, 1.18, 0.97][archetype]
        solar_scale = [1.00, 1.20, 0.45, 0.70, 0.92][archetype]
        workload_scale = [0.85, 1.00, 1.08, 1.25, 1.12][archetype]

        temporal_load = np.clip(
            load_scale * evening_shape + _smooth_noise(rng, horizon, 0.025),
            0.70,
            1.45,
        )
        spatial_noise = np.clip(rng.normal(1.0, 0.025, (horizon, len(feeder.buses))), 0.92, 1.08)
        active_load = temporal_load[:, None] * feeder.base_active_load_mw[None, :] * spatial_noise
        reactive_load = active_load * 0.28

        cloud = np.clip(1.0 + _smooth_noise(rng, horizon, 0.06), 0.65, 1.15)
        pv_available = (
            solar_scale
            * solar_shape[:, None]
            * cloud[:, None]
            * feeder.pv_capacity_mw[None, :]
        )

        workload_base = 0.38 + 0.14 * np.sin(2 * np.pi * (t + scenario_id) / horizon) ** 2
        if archetype == 3:
            workload_base[-max(2, horizon // 3) :] += 0.20
        workload = np.clip(
            workload_scale * workload_base + _smooth_noise(rng, horizon, 0.025),
            0.15,
            0.78,
        )

        price = 54.0 + 16.0 * (t / max(1, horizon - 1)) + 4.0 * rng.normal(size=horizon)
        if archetype in (2, 3):
            price[-max(2, horizon // 3) :] += 22.0
        price = np.clip(price, 30.0, 120.0)

        carbon = 0.39 + 0.08 * (t / max(1, horizon - 1)) + _smooth_noise(rng, horizon, 0.018)
        if archetype == 1:
            carbon[: horizon // 2] -= 0.14
            carbon[horizon // 2 :] += 0.15
        elif archetype == 2:
            carbon += 0.16
        elif archetype == 3:
            carbon[-max(2, horizon // 2) :] += 0.22
        elif archetype == 4:
            carbon[: max(2, horizon // 3)] -= 0.09
            carbon[-max(2, horizon // 3) :] += 0.12
        carbon = np.clip(carbon, 0.18, 0.82)

        pue = np.clip(1.18 + 0.02 * rng.normal(size=horizon), 1.12, 1.25)
        grid_available = np.ones(horizon, dtype=float)
        scenarios.append(
            Scenario(
                name=f"toy_{scenario_id:02d}_type_{archetype}",
                context=context,
                active_load_mw=active_load,
                reactive_load_mvar=reactive_load,
                pv_available_mw=pv_available,
                workload_arrival=workload,
                pue=pue,
                grid_price_per_mwh=price,
                grid_carbon_t_per_mwh=carbon,
                grid_available=grid_available,
            )
        )
    return ScenarioPool(tuple(scenarios))
