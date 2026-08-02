from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from storage_dfl.data.schema import Scenario, ScenarioPool
from storage_dfl.network import Feeder


@dataclass(frozen=True)
class ScenarioCodec:
    """Map physical trajectories to the normalized vectors used by the CVAE."""

    feeder: Feeder
    horizon: int
    trajectory_mean: np.ndarray
    trajectory_std: np.ndarray
    context_mean: np.ndarray
    context_std: np.ndarray

    @classmethod
    def fit(cls, pool: ScenarioPool, feeder: Feeder) -> "ScenarioCodec":
        raw = np.stack([cls._pack(scenario) for scenario in pool.scenarios])
        contexts = np.stack([scenario.context for scenario in pool.scenarios])
        return cls(
            feeder=feeder,
            horizon=pool.scenarios[0].horizon,
            trajectory_mean=raw.mean(axis=0),
            trajectory_std=np.maximum(raw.std(axis=0), 1.0e-4),
            context_mean=contexts.mean(axis=0),
            context_std=np.maximum(contexts.std(axis=0), 1.0e-4),
        )

    @classmethod
    def from_normalization_dict(cls, payload: dict, feeder: Feeder) -> "ScenarioCodec":
        if int(payload.get("layout_version", 0)) != 2:
            raise ValueError(
                "Normalization layout is not the phase-resolved v2 format; retrain the CVAE."
            )
        return cls(
            feeder=feeder,
            horizon=int(payload["horizon"]),
            trajectory_mean=np.asarray(payload["trajectory_mean"], dtype=np.float32),
            trajectory_std=np.asarray(payload["trajectory_std"], dtype=np.float32),
            context_mean=np.asarray(payload["context_mean"], dtype=np.float32),
            context_std=np.asarray(payload["context_std"], dtype=np.float32),
        )

    @staticmethod
    def _pack(scenario: Scenario) -> np.ndarray:
        horizon = scenario.horizon
        time_major = np.concatenate(
            (
                scenario.active_load_mw.reshape(horizon, -1),
                scenario.reactive_load_mvar.reshape(horizon, -1),
                scenario.pv_available_mw.reshape(horizon, -1),
                scenario.workload_arrival[:, None],
                scenario.pue[:, None],
                scenario.grid_price_per_mwh[:, None],
                scenario.grid_carbon_t_per_mwh[:, None],
            ),
            axis=1,
        )
        return time_major.reshape(-1).astype(np.float32)

    @property
    def trajectory_dim(self) -> int:
        return int(self.trajectory_mean.size)

    @property
    def context_dim(self) -> int:
        return int(self.context_mean.size)

    def encode_pool(self, pool: ScenarioPool) -> tuple[np.ndarray, np.ndarray]:
        trajectories = np.stack([self._pack(scenario) for scenario in pool.scenarios])
        contexts = np.stack([scenario.context for scenario in pool.scenarios])
        x = (trajectories - self.trajectory_mean) / self.trajectory_std
        c = (contexts - self.context_mean) / self.context_std
        return x.astype(np.float32), c.astype(np.float32)

    def decode_batch(
        self,
        normalized_trajectories: np.ndarray,
        normalized_contexts: np.ndarray,
        *,
        name_prefix: str,
    ) -> tuple[Scenario, ...]:
        if normalized_trajectories.ndim != 2:
            raise ValueError("Decoded trajectories must have shape [scenario, feature].")
        raw = normalized_trajectories * self.trajectory_std + self.trajectory_mean
        contexts = normalized_contexts * self.context_std + self.context_mean
        return tuple(
            self._unpack(raw[index], contexts[index], f"{name_prefix}_{index:02d}")
            for index in range(raw.shape[0])
        )

    def _unpack(self, vector: np.ndarray, context: np.ndarray, name: str) -> Scenario:
        buses = len(self.feeder.buses)
        phases = len(self.feeder.phases)
        field_size = buses * phases
        matrix = vector.reshape(self.horizon, 3 * field_size + 4)
        active = matrix[:, :field_size].reshape(self.horizon, buses, phases)
        reactive = matrix[:, field_size : 2 * field_size].reshape(
            self.horizon, buses, phases
        )
        pv = matrix[:, 2 * field_size : 3 * field_size].reshape(
            self.horizon, buses, phases
        )
        workload, pue, price, carbon = (
            matrix[:, 3 * field_size + index] for index in range(4)
        )

        active_limit = 2.0 * self.feeder.base_active_load_mw + 0.05
        reactive_limit = 2.0 * self.feeder.base_reactive_load_mvar + 0.05
        active = np.clip(active, 0.0, active_limit[None, :, :])
        reactive = np.clip(reactive, 0.0, reactive_limit[None, :, :])
        active[:, self.feeder.base_active_load_mw == 0.0] = 0.0
        reactive[:, self.feeder.base_reactive_load_mvar == 0.0] = 0.0
        pv = np.clip(pv, 0.0, 1.30 * self.feeder.pv_capacity_mw[None, :, :])
        pv[:, self.feeder.pv_capacity_mw == 0.0] = 0.0

        scenario = Scenario(
            name=name,
            context=np.asarray(context, dtype=float),
            active_load_mw=np.asarray(active, dtype=float),
            reactive_load_mvar=np.asarray(reactive, dtype=float),
            pv_available_mw=np.asarray(pv, dtype=float),
            workload_arrival=np.clip(workload, 0.05, 1.0),
            pue=np.clip(pue, 1.05, 1.50),
            grid_price_per_mwh=np.clip(price, 5.0, 300.0),
            grid_carbon_t_per_mwh=np.clip(carbon, 0.02, 1.10),
            grid_available=np.ones(self.horizon, dtype=float),
        )
        scenario.validate()
        return scenario

    def support_conditions(self, count: int) -> np.ndarray:
        """Place fixed conditional anchors at quantiles of the context manifold."""

        if count <= 0:
            raise ValueError("The support-scenario count must be positive.")
        # Contexts are standardized before this method is called by the pipeline.
        # The anchors span the dominant conditional direction without choosing any
        # realized trajectory from the historical pool.
        quantiles = np.linspace(-0.75, 0.75, count, dtype=np.float32)
        anchors = np.zeros((count, self.context_dim), dtype=np.float32)
        anchors[:, 0] = quantiles
        return anchors

    def normalization_dict(self) -> dict[str, object]:
        return {
            "layout_version": 2,
            "layout": "time-major:[P_bus_phase,Q_bus_phase,PV_bus_phase,workload,pue,price,carbon]",
            "horizon": self.horizon,
            "trajectory_mean": self.trajectory_mean.tolist(),
            "trajectory_std": self.trajectory_std.tolist(),
            "context_mean": self.context_mean.tolist(),
            "context_std": self.context_std.tolist(),
        }
