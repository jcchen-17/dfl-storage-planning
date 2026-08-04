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
    context_anchors: np.ndarray

    @staticmethod
    def _farthest_point_indices(features: np.ndarray, count: int) -> np.ndarray:
        """Select diverse observed rows without inventing off-manifold anchors."""

        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("Anchor features must be a non-empty matrix.")
        count = min(int(count), features.shape[0])
        center = features.mean(axis=0, keepdims=True)
        first = int(np.argmax(np.sum((features - center) ** 2, axis=1)))
        selected = [first]
        available = np.ones(features.shape[0], dtype=bool)
        available[first] = False
        minimum_distance = np.sum((features - features[first]) ** 2, axis=1)
        while len(selected) < count:
            next_index = int(np.argmax(np.where(available, minimum_distance, -np.inf)))
            selected.append(next_index)
            available[next_index] = False
            distance = np.sum((features - features[next_index]) ** 2, axis=1)
            minimum_distance = np.minimum(minimum_distance, distance)
        return np.asarray(selected, dtype=np.int64)

    @classmethod
    def fit(cls, pool: ScenarioPool, feeder: Feeder) -> "ScenarioCodec":
        raw = np.stack([cls._pack(scenario) for scenario in pool.scenarios])
        contexts = np.stack([scenario.context for scenario in pool.scenarios])
        context_mean = contexts.mean(axis=0)
        context_std = np.maximum(contexts.std(axis=0), 1.0e-4)
        normalized_contexts = (contexts - context_mean) / context_std
        anchor_indices = cls._farthest_point_indices(
            normalized_contexts,
            min(16, len(pool.scenarios)),
        )
        return cls(
            feeder=feeder,
            horizon=pool.scenarios[0].horizon,
            trajectory_mean=raw.mean(axis=0),
            trajectory_std=np.maximum(raw.std(axis=0), 1.0e-4),
            context_mean=context_mean,
            context_std=context_std,
            context_anchors=normalized_contexts[anchor_indices].astype(np.float32),
        )

    @classmethod
    def from_normalization_dict(cls, payload: dict, feeder: Feeder) -> "ScenarioCodec":
        layout_version = int(payload.get("layout_version", 0))
        if layout_version not in {2, 3}:
            raise ValueError(
                "Normalization layout is not a supported phase-resolved format; retrain the CVAE."
            )
        return cls(
            feeder=feeder,
            horizon=int(payload["horizon"]),
            trajectory_mean=np.asarray(payload["trajectory_mean"], dtype=np.float32),
            trajectory_std=np.asarray(payload["trajectory_std"], dtype=np.float32),
            context_mean=np.asarray(payload["context_mean"], dtype=np.float32),
            context_std=np.asarray(payload["context_std"], dtype=np.float32),
            context_anchors=np.asarray(
                payload.get("context_anchors", []),
                dtype=np.float32,
            ).reshape(-1, len(payload["context_mean"])),
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

    @classmethod
    def pack(cls, scenario: Scenario) -> np.ndarray:
        """Public packing, used to measure how much clipping a sample needed."""

        return cls._pack(scenario)

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
        """Return diverse conditions that were actually observed during fitting."""

        if count <= 0:
            raise ValueError("The support-scenario count must be positive.")
        if self.context_anchors.shape[0] == 0:
            # Backward-compatible fallback for v2 normalization artifacts. New
            # runs always persist observed anchors and should not enter here.
            quantiles = np.linspace(-0.75, 0.75, count, dtype=np.float32)
            anchors = np.zeros((count, self.context_dim), dtype=np.float32)
            anchors[:, 0] = quantiles
            return anchors
        indices = self._farthest_point_indices(self.context_anchors, count)
        return self.context_anchors[indices].copy()

    def support_indices(self, pool: ScenarioPool, count: int) -> np.ndarray:
        """Choose real scenarios spanning context and decision-relevant extremes."""

        if count <= 0:
            raise ValueError("The support-scenario count must be positive.")
        _, contexts = self.encode_pool(pool)
        metrics = []
        for scenario in pool.scenarios:
            net_load = scenario.active_load_mw.sum(axis=(1, 2)) - scenario.pv_available_mw.sum(axis=(1, 2))
            metrics.append(
                (
                    float(net_load.max()),
                    float(np.quantile(net_load, 0.95)),
                    float(np.ptp(scenario.grid_price_per_mwh)),
                    float(scenario.grid_carbon_t_per_mwh.max()),
                    float(scenario.workload_arrival.max()),
                )
            )
        decision_features = np.asarray(metrics, dtype=np.float32)
        decision_features = (decision_features - decision_features.mean(axis=0)) / np.maximum(
            decision_features.std(axis=0), 1.0e-4
        )
        return self._farthest_point_indices(
            np.concatenate((contexts, decision_features), axis=1),
            count,
        )

    def field_masks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Masks for physically present P, Q and PV node-phase channels."""

        return (
            (self.feeder.base_active_load_mw.reshape(-1) > 0.0),
            (self.feeder.base_reactive_load_mvar.reshape(-1) > 0.0),
            (self.feeder.pv_capacity_mw.reshape(-1) > 0.0),
        )

    def normalization_dict(self) -> dict[str, object]:
        return {
            "layout_version": 3,
            "layout": "time-major:[P_bus_phase,Q_bus_phase,PV_bus_phase,workload,pue,price,carbon]",
            "horizon": self.horizon,
            "trajectory_mean": self.trajectory_mean.tolist(),
            "trajectory_std": self.trajectory_std.tolist(),
            "context_mean": self.context_mean.tolist(),
            "context_std": self.context_std.tolist(),
            "context_anchors": self.context_anchors.tolist(),
        }
