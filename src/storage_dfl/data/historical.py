from __future__ import annotations

from pathlib import Path

import numpy as np

from storage_dfl.data.schema import Scenario, ScenarioPool


def load_historical_scenarios(
    path: str | Path,
    *,
    split: str | None = None,
    horizon: int | None = None,
) -> ScenarioPool:
    """Load node-phase trajectories, optionally truncating each window."""

    with np.load(Path(path), allow_pickle=False) as payload:
        available_splits = np.asarray(payload["split"]).astype(str)
        if split is None:
            selected = np.arange(available_splits.size)
        else:
            if split not in {"train", "validation", "test"}:
                raise ValueError("split must be train, validation, test, or None")
            selected = np.flatnonzero(available_splits == split)
        if selected.size == 0:
            raise ValueError(f"No scenarios found for split {split!r}")

        dataset_horizon = int(payload["workload_arrival"].shape[1])
        selected_horizon = dataset_horizon if horizon is None else int(horizon)
        if selected_horizon <= 0 or selected_horizon > dataset_horizon:
            raise ValueError(
                f"horizon must be between 1 and the dataset horizon {dataset_horizon}"
            )

        names = np.asarray(payload["scenario_name"]).astype(str)
        # Optional, so every dataset built before rare events existed keeps the
        # planning model's uniform 8760 / horizon.
        sample_weights = (
            np.asarray(payload["sample_weight"][selected], dtype=float)
            if "sample_weight" in payload.files
            else None
        )
        arrays = {
            field: np.asarray(payload[field][selected])
            for field in (
                "context",
                "active_load_phase_mw",
                "reactive_load_phase_mvar",
                "pv_available_phase_mw",
                "workload_arrival",
                "pue",
                "grid_price_per_mwh",
                "grid_carbon_t_per_mwh",
                "grid_available",
            )
        }
        scenarios = tuple(
            Scenario(
                name=str(names[source_index]),
                context=np.asarray(arrays["context"][local_index], dtype=float),
                active_load_mw=np.asarray(arrays["active_load_phase_mw"][local_index, :selected_horizon], dtype=float),
                reactive_load_mvar=np.asarray(arrays["reactive_load_phase_mvar"][local_index, :selected_horizon], dtype=float),
                pv_available_mw=np.asarray(arrays["pv_available_phase_mw"][local_index, :selected_horizon], dtype=float),
                workload_arrival=np.asarray(arrays["workload_arrival"][local_index, :selected_horizon], dtype=float),
                pue=np.asarray(arrays["pue"][local_index, :selected_horizon], dtype=float),
                grid_price_per_mwh=np.asarray(arrays["grid_price_per_mwh"][local_index, :selected_horizon], dtype=float),
                grid_carbon_t_per_mwh=np.asarray(arrays["grid_carbon_t_per_mwh"][local_index, :selected_horizon], dtype=float),
                grid_available=np.asarray(arrays["grid_available"][local_index, :selected_horizon], dtype=float),
                # Probabilities, rather than per-row annual occurrences, remain
                # composable when several normal/outage representatives share
                # one planning model.
                annual_occurrences=None,
                probability_weight=(
                    float(sample_weights[local_index])
                    if sample_weights is not None
                    else None
                ),
            )
            for local_index, source_index in enumerate(selected)
        )
    return ScenarioPool(scenarios)
