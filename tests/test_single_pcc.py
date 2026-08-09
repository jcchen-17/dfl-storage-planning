from dataclasses import replace

import numpy as np
import torch

from storage_dfl.config import load_config
from storage_dfl.data import ScenarioCodec, make_toy_scenarios
from storage_dfl.network import single_pcc_microgrid
from storage_dfl.models import ConditionalVAE, train_cvae
from storage_dfl.planning import SinglePCCPlanningOracle, StoragePlanningOracle
from storage_dfl.stages import _experiment_data


def test_single_pcc_factory_and_historical_aggregation() -> None:
    config = load_config("configs/dataset_v2_dfl_hourly_layered.yaml")
    feeder, pool = _experiment_data(config, config.data.train_split)

    assert feeder.buses == ("PCC",)
    assert feeder.lines == ()
    assert feeder.storage_candidates == ("PCC",)
    scenario = pool.scenarios[0]
    expected = config.data_center.power_mw(
        scenario.pue, scenario.workload_arrival
    )
    assert scenario.active_load_mw.shape == (config.data.horizon, 1, 3)
    assert np.allclose(scenario.active_load_mw.sum(axis=(1, 2)), expected)
    assert np.all(scenario.reactive_load_mvar == 0.0)
    assert any(np.any(item.grid_available < 0.5) for item in pool.scenarios)
    assert all(item.annual_occurrences is None for item in pool.scenarios)
    assert np.isclose(sum(pool.normalized_weights()), 1.0)
    outage_mass = sum(
        weight
        for item, weight in zip(pool.scenarios, pool.normalized_weights(), strict=True)
        if np.any(item.grid_available < 0.5)
    )
    assert 0.005 < outage_mass < 0.01

    codec = ScenarioCodec.fit(pool, feeder)
    assert codec.trajectory_dim == config.data.horizon * (3 * 1 * 3 + 4)
    oracle = StoragePlanningOracle(
        feeder, config.planning, config.costs, config.data, config.data_center
    )
    assert isinstance(oracle, SinglePCCPlanningOracle)


def test_single_pcc_layered_model_solves_small_case(monkeypatch) -> None:
    # Keep the smoke solve in-process. The production pipeline still exercises
    # the isolated worker whenever torch has initialized on Windows.
    monkeypatch.setenv("STORAGE_DFL_SOLVER_WORKER", "1")
    config = load_config("configs/dataset_v2_dfl_hourly_layered.yaml")
    feeder = single_pcc_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=4, horizon=4, seed=29)
    planning = replace(
        config.planning,
        solver_backend="scip",
        solver_time_limit_seconds=20.0,
        solver_relative_gap=0.0,
        solver_threads=1,
        dc_carbon_cap=1.10,
    )
    oracle = SinglePCCPlanningOracle(
        feeder, planning, config.costs, config.data, config.data_center
    )
    result = oracle.solve(pool.scenarios[:1], allow_carbon_slack=False)

    assert result.feasible
    assert np.isfinite(result.objective)
    assert result.design.site.keys() == {"PCC"}
    assert 0.0 <= result.peak_grid_mw <= planning.grid_limit_mw
    assert result.carbon_ledger is not None
    assert result.carbon_ledger["served_dc_energy_mwh"] > 0.0
    assert result.carbon_ledger["delivered_dc_carbon_t"] >= 0.0


def test_single_pcc_codec_trains_with_empty_reactive_mask() -> None:
    config = load_config("configs/dataset_v2_dfl_hourly_layered.yaml")
    feeder = single_pcc_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=8, horizon=4, seed=31)
    codec = ScenarioCodec.fit(pool, feeder)
    trajectories, contexts = codec.encode_pool(pool)
    model = ConditionalVAE(
        trajectory_dim=codec.trajectory_dim,
        context_dim=codec.context_dim,
        latent_dim=2,
        hidden_dim=8,
    )
    history = train_cvae(
        model,
        trajectories,
        contexts,
        replace(config.cvae, epochs=1, batch_size=4),
        horizon=4,
        device=torch.device("cpu"),
        seed=3,
        trajectory_mean=codec.trajectory_mean,
        trajectory_std=codec.trajectory_std,
        field_masks=codec.field_masks(),
    )

    assert len(history) == 1
    assert np.isfinite(history[0].loss)


def test_single_pcc_rejects_non_layered_carbon() -> None:
    config = load_config("configs/dataset_v2_dfl_hourly_layered.yaml")
    feeder = single_pcc_microgrid()
    planning = replace(config.planning, carbon_formulation="system_average")
    try:
        SinglePCCPlanningOracle(
            feeder, planning, config.costs, config.data, config.data_center
        )
    except ValueError as error:
        assert "layered_pcc" in str(error)
    else:
        raise AssertionError("single-PCC oracle accepted a non-layered carbon model")
