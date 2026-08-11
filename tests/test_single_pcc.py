from dataclasses import replace

import numpy as np
import torch

from storage_dfl.config import load_config
from storage_dfl.data import (
    ScenarioCodec,
    load_historical_scenarios,
    make_toy_scenarios,
)
from storage_dfl.network import ieee13_unbalanced_microgrid, single_pcc_microgrid
from storage_dfl.models import ConditionalVAE, train_cvae
from storage_dfl.planning import SinglePCCPlanningOracle, StoragePlanningOracle
from storage_dfl.stages import _experiment_data


def test_single_pcc_factory_and_historical_aggregation() -> None:
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")
    feeder, pool = _experiment_data(config, config.data.train_split)

    assert config.data.delta_t_hours == 1.0
    assert config.data.horizon == 48
    assert config.data.pcc_pv_capacity_mw == 5.0
    assert config.planning.grid_limit_mw == 15.0
    assert config.planning.backup_generator_mw == 6.0
    assert config.planning.outage_carbon_cap == 0.72
    assert config.planning.max_power_mw == 7.5
    assert config.planning.max_energy_mwh == 60.0
    assert feeder.buses == ("PCC",)
    assert feeder.lines == ()
    assert feeder.storage_candidates == ("PCC",)
    scenario = pool.scenarios[0]
    source = load_historical_scenarios(
        config.data.dataset_path,
        split=config.data.train_split,
        horizon=config.data.horizon,
    ).scenarios[0]
    source_feeder = ieee13_unbalanced_microgrid()
    source_index = source_feeder.bus_index[config.data.pcc_source_bus]
    expected = config.data_center.power_mw(
        scenario.pue, scenario.workload_arrival
    )
    source_capacity = source_feeder.pv_capacity_mw[source_index].sum()
    expected_pv = (
        source.pv_available_mw[:, source_index, :].sum(axis=1)
        * config.data.pcc_pv_capacity_mw
        / source_capacity
    )
    assert scenario.active_load_mw.shape == (config.data.horizon, 1, 3)
    assert np.allclose(scenario.active_load_mw.sum(axis=(1, 2)), expected)
    assert np.allclose(scenario.pv_available_mw.sum(axis=(1, 2)), expected_pv)
    assert np.isclose(feeder.pv_capacity_mw.sum(), 5.0)
    assert 7.4 < scenario.active_load_mw.sum(axis=(1, 2)).min() < 10.3
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
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")
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
    assert result.recourse_diagnostics is not None
    exposure = np.asarray(result.recourse_diagnostics.grid_carbon_exposure_mw)
    assert exposure.shape == (1, 4, 4)
    assert np.all(exposure >= 0.0)


def test_grid_connected_carbon_pressure_cannot_shed_load(monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_DFL_SOLVER_WORKER", "1")
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")
    feeder = single_pcc_microgrid()
    base = make_toy_scenarios(feeder, num_scenarios=4, horizon=4, seed=30)
    scenario = replace(
        base.scenarios[0],
        pv_available_mw=np.zeros_like(base.scenarios[0].pv_available_mw),
        grid_available=np.ones(4),
        grid_carbon_t_per_mwh=np.full(4, 0.70),
    )
    planning = replace(
        config.planning,
        solver_backend="scip",
        solver_time_limit_seconds=20.0,
        solver_relative_gap=0.0,
        solver_threads=1,
        max_storage_sites=0,
        backup_generator_mw=0.0,
        dc_carbon_cap=0.05,
    )
    oracle = SinglePCCPlanningOracle(
        feeder, planning, config.costs, config.data, config.data_center
    )
    result = oracle.solve((scenario,), allow_carbon_slack=True)
    assert result.feasible
    assert result.recourse_diagnostics is not None
    assert result.recourse_diagnostics.load_shedding_mwh < 1.0e-7
    assert result.recourse_diagnostics.carbon_excess_t > 0.0


def test_outage_carbon_budget_allows_diesel_reliability(monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_DFL_SOLVER_WORKER", "1")
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")
    feeder = single_pcc_microgrid()
    base = make_toy_scenarios(feeder, num_scenarios=4, horizon=4, seed=301)
    scenario = replace(
        base.scenarios[0],
        pv_available_mw=np.zeros_like(base.scenarios[0].pv_available_mw),
        grid_available=np.zeros(4),
    )
    peak_load = float(scenario.active_load_mw.sum(axis=(1, 2)).max())
    planning = replace(
        config.planning,
        solver_backend="scip",
        solver_time_limit_seconds=20.0,
        solver_relative_gap=0.0,
        solver_threads=1,
        max_storage_sites=0,
        backup_generator_mw=peak_load + 0.1,
        dc_carbon_cap=0.28,
        outage_carbon_cap=0.72,
        carbon_cap_scope="horizon",
    )
    result = SinglePCCPlanningOracle(
        feeder, planning, config.costs, config.data, config.data_center
    ).solve((scenario,), allow_carbon_slack=False)
    assert result.feasible
    assert result.recourse_diagnostics is not None
    assert result.recourse_diagnostics.load_shedding_mwh < 1.0e-7
    assert result.recourse_diagnostics.carbon_excess_t < 1.0e-7


def test_single_pcc_codec_trains_with_empty_reactive_mask() -> None:
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")
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


def test_single_pcc_accepts_average_carbon_baseline() -> None:
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")
    feeder = single_pcc_microgrid()
    planning = replace(config.planning, carbon_formulation="average_pcc")
    oracle = SinglePCCPlanningOracle(
        feeder, planning, config.costs, config.data, config.data_center
    )
    assert oracle.planning.carbon_formulation == "average_pcc"


def test_average_carbon_baseline_uses_source_emissions(monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_DFL_SOLVER_WORKER", "1")
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")
    feeder = single_pcc_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=4, horizon=4, seed=302)
    planning = replace(
        config.planning,
        carbon_formulation="average_pcc",
        solver_backend="scip",
        solver_time_limit_seconds=20.0,
        solver_relative_gap=0.0,
        solver_threads=1,
        dc_carbon_cap=1.10,
    )
    result = SinglePCCPlanningOracle(
        feeder, planning, config.costs, config.data, config.data_center
    ).solve(pool.scenarios[:1], allow_carbon_slack=False)
    assert result.feasible
    assert result.carbon_ledger is not None
    assert np.isclose(
        result.carbon_ledger["delivered_dc_carbon_t"],
        result.carbon_ledger["source_operational_carbon_t"],
    )
    assert result.recourse_diagnostics is not None
    exposure = np.asarray(result.recourse_diagnostics.grid_carbon_exposure_mw)
    assert exposure.shape == (1, 4, 4)
    assert np.allclose(exposure[0], np.diag(np.diag(exposure[0])))


def test_single_pcc_rejects_unknown_carbon_formulation() -> None:
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")
    feeder = single_pcc_microgrid()
    planning = replace(config.planning, carbon_formulation="unknown")
    try:
        SinglePCCPlanningOracle(
            feeder, planning, config.costs, config.data, config.data_center
        )
    except ValueError as error:
        assert "average_pcc" in str(error)
    else:
        raise AssertionError("single-PCC oracle accepted a non-layered carbon model")
