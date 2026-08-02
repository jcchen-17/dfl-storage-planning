import numpy as np
import torch

from storage_dfl.data import (
    ScenarioCodec,
    ScenarioPool,
    load_historical_scenarios,
    make_toy_scenarios,
)
from storage_dfl.dfl import DirectSupportPolicy
from storage_dfl.models import ConditionalVAE
from storage_dfl.network import ieee13_unbalanced_microgrid


def test_ieee13_is_radial() -> None:
    feeder = ieee13_unbalanced_microgrid()
    assert len(feeder.buses) == 13
    assert len(feeder.lines) == 12
    assert feeder.data_center_bus == "675"
    assert feeder.phase_mask.shape == (13, 3)
    assert feeder.base_active_load_mw.shape == (13, 3)
    assert np.isclose(feeder.base_active_load_mw.sum(), 3.466)
    assert np.isclose(feeder.base_reactive_load_mvar.sum(), 2.102)
    pv_buses = {
        bus
        for bus, capacity in zip(feeder.buses, feeder.pv_capacity_bus_mw, strict=True)
        if capacity > 0.0
    }
    assert pv_buses == {"634", "675", "680"}
    assert feeder.storage_candidates == ("632", "671", "675", "680")


def test_codec_cvae_and_direct_support() -> None:
    feeder = ieee13_unbalanced_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=10, horizon=6, seed=7)
    codec = ScenarioCodec.fit(pool, feeder)
    trajectories, contexts = codec.encode_pool(pool)
    assert trajectories.shape == (10, 6 * (3 * 13 * 3 + 4))
    assert contexts.shape == (10, 3)

    model = ConditionalVAE(
        trajectory_dim=codec.trajectory_dim,
        context_dim=codec.context_dim,
        latent_dim=4,
        hidden_dim=16,
    )
    conditions = torch.tensor(codec.support_conditions(2))
    decoded = model.decode(torch.zeros(2, 4), conditions).detach().numpy()
    generated = codec.decode_batch(decoded, conditions.numpy(), name_prefix="test")
    ScenarioPool(generated)
    assert all(scenario.num_buses == 13 for scenario in generated)
    assert all(scenario.num_phases == 3 for scenario in generated)

    policy = DirectSupportPolicy(support_count=2, latent_dim=4, seed=7)
    sample = policy.sample(exploration_std=0.2)
    assert sample.latent.shape == (2, 4)
    assert torch.isclose(sample.weights.sum(), torch.tensor(1.0))
    (-sample.log_probability).backward()
    assert policy.latent_location.grad is not None


def test_historical_loader_truncates_horizon(tmp_path) -> None:
    path = tmp_path / "windows.npz"
    scenario_count, dataset_horizon, buses, phases = 2, 5, 13, 3
    phase_values = np.zeros((scenario_count, dataset_horizon, buses, phases), dtype=np.float32)
    vectors = np.ones((scenario_count, dataset_horizon), dtype=np.float32)
    np.savez_compressed(
        path,
        split=np.asarray(["train", "test"]),
        scenario_name=np.asarray(["train_000", "test_000"]),
        context=np.zeros((scenario_count, 4), dtype=np.float32),
        active_load_phase_mw=phase_values,
        reactive_load_phase_mvar=phase_values,
        pv_available_phase_mw=phase_values,
        workload_arrival=vectors,
        pue=vectors,
        grid_price_per_mwh=vectors,
        grid_carbon_t_per_mwh=vectors,
        grid_available=vectors,
    )

    pool = load_historical_scenarios(path, split="train", horizon=3)

    assert len(pool.scenarios) == 1
    assert pool.scenarios[0].horizon == 3
