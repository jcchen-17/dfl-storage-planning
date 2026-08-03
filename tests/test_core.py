import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from storage_dfl.data import (
    ScenarioCodec,
    ScenarioPool,
    load_historical_scenarios,
    make_toy_scenarios,
)
from storage_dfl.dfl import DirectSupportPolicy
from storage_dfl.dfl.scenario_bo import (
    gp_lower_confidence_bound,
    scenario_features,
    select_supports,
)
from storage_dfl.config import CVAEConfig, DFLConfig, load_config
from storage_dfl.dfl.trainer import INFEASIBLE_LOSS, _rank_advantages, _safe_number
from storage_dfl.planning.model import StoragePlanningOracle
from storage_dfl.models import ConditionalVAE, train_cvae
from storage_dfl.network import ieee13_unbalanced_microgrid
from storage_dfl.planning import PlanningResult, StorageDesign
from storage_dfl.dfl.scenario_bo import train_scenario_bo


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
    assert any(np.allclose(conditions[0].numpy(), context) for context in contexts)
    assert any(np.allclose(conditions[1].numpy(), context) for context in contexts)
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

    single_policy = DirectSupportPolicy(support_count=1, latent_dim=4, seed=9)
    single_sample = single_policy.sample(exploration_std=0.2)
    assert torch.equal(single_sample.weights, torch.ones(1))
    (-single_sample.log_probability).backward()
    assert single_policy.latent_location.grad is not None
    assert single_policy.weight_logit_location.grad is None

    initialized = DirectSupportPolicy(
        support_count=2,
        latent_dim=4,
        seed=7,
        initial_latent=torch.ones(2, 4),
    )
    assert torch.allclose(initialized.latent_location, torch.ones(2, 4))


def test_decision_aware_cvae_loss_runs() -> None:
    feeder = ieee13_unbalanced_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=8, horizon=6, seed=11)
    codec = ScenarioCodec.fit(pool, feeder)
    trajectories, contexts = codec.encode_pool(pool)
    model = ConditionalVAE(
        trajectory_dim=codec.trajectory_dim,
        context_dim=codec.context_dim,
        latent_dim=4,
        hidden_dim=16,
    )
    history = train_cvae(
        model,
        trajectories,
        contexts,
        CVAEConfig(
            latent_dim=4,
            hidden_dim=16,
            epochs=1,
            learning_rate=1.0e-3,
            beta=0.002,
            ramp_weight=0.05,
            batch_size=4,
            kl_warmup_epochs=2,
        ),
        horizon=6,
        device=torch.device("cpu"),
        seed=7,
        trajectory_mean=codec.trajectory_mean,
        trajectory_std=codec.trajectory_std,
        field_masks=codec.field_masks(),
    )
    assert len(history) == 1
    assert np.isfinite(history[0].loss)
    assert history[0].beta == 0.001
    assert history[0].net_peak >= 0.0
    assert history[0].price_spread >= 0.0


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


def test_rank_advantages_survive_an_infeasible_sample() -> None:
    # Standardizing raw costs against INFEASIBLE_LOSS collapsed the feasible
    # samples to indistinguishable advantages; ranking must keep them separated.
    losses = np.asarray([1.0e6, 1.2e6, INFEASIBLE_LOSS])
    advantages = _rank_advantages(losses)

    assert advantages.shape == (3,)
    assert np.all(np.abs(advantages) <= 1.0)
    assert advantages[0] < advantages[1] < advantages[2]
    assert abs(advantages[1] - advantages[0]) > 0.5
    # Cheaper than the mean must be rewarded, more expensive penalized.
    assert advantages[0] < 0.0 < advantages[2]

    tied = _rank_advantages(np.asarray([5.0, 5.0, 9.0]))
    assert tied[0] == tied[1]
    assert tied[2] > tied[0]

    single = _rank_advantages(np.asarray([3.0]))
    assert single.shape == (1,)
    assert single[0] == 0.0


def test_safe_number_replaces_infeasible_sentinels() -> None:
    assert _safe_number(12.5, 0.0) == 12.5
    assert _safe_number(float("inf"), INFEASIBLE_LOSS) == INFEASIBLE_LOSS
    assert _safe_number(float("nan"), 0.0) == 0.0
    # The history is written with allow_nan=False, so this must be serializable.
    json.dumps({"objective": _safe_number(float("inf"), INFEASIBLE_LOSS)}, allow_nan=False)


def test_warm_start_cache_is_shared_across_designs() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml")
    feeder = ieee13_unbalanced_microgrid()
    oracle = StoragePlanningOracle(feeder, config.planning, config.costs, config.data)
    pool = make_toy_scenarios(feeder, num_scenarios=4, horizon=6, seed=31)

    design = StorageDesign(
        site={bus: int(bus == "680") for bus in feeder.storage_candidates},
        power_mw={bus: (0.2 if bus == "680" else 0.0) for bus in feeder.storage_candidates},
        energy_mwh={bus: (0.5 if bus == "680" else 0.0) for bus in feeder.storage_candidates},
    )
    weights = (0.25, 0.25, 0.25, 0.25)
    free_key = oracle._warm_start_key(pool.scenarios, weights, True)
    fixed_key = oracle._warm_start_key(pool.scenarios, weights, True)

    # The bootstrap never sees the design, so every design evaluated against the
    # same scenarios must reuse one cached bootstrap.
    assert free_key == fixed_key
    assert oracle._warm_start_key(pool.scenarios, weights, False) != free_key
    assert oracle._warm_start_key(pool.scenarios[:1], (1.0,), True) != free_key

    assert oracle._warm_start_supported(None)
    assert oracle._warm_start_supported(design) == (config.planning.self_discharge == 0.0)
    decaying = replace(config.planning, self_discharge=0.01)
    decaying_oracle = StoragePlanningOracle(feeder, decaying, config.costs, config.data)
    # Idle storage is not a valid seed once the inventory decays.
    assert decaying_oracle._warm_start_supported(None)
    assert not decaying_oracle._warm_start_supported(design)


def test_scenario_bo_features_selection_and_gp() -> None:
    feeder = ieee13_unbalanced_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=6, horizon=6, seed=19)
    features = scenario_features(pool.scenarios)
    assert features.shape == (6, 5)
    assert np.all(np.isfinite(features))

    parameters = np.asarray([1.0, 0.5, 2.0, 0.25, 0.1])
    selected, weights = select_supports(features, parameters, support_count=3, weight_floor=0.02)
    assert selected.shape == (3,)
    assert len(set(selected.tolist())) == 3
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.02)

    single_selected, single_weights = select_supports(
        features, parameters, support_count=1, weight_floor=0.02
    )
    assert single_selected.shape == (1,)
    assert np.array_equal(single_weights, np.ones(1))

    observed_x = np.asarray([[-1.0], [1.0]])
    observed_y = np.asarray([1.0, 3.0])
    noise = np.asarray([0.01, 0.01])
    candidates = np.asarray([[-1.0], [0.0], [1.0]])
    acquisition = gp_lower_confidence_bound(
        observed_x,
        observed_y,
        noise,
        candidates,
        length_scale=1.0,
        exploration=0.1,
    )
    assert acquisition.shape == (3,)
    assert np.all(np.isfinite(acquisition))
    assert acquisition[0] < acquisition[2]


def test_scenario_bo_training_pipeline_with_black_box_oracle() -> None:
    feeder = ieee13_unbalanced_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=8, horizon=6, seed=23)
    codec = ScenarioCodec.fit(pool, feeder)
    model = ConditionalVAE(
        trajectory_dim=codec.trajectory_dim,
        context_dim=codec.context_dim,
        latent_dim=4,
        hidden_dim=16,
    )

    class FakeOracle:
        def solve(self, scenarios, *, weights=None, fixed_design=None, **kwargs):
            scenario_tuple = tuple(scenarios)
            buses = feeder.storage_candidates
            if fixed_design is None:
                spread = max(float(np.ptp(s.grid_price_per_mwh)) for s in scenario_tuple)
                installed_bus = "680" if spread > 1.0 else None
                design = StorageDesign(
                    site={bus: int(bus == installed_bus) for bus in buses},
                    power_mw={bus: (0.2 if bus == installed_bus else 0.0) for bus in buses},
                    energy_mwh={bus: (0.5 if bus == installed_bus else 0.0) for bus in buses},
                )
            else:
                design = fixed_design
            objective = 1000.0 - 25.0 * len(design.installed_buses)
            return PlanningResult(
                status="optimal",
                objective=objective,
                investment_cost=10.0 * len(design.installed_buses),
                operating_cost=objective - 10.0 * len(design.installed_buses),
                carbon_slack_cost=0.0,
                peak_grid_mw=1.0,
                design=design,
                scenario_names=tuple(s.name for s in scenario_tuple),
                solve_time_seconds=0.01,
                relative_gap=0.0,
            )

    config = DFLConfig(
        num_support_scenarios=2,
        epochs=1,
        validation_batch_size=2,
        final_validation_size=2,
        learning_rate=0.01,
        baseline_momentum=0.75,
        initial_exploration_std=0.2,
        minimum_exploration_std=0.1,
        exploration_decay=0.9,
        diversity_margin=0.8,
        diversity_weight=0.05,
        weight_entropy_weight=0.005,
        device="cpu",
        method="scenario_bo",
        candidate_pool_size=4,
        bo_initial_evaluations=2,
        bo_iterations=1,
        bo_candidate_draws=16,
        bo_finalists=1,
    )
    result = train_scenario_bo(model, codec, pool, FakeOracle(), config, seed=7)
    assert len(result.history) == 3
    assert len(result.generated_scenarios) == 2
    assert np.isclose(sum(result.scenario_weights), 1.0)
    assert result.support_latent.shape == (2, 4)
    assert result.support_conditions.shape == (2, codec.context_dim)
    assert result.full_validation_result.feasible
