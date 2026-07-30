import numpy as np
import torch

from storage_dfl.data import ScenarioCodec, ScenarioPool, make_toy_scenarios
from storage_dfl.dfl import DirectSupportPolicy
from storage_dfl.models import ConditionalVAE
from storage_dfl.network import ieee13_balanced_microgrid


def test_ieee13_is_radial() -> None:
    feeder = ieee13_balanced_microgrid()
    assert len(feeder.buses) == 13
    assert len(feeder.lines) == 12
    assert feeder.data_center_bus in feeder.buses
    assert np.count_nonzero(feeder.pv_capacity_mw) == 3


def test_codec_cvae_and_direct_support() -> None:
    feeder = ieee13_balanced_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=10, horizon=6, seed=7)
    codec = ScenarioCodec.fit(pool, feeder)
    trajectories, contexts = codec.encode_pool(pool)
    assert trajectories.shape == (10, 6 * (2 * 13 + 4))
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

    policy = DirectSupportPolicy(support_count=2, latent_dim=4, seed=7)
    sample = policy.sample(exploration_std=0.2)
    assert sample.latent.shape == (2, 4)
    assert torch.isclose(sample.weights.sum(), torch.tensor(1.0))
    (-sample.log_probability).backward()
    assert policy.latent_location.grad is not None
