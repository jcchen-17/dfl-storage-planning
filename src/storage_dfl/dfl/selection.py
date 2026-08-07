"""Non-learned scenario-selection rules the decision-focused method must beat.

Each rule answers the same question the DFL stage answers -- which K scenarios
should the planner see -- but using only the observed distribution, never the
resulting decision.  That contrast is the point: if a rule that never looks at
the planner does just as well, the decision feedback is not earning its cost.

``farthest`` is the rule already used to seed the DFL stage.  It maximises
coverage, which makes it a poor representative at small K: the point farthest
from the centre is an extreme day, not a typical one.  ``kmeans`` and
``aggregate`` are the conventional scenario-reduction answers and are the ones a
reviewer will expect to see.
"""

from __future__ import annotations

import numpy as np

from storage_dfl.data import Scenario, ScenarioCodec, ScenarioPool

SELECTION_RULES = ("farthest", "kmeans", "aggregate", "random")

# Rules usable to SEED the DFL support. Narrower than SELECTION_RULES because
# the seed has to be a real pool member: the trainer encodes it to obtain a
# starting latent, and ``aggregate`` returns synthetic cluster means that have
# no index to encode from.
SUPPORT_INIT_RULES = ("farthest", "kmeans", "random")


def support_init_indices(
    rule: str,
    pool: ScenarioPool,
    codec: ScenarioCodec,
    count: int,
    seed: int = 0,
) -> np.ndarray:
    """Pool indices that seed the DFL support set.

    Separate from ``ScenarioCodec.support_indices`` on purpose. That method also
    chooses the fixed validation subset and the reported test subset, so changing
    it would move the evaluation sets and silently invalidate every sweep already
    recorded. This function only ever picks a starting point for the policy.

    ``farthest`` is the historical default and stays the default so existing runs
    reproduce. It is a poor seed at K=1: it returns the row farthest from the
    centroid, which in this dataset is an extreme-net-load day whose price spread
    is flat -- and price spread is what storage value tracks (r = +0.86 against
    out-of-sample storage value across the K=1 sweep rows, versus r = -0.51 for
    peak net load). The K=1 seed it picks scores 0 on its own.
    """

    if rule not in SUPPORT_INIT_RULES:
        raise ValueError(
            f"support_init_rule must be one of {SUPPORT_INIT_RULES}, got {rule!r}."
        )
    count = min(int(count), len(pool.scenarios))
    if count <= 0:
        raise ValueError("count must be positive.")

    if rule == "farthest":
        return codec.support_indices(pool, count)

    if rule == "random":
        rng = np.random.default_rng(seed)
        return np.sort(
            rng.choice(len(pool.scenarios), size=count, replace=False)
        ).astype(np.int64)

    # ``kmeans``: the medoid of each cluster, matching what select_scenarios
    # returns for the same rule so the seed and the baseline are the same points.
    features = decision_feature_matrix(pool, codec)
    labels, centres = _kmeans(features, count, seed)
    indices = []
    for index in range(count):
        members = np.flatnonzero(labels == index)
        if members.size == 0:
            members = np.arange(features.shape[0])
        distances = ((features[members] - centres[index]) ** 2).sum(axis=1)
        indices.append(int(members[int(distances.argmin())]))
    return np.asarray(indices, dtype=np.int64)


def decision_feature_matrix(pool: ScenarioPool, codec: ScenarioCodec) -> np.ndarray:
    """Standardized context plus decision-relevant summaries.

    This is the same space ``ScenarioCodec.support_indices`` selects in, so every
    rule here is comparable to the farthest-point baseline rather than being
    confounded by a different notion of distance.
    """

    _, contexts = codec.encode_pool(pool)
    metrics = []
    for scenario in pool.scenarios:
        net_load = (
            scenario.active_load_mw.sum(axis=(1, 2))
            - scenario.pv_available_mw.sum(axis=(1, 2))
        )
        metrics.append(
            (
                float(net_load.max()),
                float(np.quantile(net_load, 0.95)),
                float(np.ptp(scenario.grid_price_per_mwh)),
                float(scenario.grid_carbon_t_per_mwh.max()),
                float(scenario.workload_arrival.max()),
            )
        )
    features = np.asarray(metrics, dtype=np.float64)
    features = (features - features.mean(axis=0)) / np.maximum(features.std(axis=0), 1.0e-4)
    return np.concatenate((contexts.astype(np.float64), features), axis=1)


def _kmeans(
    features: np.ndarray,
    clusters: int,
    seed: int,
    restarts: int = 8,
    iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with k-means++ seeding; returns labels and centres."""

    rng = np.random.default_rng(seed)
    best_labels: np.ndarray | None = None
    best_centres: np.ndarray | None = None
    best_inertia = np.inf
    for _ in range(restarts):
        centres = [features[rng.integers(features.shape[0])]]
        while len(centres) < clusters:
            distances = np.min(
                np.stack([((features - c) ** 2).sum(axis=1) for c in centres]), axis=0
            )
            total = distances.sum()
            if total <= 0.0:
                centres.append(features[rng.integers(features.shape[0])])
                continue
            centres.append(features[rng.choice(features.shape[0], p=distances / total)])
        centre_array = np.stack(centres)
        labels = np.zeros(features.shape[0], dtype=int)
        for _ in range(iterations):
            distances = ((features[:, None, :] - centre_array[None, :, :]) ** 2).sum(axis=2)
            new_labels = distances.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for index in range(clusters):
                members = features[labels == index]
                if members.size:
                    centre_array[index] = members.mean(axis=0)
        inertia = float(
            ((features - centre_array[labels]) ** 2).sum()
        )
        if inertia < best_inertia:
            best_inertia, best_labels, best_centres = inertia, labels, centre_array
    assert best_labels is not None and best_centres is not None
    return best_labels, best_centres


def select_scenarios(
    rule: str,
    pool: ScenarioPool,
    codec: ScenarioCodec,
    count: int,
    seed: int = 0,
) -> tuple[tuple[Scenario, ...], tuple[float, ...], tuple[str, ...]]:
    """Return ``(scenarios, weights, source_labels)`` for one selection rule."""

    if rule not in SELECTION_RULES:
        raise ValueError(f"rule must be one of {SELECTION_RULES}, got {rule!r}.")
    count = min(int(count), len(pool.scenarios))
    if count <= 0:
        raise ValueError("count must be positive.")

    if rule == "farthest":
        indices = codec.support_indices(pool, count).tolist()
        scenarios = pool.subset(indices)
        return scenarios, (1.0 / count,) * count, tuple(pool.names(indices))

    if rule == "random":
        rng = np.random.default_rng(seed)
        indices = sorted(
            rng.choice(len(pool.scenarios), size=count, replace=False).tolist()
        )
        scenarios = pool.subset(indices)
        return scenarios, (1.0 / count,) * count, tuple(pool.names(indices))

    features = decision_feature_matrix(pool, codec)
    labels, centres = _kmeans(features, count, seed)
    # Cluster shares are the natural probabilities: they make the reduced set an
    # estimator of the observed distribution rather than an arbitrary subset.
    weights = np.array(
        [max(float((labels == index).sum()), 1.0) for index in range(count)]
    )
    weights = weights / weights.sum()

    if rule == "kmeans":
        indices = []
        for index in range(count):
            members = np.flatnonzero(labels == index)
            if members.size == 0:
                members = np.arange(features.shape[0])
            distances = ((features[members] - centres[index]) ** 2).sum(axis=1)
            indices.append(int(members[int(distances.argmin())]))
        scenarios = pool.subset(indices)
        return scenarios, tuple(weights.tolist()), tuple(pool.names(indices))

    # ``aggregate``: each representative is the pointwise mean of its cluster.
    # At count == 1 this is exactly the expected-value scenario, the classical
    # deterministic reduction. Normalization is affine, so averaging normalized
    # trajectories equals averaging physical ones.
    trajectories, contexts = codec.encode_pool(pool)
    mean_trajectories = []
    mean_contexts = []
    for index in range(count):
        members = np.flatnonzero(labels == index)
        if members.size == 0:
            members = np.arange(trajectories.shape[0])
        mean_trajectories.append(trajectories[members].mean(axis=0))
        mean_contexts.append(contexts[members].mean(axis=0))
    scenarios = codec.decode_batch(
        np.stack(mean_trajectories),
        np.stack(mean_contexts),
        name_prefix="aggregate",
    )
    labels_out = tuple(
        f"cluster{index:02d}_n{int((labels == index).sum())}" for index in range(count)
    )
    return scenarios, tuple(weights.tolist()), labels_out
