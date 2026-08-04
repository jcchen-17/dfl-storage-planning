from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import numpy as np
import torch

from storage_dfl.config import DFLConfig
from storage_dfl.data import Scenario, ScenarioCodec, ScenarioPool
from storage_dfl.models import ConditionalGenerator
from storage_dfl.planning import PlanningResult, StoragePlanningOracle

from .trainer import INFEASIBLE_LOSS, resolve_device


FEATURE_NAMES = (
    "peak_net_load",
    "q95_net_load",
    "price_spread",
    "peak_carbon",
    "pv_surplus",
)


@dataclass(frozen=True)
class BOEvaluationRecord:
    evaluation: int
    phase: str
    parameters: tuple[float, ...]
    selected_pool_indices: tuple[int, ...]
    selected_source_names: tuple[str, ...]
    scenario_weights: tuple[float, ...]
    planning_status: str
    planning_objective: float
    planning_relative_gap: float
    validation_status: str
    validation_objective: float
    validation_relative_gap: float
    observation_uncertainty: float
    installed_buses: tuple[str, ...]
    power_mw: dict[str, float]
    energy_mwh: dict[str, float]
    planning_solve_seconds: float
    validation_solve_seconds: float

    @property
    def feasible(self) -> bool:
        return isfinite(self.validation_objective) and self.validation_objective < INFEASIBLE_LOSS


@dataclass(frozen=True)
class ScenarioBOTrainingResult:
    generated_scenarios: tuple[Scenario, ...]
    scenario_weights: tuple[float, ...]
    support_latent: np.ndarray
    support_conditions: np.ndarray
    support_source_names: tuple[str, ...]
    candidate_source_names: tuple[str, ...]
    planning_result: PlanningResult
    full_validation_result: PlanningResult
    history: tuple[BOEvaluationRecord, ...]
    device: str
    feature_names: tuple[str, ...]
    best_parameters: tuple[float, ...]
    finalist_evaluations: tuple[dict[str, object], ...]

    def history_as_dicts(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self.history]


def scenario_features(scenarios: tuple[Scenario, ...]) -> np.ndarray:
    """Decision-relevant scenario descriptors used by the low-dimensional selector."""

    rows = []
    for scenario in scenarios:
        load = scenario.active_load_mw.sum(axis=(1, 2))
        pv = scenario.pv_available_mw.sum(axis=(1, 2))
        net = load - pv
        rows.append(
            (
                float(net.max()),
                float(np.quantile(net, 0.95)),
                float(np.ptp(scenario.grid_price_per_mwh)),
                float(scenario.grid_carbon_t_per_mwh.max()),
                float(np.maximum(pv - load, 0.0).max()),
            )
        )
    values = np.asarray(rows, dtype=float)
    return (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1.0e-6)


def select_supports(
    features: np.ndarray,
    parameters: np.ndarray,
    support_count: int,
    weight_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select top-scoring candidates and assign stable softmax weights."""

    if features.ndim != 2 or parameters.shape != (features.shape[1],):
        raise ValueError("Scenario feature and BO parameter shapes do not match.")
    if not 0.0 <= weight_floor < 1.0 / support_count:
        raise ValueError("bo_weight_floor must be in [0, 1 / num_support_scenarios).")
    feature_scores = features @ parameters
    # Preserve a small neighborhood around the representative ordering. Without
    # this tie-break prior, theta=0 is successful but any arbitrarily small BO
    # perturbation can replace all selected supports at once.
    rank_prior = np.linspace(0.15, 0.0, features.shape[0], dtype=float)
    scores = feature_scores + rank_prior
    # Stable ordering makes the zero-vector baseline select the representative
    # candidates in the order returned by ScenarioCodec.support_indices.
    selected = np.argsort(-scores, kind="stable")[:support_count]
    selected_scores = feature_scores[selected]
    softmax = np.exp(selected_scores - selected_scores.max())
    softmax /= softmax.sum()
    weights = weight_floor + (1.0 - support_count * weight_floor) * softmax
    weights /= weights.sum()
    return selected.astype(int), weights.astype(float)


def _kernel(first: np.ndarray, second: np.ndarray, length_scale: float) -> np.ndarray:
    distances = np.sum((first[:, None, :] - second[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * distances / max(length_scale**2, 1.0e-8))


def gp_lower_confidence_bound(
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    observed_noise: np.ndarray,
    candidate_x: np.ndarray,
    *,
    length_scale: float,
    exploration: float,
) -> np.ndarray:
    """Return a minimization LCB from a small heteroscedastic NumPy GP."""

    if observed_x.shape[0] == 0:
        return np.zeros(candidate_x.shape[0], dtype=float)
    center = float(observed_y.mean())
    scale = max(float(observed_y.std()), 1.0)
    normalized_y = (observed_y - center) / scale
    normalized_noise = np.clip(observed_noise / scale, 1.0e-6, 10.0)
    covariance = _kernel(observed_x, observed_x, length_scale)
    covariance += np.diag(normalized_noise**2 + 1.0e-8)
    jitter = 1.0e-8
    for _ in range(6):
        try:
            factor = np.linalg.cholesky(covariance + jitter * np.eye(covariance.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        raise RuntimeError("Unable to stabilize the Bayesian-optimization GP covariance.")
    cross = _kernel(observed_x, candidate_x, length_scale)
    alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, normalized_y))
    mean = cross.T @ alpha
    projected = np.linalg.solve(factor, cross)
    variance = np.maximum(1.0 - np.sum(projected**2, axis=0), 1.0e-10)
    return mean - exploration * np.sqrt(variance)


def _design_signature(result: PlanningResult) -> tuple[object, ...]:
    return (
        tuple(sorted(result.design.installed_buses)),
        tuple(round(result.design.power_mw[bus], 5) for bus in result.design.power_mw),
        tuple(round(result.design.energy_mwh[bus], 5) for bus in result.design.energy_mwh),
    )


def _safe_number(value: float, fallback: float) -> float:
    numeric = float(value)
    return numeric if isfinite(numeric) else float(fallback)


def _representative_scenarios(
    codec: ScenarioCodec,
    pool: ScenarioPool,
    count: int,
) -> tuple[Scenario, ...]:
    indices = codec.support_indices(pool, min(count, len(pool.scenarios)))
    return pool.subset(indices.tolist())


@torch.no_grad()
def _candidate_pool(
    cvae: ConditionalGenerator,
    codec: ScenarioCodec,
    observed_pool: ScenarioPool,
    count: int,
    device: torch.device,
) -> tuple[tuple[Scenario, ...], np.ndarray, np.ndarray, tuple[str, ...]]:
    trajectories, contexts = codec.encode_pool(observed_pool)
    indices = codec.support_indices(observed_pool, min(count, len(observed_pool.scenarios)))
    conditions = torch.as_tensor(contexts[indices], dtype=torch.float32, device=device)
    latent, _ = cvae.encode(
        torch.as_tensor(trajectories[indices], dtype=torch.float32, device=device),
        conditions,
    )
    decoded = cvae.decode(latent, conditions).cpu().numpy()
    scenarios = codec.decode_batch(decoded, conditions.cpu().numpy(), name_prefix="bo_pool")
    source_names = tuple(observed_pool.scenarios[int(index)].name for index in indices)
    return (
        scenarios,
        latent.cpu().numpy().copy(),
        conditions.cpu().numpy().copy(),
        source_names,
    )


def _initial_parameters(dimension: int, count: int, bound: float, rng: np.random.Generator) -> list[np.ndarray]:
    points = [np.zeros(dimension, dtype=float)]
    for axis in range(dimension):
        point = np.zeros(dimension, dtype=float)
        point[axis] = bound
        points.append(point)
        if len(points) >= count:
            return points
    while len(points) < count:
        points.append(rng.uniform(-bound, bound, size=dimension))
    return points


def train_scenario_bo(
    cvae: ConditionalGenerator,
    codec: ScenarioCodec,
    observed_pool: ScenarioPool,
    oracle: StoragePlanningOracle,
    config: DFLConfig,
    seed: int,
    writer: Any | None = None,
) -> ScenarioBOTrainingResult:
    """Decision-focused candidate selection and weight search around a frozen CVAE."""

    if config.candidate_pool_size < config.num_support_scenarios:
        raise ValueError("candidate_pool_size must be at least num_support_scenarios.")
    if config.bo_initial_evaluations <= 0 or config.bo_iterations < 0:
        raise ValueError("BO evaluation counts must be nonnegative with at least one initial point.")
    device = resolve_device(config.device)
    cvae.to(device).eval()
    rng = np.random.default_rng(seed + 17)
    pool, pool_latent, pool_conditions, pool_names = _candidate_pool(
        cvae,
        codec,
        observed_pool,
        config.candidate_pool_size,
        device,
    )
    features = scenario_features(pool)
    fixed_validation = _representative_scenarios(
        codec, observed_pool, config.validation_batch_size
    )
    final_validation = _representative_scenarios(
        codec, observed_pool, config.final_validation_size
    )
    total_evaluations = config.bo_initial_evaluations + config.bo_iterations
    initial = _initial_parameters(
        features.shape[1], config.bo_initial_evaluations, config.bo_parameter_bound, rng
    )
    parameters_seen: list[np.ndarray] = []
    losses: list[float] = []
    noises: list[float] = []
    records: list[BOEvaluationRecord] = []
    evaluation_payloads: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Scenario, ...], PlanningResult, PlanningResult]
    ] = []
    selection_keys: set[tuple[object, ...]] = set()

    for evaluation in range(total_evaluations):
        if evaluation < len(initial):
            parameters = initial[evaluation]
            phase = "initial"
        else:
            random_candidates = rng.uniform(
                -config.bo_parameter_bound,
                config.bo_parameter_bound,
                size=(config.bo_candidate_draws, features.shape[1]),
            )
            best_index = int(np.argmin(np.asarray(losses) + np.asarray(noises)))
            local = np.clip(
                parameters_seen[best_index]
                + rng.normal(
                    0.0,
                    0.03 * config.bo_parameter_bound,
                    size=(max(64, config.bo_candidate_draws // 2), features.shape[1]),
                ),
                -config.bo_parameter_bound,
                config.bo_parameter_bound,
            )
            candidates = np.concatenate((random_candidates, local), axis=0)
            observed_x = np.asarray(parameters_seen) / config.bo_parameter_bound
            candidate_x = candidates / config.bo_parameter_bound
            acquisition = gp_lower_confidence_bound(
                observed_x,
                np.asarray(losses),
                np.asarray(noises),
                candidate_x,
                length_scale=config.bo_kernel_length_scale,
                exploration=config.bo_exploration,
            )
            parameters = candidates[int(np.argmin(acquisition))]
            phase = "bo"

        selected, weights = select_supports(
            features,
            np.asarray(parameters),
            config.num_support_scenarios,
            config.bo_weight_floor,
        )
        key = (tuple(selected.tolist()), tuple(np.round(weights, 4).tolist()))
        if key in selection_keys:
            # The exact downstream map is piecewise constant. Jitter duplicate
            # selector outputs so an expensive SCIP call is not knowingly spent
            # on an identical scenario set and weight vector.
            for _ in range(128):
                proposal = rng.uniform(
                    -config.bo_parameter_bound,
                    config.bo_parameter_bound,
                    size=features.shape[1],
                )
                proposal_selected, proposal_weights = select_supports(
                    features,
                    proposal,
                    config.num_support_scenarios,
                    config.bo_weight_floor,
                )
                proposal_key = (
                    tuple(proposal_selected.tolist()),
                    tuple(np.round(proposal_weights, 4).tolist()),
                )
                if proposal_key not in selection_keys:
                    parameters, selected, weights, key = (
                        proposal,
                        proposal_selected,
                        proposal_weights,
                        proposal_key,
                    )
                    break
        selection_keys.add(key)
        scenarios = tuple(pool[int(index)] for index in selected)
        print(
            f"DFL-BO evaluation {evaluation + 1}/{total_evaluations} ({phase}): "
            f"supports={selected.tolist()}, weights={np.round(weights, 3).tolist()}",
            flush=True,
        )
        plan = oracle.solve(scenarios, weights=weights.tolist())
        validation = (
            oracle.solve(
                fixed_validation,
                fixed_design=plan.design,
                allow_carbon_slack=True,
                use_cache=True,
            )
            if plan.feasible
            else plan
        )
        loss = (
            float(validation.objective)
            if validation.feasible and isfinite(validation.objective)
            else INFEASIBLE_LOSS
        )
        relative_gap = (
            max(0.0, float(validation.relative_gap))
            if validation.feasible and isfinite(validation.relative_gap)
            else 1.0
        )
        planning_gap = (
            max(0.0, float(plan.relative_gap))
            if plan.feasible and isfinite(plan.relative_gap)
            else 1.0
        )
        uncertainty = max(
            abs(loss) * relative_gap,
            abs(float(plan.objective)) * planning_gap
            if plan.feasible and isfinite(plan.objective)
            else abs(loss),
            abs(loss) * config.decision_deadband_relative,
            1.0,
        )
        parameters_seen.append(np.asarray(parameters, dtype=float).copy())
        losses.append(loss)
        noises.append(uncertainty)
        record = BOEvaluationRecord(
            evaluation=evaluation,
            phase=phase,
            parameters=tuple(float(value) for value in parameters),
            selected_pool_indices=tuple(int(index) for index in selected),
            selected_source_names=tuple(pool_names[int(index)] for index in selected),
            scenario_weights=tuple(float(value) for value in weights),
            planning_status=plan.status,
            planning_objective=_safe_number(plan.objective, INFEASIBLE_LOSS),
            planning_relative_gap=_safe_number(plan.relative_gap, 1.0),
            validation_status=validation.status,
            validation_objective=_safe_number(validation.objective, INFEASIBLE_LOSS),
            validation_relative_gap=_safe_number(validation.relative_gap, 1.0),
            observation_uncertainty=float(uncertainty),
            installed_buses=tuple(plan.design.installed_buses),
            power_mw=dict(plan.design.power_mw),
            energy_mwh=dict(plan.design.energy_mwh),
            planning_solve_seconds=float(plan.solve_time_seconds),
            validation_solve_seconds=float(validation.solve_time_seconds),
        )
        records.append(record)
        evaluation_payloads.append((np.asarray(parameters), selected, weights, scenarios, plan, validation))
        print(
            f"  plan={plan.status}/{plan.objective:.6g} gap={plan.relative_gap:.4g}; "
            f"validation={validation.status}/{validation.objective:.6g} "
            f"gap={validation.relative_gap:.4g}; installed={plan.design.installed_buses}, "
            f"P={sum(plan.design.power_mw.values()):.4f} MW, "
            f"E={sum(plan.design.energy_mwh.values()):.4f} MWh",
            flush=True,
        )
        if writer is not None:
            writer.add_scalar("bo/validation_objective", loss, evaluation)
            writer.add_scalar("bo/observation_uncertainty", uncertainty, evaluation)
            writer.add_scalar("bo/planning_gap", float(plan.relative_gap), evaluation)
            writer.add_scalar("bo/validation_gap", relative_gap, evaluation)
            writer.add_scalar("design/installed_sites", len(plan.design.installed_buses), evaluation)
            writer.flush()

    feasible_indices = [
        index
        for index, payload in enumerate(evaluation_payloads)
        if payload[4].feasible and payload[5].feasible
    ]
    if not feasible_indices:
        raise RuntimeError("Scenario BO did not produce a feasible planning and validation pair.")
    # A low validation incumbent reached from a poorly solved planning model is
    # not equally trustworthy. Rank finalists by a conservative upper score;
    # the final fixed-design evaluation then compares their realized costs.
    ranked = sorted(
        feasible_indices,
        key=lambda index: losses[index] + noises[index],
    )
    finalist_indices: list[int] = []
    finalist_designs: set[tuple[object, ...]] = set()
    for index in ranked:
        signature = _design_signature(evaluation_payloads[index][4])
        if signature not in finalist_designs:
            finalist_indices.append(index)
            finalist_designs.add(signature)
        if len(finalist_indices) >= max(1, config.bo_finalists):
            break

    chosen_index = finalist_indices[0]
    chosen_full_validation: PlanningResult | None = None
    chosen_full_loss = float("inf")
    finalist_evaluations: list[dict[str, object]] = []
    for index in finalist_indices:
        plan = evaluation_payloads[index][4]
        full_validation = oracle.solve(
            final_validation,
            fixed_design=plan.design,
            allow_carbon_slack=True,
            use_cache=True,
        )
        full_loss = (
            float(full_validation.objective)
            if full_validation.feasible
            else INFEASIBLE_LOSS
        )
        finalist_evaluations.append(
            {
                "evaluation": int(index),
                "installed_buses": list(plan.design.installed_buses),
                "power_mw": dict(plan.design.power_mw),
                "energy_mwh": dict(plan.design.energy_mwh),
                "validation_status": full_validation.status,
                "validation_objective": float(full_validation.objective),
                "validation_relative_gap": float(full_validation.relative_gap),
                "validation_solve_seconds": float(full_validation.solve_time_seconds),
            }
        )
        print(
            f"DFL-BO finalist evaluation {index + 1}: "
            f"validation={full_validation.status}/{full_validation.objective:.6g} "
            f"gap={full_validation.relative_gap:.4g}; "
            f"installed={plan.design.installed_buses}",
            flush=True,
        )
        deadband = max(
            config.decision_deadband_relative * max(abs(full_loss), abs(chosen_full_loss) if isfinite(chosen_full_loss) else 0.0),
            1.0,
        )
        if chosen_full_validation is None or full_loss < chosen_full_loss - deadband:
            chosen_index = index
            chosen_full_validation = full_validation
            chosen_full_loss = full_loss

    if chosen_full_validation is None or not chosen_full_validation.feasible:
        raise RuntimeError("Scenario BO finalists did not yield a feasible full validation result.")
    parameters, selected, weights, scenarios, plan, _ = evaluation_payloads[chosen_index]
    return ScenarioBOTrainingResult(
        generated_scenarios=scenarios,
        scenario_weights=tuple(float(value) for value in weights),
        support_latent=pool_latent[selected].copy(),
        support_conditions=pool_conditions[selected].copy(),
        support_source_names=tuple(pool_names[int(index)] for index in selected),
        candidate_source_names=pool_names,
        planning_result=plan,
        full_validation_result=chosen_full_validation,
        history=tuple(records),
        device=str(device),
        feature_names=FEATURE_NAMES,
        best_parameters=tuple(float(value) for value in parameters),
        finalist_evaluations=tuple(finalist_evaluations),
    )
