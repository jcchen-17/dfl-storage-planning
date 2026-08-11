"""Direct CVAE fine-tuning with fixed-design MILP recourse feedback."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Any

import numpy as np
import torch

from storage_dfl.config import CVAEConfig, DFLConfig
from storage_dfl.data import Scenario, ScenarioCodec, ScenarioPool
from storage_dfl.dfl.decision_loss import RecourseFeasibilityLoss
from storage_dfl.dfl.selection import select_scenarios
from storage_dfl.dfl.solution_cache import PerfectInformationCache
from storage_dfl.models import (
    ConditionalVAE,
    ShapeStatistics,
    TrajectoryLayout,
    cvae_loss_components,
)
from storage_dfl.planning import (
    PlanningResult,
    evaluate_fixed_design_recourse,
)



def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


@dataclass(frozen=True)
class RecourseDFLEpoch:
    epoch: int
    cvae_loss: float
    feasibility_loss: float
    total_loss: float
    load_underestimation_loss: float
    pv_overestimation_loss: float
    carbon_underestimation_loss: float
    load_shedding_fraction: float
    carbon_excess_t_per_mwh: float
    pv_curtailment_mwh: float
    decision_regret: float
    planning_objective: float
    recourse_objective: float
    reference_objective: float
    feasibility_gradient_norm: float
    statistical_gradient_norm: float
    statistical_loss_weight: float
    gradient_norm: float
    effective_lambda_dfl: float
    infeasibility_penalty_loss: float
    optimality_preserving_loss: float
    validation_load_shedding_fraction: float
    validation_carbon_excess_t_per_mwh: float
    validation_decision_regret: float
    validation_objective: float
    is_best_checkpoint: bool
    dfl_evaluated: bool


@dataclass(frozen=True)
class RecourseDFLTrainingResult:
    generated_scenarios: tuple[Scenario, ...]
    scenario_weights: tuple[float, ...]
    support_latent: np.ndarray
    support_conditions: np.ndarray
    planning_result: PlanningResult
    full_validation_result: PlanningResult
    reference_result: PlanningResult
    decision_regret: float
    history: tuple[RecourseDFLEpoch, ...]
    device: str
    best_epoch: int

    def history_as_dicts(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self.history]


def normalized_decision_regret(
    evaluated: PlanningResult,
    reference: PlanningResult,
    epsilon: float = 1.0e-6,
) -> float:
    """Exact forward MILP regret; never used as an autograd loss here."""

    if not evaluated.feasible or not reference.feasible:
        return float("inf")
    return float(
        (evaluated.objective - reference.objective)
        / (abs(reference.objective) + epsilon)
    )


def _validation_checkpoint_score(
    regret: float,
    shed_fraction: float,
    carbon_rate: float,
    carbon_cap: float,
    *,
    selection: str = "economic",
    shedding_tolerance: float = 1.0e-4,
) -> tuple[float, ...]:
    """Rank validation checkpoints under the configured research objective."""

    violation = shed_fraction + carbon_rate / max(float(carbon_cap), 1.0e-6)
    if selection == "economic":
        return regret, violation
    if selection == "carbon_first":
        reliability_excess = max(0.0, shed_fraction - shedding_tolerance)
        return (
            float(shed_fraction > shedding_tolerance),
            reliability_excess,
            carbon_rate,
            regret,
        )
    raise ValueError(
        "dfl.checkpoint_selection must be 'economic' or 'carbon_first'."
    )


def _validation_checkpoint_is_better(
    *,
    selection: str,
    candidate_score: tuple[float, ...],
    incumbent_score: tuple[float, ...],
    candidate_shed: float,
    candidate_carbon: float,
    candidate_regret: float,
    incumbent_shed: float,
    incumbent_carbon: float,
    incumbent_regret: float,
    shedding_tolerance: float,
    carbon_tolerance: float,
    reliable_carbon_floor: float,
) -> tuple[bool, float]:
    """Compare checkpoints without allowing carbon-tolerance ratcheting."""

    if selection != "carbon_first" or carbon_tolerance <= 0.0:
        return candidate_score < incumbent_score, reliable_carbon_floor

    candidate_reliable = candidate_shed <= shedding_tolerance
    incumbent_reliable = incumbent_shed <= shedding_tolerance
    if candidate_reliable:
        reliable_carbon_floor = min(reliable_carbon_floor, candidate_carbon)

    if candidate_reliable != incumbent_reliable:
        return candidate_reliable, reliable_carbon_floor
    if not candidate_reliable:
        return candidate_score < incumbent_score, reliable_carbon_floor

    carbon_ceiling = reliable_carbon_floor + carbon_tolerance
    candidate_eligible = candidate_carbon <= carbon_ceiling
    incumbent_eligible = incumbent_carbon <= carbon_ceiling
    if candidate_eligible != incumbent_eligible:
        return candidate_eligible, reliable_carbon_floor
    if candidate_eligible:
        return (
            (candidate_regret, candidate_carbon)
            < (incumbent_regret, incumbent_carbon),
            reliable_carbon_floor,
        )
    return candidate_score < incumbent_score, reliable_carbon_floor


def _gradient_norm(parameters) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().square().sum().cpu())
    return squared**0.5


def _tensor_gradient_norm(gradients) -> float:
    squared = 0.0
    for gradient in gradients:
        if gradient is not None:
            squared += float(gradient.detach().square().sum().cpu())
    return squared**0.5


def _decode_scenarios(
    model: ConditionalVAE,
    codec: ScenarioCodec,
    latent: torch.Tensor,
    conditions: torch.Tensor,
    prefix: str,
    exogenous: tuple[Scenario, ...] | None = None,
) -> tuple[torch.Tensor, tuple[Scenario, ...]]:
    normalized = model.decode(latent, conditions)
    scenarios = codec.decode_batch(
        normalized.detach().cpu().numpy(),
        conditions.detach().cpu().numpy(),
        name_prefix=prefix,
    )
    if exogenous is not None:
        if len(exogenous) != len(scenarios):
            raise ValueError("Generated and exogenous scenario counts must match.")
        scenarios = tuple(
            replace(
                generated,
                # Grid availability is an observed exogenous contingency, not
                # a differentiable CVAE output. The generated load/PV/carbon
                # trajectories remain independent Torch tensors.
                grid_available=np.asarray(source.grid_available, dtype=float).copy(),
                annual_occurrences=None,
                probability_weight=source.probability_weight,
            )
            for generated, source in zip(scenarios, exogenous, strict=True)
        )
    return normalized, scenarios


def _stratified_decision_indices(
    pool: ScenarioPool,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Preserve the file's outage oversampling in every small DFL batch."""

    outage = np.asarray(
        [np.any(scenario.grid_available < 0.5) for scenario in pool.scenarios]
    )
    outage_indices = np.flatnonzero(outage)
    normal_indices = np.flatnonzero(~outage)
    if count < 2 or outage_indices.size == 0 or normal_indices.size == 0:
        return rng.choice(len(pool.scenarios), size=count, replace=False)
    empirical_outage_share = outage_indices.size / len(pool.scenarios)
    outage_count = int(np.clip(round(count * empirical_outage_share), 1, count - 1))
    outage_count = min(outage_count, outage_indices.size)
    normal_count = min(count - outage_count, normal_indices.size)
    if outage_count + normal_count < count:
        outage_count = min(count - normal_count, outage_indices.size)
    selected = np.concatenate(
        (
            rng.choice(outage_indices, size=outage_count, replace=False),
            rng.choice(normal_indices, size=normal_count, replace=False),
        )
    )
    rng.shuffle(selected)
    return selected.astype(np.int64)


def _final_support_indices(
    pool: ScenarioPool,
    codec: ScenarioCodec,
    count: int,
) -> np.ndarray:
    indices = codec.support_indices(pool, count)
    outage = np.asarray(
        [np.any(scenario.grid_available < 0.5) for scenario in pool.scenarios]
    )
    if count >= 2 and outage.any() and (~outage).any():
        if not outage[indices].any():
            candidates = np.flatnonzero(outage)
            hours = np.asarray(
                [np.count_nonzero(pool.scenarios[index].grid_available < 0.5)
                 for index in candidates]
            )
            indices[-1] = candidates[int(hours.argmax())]
        if outage[indices].all():
            indices[-1] = np.flatnonzero(~outage)[0]
    return indices


def _contingency_preserving_weights(
    pool: ScenarioPool,
    indices: np.ndarray,
) -> tuple[float, ...]:
    """Preserve total normal/outage probability under reduced supports."""

    full_weights = np.asarray(pool.normalized_weights(), dtype=float)
    full_outage = np.asarray(
        [np.any(scenario.grid_available < 0.5) for scenario in pool.scenarios]
    )
    selected_outage = full_outage[indices]
    if not full_outage.any() or full_outage.all():
        return pool.normalized_weights(indices.tolist())
    values = np.zeros(len(indices), dtype=float)
    for is_outage in (False, True):
        selected = selected_outage == is_outage
        if not selected.any():
            continue
        class_mass = float(full_weights[full_outage == is_outage].sum())
        values[selected] = class_mass / int(selected.sum())
    if values.sum() <= 0.0:
        raise ValueError("Selected supports carry no probability mass.")
    values /= values.sum()
    return tuple(float(value) for value in values)


def train_recourse_feasibility_cvae(
    model: ConditionalVAE,
    codec: ScenarioCodec,
    observed_pool: ScenarioPool,
    oracle,
    cvae_config: CVAEConfig,
    dfl_config: DFLConfig,
    seed: int,
    *,
    validation_pool: ScenarioPool | None = None,
    validation_oracle=None,
    writer: Any | None = None,
) -> RecourseDFLTrainingResult:
    """Fine-tune the CVAE decoder using exact recourse severities.

    The planning and recourse MILPs are detached black boxes.  The differentiable
    loss only sees their hourly violation magnitudes as constant weights.
    """

    if not isinstance(model, ConditionalVAE):
        raise TypeError("recourse_feasibility currently supports the CVAE only.")
    if dfl_config.lambda_dfl < 0.0:
        raise ValueError("dfl.lambda_dfl must be nonnegative.")
    if dfl_config.dfl_eval_interval <= 0 or dfl_config.validation_interval <= 0:
        raise ValueError("DFL and validation intervals must be positive.")
    if dfl_config.decision_batch_size <= 0:
        raise ValueError("dfl.decision_batch_size must be positive.")
    if not 0.0 <= dfl_config.dfl_statistical_weight <= 1.0:
        raise ValueError("dfl.dfl_statistical_weight must lie in [0, 1].")
    if dfl_config.gradient_balance_min_scale > dfl_config.gradient_balance_max_scale:
        raise ValueError("gradient balance minimum cannot exceed maximum.")
    if dfl_config.checkpoint_shedding_tolerance < 0.0:
        raise ValueError("checkpoint shedding tolerance must be nonnegative.")
    if dfl_config.checkpoint_carbon_tolerance < 0.0:
        raise ValueError("checkpoint carbon tolerance must be nonnegative.")
    if dfl_config.checkpoint_selection not in {"economic", "carbon_first"}:
        raise ValueError(
            "dfl.checkpoint_selection must be 'economic' or 'carbon_first'."
        )

    # Backward compatibility for callers that have only one toy pool.  The
    # production stage always supplies a disjoint validation split.
    validation_pool = validation_pool or observed_pool
    validation_oracle = validation_oracle or oracle
    device = resolve_device(dfl_config.device)
    torch.manual_seed(seed + 17)
    rng = np.random.default_rng(seed + 17)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.train()

    trajectories_np, contexts_np = codec.encode_pool(observed_pool)
    trajectories = torch.as_tensor(trajectories_np, dtype=torch.float32, device=device)
    contexts = torch.as_tensor(contexts_np, dtype=torch.float32, device=device)
    layout = TrajectoryLayout.build(
        codec.trajectory_dim,
        codec.horizon,
        device,
        trajectory_mean=codec.trajectory_mean,
        trajectory_std=codec.trajectory_std,
        field_masks=codec.field_masks(),
    )
    shape = ShapeStatistics.fit(trajectories, layout)
    optimizer = torch.optim.Adam(model.parameters(), lr=cvae_config.learning_rate)
    feasibility = RecourseFeasibilityLoss(
        load_weight=dfl_config.feasibility_load_weight,
        pv_weight=dfl_config.feasibility_pv_weight,
        carbon_weight=dfl_config.feasibility_carbon_weight,
        carbon_cap_t_per_mwh=oracle.planning.dc_carbon_cap,
        infeasibility_aversion_alpha=dfl_config.infeasibility_aversion_alpha,
        margin=dfl_config.feasibility_margin,
    )
    reference_cache = PerfectInformationCache(dfl_config.use_solution_cache)
    validation_reference_cache = PerfectInformationCache(
        dfl_config.use_solution_cache
    )
    history: list[RecourseDFLEpoch] = []
    decision_count = min(dfl_config.decision_batch_size, len(observed_pool.scenarios))
    reconstruction_count = dfl_config.reconstruction_batch_size or cvae_config.batch_size
    reconstruction_count = min(reconstruction_count, len(observed_pool.scenarios))

    # These supports and this held-out bag never change between epochs.  Thus a
    # lower validation score means a better model, not an easier random draw.
    support_indices = _final_support_indices(
        observed_pool, codec, dfl_config.num_support_scenarios
    )
    support_conditions = torch.as_tensor(
        contexts_np[support_indices], dtype=torch.float32, device=device
    )
    support_exogenous = observed_pool.subset(support_indices.tolist())
    support_weights = _contingency_preserving_weights(observed_pool, support_indices)
    support_generator = torch.Generator(device=device)
    support_generator.manual_seed(seed + 314159)
    support_latent = model.sample_latent(
        dfl_config.num_support_scenarios,
        device=device,
        generator=support_generator,
    ).detach()
    # In stable-anchor mode, train on exactly the support scenarios that are
    # checkpointed and used for final planning.  A separate fixed random bank
    # would be less noisy than resampling every epoch, but it would still
    # optimize a different decision problem from the one ultimately reported.
    fixed_decision_indices_np = support_indices
    fixed_decision_latent = support_latent
    cvae_only = dfl_config.lambda_dfl == 0.0
    validation_scenarios, validation_weights, _ = select_scenarios(
        dfl_config.evaluation_selection_rule,
        validation_pool,
        codec,
        min(dfl_config.final_validation_size, len(validation_pool.scenarios)),
        seed=seed + 2718,
    )
    validation_reference = None

    def evaluate_fixed_checkpoint(prefix: str):
        nonlocal validation_reference
        if validation_reference is None:
            validation_reference = validation_reference_cache.solve(
                validation_oracle,
                validation_scenarios,
                validation_weights,
                allow_carbon_slack=True,
            )
        model.eval()
        with torch.no_grad():
            _, generated = _decode_scenarios(
                model, codec, support_latent, support_conditions, prefix,
                exogenous=support_exogenous,
            )
        plan = validation_oracle.solve(
            generated,
            weights=support_weights,
            allow_carbon_slack=dfl_config.training_allow_carbon_slack,
            use_cache=dfl_config.use_solution_cache,
        )
        if not plan.feasible:
            raise RuntimeError(f"Fixed-support planning failed: {plan.status}.")
        evaluated, _ = evaluate_fixed_design_recourse(
            validation_oracle,
            validation_scenarios,
            validation_weights,
            plan.design,
            allow_carbon_slack=True,
            use_cache=dfl_config.use_solution_cache,
            accepted_relative_gap=validation_oracle.planning.solver_relative_gap,
        )
        diagnostics = evaluated.recourse_diagnostics
        if diagnostics is None:
            raise ValueError("Validation result has no recourse diagnostics.")
        demand = diagnostics.served_demand_mwh + diagnostics.load_shedding_mwh
        shed_fraction = diagnostics.load_shedding_mwh / max(demand, 1.0e-6)
        carbon_rate = diagnostics.carbon_excess_t / max(
            diagnostics.served_demand_mwh, 1.0e-6
        )
        regret = normalized_decision_regret(evaluated, validation_reference)
        # Rank the fixed validation result under the configured economic or
        # carbon-first rule. The stateful carbon tolerance is applied later,
        # when this candidate is compared with the incumbent checkpoint.
        score = _validation_checkpoint_score(
            regret,
            shed_fraction,
            carbon_rate,
            validation_oracle.planning.dc_carbon_cap,
            selection=dfl_config.checkpoint_selection,
            shedding_tolerance=dfl_config.checkpoint_shedding_tolerance,
        )
        model.train()
        return generated, plan, evaluated, shed_fraction, carbon_rate, regret, score

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = -1
    if cvae_only:
        # The statistical baseline trains first, then invokes planning once for
        # its final generated scenarios. No decision result influences training
        # or checkpoint selection.
        best_score = (float("inf"),)
        best_shed = best_carbon = best_regret = float("inf")
        reliable_carbon_floor = float("inf")
    else:
        initial_eval = evaluate_fixed_checkpoint("recourse_dfl_initial")
        best_score = initial_eval[-1]
        best_shed = float(initial_eval[3])
        best_carbon = float(initial_eval[4])
        best_regret = float(initial_eval[5])
        reliable_carbon_floor = (
            best_carbon
            if best_shed <= dfl_config.checkpoint_shedding_tolerance
            else float("inf")
        )
        initialization_label = (
            "pretrained"
            if dfl_config.initialize_from_pretrained
            else "random-initialized"
        )
        print(
            f"Fixed validation baseline ({initialization_label} epoch -1): "
            f"shed={initial_eval[3]:.3g}, carbon={initial_eval[4]:.3g} t/MWh, "
            f"regret={initial_eval[5]:.3g}",
            flush=True,
        )

    # With scratch initialization, lambda=0 is the CVAE-only/Predict-then-
    # Optimize baseline: train the statistical objective for the same epoch
    # budget, but never request decision feedback. With a separately pretrained
    # generator it remains the historical exact no-op ablation.
    fine_tune_epochs = (
        0 if cvae_only and dfl_config.initialize_from_pretrained
        else dfl_config.epochs
    )
    parameters = tuple(model.parameters())
    # Decision feedback reaches the generator through decode().  Balancing it
    # against encoder gradients would compare disjoint parameter subspaces and
    # produce a misleading scale, so use decoder gradients for both norms.
    decoder_parameters = tuple(model.decoder.parameters())
    for epoch in range(fine_tune_epochs):
        decision_indices_np = (
            fixed_decision_indices_np
            if dfl_config.fixed_decision_anchors
            else _stratified_decision_indices(observed_pool, decision_count, rng)
        )
        reconstruction_indices_np = rng.choice(
            len(observed_pool.scenarios), size=reconstruction_count, replace=False
        )
        decision_indices = torch.as_tensor(
            decision_indices_np, dtype=torch.long, device=device
        )
        reconstruction_indices = torch.as_tensor(
            reconstruction_indices_np, dtype=torch.long, device=device
        )
        decision_x = trajectories[decision_indices]
        decision_c = contexts[decision_indices]
        real_scenarios = observed_pool.subset(decision_indices_np.tolist())
        weights = (
            support_weights
            if dfl_config.fixed_decision_anchors
            else _contingency_preserving_weights(observed_pool, decision_indices_np)
        )

        optimizer.zero_grad(set_to_none=True)
        statistical = cvae_loss_components(
            model,
            trajectories[reconstruction_indices],
            contexts[reconstruction_indices],
            layout,
            shape,
            cvae_config.field_weights(),
            cvae_config,
            codec.horizon,
            cvae_config.beta,
            device,
            sample_latent=True,
        )
        cvae_loss = statistical["loss"]
        statistical_weight = (
            1.0
            if cvae_only or epoch < dfl_config.dfl_start_epoch
            else float(dfl_config.dfl_statistical_weight)
        )
        weighted_statistical_loss = statistical_weight * cvae_loss
        statistical_gradients = torch.autograd.grad(
            weighted_statistical_loss,
            decoder_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        statistical_gradient_norm = _tensor_gradient_norm(statistical_gradients)
        evaluate_dfl = (
            not cvae_only
            and epoch >= dfl_config.dfl_start_epoch
            and (epoch - dfl_config.dfl_start_epoch) % dfl_config.dfl_eval_interval == 0
        )
        feasibility_loss = torch.zeros((), dtype=cvae_loss.dtype, device=device)
        component_values = {
            "load": 0.0, "pv": 0.0, "carbon": 0.0, "ipl": 0.0, "opl": 0.0,
            "shed_fraction": 0.0, "carbon_rate": 0.0, "curtailment": 0.0,
        }
        planning_objective = recourse_objective = reference_objective = float("nan")
        regret = float("nan")
        feasibility_gradient_norm = 0.0

        if evaluate_dfl:
            latent = (
                fixed_decision_latent
                if dfl_config.fixed_decision_anchors
                else model.sample_latent(decision_count, device=device)
            )
            generated_normalized, generated_scenarios = _decode_scenarios(
                model, codec, latent, decision_c, f"recourse_dfl_e{epoch:03d}",
                exogenous=real_scenarios,
            )
            generated_physical = codec.physical_torch(generated_normalized)
            true_physical = codec.physical_torch(decision_x).detach()
            plan = oracle.solve(
                generated_scenarios,
                weights=weights,
                allow_carbon_slack=dfl_config.training_allow_carbon_slack,
                use_cache=dfl_config.use_solution_cache,
            )
            if not plan.feasible:
                raise RuntimeError(
                    f"Generated-scenario planning failed at epoch {epoch}: {plan.status}."
                )
            recourse, recourse_results = evaluate_fixed_design_recourse(
                oracle, real_scenarios, weights, plan.design,
                allow_carbon_slack=True,
                use_cache=dfl_config.use_solution_cache,
                accepted_relative_gap=oracle.planning.solver_relative_gap,
            )
            reference = reference_cache.solve(
                oracle, real_scenarios, weights, allow_carbon_slack=True
            )
            if not recourse.feasible or not reference.feasible:
                raise RuntimeError("Recourse or perfect-information MILP has no feasible solution.")
            preservation = preservation_results = None
            if dfl_config.infeasibility_aversion_alpha < 1.0:
                preservation, preservation_results = evaluate_fixed_design_recourse(
                    oracle, generated_scenarios, weights, reference.design,
                    allow_carbon_slack=True,
                    use_cache=dfl_config.use_solution_cache,
                    accepted_relative_gap=oracle.planning.solver_relative_gap,
                )
            feasibility_output = feasibility(
                generated_physical,
                true_physical,
                recourse,
                recourse_results,
                preservation,
                preservation_results,
                grid_connected_mask=torch.as_tensor(
                    np.stack(
                        [scenario.grid_available for scenario in real_scenarios]
                    ),
                    dtype=true_physical.grid_carbon_t_per_mwh.dtype,
                    device=device,
                ),
            )
            feasibility_loss = feasibility_output.loss
            feasibility_gradients = torch.autograd.grad(
                feasibility_loss,
                decoder_parameters,
                retain_graph=True,
                allow_unused=True,
            )
            feasibility_gradient_norm = _tensor_gradient_norm(feasibility_gradients)
            diagnostics = feasibility_output.diagnostics()
            component_values = {
                "load": diagnostics["surrogate_load_underestimation"],
                "pv": diagnostics["surrogate_pv_overestimation"],
                "carbon": diagnostics["surrogate_carbon_underestimation"],
                "ipl": diagnostics["surrogate_ipl"],
                "opl": diagnostics["surrogate_opl"],
                "shed_fraction": diagnostics["forward_load_shedding_fraction"],
                "carbon_rate": diagnostics["forward_carbon_excess_t_per_mwh"],
                "curtailment": diagnostics["forward_pv_curtailment_mwh"],
            }
            planning_objective = float(plan.objective)
            recourse_objective = float(recourse.objective)
            reference_objective = float(reference.objective)
            regret = normalized_decision_regret(recourse, reference)

        effective_lambda = float(dfl_config.lambda_dfl)
        if (
            evaluate_dfl
            and dfl_config.gradient_balance_ratio > 0.0
            and feasibility_gradient_norm > 0.0
        ):
            balance = (
                dfl_config.gradient_balance_ratio
                * statistical_gradient_norm
                / feasibility_gradient_norm
            )
            balance = float(np.clip(
                balance,
                dfl_config.gradient_balance_min_scale,
                dfl_config.gradient_balance_max_scale,
            ))
            effective_lambda *= balance
        total_loss = weighted_statistical_loss + effective_lambda * feasibility_loss
        total_loss.backward()
        gradient_norm = _gradient_norm(parameters)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
        optimizer.step()

        validate_now = not cvae_only and (
            (epoch % dfl_config.validation_interval == 0)
            or epoch == fine_tune_epochs - 1
        )
        val_shed = val_carbon = val_regret = val_objective = float("nan")
        is_best = False
        if cvae_only:
            # A statistical baseline must not use downstream decision quality
            # for model selection. Retain the final statistical-training epoch.
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if validate_now:
            checkpoint_eval = evaluate_fixed_checkpoint(
                f"recourse_dfl_validation_e{epoch:03d}"
            )
            _, _, validation_result, val_shed, val_carbon, val_regret, score = checkpoint_eval
            val_objective = float(validation_result.objective)
            is_better, reliable_carbon_floor = (
                _validation_checkpoint_is_better(
                    selection=dfl_config.checkpoint_selection,
                    candidate_score=score,
                    incumbent_score=best_score,
                    candidate_shed=val_shed,
                    candidate_carbon=val_carbon,
                    candidate_regret=val_regret,
                    incumbent_shed=best_shed,
                    incumbent_carbon=best_carbon,
                    incumbent_regret=best_regret,
                    shedding_tolerance=(
                        dfl_config.checkpoint_shedding_tolerance
                    ),
                    carbon_tolerance=dfl_config.checkpoint_carbon_tolerance,
                    reliable_carbon_floor=reliable_carbon_floor,
                )
            )
            if is_better:
                best_score = score
                best_shed = val_shed
                best_carbon = val_carbon
                best_regret = val_regret
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                is_best = True

        record = RecourseDFLEpoch(
            epoch=epoch,
            cvae_loss=float(cvae_loss.detach().cpu()),
            feasibility_loss=float(feasibility_loss.detach().cpu()),
            total_loss=float(total_loss.detach().cpu()),
            load_underestimation_loss=component_values["load"],
            pv_overestimation_loss=component_values["pv"],
            carbon_underestimation_loss=component_values["carbon"],
            load_shedding_fraction=component_values["shed_fraction"],
            carbon_excess_t_per_mwh=component_values["carbon_rate"],
            pv_curtailment_mwh=component_values["curtailment"],
            decision_regret=regret,
            planning_objective=planning_objective,
            recourse_objective=recourse_objective,
            reference_objective=reference_objective,
            feasibility_gradient_norm=feasibility_gradient_norm,
            statistical_gradient_norm=statistical_gradient_norm,
            statistical_loss_weight=statistical_weight,
            gradient_norm=gradient_norm,
            effective_lambda_dfl=effective_lambda,
            infeasibility_penalty_loss=component_values["ipl"],
            optimality_preserving_loss=component_values["opl"],
            validation_load_shedding_fraction=val_shed,
            validation_carbon_excess_t_per_mwh=val_carbon,
            validation_decision_regret=val_regret,
            validation_objective=val_objective,
            is_best_checkpoint=is_best,
            dfl_evaluated=evaluate_dfl,
        )
        history.append(record)
        epoch_label = "CVAE-only" if cvae_only else "Recourse DFL"
        print(
            f"{epoch_label} epoch {epoch + 1}/{fine_tune_epochs}: "
            f"CVAE={record.cvae_loss:.6g}, IPL={record.infeasibility_penalty_loss:.3g}, "
            f"OPL={record.optimality_preserving_loss:.3g}, "
            f"lambda_eff={record.effective_lambda_dfl:.3g}; "
            f"train_shed={record.load_shedding_fraction:.3g}, "
            f"train_regret={record.decision_regret:.3g}; "
            f"fixed_val_shed={record.validation_load_shedding_fraction:.3g}, "
            f"fixed_val_carbon={record.validation_carbon_excess_t_per_mwh:.3g}, "
            f"fixed_val_regret={record.validation_decision_regret:.3g}"
            f"{' [best]' if is_best else ''}",
            flush=True,
        )
        if writer is not None:
            for name, value in asdict(record).items():
                if isinstance(value, (int, float)) and isfinite(float(value)):
                    writer.add_scalar(f"recourse_dfl/{name}", float(value), epoch)

    # Restore the fixed-validation winner (which may be the pretrained epoch
    # -1), then generate and solve the exact supports that evaluation will use.
    model.load_state_dict(best_state)
    final_eval = evaluate_fixed_checkpoint("recourse_dfl_final")
    assert validation_reference is not None
    final_scenarios, final_plan, full_validation, _, _, final_regret, _ = final_eval
    model.freeze()
    if writer is not None:
        writer.flush()
    return RecourseDFLTrainingResult(
        generated_scenarios=final_scenarios,
        scenario_weights=support_weights,
        support_latent=support_latent.detach().cpu().numpy(),
        support_conditions=support_conditions.detach().cpu().numpy(),
        planning_result=final_plan,
        full_validation_result=full_validation,
        reference_result=validation_reference,
        decision_regret=final_regret,
        history=tuple(history),
        device=str(device),
        best_epoch=best_epoch,
    )
