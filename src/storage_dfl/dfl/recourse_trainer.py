"""Direct CVAE fine-tuning with fixed-design MILP recourse feedback."""

from __future__ import annotations

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
    gradient_norm: float
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
    if dfl_config.dfl_eval_interval <= 0:
        raise ValueError("dfl.dfl_eval_interval must be positive.")
    if dfl_config.decision_batch_size <= 0:
        raise ValueError("dfl.decision_batch_size must be positive.")

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
    )
    reference_cache = PerfectInformationCache(dfl_config.use_solution_cache)
    history: list[RecourseDFLEpoch] = []
    count = min(dfl_config.decision_batch_size, len(observed_pool.scenarios))

    # lambda=0 is the exact pretrained CVAE baseline: do not add hidden extra
    # epochs after stage-one training and do not consume a different RNG stream.
    fine_tune_epochs = dfl_config.epochs if dfl_config.lambda_dfl > 0.0 else 0
    for epoch in range(fine_tune_epochs):
        indices_np = _stratified_decision_indices(observed_pool, count, rng)
        indices = torch.as_tensor(indices_np, dtype=torch.long, device=device)
        batch_x = trajectories[indices]
        batch_c = contexts[indices]
        real_scenarios = observed_pool.subset(indices_np.tolist())
        weights = _contingency_preserving_weights(observed_pool, indices_np)

        optimizer.zero_grad(set_to_none=True)
        statistical = cvae_loss_components(
            model,
            batch_x,
            batch_c,
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
        evaluate_dfl = (
            epoch >= dfl_config.dfl_start_epoch
            and (epoch - dfl_config.dfl_start_epoch) % dfl_config.dfl_eval_interval == 0
        )
        feasibility_loss = torch.zeros((), dtype=cvae_loss.dtype, device=device)
        component_values = {
            "load": 0.0,
            "pv": 0.0,
            "carbon": 0.0,
            "shed_fraction": 0.0,
            "carbon_rate": 0.0,
            "curtailment": 0.0,
        }
        planning_objective = recourse_objective = reference_objective = float("nan")
        regret = float("nan")
        feasibility_gradient_norm = 0.0

        if evaluate_dfl:
            # The DFL samples come directly from the CVAE prior. There is no
            # latent selector or scenario-selection policy in this path.
            latent = model.sample_latent(count, device=device)
            generated_normalized, generated_scenarios = _decode_scenarios(
                model,
                codec,
                latent,
                batch_c,
                f"recourse_dfl_e{epoch:03d}",
                exogenous=real_scenarios,
            )
            generated_physical = codec.physical_torch(generated_normalized)
            true_physical = codec.physical_torch(batch_x).detach()
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
                oracle,
                real_scenarios,
                weights,
                plan.design,
                allow_carbon_slack=True,
                use_cache=dfl_config.use_solution_cache,
                accepted_relative_gap=oracle.planning.solver_relative_gap,
            )
            reference = reference_cache.solve(
                oracle,
                real_scenarios,
                weights,
                allow_carbon_slack=True,
            )
            if not recourse.feasible or not reference.feasible:
                raise RuntimeError("Recourse or perfect-information MILP has no feasible solution.")
            feasibility_output = feasibility(
                generated_physical,
                true_physical,
                recourse,
                recourse_results,
            )
            feasibility_loss = feasibility_output.loss
            feasibility_gradients = torch.autograd.grad(
                feasibility_loss,
                tuple(model.parameters()),
                retain_graph=True,
                allow_unused=True,
            )
            feasibility_gradient_norm = _tensor_gradient_norm(
                feasibility_gradients
            )
            diagnostics = feasibility_output.diagnostics()
            component_values = {
                "load": diagnostics["surrogate_load_underestimation"],
                "pv": diagnostics["surrogate_pv_overestimation"],
                "carbon": diagnostics["surrogate_carbon_underestimation"],
                "shed_fraction": diagnostics["forward_load_shedding_fraction"],
                "carbon_rate": diagnostics["forward_carbon_excess_t_per_mwh"],
                "curtailment": diagnostics["forward_pv_curtailment_mwh"],
            }
            planning_objective = float(plan.objective)
            recourse_objective = float(recourse.objective)
            reference_objective = float(reference.objective)
            regret = normalized_decision_regret(recourse, reference)

        total_loss = cvae_loss + dfl_config.lambda_dfl * feasibility_loss
        total_loss.backward()
        gradient_norm = _gradient_norm(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
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
            gradient_norm=gradient_norm,
            dfl_evaluated=evaluate_dfl,
        )
        history.append(record)
        print(
            f"Recourse DFL epoch {epoch + 1}/{fine_tune_epochs}: "
            f"CVAE={record.cvae_loss:.6g}, feas={record.feasibility_loss:.6g}, "
            f"shed={record.load_shedding_fraction:.3g}, "
            f"carbon_excess={record.carbon_excess_t_per_mwh:.3g} t/MWh, "
            f"regret={record.decision_regret:.3g}, "
            f"feas_grad={record.feasibility_gradient_norm:.3g}, "
            f"total_grad={record.gradient_norm:.3g}",
            flush=True,
        )
        if writer is not None:
            for name, value in asdict(record).items():
                if isinstance(value, (int, float)) and isfinite(float(value)):
                    writer.add_scalar(f"recourse_dfl/{name}", float(value), epoch)

    # A fixed seed makes the final direct CVAE sample reproducible. Conditions
    # are observed anchors, but no observed trajectory is selected or encoded.
    final_indices = _final_support_indices(
        observed_pool, codec, dfl_config.num_support_scenarios
    )
    final_conditions_np = contexts_np[final_indices]
    final_exogenous = observed_pool.subset(final_indices.tolist())
    final_conditions = torch.as_tensor(
        final_conditions_np, dtype=torch.float32, device=device
    )
    final_generator = torch.Generator(device=device)
    final_generator.manual_seed(seed + 314159)
    final_latent = model.sample_latent(
        dfl_config.num_support_scenarios,
        device=device,
        generator=final_generator,
    )
    model.eval()
    with torch.no_grad():
        _, final_scenarios = _decode_scenarios(
            model,
            codec,
            final_latent,
            final_conditions,
            "recourse_dfl_final",
            exogenous=final_exogenous,
        )
    final_weights = _contingency_preserving_weights(observed_pool, final_indices)
    final_plan = oracle.solve(
        final_scenarios,
        weights=final_weights,
        allow_carbon_slack=dfl_config.training_allow_carbon_slack,
        use_cache=dfl_config.use_solution_cache,
    )
    validation_scenarios, validation_weights, _ = select_scenarios(
        dfl_config.evaluation_selection_rule,
        observed_pool,
        codec,
        min(dfl_config.final_validation_size, len(observed_pool.scenarios)),
        seed=seed,
    )
    full_validation, _ = evaluate_fixed_design_recourse(
        oracle,
        validation_scenarios,
        validation_weights,
        final_plan.design,
        allow_carbon_slack=True,
        use_cache=dfl_config.use_solution_cache,
    )
    reference = reference_cache.solve(
        oracle,
        validation_scenarios,
        validation_weights,
        allow_carbon_slack=True,
    )
    final_regret = normalized_decision_regret(full_validation, reference)
    model.freeze()
    if writer is not None:
        writer.flush()
    return RecourseDFLTrainingResult(
        generated_scenarios=final_scenarios,
        scenario_weights=final_weights,
        support_latent=final_latent.detach().cpu().numpy(),
        support_conditions=final_conditions.detach().cpu().numpy(),
        planning_result=final_plan,
        full_validation_result=full_validation,
        reference_result=reference,
        decision_regret=final_regret,
        history=tuple(history),
        device=str(device),
    )
