from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import numpy as np
import torch

from storage_dfl.config import DFLConfig
from storage_dfl.data import Scenario, ScenarioCodec, ScenarioPool
from storage_dfl.dfl.support import DirectSupportPolicy
from storage_dfl.models import ConditionalVAE
from storage_dfl.planning import PlanningResult, StoragePlanningOracle


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    exploration_std: float
    scenario_weights: tuple[float, ...]
    planning_objective: float
    validation_objective: float
    carbon_slack_cost: float
    installed_buses: tuple[str, ...]
    planning_solve_seconds: float
    validation_solve_seconds: float


@dataclass(frozen=True)
class DFLTrainingResult:
    generated_scenarios: tuple[Scenario, ...]
    scenario_weights: tuple[float, ...]
    support_latent: np.ndarray
    support_conditions: np.ndarray
    planning_result: PlanningResult
    full_validation_result: PlanningResult
    history: tuple[EpochRecord, ...]
    device: str

    def history_as_dicts(self) -> list[dict]:
        return [asdict(record) for record in self.history]


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def _validation_scenarios(pool: ScenarioPool, epoch: int, batch_size: int) -> tuple[Scenario, ...]:
    size = min(batch_size, len(pool.scenarios))
    start = (epoch * size) % len(pool.scenarios)
    ordered = pool.scenarios[start:] + pool.scenarios[:start]
    return tuple(ordered[:size])


@torch.no_grad()
def _decode_support(
    cvae: ConditionalVAE,
    codec: ScenarioCodec,
    latent: torch.Tensor,
    support_conditions: torch.Tensor,
    name_prefix: str,
) -> tuple[Scenario, ...]:
    decoded = cvae.decode(latent, support_conditions).cpu().numpy()
    return codec.decode_batch(
        decoded,
        support_conditions.cpu().numpy(),
        name_prefix=name_prefix,
    )


def _finite_loss(result: PlanningResult) -> float:
    return float(result.objective) if result.feasible and isfinite(result.objective) else 1.0e12


def train_direct_generator(
    policy: DirectSupportPolicy,
    cvae: ConditionalVAE,
    codec: ScenarioCodec,
    observed_pool: ScenarioPool,
    oracle: StoragePlanningOracle,
    config: DFLConfig,
    seed: int,
    writer: Any | None = None,
) -> DFLTrainingResult:
    """Tune generated scenarios with black-box planning feedback.

    The planner is a nonconvex MINLP, so the implementation uses a score-function
    estimator.  SCIP is never differentiated and can be replaced independently.
    """

    device = resolve_device(config.device)
    torch.manual_seed(seed + 1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + 1)
    policy.to(device)
    cvae.to(device)
    cvae.eval()
    conditions = torch.as_tensor(
        codec.support_conditions(config.num_support_scenarios),
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    baseline: float | None = None
    best_loss = float("inf")
    best_scenarios: tuple[Scenario, ...] | None = None
    best_weights: tuple[float, ...] | None = None
    best_latent: np.ndarray | None = None
    best_plan: PlanningResult | None = None
    records: list[EpochRecord] = []

    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        exploration_std = max(
            config.minimum_exploration_std,
            config.initial_exploration_std * config.exploration_decay**epoch,
        )
        sample = policy.sample(exploration_std)
        generated = _decode_support(cvae, codec, sample.latent, conditions, f"dfl_e{epoch:03d}")
        weights = tuple(float(value) for value in sample.weights.cpu())
        plan = oracle.solve(generated, weights=weights)

        if plan.feasible:
            validation = oracle.solve(
                _validation_scenarios(observed_pool, epoch, config.validation_batch_size),
                fixed_design=plan.design,
                allow_carbon_slack=True,
            )
        else:
            validation = plan
        decision_loss = _finite_loss(validation)

        if baseline is None:
            baseline = decision_loss
        scale = max(abs(baseline), 1.0)
        advantage = float(np.clip((decision_loss - baseline) / scale, -5.0, 5.0))
        loss = advantage * sample.log_probability
        loss = loss + config.diversity_weight * policy.diversity_penalty(config.diversity_margin)
        loss = loss - config.weight_entropy_weight * policy.weight_entropy()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        baseline = config.baseline_momentum * baseline + (1.0 - config.baseline_momentum) * decision_loss

        if plan.feasible and decision_loss < best_loss:
            best_loss = decision_loss
            best_scenarios = generated
            best_weights = weights
            best_latent = sample.latent.cpu().numpy().copy()
            best_plan = plan

        records.append(
            EpochRecord(
                epoch=epoch,
                exploration_std=exploration_std,
                scenario_weights=weights,
                planning_objective=float(plan.objective),
                validation_objective=float(validation.objective),
                carbon_slack_cost=float(validation.carbon_slack_cost),
                installed_buses=tuple(plan.design.installed_buses),
                planning_solve_seconds=float(plan.solve_time_seconds),
                validation_solve_seconds=float(validation.solve_time_seconds),
            )
        )
        if writer is not None:
            writer.add_scalar("loss/decision", decision_loss, epoch)
            writer.add_scalar("objective/planning", float(plan.objective), epoch)
            writer.add_scalar("objective/validation", float(validation.objective), epoch)
            writer.add_scalar("objective/carbon_slack", float(validation.carbon_slack_cost), epoch)
            writer.add_scalar("policy/exploration_std", exploration_std, epoch)
            writer.add_scalar("design/installed_sites", len(plan.design.installed_buses), epoch)
            writer.add_scalar("design/power_mw", sum(plan.design.power_mw.values()), epoch)
            writer.add_scalar("design/energy_mwh", sum(plan.design.energy_mwh.values()), epoch)
            for index, weight in enumerate(weights):
                writer.add_scalar(f"scenario_weight/support_{index}", weight, epoch)
            writer.flush()

    deterministic_latent, deterministic_weights_tensor = policy.deterministic()
    deterministic_scenarios = _decode_support(
        cvae,
        codec,
        deterministic_latent,
        conditions,
        "dfl_final",
    )
    deterministic_weights = tuple(float(value) for value in deterministic_weights_tensor.cpu())
    deterministic_plan = oracle.solve(deterministic_scenarios, weights=deterministic_weights)
    if deterministic_plan.feasible:
        deterministic_validation = oracle.solve(
            observed_pool.scenarios,
            fixed_design=deterministic_plan.design,
            allow_carbon_slack=True,
        )
        deterministic_loss = _finite_loss(deterministic_validation)
    else:
        deterministic_validation = deterministic_plan
        deterministic_loss = float("inf")

    if best_plan is not None:
        best_full_validation = oracle.solve(
            observed_pool.scenarios,
            fixed_design=best_plan.design,
            allow_carbon_slack=True,
        )
        best_full_loss = _finite_loss(best_full_validation)
    else:
        best_full_validation = None
        best_full_loss = float("inf")

    if best_plan is None or deterministic_loss <= best_full_loss:
        chosen_scenarios = deterministic_scenarios
        chosen_weights = deterministic_weights
        chosen_plan = deterministic_plan
        full_validation = deterministic_validation
        chosen_latent = deterministic_latent.cpu().numpy().copy()
    else:
        chosen_scenarios = best_scenarios
        chosen_weights = best_weights
        chosen_plan = best_plan
        full_validation = best_full_validation
        chosen_latent = best_latent

    if (
        chosen_scenarios is None
        or chosen_weights is None
        or chosen_latent is None
        or full_validation is None
    ):
        raise RuntimeError("DFL did not produce a scenario set.")
    if writer is not None:
        writer.flush()
    return DFLTrainingResult(
        generated_scenarios=chosen_scenarios,
        scenario_weights=chosen_weights,
        support_latent=chosen_latent,
        support_conditions=conditions.detach().cpu().numpy().copy(),
        planning_result=chosen_plan,
        full_validation_result=full_validation,
        history=tuple(records),
        device=str(device),
    )
