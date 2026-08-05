from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import numpy as np
import torch

from storage_dfl.config import DFLConfig
from storage_dfl.data import Scenario, ScenarioCodec, ScenarioPool
from storage_dfl.dfl.support import DirectSupportPolicy
from storage_dfl.models import ConditionalGenerator
from storage_dfl.planning import PlanningJob, PlanningResult, StoragePlanningOracle


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
    policy_samples: int
    validation_objective_mean: float
    distinct_designs: int


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


@torch.no_grad()
def _decode_support(
    cvae: ConditionalGenerator,
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


INFEASIBLE_LOSS = 1.0e12


def _finite_loss(result: PlanningResult) -> float:
    return float(result.objective) if result.feasible and isfinite(result.objective) else INFEASIBLE_LOSS


def _safe_number(value: float, fallback: float) -> float:
    """Keep infeasible sentinels out of the JSON history.

    ``infeasible_result`` reports infinite costs, and the history is written with
    ``allow_nan=False``.  Without this the whole run's artifacts are lost at the
    final write because a single epoch failed to find a feasible plan.
    """

    numeric = float(value)
    return numeric if isfinite(numeric) else float(fallback)


def _design_summary(result: PlanningResult) -> str:
    installed = list(result.design.installed_buses)
    power = sum(float(result.design.power_mw[bus]) for bus in installed)
    energy = sum(float(result.design.energy_mwh[bus]) for bus in installed)
    return f"installed={installed}, P={power:.4f} MW, E={energy:.4f} MWh"


def _rank_advantages(losses: np.ndarray) -> np.ndarray:
    """Scale-free advantages in [-1, 1] from the ordering of the losses alone.

    Standardizing raw costs let one infeasible sample (INFEASIBLE_LOSS) inflate
    the standard deviation by six orders of magnitude, which collapsed the
    advantage of every feasible sample to roughly zero and wasted the epoch.
    Ranking keeps the feasible samples fully separated no matter how the
    infeasible sentinel is scaled.
    """

    count = losses.shape[0]
    order = np.argsort(losses, kind="stable")
    ranks = np.empty(count, dtype=float)
    ranks[order] = np.arange(count, dtype=float)
    # Tied costs must receive identical advantages; distinct ranks would push
    # the policy away from one of two indistinguishable outcomes.
    for value in np.unique(losses):
        tied = losses == value
        if int(tied.sum()) > 1:
            ranks[tied] = float(ranks[tied].mean())
    center = (count - 1) / 2.0
    return (ranks - center) / max(center, 1.0)


def train_direct_generator(
    policy: DirectSupportPolicy,
    cvae: ConditionalGenerator,
    codec: ScenarioCodec,
    observed_pool: ScenarioPool,
    oracle: StoragePlanningOracle,
    config: DFLConfig,
    seed: int,
    support_conditions: torch.Tensor | np.ndarray | None = None,
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
        codec.support_conditions(config.num_support_scenarios)
        if support_conditions is None
        else support_conditions,
        dtype=torch.float32,
        device=device,
    )
    if tuple(conditions.shape) != (config.num_support_scenarios, codec.context_dim):
        raise ValueError("support_conditions do not match the configured support count and context size")
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    # Keep the downstream reward distribution fixed across policy updates.
    # Rotating contiguous windows made epoch losses incomparable and, in the
    # price-regime dataset, could expose an epoch to only one tariff block.
    validation_indices = codec.support_indices(
        observed_pool,
        min(config.validation_batch_size, len(observed_pool.scenarios)),
    )
    fixed_validation_scenarios = observed_pool.subset(validation_indices.tolist())

    baseline: float | None = None
    infeasible_samples = 0
    best_loss = float("inf")
    best_scenarios: tuple[Scenario, ...] | None = None
    best_weights: tuple[float, ...] | None = None
    best_latent: np.ndarray | None = None
    best_plan: PlanningResult | None = None
    records: list[EpochRecord] = []

    for epoch in range(config.epochs):
        print(f"DFL epoch {epoch + 1}/{config.epochs}: solving planning candidates...", flush=True)
        optimizer.zero_grad(set_to_none=True)
        exploration_std = max(
            config.minimum_exploration_std,
            config.initial_exploration_std * config.exploration_decay**epoch,
        )
        validation_scenarios = fixed_validation_scenarios
        candidates = []
        infeasible_in_epoch = 0
        # Every sample of an epoch is drawn before any is solved. The draws keep
        # their original order, so the policy's random stream is unchanged, but
        # the planning solves can then be dispatched together.
        samples = []
        for sample_index in range(max(1, config.policy_samples_per_epoch)):
            sample = policy.sample(exploration_std)
            generated = _decode_support(
                cvae,
                codec,
                sample.latent,
                conditions,
                f"dfl_e{epoch:03d}_s{sample_index:02d}",
            )
            weights = tuple(float(value) for value in sample.weights.cpu())
            samples.append((sample, generated, weights))

        plans = oracle.solve_many(
            [PlanningJob(generated, weights) for _, generated, weights in samples]
        )
        # The validations of one epoch share the fixed scenario set and differ
        # only in the design they fix, so they batch exactly like the planning
        # solves above.  Solved inside the loop below they left every worker but
        # one idle for that half of the epoch.
        validated_indices = [index for index, plan in enumerate(plans) if plan.feasible]
        validations = dict(
            zip(
                validated_indices,
                oracle.solve_many(
                    [
                        PlanningJob(
                            validation_scenarios, fixed_design=plans[index].design
                        )
                        for index in validated_indices
                    ],
                    allow_carbon_slack=True,
                    use_cache=True,
                ),
                strict=True,
            )
        )
        for sample_index, ((sample, generated, weights), plan) in enumerate(
            zip(samples, plans, strict=True)
        ):
            # An infeasible plan has no design to validate, so it stands in for
            # its own validation, as it did when the two solves were adjacent.
            validation = validations.get(sample_index, plan)
            decision_loss = _finite_loss(validation)
            if decision_loss >= INFEASIBLE_LOSS:
                infeasible_samples += 1
                infeasible_in_epoch += 1
            candidates.append((sample, generated, weights, plan, validation, decision_loss))
            print(
                f"DFL epoch {epoch + 1}/{config.epochs} sample "
                f"{sample_index + 1}/{max(1, config.policy_samples_per_epoch)}: "
                f"plan={plan.status}/{plan.objective:.6g}, "
                f"validation={validation.status}/{validation.objective:.6g}; "
                f"{_design_summary(plan)}",
                flush=True,
            )

        decision_losses = np.asarray([candidate[5] for candidate in candidates], dtype=float)
        design_signatures = {
            (
                tuple(sorted(candidate[3].design.installed_buses)),
                tuple(round(value, 5) for value in candidate[3].design.power_mw.values()),
                tuple(round(value, 5) for value in candidate[3].design.energy_mwh.values()),
            )
            for candidate in candidates
        }
        # The baseline tracks attainable cost, so an infeasible sentinel must not
        # enter it; otherwise one failed sample poisons every later comparison.
        feasible_losses = decision_losses[decision_losses < INFEASIBLE_LOSS]
        baseline_target = float(
            feasible_losses.mean() if feasible_losses.size else decision_losses.mean()
        )
        if len(candidates) > 1:
            advantages = _rank_advantages(decision_losses)
            # Identical downstream designs carry no action-dependent signal.
            if len(design_signatures) == 1:
                advantages = np.zeros_like(advantages)
        else:
            decision_loss = float(decision_losses[0])
            if baseline is None:
                baseline = baseline_target
            scale = max(abs(baseline), 1.0)
            # Bounded like the ranked branch so both keep the same effective
            # policy step size for a given learning rate.
            advantages = np.asarray(
                [float(np.clip((decision_loss - baseline) / scale, -1.0, 1.0))]
            )

        loss = torch.stack(
            [
                float(advantage) * candidate[0].log_probability
                for advantage, candidate in zip(advantages, candidates, strict=True)
            ]
        ).mean()
        loss = loss + config.diversity_weight * policy.diversity_penalty(config.diversity_margin)
        loss = loss - config.weight_entropy_weight * policy.weight_entropy()
        loss = loss + config.latent_prior_weight * policy.latent_prior_penalty()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        mean_decision_loss = float(decision_losses.mean())
        if baseline is None:
            baseline = baseline_target
        baseline = config.baseline_momentum * baseline + (1.0 - config.baseline_momentum) * baseline_target

        epoch_best = candidates[int(np.argmin(decision_losses))]
        sample, generated, weights, plan, validation, decision_loss = epoch_best

        if plan.feasible and decision_loss < best_loss:
            best_loss = decision_loss
            best_scenarios = generated
            best_weights = weights
            best_latent = sample.latent.cpu().numpy().copy()
            best_plan = plan

        print(
            f"DFL epoch {epoch + 1}/{config.epochs} best: "
            f"validation={validation.status}/{validation.objective:.6g}; "
            f"{_design_summary(plan)}; distinct_designs={len(design_signatures)}, "
            f"exploration_std={exploration_std:.4f}",
            flush=True,
        )

        records.append(
            EpochRecord(
                epoch=epoch,
                exploration_std=exploration_std,
                scenario_weights=weights,
                planning_objective=_safe_number(plan.objective, INFEASIBLE_LOSS),
                validation_objective=_safe_number(validation.objective, INFEASIBLE_LOSS),
                carbon_slack_cost=_safe_number(validation.carbon_slack_cost, 0.0),
                installed_buses=tuple(plan.design.installed_buses),
                planning_solve_seconds=float(plan.solve_time_seconds),
                validation_solve_seconds=float(validation.solve_time_seconds),
                policy_samples=len(candidates),
                validation_objective_mean=_safe_number(mean_decision_loss, INFEASIBLE_LOSS),
                distinct_designs=len(design_signatures),
            )
        )
        if writer is not None:
            # The infeasible sentinel is 1e12 and real objectives are around 1e6,
            # so writing it as a value put the y-axis six orders of magnitude
            # above every real point and flattened the curve to nothing. Averaging
            # it in was just as destructive: one infeasible sample out of five
            # still lands the mean at 2e11. Only feasible costs are plotted, and
            # the failures are plotted as a count on their own axis. The JSON
            # history keeps the sentinel form, which it needs to stay NaN-free.
            feasible_mean = (
                float(feasible_losses.mean()) if feasible_losses.size else float("inf")
            )
            for tag, value in (
                ("loss/decision", feasible_mean),
                ("objective/planning", plan.objective),
                ("objective/validation", validation.objective),
                ("objective/carbon_slack", validation.carbon_slack_cost),
            ):
                if isfinite(value) and value < INFEASIBLE_LOSS:
                    writer.add_scalar(tag, float(value), epoch)
            writer.add_scalar("loss/infeasible_samples", infeasible_in_epoch, epoch)
            writer.add_scalar("policy/exploration_std", exploration_std, epoch)
            writer.add_scalar("policy/distinct_designs", len(design_signatures), epoch)
            writer.add_scalar("policy/latent_prior", float(policy.latent_prior_penalty().detach().cpu()), epoch)
            writer.add_scalar("design/installed_sites", len(plan.design.installed_buses), epoch)
            writer.add_scalar("design/power_mw", sum(plan.design.power_mw.values()), epoch)
            writer.add_scalar("design/energy_mwh", sum(plan.design.energy_mwh.values()), epoch)
            for index, weight in enumerate(weights):
                writer.add_scalar(f"scenario_weight/support_{index}", weight, epoch)
            writer.flush()

    total_policy_samples = config.epochs * max(1, config.policy_samples_per_epoch)
    if infeasible_samples == total_policy_samples:
        raise RuntimeError(
            f"All {config.epochs} DFL epochs ended without a feasible plan, so the "
            "policy was never exposed to a planning gradient and this run carries "
            "no decision-focused signal.  Establish that the oracle can produce a "
            "feasible incumbent before training: raise "
            "planning.solver_time_limit_seconds, relax planning.solver_relative_gap, "
            "or fall back along carbon_formulation "
            "(system_average -> aggregate_mccormick -> mccormick -> exact)."
        )

    deterministic_latent, deterministic_weights_tensor = policy.deterministic()
    deterministic_scenarios = _decode_support(
        cvae,
        codec,
        deterministic_latent,
        conditions,
        "dfl_final",
    )
    deterministic_weights = tuple(float(value) for value in deterministic_weights_tensor.cpu())
    print("DFL final: solving deterministic planning model...", flush=True)
    deterministic_plan = oracle.solve(deterministic_scenarios, weights=deterministic_weights)
    final_validation_indices = codec.support_indices(
        observed_pool,
        min(config.final_validation_size, len(observed_pool.scenarios)),
    )
    final_validation_scenarios = observed_pool.subset(
        final_validation_indices.tolist()
    )
    # The deterministic and best-epoch designs are frequently the ones already
    # validated during training; when final_validation_size matches
    # validation_batch_size these calls then cost nothing at all.
    if deterministic_plan.feasible:
        deterministic_validation = oracle.solve(
            final_validation_scenarios,
            fixed_design=deterministic_plan.design,
            allow_carbon_slack=True,
            use_cache=True,
        )
        deterministic_loss = _finite_loss(deterministic_validation)
    else:
        deterministic_validation = deterministic_plan
        deterministic_loss = float("inf")

    print(
        "DFL final deterministic: "
        f"plan={deterministic_plan.status}/{deterministic_plan.objective:.6g}, "
        f"validation={deterministic_validation.status}/"
        f"{deterministic_validation.objective:.6g}; "
        f"{_design_summary(deterministic_plan)}",
        flush=True,
    )

    if best_plan is not None:
        best_full_validation = oracle.solve(
            final_validation_scenarios,
            fixed_design=best_plan.design,
            allow_carbon_slack=True,
            use_cache=True,
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
        chosen_source = "deterministic policy"
    else:
        chosen_scenarios = best_scenarios
        chosen_weights = best_weights
        chosen_plan = best_plan
        full_validation = best_full_validation
        chosen_latent = best_latent
        chosen_source = "best sampled epoch"

    if (
        chosen_scenarios is None
        or chosen_weights is None
        or chosen_latent is None
        or full_validation is None
    ):
        raise RuntimeError("DFL did not produce a scenario set.")
    if not chosen_plan.feasible:
        raise RuntimeError(
            f"DFL planning produced no feasible solution: status={chosen_plan.status!r}, "
            f"seconds={chosen_plan.solve_time_seconds:.1f}."
        )
    if not full_validation.feasible:
        raise RuntimeError(
            f"DFL validation produced no feasible solution: status={full_validation.status!r}, "
            f"seconds={full_validation.solve_time_seconds:.1f}."
        )
    print(
        f"DFL selected ({chosen_source}): "
        f"validation={full_validation.status}/{full_validation.objective:.6g}; "
        f"{_design_summary(chosen_plan)}",
        flush=True,
    )
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
