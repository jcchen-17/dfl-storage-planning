from __future__ import annotations

import time
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


@dataclass(frozen=True)
class _ReinforceFinalist:
    scenarios: tuple[Scenario, ...]
    weights: tuple[float, ...]
    latent: np.ndarray
    plan: PlanningResult
    validation_loss: float
    source: str


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


def _converged(result: PlanningResult) -> bool:
    """Return whether an objective is supported by a valid termination bound."""

    return (
        result.feasible
        and isfinite(result.objective)
        and result.status in {"optimal", "gaplimit"}
        and isfinite(result.relative_gap)
    )


def _finite_loss(result: PlanningResult, *, require_converged: bool = False) -> float:
    usable = _converged(result) if require_converged else result.feasible
    return float(result.objective) if usable and isfinite(result.objective) else INFEASIBLE_LOSS


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


def _rank_advantages(
    losses: np.ndarray,
    relative_deadband: float = 0.0,
) -> np.ndarray:
    """Scale-free advantages in [-1, 1] from the ordering of the losses alone.

    Standardizing raw costs let one infeasible sample (INFEASIBLE_LOSS) inflate
    the standard deviation by six orders of magnitude, which collapsed the
    advantage of every feasible sample to roughly zero and wasted the epoch.
    Ranking keeps the feasible samples fully separated no matter how the
    infeasible sentinel is scaled.
    """

    count = losses.shape[0]
    if count == 0:
        return np.empty(0, dtype=float)
    order = np.argsort(losses, kind="stable")
    ranks = np.empty(count, dtype=float)
    ranks[order] = np.arange(count, dtype=float)
    # Solver objectives inside the requested MIP gap are not measurably
    # different. Treat neighbouring values inside that deadband as ties rather
    # than turning numerical ordering into a policy gradient.
    finite = losses[np.isfinite(losses) & (losses < INFEASIBLE_LOSS)]
    scale = max(float(np.median(np.abs(finite))) if finite.size else 1.0, 1.0)
    tolerance = max(float(relative_deadband), 0.0) * scale
    group_start = 0
    while group_start < count:
        group_end = group_start + 1
        anchor = float(losses[order[group_start]])
        while group_end < count:
            value = float(losses[order[group_end]])
            if value - anchor > tolerance:
                break
            group_end += 1
        members = order[group_start:group_end]
        if len(members) > 1:
            ranks[members] = float(ranks[members].mean())
        group_start = group_end
    center = (count - 1) / 2.0
    return (ranks - center) / max(center, 1.0)


def _design_signature(result: PlanningResult) -> tuple:
    """Decision signature at meaningful engineering resolution.

    Five-decimal capacity differences made practically identical designs look
    distinct and created a false action-dependent signal. 10 kW / 10 kWh is
    already finer than the accuracy supported by these planning solves.
    """

    buses = tuple(result.design.power_mw)
    return (
        tuple(sorted(result.design.installed_buses)),
        tuple(round(float(result.design.power_mw[bus]), 2) for bus in buses),
        tuple(round(float(result.design.energy_mwh[bus]), 2) for bus in buses),
    )


def _retain_finalist(
    finalists: dict[tuple, _ReinforceFinalist],
    candidate: _ReinforceFinalist,
    limit: int,
) -> None:
    signature = _design_signature(candidate.plan)
    previous = finalists.get(signature)
    if previous is None or candidate.validation_loss < previous.validation_loss:
        finalists[signature] = candidate
    if len(finalists) > max(1, limit):
        worst = max(finalists, key=lambda key: finalists[key].validation_loss)
        del finalists[worst]


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

    # Displacement of the policy mean from where it started. Without it a run
    # gives no way to tell "learned something" from "never moved": the score-
    # function estimator normalizes through Adam, so a small step size can leave
    # the mean well inside the exploration noise for the whole run while every
    # other curve still looks alive.
    initial_latent = policy.latent_location.detach().clone()
    baseline: float | None = None
    infeasible_samples = 0
    finalists: dict[tuple, _ReinforceFinalist] = {}
    records: list[EpochRecord] = []
    stopping_reference = float("inf")
    epochs_without_material_improvement = 0
    evaluated_policy_samples = 0

    epoch_seconds: list[float] = []
    for epoch in range(config.epochs):
        print(f"DFL epoch {epoch + 1}/{config.epochs}: solving planning candidates...", flush=True)
        epoch_started = time.perf_counter()
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

        planning_started = time.perf_counter()
        plans = oracle.solve_many(
            [PlanningJob(generated, weights) for _, generated, weights in samples]
        )
        planning_wall_seconds = time.perf_counter() - planning_started
        # The validations of one epoch share the fixed scenario set and differ
        # only in the design they fix, so they batch exactly like the planning
        # solves above.  Solved inside the loop below they left every worker but
        # one idle for that half of the epoch.
        validated_indices = [index for index, plan in enumerate(plans) if plan.feasible]
        validation_started = time.perf_counter()
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
        validation_wall_seconds = time.perf_counter() - validation_started
        for sample_index, ((sample, generated, weights), plan) in enumerate(
            zip(samples, plans, strict=True)
        ):
            # An infeasible plan has no design to validate, so it stands in for
            # its own validation, as it did when the two solves were adjacent.
            validation = validations.get(sample_index, plan)
            decision_loss = _finite_loss(validation, require_converged=True)
            if decision_loss >= INFEASIBLE_LOSS:
                infeasible_samples += 1
                infeasible_in_epoch += 1
            candidates.append((sample, generated, weights, plan, validation, decision_loss))
            evaluated_policy_samples += 1
            print(
                f"DFL epoch {epoch + 1}/{config.epochs} sample "
                f"{sample_index + 1}/{max(1, config.policy_samples_per_epoch)}: "
                f"plan={plan.status}/{plan.objective:.6g}, "
                f"validation={validation.status}/{validation.objective:.6g}; "
                f"{_design_summary(plan)}",
                flush=True,
            )

        decision_losses = np.asarray([candidate[5] for candidate in candidates], dtype=float)
        design_signatures = {_design_signature(candidate[3]) for candidate in candidates}
        # The baseline tracks attainable cost, so an infeasible sentinel must not
        # enter it; otherwise one failed sample poisons every later comparison.
        feasible_losses = decision_losses[decision_losses < INFEASIBLE_LOSS]
        baseline_target = float(
            feasible_losses.mean() if feasible_losses.size else decision_losses.mean()
        )
        if len(candidates) > 1:
            advantages = _rank_advantages(
                decision_losses,
                relative_deadband=config.decision_deadband_relative,
            )
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

        material_improvement = False
        if decision_loss < INFEASIBLE_LOSS:
            if not isfinite(stopping_reference):
                stopping_reference = float(decision_loss)
                material_improvement = True
            else:
                tolerance = config.decision_deadband_relative * max(
                    abs(stopping_reference), 1.0
                )
                if decision_loss < stopping_reference - tolerance:
                    stopping_reference = float(decision_loss)
                    material_improvement = True
        if material_improvement:
            epochs_without_material_improvement = 0
        else:
            epochs_without_material_improvement += 1

        for sample, generated, weights, plan, _, decision_loss in candidates:
            if plan.feasible and decision_loss < INFEASIBLE_LOSS:
                _retain_finalist(
                    finalists,
                    _ReinforceFinalist(
                        scenarios=generated,
                        weights=weights,
                        latent=sample.latent.cpu().numpy().copy(),
                        plan=plan,
                        validation_loss=float(decision_loss),
                        source=f"epoch {epoch + 1} sample",
                    ),
                    config.reinforce_finalists,
                )

        latent_shift = float(
            (policy.latent_location.detach() - initial_latent).abs().mean()
        )
        epoch_wall_seconds = time.perf_counter() - epoch_started
        epoch_seconds.append(epoch_wall_seconds)
        print(
            f"DFL epoch {epoch + 1}/{config.epochs} best: "
            f"validation={validation.status}/{validation.objective:.6g}; "
            f"{_design_summary(plan)}; distinct_designs={len(design_signatures)}, "
            f"exploration_std={exploration_std:.4f}",
            flush=True,
        )
        # Wall clock, not the solvers' own reported times: those exclude the
        # queueing behind solver_max_parallel_workers, which is most of the
        # difference between a batch and a serial loop. The remaining estimate
        # uses the mean of the epochs completed so far, so it settles quickly.
        remaining = (config.epochs - epoch - 1) * (
            sum(epoch_seconds) / len(epoch_seconds)
        )
        print(
            f"DFL epoch {epoch + 1}/{config.epochs} timing: "
            f"{epoch_wall_seconds:.1f}s "
            f"(planning {planning_wall_seconds:.1f}s + "
            f"validation {validation_wall_seconds:.1f}s), "
            f"latent_shift={latent_shift:.4f}, "
            f"est. remaining {remaining / 60.0:.1f} min",
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
            # Read against exploration_std on the same chart: while the shift
            # stays far below it, the mean has not left the cloud it is sampling
            # from and no amount of further epochs will show a decision effect.
            writer.add_scalar("policy/latent_shift", latent_shift, epoch)
            if isfinite(stopping_reference):
                writer.add_scalar(
                    "convergence/best_material_validation",
                    stopping_reference,
                    epoch,
                )
            writer.add_scalar(
                "convergence/epochs_without_material_improvement",
                epochs_without_material_improvement,
                epoch,
            )
            writer.add_scalar("time/epoch_seconds", epoch_wall_seconds, epoch)
            writer.add_scalar("time/planning_seconds", planning_wall_seconds, epoch)
            writer.add_scalar("time/validation_seconds", validation_wall_seconds, epoch)
            writer.add_scalar("design/installed_sites", len(plan.design.installed_buses), epoch)
            writer.add_scalar("design/power_mw", sum(plan.design.power_mw.values()), epoch)
            writer.add_scalar("design/energy_mwh", sum(plan.design.energy_mwh.values()), epoch)
            for index, weight in enumerate(weights):
                writer.add_scalar(f"scenario_weight/support_{index}", weight, epoch)
            writer.flush()

        if (
            config.early_stopping_patience > 0
            and epoch + 1 >= config.early_stopping_min_epochs
            and epochs_without_material_improvement
            >= config.early_stopping_patience
        ):
            print(
                f"DFL early stopping after epoch {epoch + 1}: no validation "
                f"improvement larger than {config.decision_deadband_relative:.4g} "
                f"relative for {epochs_without_material_improvement} epochs.",
                flush=True,
            )
            break

    if infeasible_samples == evaluated_policy_samples:
        raise RuntimeError(
            f"All {config.epochs} DFL epochs ended without a converged validation "
            "reward, so the policy was never exposed to a trustworthy planning "
            "gradient and this run carries no decision-focused signal. Establish "
            "that the oracle can produce a bounded incumbent before training: raise "
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
    candidate_by_signature: dict[tuple, _ReinforceFinalist] = {}
    if deterministic_plan.feasible:
        deterministic_candidate = _ReinforceFinalist(
            scenarios=deterministic_scenarios,
            weights=deterministic_weights,
            latent=deterministic_latent.cpu().numpy().copy(),
            plan=deterministic_plan,
            validation_loss=float("inf"),
            source="deterministic policy",
        )
        candidate_by_signature[_design_signature(deterministic_plan)] = (
            deterministic_candidate
        )
    # One representative per engineering-distinct design is enough. The fixed-
    # design objective depends on the design, not on which latent generated it.
    for signature, finalist in finalists.items():
        candidate_by_signature.setdefault(signature, finalist)

    final_results: list[tuple[_ReinforceFinalist, PlanningResult, float]] = []
    for finalist_index, finalist in enumerate(candidate_by_signature.values(), start=1):
        validation = oracle.solve(
            final_validation_scenarios,
            fixed_design=finalist.plan.design,
            allow_carbon_slack=True,
            use_cache=True,
        )
        final_loss = _finite_loss(validation, require_converged=True)
        print(
            f"DFL finalist {finalist_index}/{len(candidate_by_signature)} "
            f"({finalist.source}): validation={validation.status}/"
            f"{validation.objective:.6g}; {_design_summary(finalist.plan)}",
            flush=True,
        )
        if final_loss >= INFEASIBLE_LOSS:
            print(
                f"WARNING: excluded finalist because validation stopped at "
                f"{validation.status!r} without a usable bound over "
                f"{len(final_validation_scenarios)} scenarios.",
                flush=True,
            )
        else:
            final_results.append((finalist, validation, final_loss))

    if not final_results:
        raise RuntimeError(
            "No REINFORCE finalist completed the common final validation with a "
            "usable optimal/gap-limit bound. No checkpoint was selected from "
            "unproven incumbents. Set final_validation_size equal to "
            "validation_batch_size, reduce solver_max_parallel_workers and give "
            "each solve more memory, or raise solver_memory_limit_mb."
        )

    best_final_loss = min(item[2] for item in final_results)
    final_deadband = max(config.decision_deadband_relative, 0.0) * max(
        abs(best_final_loss), 1.0
    )
    statistically_tied = [
        item for item in final_results if item[2] <= best_final_loss + final_deadband
    ]
    # If the objective cannot distinguish finalists at solver accuracy, prefer
    # the cheaper/smaller design rather than rewarding oversizing by noise.
    chosen, full_validation, _ = min(
        statistically_tied,
        key=lambda item: (
            float(item[0].plan.investment_cost),
            sum(item[0].plan.design.energy_mwh.values()),
            sum(item[0].plan.design.power_mw.values()),
        ),
    )
    chosen_scenarios = chosen.scenarios
    chosen_weights = chosen.weights
    chosen_plan = chosen.plan
    chosen_latent = chosen.latent
    chosen_source = chosen.source

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
    if not _converged(full_validation):
        raise RuntimeError(
            f"DFL validation did not converge: status={full_validation.status!r}, "
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
