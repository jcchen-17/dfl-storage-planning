"""Observed-scenario baselines for the single-PCC planning experiment.

The learned methods construct support scenarios.  These baselines answer the
same planning question with scenarios selected only from the observed training
library.  Every resulting design is fixed before it is evaluated on one common
held-out test set.
"""

from __future__ import annotations

import time
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from storage_dfl.config import ExperimentConfig, load_config
from storage_dfl.data import ScenarioCodec
from storage_dfl.dfl import select_scenarios
from storage_dfl.planning import StorageDesign, evaluate_fixed_design_recourse
from storage_dfl.planning import make_planning_oracle
from storage_dfl.stages import _experiment_data, _fit_codec, _scenario_set_summary


def _effective_config(
    config_path: str | Path,
    *,
    solver_workers: int | None = None,
    solver_threads: int | None = None,
    solver_relative_gap: float | None = None,
    solver_time_limit_seconds: float | None = None,
) -> ExperimentConfig:
    config = load_config(config_path)
    overrides: dict[str, Any] = {}
    if solver_workers is not None:
        overrides["solver_max_parallel_workers"] = int(solver_workers)
    if solver_threads is not None:
        overrides["solver_threads"] = int(solver_threads)
    if solver_relative_gap is not None:
        overrides["solver_relative_gap"] = float(solver_relative_gap)
    if solver_time_limit_seconds is not None:
        overrides["solver_time_limit_seconds"] = float(solver_time_limit_seconds)
    if overrides:
        config = replace(config, planning=replace(config.planning, **overrides))
    return config


def _evaluation_set(config: ExperimentConfig, codec: ScenarioCodec, count: int):
    _, test_pool = _experiment_data(config, config.data.test_split)
    if count <= 0 or count >= len(test_pool.scenarios):
        return (
            test_pool.scenarios,
            test_pool.normalized_weights(),
            tuple(scenario.name for scenario in test_pool.scenarios),
            len(test_pool.scenarios),
        )
    # The evaluation seed is deliberately fixed by the base config.  Baseline
    # seeds may change training supports, never the held-out test distribution.
    scenarios, weights, names = select_scenarios(
        config.dfl.evaluation_selection_rule,
        test_pool,
        codec,
        count,
        seed=config.seed,
    )
    return scenarios, weights, names, len(test_pool.scenarios)


def _compact_result(result) -> dict[str, Any]:
    """Keep scalar diagnostics while omitting very large hourly tensors."""

    payload = result.to_dict()
    diagnostics = payload.get("recourse_diagnostics")
    if diagnostics is not None:
        diagnostics.pop("hourly_load_shedding_mw", None)
        diagnostics.pop("hourly_carbon_excess_t_per_hour", None)
        diagnostics.pop("grid_carbon_exposure_mw", None)
    return payload


def _bounded(payload: dict[str, Any], accepted_gap: float) -> bool:
    objective = payload.get("objective")
    gap = payload.get("relative_gap")
    if objective is None or gap is None:
        return False
    if not isfinite(float(objective)) or not isfinite(float(gap)):
        return False
    if payload.get("status") in {"optimal", "gaplimit"}:
        return True
    return accepted_gap > 0.0 and float(gap) <= accepted_gap


def _risk_metrics(
    scenario_results: list[dict[str, Any]], weights: tuple[float, ...]
) -> dict[str, float | None]:
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= probabilities.sum()
    shedding_probability = 0.0
    carbon_excess_probability = 0.0
    available = True
    for probability, result in zip(probabilities, scenario_results, strict=True):
        diagnostics = result.get("recourse_diagnostics")
        if diagnostics is None:
            available = False
            break
        if float(diagnostics["load_shedding_mwh"]) > 1.0e-6:
            shedding_probability += float(probability)
        if float(diagnostics["carbon_excess_t"]) > 1.0e-6:
            carbon_excess_probability += float(probability)
    return {
        "load_shedding_probability": shedding_probability if available else None,
        "carbon_excess_probability": carbon_excess_probability if available else None,
    }


def run_reference(
    config_path: str | Path,
    *,
    test_scenarios: int = 0,
    perfect_information: bool = False,
    solver_workers: int | None = None,
    solver_threads: int | None = None,
    solver_relative_gap: float | None = None,
    solver_time_limit_seconds: float | None = None,
) -> dict[str, Any]:
    """Compute the common held-out set and no-storage/oracle references once."""

    config = _effective_config(
        config_path,
        solver_workers=solver_workers,
        solver_threads=solver_threads,
        solver_relative_gap=solver_relative_gap,
        solver_time_limit_seconds=solver_time_limit_seconds,
    )
    feeder, train_pool = _experiment_data(config, config.data.train_split)
    codec = _fit_codec(config, train_pool, feeder)
    scenarios, weights, names, total_test_scenarios = _evaluation_set(
        config, codec, test_scenarios
    )
    oracle = make_planning_oracle(
        feeder, config.planning, config.costs, config.data, config.data_center
    )
    design = StorageDesign(
        site={bus: 0 for bus in feeder.storage_candidates},
        power_mw={bus: 0.0 for bus in feeder.storage_candidates},
        energy_mwh={bus: 0.0 for bus in feeder.storage_candidates},
    )
    started = time.perf_counter()
    reference, rows = evaluate_fixed_design_recourse(
        oracle,
        scenarios,
        weights,
        design,
        allow_carbon_slack=True,
        use_cache=False,
        accepted_relative_gap=config.planning.solver_relative_gap,
    )
    reference_wall_seconds = time.perf_counter() - started
    compact_rows = [_compact_result(row) for row in rows]

    perfect = None
    perfect_wall_seconds = None
    if perfect_information:
        started = time.perf_counter()
        perfect_result = oracle.solve(
            scenarios,
            weights=weights,
            allow_carbon_slack=True,
            use_cache=False,
        )
        perfect_wall_seconds = time.perf_counter() - started
        perfect = _compact_result(perfect_result)

    return {
        "kind": "scenario_selection_reference",
        "base_config": str(Path(config_path).resolve()),
        "test_split": config.data.test_split,
        "test_scenarios_evaluated": len(scenarios),
        "evaluation_selection_rule": (
            "all"
            if len(scenarios) == total_test_scenarios
            else config.dfl.evaluation_selection_rule
        ),
        "evaluation_scenario_names": list(names),
        "evaluation_scenario_weights": list(weights),
        "solver_relative_gap": config.planning.solver_relative_gap,
        "no_storage_reference": _compact_result(reference),
        "no_storage_scenario_results": compact_rows,
        "no_storage_wall_seconds": reference_wall_seconds,
        "perfect_information_reference": perfect,
        "perfect_information_wall_seconds": perfect_wall_seconds,
        "risk": _risk_metrics(compact_rows, weights),
    }


def run_selection_baseline(
    config_path: str | Path,
    *,
    rule: str,
    seed: int,
    support_scenarios: int,
    test_scenarios: int = 0,
    solver_workers: int | None = None,
    solver_threads: int | None = None,
    solver_relative_gap: float | None = None,
    solver_time_limit_seconds: float | None = None,
) -> dict[str, Any]:
    """Plan on selected observed training scenarios and test the fixed design."""

    config = _effective_config(
        config_path,
        solver_workers=solver_workers,
        solver_threads=solver_threads,
        solver_relative_gap=solver_relative_gap,
        solver_time_limit_seconds=solver_time_limit_seconds,
    )
    feeder, train_pool = _experiment_data(config, config.data.train_split)
    codec = _fit_codec(config, train_pool, feeder)
    supports, support_weights, support_names = select_scenarios(
        rule, train_pool, codec, support_scenarios, seed=seed
    )
    evaluation, evaluation_weights, evaluation_names, _ = _evaluation_set(
        config, codec, test_scenarios
    )
    oracle = make_planning_oracle(
        feeder, config.planning, config.costs, config.data, config.data_center
    )

    planning_started = time.perf_counter()
    planning = oracle.solve(
        supports,
        weights=support_weights,
        allow_carbon_slack=config.dfl.training_allow_carbon_slack,
        use_cache=False,
    )
    planning_wall_seconds = time.perf_counter() - planning_started
    if not planning.feasible:
        raise RuntimeError(
            f"{rule} seed={seed} planning failed with status {planning.status!r}."
        )

    evaluation_started = time.perf_counter()
    evaluated, rows = evaluate_fixed_design_recourse(
        oracle,
        evaluation,
        evaluation_weights,
        planning.design,
        allow_carbon_slack=True,
        use_cache=False,
        accepted_relative_gap=config.planning.solver_relative_gap,
    )
    evaluation_wall_seconds = time.perf_counter() - evaluation_started
    compact_rows = [_compact_result(row) for row in rows]
    return {
        "kind": "observed_scenario_selection_baseline",
        "base_config": str(Path(config_path).resolve()),
        "rule": rule,
        "seed": int(seed),
        "support_scenario_count": len(supports),
        "support_scenario_names": list(support_names),
        "support_scenario_weights": list(support_weights),
        "support_summary": _scenario_set_summary(supports, support_weights),
        "planning": _compact_result(planning),
        "planning_wall_seconds": planning_wall_seconds,
        "test_split": config.data.test_split,
        "test_scenarios_evaluated": len(evaluation),
        "evaluation_scenario_names": list(evaluation_names),
        "evaluation_scenario_weights": list(evaluation_weights),
        "out_of_sample_evaluation": _compact_result(evaluated),
        "out_of_sample_scenario_results": compact_rows,
        "out_of_sample_wall_seconds": evaluation_wall_seconds,
        "solver_relative_gap": config.planning.solver_relative_gap,
        "risk": _risk_metrics(compact_rows, evaluation_weights),
    }


def comparison_metrics(
    baseline: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    """Derive common-reference headline metrics for one completed baseline."""

    if baseline["evaluation_scenario_names"] != reference["evaluation_scenario_names"]:
        raise ValueError("Baseline and reference use different evaluation scenarios.")
    if not np.allclose(
        baseline["evaluation_scenario_weights"],
        reference["evaluation_scenario_weights"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("Baseline and reference use different evaluation weights.")
    accepted_gap = float(baseline["solver_relative_gap"])
    evaluated = baseline["out_of_sample_evaluation"]
    no_storage = reference["no_storage_reference"]
    comparable = _bounded(evaluated, accepted_gap) and _bounded(
        no_storage, accepted_gap
    )
    storage_value = (
        float(no_storage["objective"]) - float(evaluated["objective"])
        if comparable
        else None
    )
    perfect = reference.get("perfect_information_reference")
    regret = None
    if perfect is not None and _bounded(perfect, accepted_gap) and comparable:
        regret = (
            float(evaluated["objective"]) - float(perfect["objective"])
        ) / (abs(float(perfect["objective"])) + 1.0e-6)
    diagnostics = evaluated.get("recourse_diagnostics") or {}
    design = evaluated["design"]
    planning = baseline["planning"]
    return {
        "objectives_comparable": comparable,
        "planning_converged": _bounded(planning, accepted_gap),
        "planning_status": planning.get("status"),
        "planning_relative_gap": planning.get("relative_gap"),
        "test_status": evaluated.get("status"),
        "test_relative_gap": evaluated.get("relative_gap"),
        "test_objective": evaluated.get("objective"),
        "storage_value": storage_value,
        "decision_regret": regret,
        "power_mw": design["power_mw"].get("PCC"),
        "energy_mwh": design["energy_mwh"].get("PCC"),
        "load_shedding_mwh": diagnostics.get("load_shedding_mwh"),
        "load_shedding_probability": baseline["risk"].get(
            "load_shedding_probability"
        ),
        "carbon_excess_t": diagnostics.get("carbon_excess_t"),
        "carbon_excess_probability": baseline["risk"].get(
            "carbon_excess_probability"
        ),
        "planning_wall_seconds": baseline.get("planning_wall_seconds"),
        "evaluation_wall_seconds": baseline.get("out_of_sample_wall_seconds"),
    }
