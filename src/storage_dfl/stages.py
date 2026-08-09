from __future__ import annotations

import json
import csv
import time
from dataclasses import asdict, dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import torch

from storage_dfl.config import ExperimentConfig, load_config
from storage_dfl.data import (
    Scenario,
    ScenarioCodec,
    ScenarioPool,
    load_historical_scenarios,
)
from storage_dfl.dfl import (
    DirectSupportPolicy,
    resolve_device,
    support_init_indices,
    select_scenarios,
    train_direct_generator,
    train_scenario_bo,
)
from storage_dfl.models import (
    GENERATOR_KINDS,
    ConditionalGenerator,
    build_generator,
    generator_from_checkpoint,
    save_generator,
    train_generator,
)
from storage_dfl.network import Feeder, ieee13_unbalanced_microgrid
from storage_dfl.planning import (
    PlanningJob,
    PlanningResult,
    StorageDesign,
    StoragePlanningOracle,
    make_planning_oracle,
)


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def cvae_checkpoint(self) -> Path:
        """Legacy CVAE checkpoint path, kept readable for pre-existing runs."""
        return self.root / "cvae.pt"

    def generator_checkpoint_for(self, kind: str) -> Path:
        # The CVAE keeps its historical file name so runs produced before other
        # generators existed can still be evaluated without retraining.
        return self.cvae_checkpoint if kind == "cvae" else self.root / f"{kind}.pt"

    def generator_json_for(self, stem: str, kind: str) -> Path:
        return self.root / f"{kind}_{stem}.json"

    @property
    def dfl_checkpoint(self) -> Path:
        """Legacy, unscoped DFL checkpoint path."""
        return self.root / "dfl_support.pt"

    def dfl_checkpoint_for(self, method: str) -> Path:
        return self.root / f"dfl_support_{method}.pt"

    def dfl_json_for(self, stem: str, method: str) -> Path:
        return self.root / f"{stem}_{method}.json"

    @property
    def normalization(self) -> Path:
        return self.root / "normalization.json"

    @property
    def tensorboard(self) -> Path:
        return self.root / "tensorboard"


def _json_safe(value: Any, path: str, offenders: list[str]) -> Any:
    """Replace non-finite floats with null, recording where they were.

    ``allow_nan=False`` is deliberate -- the artifacts are read by tools that
    reject NaN and Infinity -- but it made the write the most expensive possible
    place to fail: a single unbounded field, typically ``relative_gap`` from a
    solve that kept an incumbent without proving a bound, discarded a whole run's
    results after every solve had already been paid for.  Null says "not
    available" without inventing a number, and the caller reports what was lost.
    """

    if isinstance(value, float):
        if not isfinite(value):
            offenders.append(path or "<root>")
            return None
        return value
    if isinstance(value, dict):
        return {
            key: _json_safe(item, f"{path}.{key}" if path else str(key), offenders)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, f"{path}[{index}]", offenders)
            for index, item in enumerate(value)
        ]
    return value


def _write_json(path: Path, payload: object) -> None:
    offenders: list[str] = []
    sanitized = _json_safe(payload, "", offenders)
    if offenders:
        print(
            f"WARNING: {path.name} contains {len(offenders)} non-finite value(s), "
            f"written as null: {', '.join(offenders)}. A non-finite relative_gap "
            "means the solve stopped without proving a bound -- check the status "
            "field, and if it is 'memlimit' raise solver_memory_limit_mb or "
            "shrink the scenario set.",
            flush=True,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(sanitized, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _summary_writer(path: Path, enabled: bool, config: ExperimentConfig):
    """Open a TensorBoard writer in a directory unique to this run.

    TensorBoard treats a directory as one run and merges every event file it
    holds, so writing successive runs to a fixed path drew them as a single
    series whose step counter restarted at zero each time.  A timestamped leaf
    separates them in the run picker.

    The leaf is only the timestamp, not the config name: the config already
    determines ``output_dir``, so its name is present further up the path, and
    the event file names TensorBoard generates are long enough that a second
    label risks the 260-character path limit on Windows.
    """

    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard is not installed. Run `python -m pip install tensorboard`."
        ) from exc
    writer = SummaryWriter(log_dir=str(path / time.strftime("%Y%m%d-%H%M%S")))
    writer.add_text("experiment/config", f"```json\n{json.dumps(asdict(config), default=str, indent=2)}\n```")
    return writer


def _experiment_data(
    config: ExperimentConfig,
    split: str,
) -> tuple[Feeder, ScenarioPool]:
    feeder = ieee13_unbalanced_microgrid()
    pool = load_historical_scenarios(
        config.data.dataset_path,
        split=split,
        horizon=config.data.horizon,
    )
    if config.data.tariff_spread_scale <= 0.0:
        raise ValueError("data.tariff_spread_scale must be positive.")
    if abs(config.data.tariff_spread_scale - 1.0) > 1.0e-12:
        reference = float(config.data.tariff_reference_price_per_mwh)
        if reference <= 0.0:
            raise ValueError(
                "A positive tariff_reference_price_per_mwh is required when "
                "tariff_spread_scale differs from one."
            )
        pool = ScenarioPool(
            tuple(
                replace(
                    scenario,
                    grid_price_per_mwh=np.maximum(
                        0.0,
                        reference
                        + config.data.tariff_spread_scale
                        * (scenario.grid_price_per_mwh - reference),
                    ),
                )
                for scenario in pool.scenarios
            )
        )
    first = pool.scenarios[0]
    if first.horizon != config.data.horizon:
        raise ValueError(
            f"Configured horizon {config.data.horizon} does not match dataset horizon "
            f"{first.horizon}."
        )
    if first.num_buses != len(feeder.buses) or first.num_phases != len(feeder.phases):
        raise ValueError("Historical scenario dimensions do not match the IEEE13 feeder.")
    for scenario in pool.scenarios:
        if np.any(scenario.active_load_mw[:, ~feeder.phase_mask] != 0.0):
            raise ValueError(f"{scenario.name}: load is nonzero on an absent phase")
        if np.any(scenario.pv_available_mw[:, ~feeder.phase_mask] != 0.0):
            raise ValueError(f"{scenario.name}: PV is nonzero on an absent phase")
    return feeder, pool


def _load_torch(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_codec(paths: ArtifactPaths, feeder: Feeder) -> ScenarioCodec:
    if not paths.normalization.exists():
        raise FileNotFoundError(
            f"Normalization file not found: {paths.normalization}. Train the CVAE first."
        )
    return ScenarioCodec.from_normalization_dict(_read_json(paths.normalization), feeder)


def _require_current_codec(config: ExperimentConfig, codec: ScenarioCodec) -> None:
    """Reject stale normalization when deterministic tariff restoration is required."""

    if config.data.deterministic_price and (
        codec.tariff_contexts is None or codec.tariff_templates is None
    ):
        raise RuntimeError(
            "This configuration treats price as a deterministic tariff, but the "
            "normalization file predates tariff-template restoration. Retrain the "
            "CVAE before running DFL or evaluation."
        )
    if config.data.deterministic_price and (
        abs(codec.tariff_spread_scale - config.data.tariff_spread_scale) > 1.0e-12
        or abs(
            codec.tariff_reference_price_per_mwh
            - config.data.tariff_reference_price_per_mwh
        )
        > 1.0e-9
    ):
        raise RuntimeError(
            "The saved tariff templates were built for a different price-spread "
            "case. Retrain the CVAE before running DFL or evaluation."
        )


def _apply_generator_override(
    config: ExperimentConfig,
    generator_override: str | None,
) -> ExperimentConfig:
    if generator_override is None:
        return config
    if generator_override not in GENERATOR_KINDS:
        raise ValueError(f"generator_override must be one of {GENERATOR_KINDS}.")
    return replace(
        config, generator=replace(config.generator, kind=generator_override)
    )


def load_generator(
    paths: ArtifactPaths,
    kind: str,
    device: torch.device,
) -> ConditionalGenerator:
    checkpoint = _load_torch(paths.generator_checkpoint_for(kind), device)
    stored_kind = str(checkpoint.get("kind", "cvae"))
    if stored_kind != kind:
        raise RuntimeError(
            f"Requested generator {kind!r}, but the checkpoint holds {stored_kind!r}."
        )
    return generator_from_checkpoint(checkpoint, device)


def train_generator_stage(
    config_path: str | Path,
    *,
    tensorboard: bool = True,
    generator_override: str | None = None,
) -> dict:
    config = _apply_generator_override(load_config(config_path), generator_override)
    kind = config.generator.kind
    paths = ArtifactPaths(config.output_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    feeder, observed_pool = _experiment_data(config, config.data.train_split)
    codec = ScenarioCodec.fit(
        observed_pool,
        feeder,
        deterministic_price=config.data.deterministic_price,
        tariff_spread_scale=config.data.tariff_spread_scale,
        tariff_reference_price_per_mwh=(
            config.data.tariff_reference_price_per_mwh
        ),
    )
    trajectories, contexts = codec.encode_pool(observed_pool)
    # Encoded with the codec fitted on the training split, never one refitted
    # here: the held-out curve has to describe the generator downstream stages
    # load, and they all share this one normalization.
    _, validation_pool = _experiment_data(config, config.data.validation_split)
    validation_trajectories, validation_contexts = codec.encode_pool(validation_pool)
    device = resolve_device(config.dfl.device)
    model = build_generator(config, codec.trajectory_dim, codec.context_dim)
    writer = _summary_writer(paths.tensorboard / kind, tensorboard, config)
    try:
        history = train_generator(
            model,
            trajectories,
            contexts,
            config,
            device,
            trajectory_mean=codec.trajectory_mean,
            trajectory_std=codec.trajectory_std,
            field_masks=codec.field_masks(),
            writer=writer,
            validation_trajectories=validation_trajectories,
            validation_contexts=validation_contexts,
        )
    finally:
        if writer is not None:
            writer.close()

    checkpoint_path = paths.generator_checkpoint_for(kind)
    save_generator(model, checkpoint_path)
    # Every generator shares one normalization, so the codec stays interchangeable
    # and generated scenarios from different models remain directly comparable.
    _write_json(paths.normalization, codec.normalization_dict())
    _write_json(
        paths.generator_json_for("history", kind),
        [asdict(record) for record in history],
    )
    payload = {
        "generator": kind,
        "device": str(device),
        "dataset": str(config.data.dataset_path),
        "split": config.data.train_split,
        "scenarios": len(observed_pool.scenarios),
        "validation_split": config.data.validation_split,
        "validation_scenarios": len(validation_pool.scenarios),
        "trajectory_dim": codec.trajectory_dim,
        "latent_dim": model.latent_dim,
        "epochs": len(history),
        "initial_loss": history[0].loss,
        "final_loss": history[-1].loss,
        "final_metrics": asdict(history[-1]),
        "checkpoint": str(checkpoint_path),
    }
    # Training runs a fixed number of epochs, so where each held-out curve turned
    # is reported rather than acted on. A best epoch well before the last one is
    # what overfitting looks like here, and it is the evidence an early-stopping
    # point would be chosen from.
    validation_minima: dict[str, dict[str, float]] = {}
    for name in sorted(history[-1].metrics):
        if not name.startswith("validation_"):
            continue
        best = min(history, key=lambda record, key=name: record.metrics[key])
        validation_minima[name] = {
            "epoch": best.epoch,
            "value": best.metrics[name],
            "final": history[-1].metrics[name],
        }
    payload["validation_minima"] = validation_minima
    _write_json(paths.generator_json_for("result", kind), payload)
    return payload


def train_cvae_stage(
    config_path: str | Path,
    *,
    tensorboard: bool = True,
    generator_override: str | None = None,
) -> dict:
    """Backward-compatible alias for :func:`train_generator_stage`."""

    return train_generator_stage(
        config_path,
        tensorboard=tensorboard,
        generator_override=generator_override,
    )


def _method_tag(config: ExperimentConfig) -> str:
    """Artifact suffix identifying both the generator and the DFL method.

    The CVAE keeps the bare method name so runs made before other generators
    existed are not orphaned; other generators get a prefixed tag so a GAN run
    never overwrites a CVAE run in the same output directory.
    """

    kind = config.generator.kind
    base = config.dfl.method if kind == "cvae" else f"{kind}_{config.dfl.method}"
    scale = float(config.costs.battery_capex_scale)
    if abs(scale - 1.0) <= 1.0e-12:
        return base
    scale_tag = f"{100.0 * scale:g}".replace(".", "p")
    return f"{base}_capex{scale_tag}"


def train_dfl_stage(
    config_path: str | Path,
    *,
    tensorboard: bool = True,
    method_override: str | None = None,
    generator_override: str | None = None,
    battery_capex_scale: float | None = None,
) -> dict:
    config = _apply_generator_override(load_config(config_path), generator_override)
    if battery_capex_scale is not None:
        if battery_capex_scale <= 0.0:
            raise ValueError("battery_capex_scale must be positive.")
        config = replace(
            config,
            costs=replace(
                config.costs, battery_capex_scale=float(battery_capex_scale)
            ),
        )
    if method_override is not None:
        if method_override not in {"reinforce", "scenario_bo"}:
            raise ValueError("method_override must be 'reinforce' or 'scenario_bo'.")
        config = replace(config, dfl=replace(config.dfl, method=method_override))
    tag = _method_tag(config)
    paths = ArtifactPaths(config.output_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    feeder, observed_pool = _experiment_data(config, config.data.validation_split)
    codec = _load_codec(paths, feeder)
    _require_current_codec(config, codec)
    device = resolve_device(config.dfl.device)
    cvae = load_generator(paths, config.generator.kind, device)
    if cvae.latent_dim > codec.trajectory_dim // 2:
        # A diffusion model in ``latent_mode: full`` lands here. Its latent is the
        # whole trajectory, which the score-function policy cannot search.
        raise RuntimeError(
            f"Generator {cvae.kind!r} exposes a {cvae.latent_dim}-dimensional latent, "
            "which is too large for the DFL policy. Use a projected latent."
        )
    # The training loop may run at a looser tolerance than the reported numbers.
    # evaluate_stage builds its own oracle from config.planning, so whatever is
    # set here never reaches a reported objective.
    training_planning = config.planning
    if (
        config.dfl.training_relative_gap > 0.0
        and not config.planning.enumerate_storage_sites
    ):
        training_planning = replace(
            config.planning,
            solver_relative_gap=config.dfl.training_relative_gap,
        )
        print(
            f"training solves at relative gap "
            f"{config.dfl.training_relative_gap:g}; evaluation stays at "
            f"{config.planning.solver_relative_gap:g}",
            flush=True,
        )
    if training_planning.enumerate_storage_sites:
        print(
            "free planning uses exact site enumeration: none + "
            f"{len(feeder.storage_candidates)} forced-site parts, "
            f"{training_planning.enumeration_site_time_limit_seconds:g}s per site",
            flush=True,
        )
    oracle = make_planning_oracle(
        feeder, training_planning, config.costs, config.data, config.data_center
    )
    writer = _summary_writer(paths.tensorboard / f"dfl_{tag}", tensorboard, config)
    policy: DirectSupportPolicy | None = None
    support_source_names: list[str] = []
    try:
        if config.dfl.method == "scenario_bo":
            result = train_scenario_bo(
                cvae,
                codec,
                observed_pool,
                oracle,
                config.dfl,
                config.seed,
                writer=writer,
            )
            support_source_names = list(result.support_source_names)
        elif config.dfl.method == "reinforce":
            validation_trajectories, validation_contexts = codec.encode_pool(observed_pool)
            # Only the policy's starting point. codec.support_indices still picks
            # the fixed validation subset and the reported test subset, so those
            # stay put and results remain comparable across support_init_rule.
            support_indices = support_init_indices(
                config.dfl.support_init_rule,
                observed_pool,
                codec,
                config.dfl.num_support_scenarios,
                seed=config.seed,
            )
            support_source_names = observed_pool.names(support_indices.tolist())
            support_conditions = torch.as_tensor(
                validation_contexts[support_indices],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                initial_latent, _ = cvae.encode(
                    torch.as_tensor(
                        validation_trajectories[support_indices],
                        dtype=torch.float32,
                        device=device,
                    ),
                    support_conditions,
                )
            policy = DirectSupportPolicy(
                support_count=config.dfl.num_support_scenarios,
                latent_dim=cvae.latent_dim,
                seed=config.seed,
                initial_latent=initial_latent.cpu(),
            )
            result = train_direct_generator(
                policy,
                cvae,
                codec,
                observed_pool,
                oracle,
                config.dfl,
                config.seed,
                support_conditions=support_conditions,
                writer=writer,
            )
        else:
            raise ValueError("dfl.method must be 'scenario_bo' or 'reinforce'.")
    finally:
        if writer is not None:
            writer.close()

    checkpoint = {
        "method": config.dfl.method,
        "generator": config.generator.kind,
        "support_count": config.dfl.num_support_scenarios,
        "latent_dim": cvae.latent_dim,
        "support_latent": torch.as_tensor(result.support_latent),
        "support_conditions": torch.as_tensor(result.support_conditions),
        "scenario_weights": torch.as_tensor(result.scenario_weights),
        "support_source_names": support_source_names,
        # Evaluation decodes the supports and replans. Persist the design chosen
        # during training so a near-degenerate replan that flips bus or capacity
        # is visible instead of looking like a change learned by DFL.
        "selected_training_design": result.planning_result.to_dict()["design"],
    }
    if policy is not None:
        checkpoint["policy_state_dict"] = policy.state_dict()
    if config.dfl.method == "scenario_bo":
        checkpoint["candidate_source_names"] = list(result.candidate_source_names)
        checkpoint["selector_feature_names"] = list(result.feature_names)
        checkpoint["selector_parameters"] = torch.as_tensor(result.best_parameters)
    dfl_checkpoint = paths.dfl_checkpoint_for(tag)
    torch.save(checkpoint, dfl_checkpoint)
    _write_json(
        paths.dfl_json_for("dfl_history", tag),
        result.history_as_dicts(),
    )
    payload = {
        "device": result.device,
        "split": config.data.validation_split,
        "validation_scenarios": len(observed_pool.scenarios),
        "method": config.dfl.method,
        "generator": config.generator.kind,
        "epochs": len(result.history),
        # Both of these move the starting point, so a run is only reproducible
        # and only comparable to another run when they are recorded alongside it.
        "support_init_rule": config.dfl.support_init_rule,
        "seed": config.seed,
        # Which tolerance the loop actually ran at. Two runs at different values
        # searched different landscapes even with everything else identical.
        "training_relative_gap": float(
            0.0
            if training_planning.enumerate_storage_sites
            else training_planning.solver_relative_gap
        ),
        "planning_oracle": (
            "exact_site_enumeration"
            if training_planning.enumerate_storage_sites
            else "joint_siting"
        ),
        "enumeration_site_time_limit_seconds": (
            float(training_planning.enumeration_site_time_limit_seconds)
            if training_planning.enumerate_storage_sites
            else None
        ),
        "training_validation_relative_gap": float(
            config.dfl.validation_relative_gap
            if config.dfl.validation_relative_gap > 0.0
            else training_planning.solver_relative_gap
        ),
        "battery_capex_scale": float(config.costs.battery_capex_scale),
        "scenario_weights": list(result.scenario_weights),
        "support_source_names": checkpoint["support_source_names"],
        "planning": result.planning_result.to_dict(),
        "training_validation": result.full_validation_result.to_dict(),
        "checkpoint": str(dfl_checkpoint),
    }
    if config.dfl.method == "scenario_bo":
        payload["finalist_evaluations"] = list(result.finalist_evaluations)
    _write_json(paths.dfl_json_for("dfl_result", tag), payload)
    return payload


def _write_trajectories(
    path: Path,
    observed: ScenarioPool,
    generated: tuple[Scenario, ...],
    weights: tuple[float, ...],
    buses: tuple[str, ...],
    phases: tuple[str, ...],
) -> None:
    fieldnames = [
        "source", "scenario_name", "scenario_weight", "time", "bus", "phase",
        "active_load_mw", "reactive_load_mvar", "pv_available_mw",
        "workload_arrival", "pue", "grid_price_per_mwh", "grid_carbon_t_per_mwh",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for source, scenarios, scenario_weights in (
            ("observed_test", observed.scenarios, (None,) * len(observed.scenarios)),
            ("dfl_generated", generated, weights),
        ):
            for scenario, weight in zip(scenarios, scenario_weights):
                for time in range(scenario.horizon):
                    for bus_index, bus in enumerate(buses):
                        for phase_index, phase in enumerate(phases):
                            writer.writerow(
                                {
                                    "source": source,
                                    "scenario_name": scenario.name,
                                    "scenario_weight": weight,
                                    "time": time,
                                    "bus": bus,
                                    "phase": phase,
                                    "active_load_mw": float(scenario.active_load_mw[time, bus_index, phase_index]),
                                    "reactive_load_mvar": float(scenario.reactive_load_mvar[time, bus_index, phase_index]),
                                    "pv_available_mw": float(scenario.pv_available_mw[time, bus_index, phase_index]),
                                    "workload_arrival": float(scenario.workload_arrival[time]),
                                    "pue": float(scenario.pue[time]),
                                    "grid_price_per_mwh": float(scenario.grid_price_per_mwh[time]),
                                    "grid_carbon_t_per_mwh": float(scenario.grid_carbon_t_per_mwh[time]),
                                }
                            )


def _scenario_set_summary(
    scenarios: tuple[Scenario, ...],
    weights: tuple[float, ...] | None = None,
) -> dict[str, float]:
    if weights is None:
        normalized_weights = np.full(len(scenarios), 1.0 / len(scenarios))
    else:
        normalized_weights = np.asarray(weights, dtype=float)
        normalized_weights = normalized_weights / normalized_weights.sum()
    peak_load = []
    peak_net_load = []
    price_spread = []
    for scenario in scenarios:
        load = scenario.active_load_mw.sum(axis=(1, 2))
        pv = scenario.pv_available_mw.sum(axis=(1, 2))
        peak_load.append(float(load.max()))
        peak_net_load.append(float((load - pv).max()))
        price_spread.append(float(np.ptp(scenario.grid_price_per_mwh)))
    return {
        "weighted_peak_load_mw": float(np.dot(normalized_weights, peak_load)),
        "maximum_peak_load_mw": float(max(peak_load)),
        "weighted_peak_net_load_mw": float(np.dot(normalized_weights, peak_net_load)),
        "maximum_peak_net_load_mw": float(max(peak_net_load)),
        "weighted_price_spread_per_mwh": float(np.dot(normalized_weights, price_spread)),
        "maximum_price_spread_per_mwh": float(max(price_spread)),
    }


def _bounded_result(result, accepted_relative_gap: float | None = None) -> bool:
    """Whether an objective is comparable rather than just a finite incumbent."""

    base = (
        result.feasible
        and isfinite(result.objective)
        and isfinite(result.relative_gap)
    )
    if not base:
        return False
    if result.status in {"optimal", "gaplimit"}:
        return True
    return (
        accepted_relative_gap is not None
        and accepted_relative_gap > 0.0
        and result.relative_gap <= accepted_relative_gap
    )


def _aggregate_scenario_wise_results(
    results: list[PlanningResult],
    weights: tuple[float, ...],
    design,
    demand_dollars_per_mw_year: float,
    accepted_relative_gap: float,
) -> PlanningResult:
    """Reconstruct the fixed-design multi-scenario objective from small solves.

    Investment is paid once. Interval operating and carbon costs are expected
    values, while the original joint model uses one peak variable shared by all
    scenarios, so annual demand cost is based on the maximum scenario peak.
    """

    normalized = np.asarray(weights, dtype=float)
    normalized /= normalized.sum()
    names = tuple(name for result in results for name in result.scenario_names)
    if not results or any(not result.feasible for result in results):
        return PlanningResult(
            status="infeasible",
            objective=float("inf"),
            investment_cost=float("inf"),
            operating_cost=float("inf"),
            carbon_slack_cost=float("inf"),
            peak_grid_mw=float("inf"),
            design=design,
            scenario_names=names,
            solve_time_seconds=sum(result.solve_time_seconds for result in results),
            relative_gap=float("inf"),
        )

    peak = max(float(result.peak_grid_mw) for result in results)
    non_demand_operating = sum(
        float(weight)
        * (
            float(result.operating_cost)
            - demand_dollars_per_mw_year * float(result.peak_grid_mw)
        )
        for weight, result in zip(normalized, results, strict=True)
    )
    operating = demand_dollars_per_mw_year * peak + non_demand_operating
    carbon = sum(
        float(weight) * float(result.carbon_slack_cost)
        for weight, result in zip(normalized, results, strict=True)
    )
    investment = float(results[0].investment_cost)
    bounded = all(
        _bounded_result(result, accepted_relative_gap) for result in results
    )
    return PlanningResult(
        status="optimal" if bounded else "scenario_limit",
        objective=investment + operating + carbon,
        investment_cost=investment,
        operating_cost=operating,
        carbon_slack_cost=carbon,
        peak_grid_mw=peak,
        design=design,
        scenario_names=names,
        # Sum is total solver effort across parallel workers, not wall time.
        solve_time_seconds=sum(result.solve_time_seconds for result in results),
        relative_gap=max(float(result.relative_gap) for result in results),
    )


def _solve_fixed_design_scenario_wise(
    oracle: StoragePlanningOracle,
    scenarios: tuple[Scenario, ...],
    weights: tuple[float, ...],
    design,
    demand_dollars_per_mw_year: float,
    accepted_relative_gap: float,
) -> tuple[PlanningResult, list[PlanningResult], float]:
    started = time.perf_counter()
    results = oracle.solve_many(
        [PlanningJob((scenario,), fixed_design=design) for scenario in scenarios],
        allow_carbon_slack=True,
    )
    wall_seconds = time.perf_counter() - started
    aggregate = _aggregate_scenario_wise_results(
        results,
        weights,
        design,
        demand_dollars_per_mw_year,
        accepted_relative_gap,
    )
    return aggregate, results, wall_seconds


@torch.no_grad()
def evaluate_stage(
    config_path: str | Path,
    *,
    method_override: str | None = None,
    generator_override: str | None = None,
    memory_limit_mb: float | None = None,
    scenarios: int | None = None,
) -> dict:
    config = _apply_generator_override(load_config(config_path), generator_override)
    if scenarios is not None:
        # final_validation_size is read here and by the end-of-training finalist
        # comparison, which runs under the training memory budget. Overriding it
        # only for evaluation lets the config keep a size that is safe there
        # while the reported number is measured on as many scenarios as the
        # serial solve can afford.
        config = replace(
            config, dfl=replace(config.dfl, final_validation_size=scenarios)
        )
    if memory_limit_mb is not None:
        # solver_memory_limit_mb is sized for solver_max_parallel_workers solves
        # running at once during training. Evaluation solves one model at a time
        # over final_validation_size scenarios, which is the larger model and the
        # smaller budget -- at 32 scenarios every solve here stopped on memlimit
        # while training was fine. The serial path can safely be given what the
        # whole concurrent set gets.
        config = replace(
            config, planning=replace(config.planning, solver_memory_limit_mb=memory_limit_mb)
        )
    if method_override is not None:
        if method_override not in {"reinforce", "scenario_bo"}:
            raise ValueError("method_override must be 'reinforce' or 'scenario_bo'.")
        config = replace(config, dfl=replace(config.dfl, method=method_override))
    tag = _method_tag(config)
    paths = ArtifactPaths(config.output_dir)
    feeder, observed_pool = _experiment_data(config, config.data.test_split)
    codec = _load_codec(paths, feeder)
    _require_current_codec(config, codec)
    evaluation_scenarios, evaluation_weights, evaluation_names = select_scenarios(
        config.dfl.evaluation_selection_rule,
        observed_pool,
        codec,
        min(config.dfl.final_validation_size, len(observed_pool.scenarios)),
        seed=config.seed,
    )
    evaluation_pool = ScenarioPool(evaluation_scenarios)
    device = resolve_device(config.dfl.device)
    cvae = load_generator(paths, config.generator.kind, device)
    checkpoint_path = paths.dfl_checkpoint_for(tag)
    # Read old runs when no method-scoped checkpoint has been created yet.
    if not checkpoint_path.exists() and paths.dfl_checkpoint.exists():
        checkpoint_path = paths.dfl_checkpoint
    checkpoint = _load_torch(checkpoint_path, device)
    checkpoint_method = checkpoint.get("method", "reinforce")
    if checkpoint_method != config.dfl.method:
        raise RuntimeError(
            f"Requested DFL method {config.dfl.method!r}, but checkpoint "
            f"{checkpoint_path} contains {checkpoint_method!r}. Train that method first."
        )
    checkpoint_generator = str(checkpoint.get("generator", "cvae"))
    if checkpoint_generator != config.generator.kind:
        raise RuntimeError(
            f"Requested generator {config.generator.kind!r}, but checkpoint "
            f"{checkpoint_path} was trained with {checkpoint_generator!r}."
        )
    latent = checkpoint["support_latent"].to(device=device, dtype=torch.float32)
    conditions = checkpoint["support_conditions"].to(device=device, dtype=torch.float32)
    weights = tuple(float(value) for value in checkpoint["scenario_weights"].cpu())
    decoded = cvae.decode(latent, conditions).cpu().numpy()
    generated = codec.decode_batch(
        decoded,
        conditions.cpu().numpy(),
        name_prefix="dfl_evaluation",
    )

    oracle = make_planning_oracle(
        feeder, config.planning, config.costs, config.data, config.data_center
    )
    planning = oracle.solve(
        generated,
        weights=weights,
        allow_carbon_slack=config.dfl.training_allow_carbon_slack,
    )
    if not planning.feasible:
        raise RuntimeError(
            f"Evaluation planning failed with status {planning.status!r} after "
            f"{planning.solve_time_seconds:.1f} seconds."
        )
    training_design = checkpoint.get("selected_training_design")
    design_changed_on_replan = None
    evaluation_design = planning.design
    evaluated_design_source = "replanned_support"
    if training_design is not None:
        replanned_design = planning.to_dict()["design"]
        training_installed = {
            bus for bus, value in training_design["site"].items() if value > 0
        }
        replanned_installed = {
            bus for bus, value in replanned_design["site"].items() if value > 0
        }
        capacity_delta = max(
            (
                abs(float(replanned_design[field][bus]) - float(training_design[field][bus]))
                for field in ("power_mw", "energy_mwh")
                for bus in training_design[field]
            ),
            default=0.0,
        )
        design_changed_on_replan = (
            training_installed != replanned_installed or capacity_delta > 0.01
        )
        evaluation_design = StorageDesign(
            site={bus: int(value) for bus, value in training_design["site"].items()},
            power_mw={
                bus: float(value)
                for bus, value in training_design["power_mw"].items()
            },
            energy_mwh={
                bus: float(value)
                for bus, value in training_design["energy_mwh"].items()
            },
        )
        evaluated_design_source = "selected_training_design"
        if design_changed_on_replan:
            print(
                "WARNING: replanning the saved support changed the selected "
                f"training design: buses {sorted(training_installed)} -> "
                f"{sorted(replanned_installed)}, maximum capacity change "
                f"{capacity_delta:.4f}. This indicates solver/tie instability, "
                "not a new DFL update.",
                flush=True,
            )
    print(
        f"Out-of-sample evaluation: solving {len(evaluation_pool.scenarios)} "
        "scenarios separately...",
        flush=True,
    )
    validation, validation_scenario_results, validation_wall_seconds = (
        _solve_fixed_design_scenario_wise(
            oracle,
            evaluation_pool.scenarios,
            evaluation_weights,
            evaluation_design,
            config.costs.demand_dollars_per_mw_year,
            config.planning.solver_relative_gap,
        )
    )
    if not validation.feasible:
        raise RuntimeError(
            f"Out-of-sample validation failed with status {validation.status!r} after "
            f"{validation.solve_time_seconds:.1f} seconds."
        )
    # What the same scenarios cost with no storage at all. Total system cost is
    # a poor denominator for comparing designs here -- the storage decision moves
    # a few percent of it and every design pays the same untouchable remainder --
    # so the difference against this reference is reported alongside it.
    reference_oracle = StoragePlanningOracle(
        feeder,
        replace(config.planning, max_storage_sites=0),
        config.costs,
        config.data,
        config.data_center,
    )
    no_storage_design = StorageDesign(
        site={bus: 0 for bus in feeder.storage_candidates},
        power_mw={bus: 0.0 for bus in feeder.storage_candidates},
        energy_mwh={bus: 0.0 for bus in feeder.storage_candidates},
    )
    print(
        f"No-storage reference: solving {len(evaluation_pool.scenarios)} "
        "scenarios separately...",
        flush=True,
    )
    reference, reference_scenario_results, reference_wall_seconds = (
        _solve_fixed_design_scenario_wise(
            reference_oracle,
            evaluation_pool.scenarios,
            evaluation_weights,
            no_storage_design,
            config.costs.demand_dollars_per_mw_year,
            config.planning.solver_relative_gap,
        )
    )
    payload = {
        "method": checkpoint.get("method", "reinforce"),
        "generator": checkpoint_generator,
        "device": str(device),
        "test_split": config.data.test_split,
        "test_scenarios_evaluated": len(evaluation_pool.scenarios),
        "evaluation_relative_gap": config.planning.solver_relative_gap,
        "planning_oracle": (
            "exact_site_enumeration"
            if config.planning.enumerate_storage_sites
            else "joint_siting"
        ),
        "enumeration_site_time_limit_seconds": (
            float(config.planning.enumeration_site_time_limit_seconds)
            if config.planning.enumerate_storage_sites
            else None
        ),
        "evaluation_selection_rule": config.dfl.evaluation_selection_rule,
        "evaluation_scenario_names": list(evaluation_names),
        "evaluation_scenario_weights": list(evaluation_weights),
        "no_storage_reference": reference.to_dict(),
        "no_storage_scenario_results": [
            result.to_dict() for result in reference_scenario_results
        ],
        "no_storage_wall_seconds": reference_wall_seconds,
        "objectives_comparable": (
            _bounded_result(reference, config.planning.solver_relative_gap)
            and _bounded_result(validation, config.planning.solver_relative_gap)
        ),
        "storage_value": (
            float(reference.objective) - float(validation.objective)
            if _bounded_result(reference, config.planning.solver_relative_gap)
            and _bounded_result(validation, config.planning.solver_relative_gap)
            else None
        ),
        "generated_scenarios": [
            {"name": scenario.name, "weight": weight, "context": scenario.context.tolist()}
            for scenario, weight in zip(generated, weights)
        ],
        "scenario_summary": {
            "generated": _scenario_set_summary(generated, weights),
            "observed_test": _scenario_set_summary(
                evaluation_pool.scenarios, evaluation_weights
            ),
        },
        "planning": planning.to_dict(),
        "selected_training_design": training_design,
        "evaluated_design_source": evaluated_design_source,
        "design_changed_on_replan": design_changed_on_replan,
        "out_of_sample_validation": validation.to_dict(),
        "out_of_sample_scenario_results": [
            result.to_dict() for result in validation_scenario_results
        ],
        "out_of_sample_wall_seconds": validation_wall_seconds,
    }
    _write_json(paths.dfl_json_for("result", tag), payload)
    _write_trajectories(
        paths.root / f"scenario_trajectories_{tag}.csv",
        evaluation_pool,
        generated,
        weights,
        feeder.buses,
        feeder.phases,
    )
    return payload
