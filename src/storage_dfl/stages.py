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
from storage_dfl.planning import StoragePlanningOracle


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
    codec = ScenarioCodec.fit(observed_pool, feeder)
    trajectories, contexts = codec.encode_pool(observed_pool)
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
        "trajectory_dim": codec.trajectory_dim,
        "latent_dim": model.latent_dim,
        "epochs": len(history),
        "initial_loss": history[0].loss,
        "final_loss": history[-1].loss,
        "final_metrics": asdict(history[-1]),
        "checkpoint": str(checkpoint_path),
    }
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
    return config.dfl.method if kind == "cvae" else f"{kind}_{config.dfl.method}"


def train_dfl_stage(
    config_path: str | Path,
    *,
    tensorboard: bool = True,
    method_override: str | None = None,
    generator_override: str | None = None,
) -> dict:
    config = _apply_generator_override(load_config(config_path), generator_override)
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
    device = resolve_device(config.dfl.device)
    cvae = load_generator(paths, config.generator.kind, device)
    if cvae.latent_dim > codec.trajectory_dim // 2:
        # A diffusion model in ``latent_mode: full`` lands here. Its latent is the
        # whole trajectory, which the score-function policy cannot search.
        raise RuntimeError(
            f"Generator {cvae.kind!r} exposes a {cvae.latent_dim}-dimensional latent, "
            "which is too large for the DFL policy. Use a projected latent."
        )
    oracle = StoragePlanningOracle(
        feeder, config.planning, config.costs, config.data, config.data_center
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
            support_indices = codec.support_indices(
                observed_pool,
                config.dfl.num_support_scenarios,
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


@torch.no_grad()
def evaluate_stage(
    config_path: str | Path,
    *,
    method_override: str | None = None,
    generator_override: str | None = None,
    memory_limit_mb: float | None = None,
) -> dict:
    config = _apply_generator_override(load_config(config_path), generator_override)
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
    evaluation_indices = codec.support_indices(
        observed_pool,
        min(config.dfl.final_validation_size, len(observed_pool.scenarios)),
    )
    evaluation_pool = ScenarioPool(observed_pool.subset(evaluation_indices.tolist()))
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

    oracle = StoragePlanningOracle(
        feeder, config.planning, config.costs, config.data, config.data_center
    )
    planning = oracle.solve(generated, weights=weights)
    if not planning.feasible:
        raise RuntimeError(
            f"Evaluation planning failed with status {planning.status!r} after "
            f"{planning.solve_time_seconds:.1f} seconds."
        )
    validation = oracle.solve(
        evaluation_pool.scenarios,
        fixed_design=planning.design,
        allow_carbon_slack=True,
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
    reference = StoragePlanningOracle(
        feeder,
        replace(config.planning, max_storage_sites=0),
        config.costs,
        config.data,
        config.data_center,
    ).solve(evaluation_pool.scenarios, allow_carbon_slack=True)
    payload = {
        "method": checkpoint.get("method", "reinforce"),
        "generator": checkpoint_generator,
        "device": str(device),
        "test_split": config.data.test_split,
        "test_scenarios_evaluated": len(evaluation_pool.scenarios),
        "no_storage_reference": reference.to_dict(),
        "storage_value": (
            float(reference.objective) - float(validation.objective)
            if reference.feasible
            else None
        ),
        "generated_scenarios": [
            {"name": scenario.name, "weight": weight, "context": scenario.context.tolist()}
            for scenario, weight in zip(generated, weights)
        ],
        "scenario_summary": {
            "generated": _scenario_set_summary(generated, weights),
            "observed_test": _scenario_set_summary(evaluation_pool.scenarios),
        },
        "planning": planning.to_dict(),
        "out_of_sample_validation": validation.to_dict(),
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
