from __future__ import annotations

import json
import csv
from dataclasses import asdict, dataclass, replace
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
from storage_dfl.models import ConditionalVAE, train_cvae
from storage_dfl.network import Feeder, ieee13_unbalanced_microgrid
from storage_dfl.planning import StoragePlanningOracle


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def cvae_checkpoint(self) -> Path:
        return self.root / "cvae.pt"

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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _summary_writer(path: Path, enabled: bool, config: ExperimentConfig):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard is not installed. Run `python -m pip install tensorboard`."
        ) from exc
    writer = SummaryWriter(log_dir=str(path))
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


def _load_cvae(paths: ArtifactPaths, device: torch.device) -> ConditionalVAE:
    checkpoint = _load_torch(paths.cvae_checkpoint, device)
    model = ConditionalVAE(
        trajectory_dim=int(checkpoint["trajectory_dim"]),
        context_dim=int(checkpoint["context_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def train_cvae_stage(
    config_path: str | Path,
    *,
    tensorboard: bool = True,
) -> dict:
    config = load_config(config_path)
    paths = ArtifactPaths(config.output_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    feeder, observed_pool = _experiment_data(config, config.data.train_split)
    codec = ScenarioCodec.fit(observed_pool, feeder)
    trajectories, contexts = codec.encode_pool(observed_pool)
    device = resolve_device(config.dfl.device)
    model = ConditionalVAE(
        trajectory_dim=codec.trajectory_dim,
        context_dim=codec.context_dim,
        latent_dim=config.cvae.latent_dim,
        hidden_dim=config.cvae.hidden_dim,
    )
    writer = _summary_writer(paths.tensorboard / "cvae", tensorboard, config)
    try:
        history = train_cvae(
            model,
            trajectories,
            contexts,
            config.cvae,
            horizon=config.data.horizon,
            device=device,
            seed=config.seed,
            trajectory_mean=codec.trajectory_mean,
            trajectory_std=codec.trajectory_std,
            field_masks=codec.field_masks(),
            writer=writer,
        )
    finally:
        if writer is not None:
            writer.close()

    torch.save(
        {
            "state_dict": model.state_dict(),
            "trajectory_dim": codec.trajectory_dim,
            "context_dim": codec.context_dim,
            "latent_dim": config.cvae.latent_dim,
            "hidden_dim": config.cvae.hidden_dim,
        },
        paths.cvae_checkpoint,
    )
    _write_json(paths.normalization, codec.normalization_dict())
    _write_json(paths.root / "cvae_history.json", [asdict(record) for record in history])
    payload = {
        "device": str(device),
        "dataset": str(config.data.dataset_path),
        "split": config.data.train_split,
        "scenarios": len(observed_pool.scenarios),
        "trajectory_dim": codec.trajectory_dim,
        "epochs": len(history),
        "initial_loss": history[0].loss,
        "final_loss": history[-1].loss,
        "final_metrics": asdict(history[-1]),
        "checkpoint": str(paths.cvae_checkpoint),
    }
    _write_json(paths.root / "cvae_result.json", payload)
    return payload


def train_dfl_stage(
    config_path: str | Path,
    *,
    tensorboard: bool = True,
    method_override: str | None = None,
) -> dict:
    config = load_config(config_path)
    if method_override is not None:
        if method_override not in {"reinforce", "scenario_bo"}:
            raise ValueError("method_override must be 'reinforce' or 'scenario_bo'.")
        config = replace(config, dfl=replace(config.dfl, method=method_override))
    paths = ArtifactPaths(config.output_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    feeder, observed_pool = _experiment_data(config, config.data.validation_split)
    codec = _load_codec(paths, feeder)
    device = resolve_device(config.dfl.device)
    cvae = _load_cvae(paths, device)
    oracle = StoragePlanningOracle(feeder, config.planning, config.costs, config.data)
    writer = _summary_writer(
        paths.tensorboard / f"dfl_{config.dfl.method}", tensorboard, config
    )
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
    dfl_checkpoint = paths.dfl_checkpoint_for(config.dfl.method)
    torch.save(checkpoint, dfl_checkpoint)
    _write_json(
        paths.dfl_json_for("dfl_history", config.dfl.method),
        result.history_as_dicts(),
    )
    payload = {
        "device": result.device,
        "split": config.data.validation_split,
        "validation_scenarios": len(observed_pool.scenarios),
        "method": config.dfl.method,
        "epochs": len(result.history),
        "scenario_weights": list(result.scenario_weights),
        "support_source_names": checkpoint["support_source_names"],
        "planning": result.planning_result.to_dict(),
        "training_validation": result.full_validation_result.to_dict(),
        "checkpoint": str(dfl_checkpoint),
    }
    if config.dfl.method == "scenario_bo":
        payload["finalist_evaluations"] = list(result.finalist_evaluations)
    _write_json(paths.dfl_json_for("dfl_result", config.dfl.method), payload)
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
) -> dict:
    config = load_config(config_path)
    if method_override is not None:
        if method_override not in {"reinforce", "scenario_bo"}:
            raise ValueError("method_override must be 'reinforce' or 'scenario_bo'.")
        config = replace(config, dfl=replace(config.dfl, method=method_override))
    paths = ArtifactPaths(config.output_dir)
    feeder, observed_pool = _experiment_data(config, config.data.test_split)
    codec = _load_codec(paths, feeder)
    evaluation_indices = codec.support_indices(
        observed_pool,
        min(config.dfl.final_validation_size, len(observed_pool.scenarios)),
    )
    evaluation_pool = ScenarioPool(observed_pool.subset(evaluation_indices.tolist()))
    device = resolve_device(config.dfl.device)
    cvae = _load_cvae(paths, device)
    checkpoint_path = paths.dfl_checkpoint_for(config.dfl.method)
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
    latent = checkpoint["support_latent"].to(device=device, dtype=torch.float32)
    conditions = checkpoint["support_conditions"].to(device=device, dtype=torch.float32)
    weights = tuple(float(value) for value in checkpoint["scenario_weights"].cpu())
    decoded = cvae.decode(latent, conditions).cpu().numpy()
    generated = codec.decode_batch(
        decoded,
        conditions.cpu().numpy(),
        name_prefix="dfl_evaluation",
    )

    oracle = StoragePlanningOracle(feeder, config.planning, config.costs, config.data)
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
    payload = {
        "method": checkpoint.get("method", "reinforce"),
        "device": str(device),
        "test_split": config.data.test_split,
        "test_scenarios_evaluated": len(evaluation_pool.scenarios),
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
    _write_json(paths.dfl_json_for("result", config.dfl.method), payload)
    _write_trajectories(
        paths.root / f"scenario_trajectories_{config.dfl.method}.csv",
        evaluation_pool,
        generated,
        weights,
        feeder.buses,
        feeder.phases,
    )
    return payload
