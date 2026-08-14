from __future__ import annotations

import json
import csv
import hashlib
import os
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
    normalized_decision_regret,
    resolve_device,
    select_scenarios,
    train_recourse_feasibility_cvae,
)
from storage_dfl.models import (
    GENERATOR_KINDS,
    ConditionalGenerator,
    build_generator,
    generator_from_checkpoint,
    save_generator,
    train_generator,
)
from storage_dfl.network import (
    Feeder,
    ieee13_unbalanced_microgrid,
    single_pcc_microgrid,
)
from storage_dfl.planning import (
    PlanningResult,
    StorageDesign,
    StoragePlanningOracle,
    evaluate_fixed_design_recourse,
    make_planning_oracle,
    weighted_carbon_ledger,
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

    def dfl_run_dir(self, run_id: str) -> Path:
        """One self-contained timestamped DFL run."""
        return self.root / "runs" / run_id

    @property
    def dfl_latest(self) -> Path:
        return self.root / "latest.json"

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

    if isinstance(value, Path):
        return str(value)
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


def _write_json_atomic(path: Path, payload: object) -> None:
    """Atomically replace a small JSON manifest shared by concurrent runs."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_json(temporary, payload)
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _summary_writer(
    path: Path,
    enabled: bool,
    config: ExperimentConfig,
    *,
    timestamped: bool = True,
):
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
    log_dir = path / time.strftime("%Y%m%d-%H%M%S") if timestamped else path
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("experiment/config", f"```json\n{json.dumps(asdict(config), default=str, indent=2)}\n```")
    return writer


def _compact_run_name(prefix: str, tag: str, max_length: int = 40) -> str:
    """Keep TensorBoard event paths below the Windows legacy path limit."""

    candidate = f"{prefix}_{tag}"
    if len(candidate) <= max_length:
        return candidate
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:8]
    readable_length = max_length - len(prefix) - len(digest) - 2
    return f"{prefix}_{tag[:readable_length]}_{digest}"


def _experiment_data(
    config: ExperimentConfig,
    split: str,
) -> tuple[Feeder, ScenarioPool]:
    pool = load_historical_scenarios(
        config.data.dataset_path,
        split=split,
        horizon=config.data.horizon,
    )
    if config.planning.topology != "single_pcc":
        raise ValueError("Only planning.topology='single_pcc' is retained.")
    source_feeder = ieee13_unbalanced_microgrid()
    source_bus = config.data.pcc_source_bus
    if source_bus not in source_feeder.bus_index:
        raise ValueError(f"Unknown PCC source bus {source_bus!r}.")
    source_index = source_feeder.bus_index[source_bus]
    source_pv_capacity = float(source_feeder.pv_capacity_mw[source_index].sum())
    target_pv_capacity = float(config.data.pcc_pv_capacity_mw)
    if source_pv_capacity <= 0.0:
        raise ValueError(f"PCC source bus {source_bus!r} has no PV capacity.")
    if target_pv_capacity <= 0.0:
        raise ValueError("data.pcc_pv_capacity_mw must be positive.")
    pv_scale = target_pv_capacity / source_pv_capacity
    feeder = single_pcc_microgrid(pv_capacity_mw=target_pv_capacity)
    aggregated = []
    for scenario in pool.scenarios:
        # The source data contain no metered data-center power. Build a
        # calibrated 10 MW-class facility trajectory from the Azure workload
        # proxy and temperature-derived PUE. Keep only bus 675's co-located PV,
        # rescaled from its original 0.65 MW rating to the configured 5 MW
        # installation; never sum feeder loads or PV into the PCC.
        dc_load = np.asarray(
            config.data_center.power_mw(
                scenario.pue, scenario.workload_arrival
            ),
            dtype=float,
        )
        pv = scenario.pv_available_mw[:, source_index, :].sum(axis=1) * pv_scale
        active = np.repeat(dc_load[:, None, None] / 3.0, 3, axis=2)
        pcc_pv = np.repeat(pv[:, None, None] / 3.0, 3, axis=2)
        aggregated.append(
            replace(
                scenario,
                active_load_mw=active,
                reactive_load_mvar=np.zeros_like(active),
                pv_available_mw=pcc_pv,
            )
        )
    pool = ScenarioPool(tuple(aggregated))
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
        raise ValueError("Scenario dimensions do not match the configured topology.")
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


def _fit_codec(config: ExperimentConfig, pool: ScenarioPool, feeder: Feeder) -> ScenarioCodec:
    """Fit the shared training-split normalization used by joint training."""

    return ScenarioCodec.fit(
        pool,
        feeder,
        deterministic_price=config.data.deterministic_price,
        tariff_spread_scale=config.data.tariff_spread_scale,
        tariff_reference_price_per_mwh=(
            config.data.tariff_reference_price_per_mwh
        ),
    )


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
    if generator_override not in {None, "cvae"}:
        raise ValueError("Only the CVAE generator is retained.")
    return config


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
    resume_checkpoint: str | Path | None = None,
    additional_epochs: int | None = None,
) -> dict:
    config = _apply_generator_override(load_config(config_path), generator_override)
    if additional_epochs is not None:
        if additional_epochs <= 0:
            raise ValueError("additional_epochs must be positive.")
        config = replace(
            config, cvae=replace(config.cvae, epochs=int(additional_epochs))
        )
    kind = config.generator.kind
    paths = ArtifactPaths(config.output_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    feeder, observed_pool = _experiment_data(config, config.data.train_split)
    codec = (
        _load_codec(paths, feeder)
        if resume_checkpoint is not None and paths.normalization.exists()
        else _fit_codec(config, observed_pool, feeder)
    )
    trajectories, contexts = codec.encode_pool(observed_pool)
    # Encoded with the codec fitted on the training split, never one refitted
    # here: the held-out curve has to describe the generator downstream stages
    # load, and they all share this one normalization.
    _, validation_pool = _experiment_data(config, config.data.validation_split)
    validation_trajectories, validation_contexts = codec.encode_pool(validation_pool)
    device = resolve_device(config.dfl.device)
    start_epoch = 0
    optimizer_state = None
    torch_rng_state = None
    cuda_rng_state_all = None
    if resume_checkpoint is None:
        model = build_generator(config, codec.trajectory_dim, codec.context_dim)
    else:
        checkpoint = _load_torch(Path(resume_checkpoint), device)
        model = generator_from_checkpoint(checkpoint, device)
        if model.kind != kind:
            raise RuntimeError(
                f"Resume checkpoint contains {model.kind!r}, expected {kind!r}."
            )
        start_epoch = int(checkpoint.get("completed_epochs", 0))
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is None:
            raise ValueError(
                "Resume checkpoint has no optimizer_state_dict; retrain the shared "
                "pretraining stage with the current code."
            )
        torch_rng_state = checkpoint.get("torch_rng_state")
        cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
        print(
            f"Resuming {kind} from epoch {start_epoch} for "
            f"{config.cvae.epochs} additional epochs.",
            flush=True,
        )
    writer = _summary_writer(paths.tensorboard / kind, tensorboard, config)
    training_state: dict = {}
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
            start_epoch=start_epoch,
            optimizer_state=optimizer_state,
            torch_rng_state=torch_rng_state,
            cuda_rng_state_all=cuda_rng_state_all,
            training_state_out=training_state,
        )
    finally:
        if writer is not None:
            writer.close()

    checkpoint_path = paths.generator_checkpoint_for(kind)
    save_generator(model, checkpoint_path, training_state=training_state)
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
        "start_epoch": start_epoch,
        "completed_epochs": int(training_state["completed_epochs"]),
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
    """Artifact suffix for the retained feasibility method and CAPEX case."""

    base = config.dfl.method
    if config.planning.carbon_formulation != "layered_pcc":
        base = f"{base}_{config.planning.carbon_formulation}"
    if config.planning.carbon_cap_scope == "horizon":
        base = f"{base}_horizon_cap"
    else:
        base = f"{base}_hourly_cap"
    if config.planning.outage_carbon_cap > 0.0:
        outage_tag = f"{100.0 * config.planning.outage_carbon_cap:g}".replace(".", "p")
        base = f"{base}_outage_cap{outage_tag}"
    if not config.dfl.initialize_from_pretrained:
        base = f"{base}_joint_scratch"
    if config.dfl.fixed_decision_anchors:
        base = f"{base}_aligned_supports"
    if config.dfl.checkpoint_selection == "carbon_first":
        base = f"{base}_carbon_first"
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
    carbon_formulation_override: str | None = None,
    carbon_cap_scope_override: str | None = None,
    run_id: str | None = None,
) -> dict:
    config = _apply_generator_override(load_config(config_path), generator_override)
    if carbon_formulation_override is not None:
        config = replace(
            config,
            planning=replace(
                config.planning, carbon_formulation=carbon_formulation_override
            ),
        )
    if carbon_cap_scope_override is not None:
        config = replace(
            config,
            planning=replace(
                config.planning, carbon_cap_scope=carbon_cap_scope_override
            ),
        )
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
        if method_override != "recourse_feasibility":
            raise ValueError("Only method_override='recourse_feasibility' is retained.")
        config = replace(config, dfl=replace(config.dfl, method=method_override))
    if config.dfl.method != "recourse_feasibility":
        raise ValueError("Only dfl.method='recourse_feasibility' is retained.")
    paths = ArtifactPaths(config.output_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = time.strftime("%Y%m%d-%H%M%S")
    allowed_run_id_characters = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
    )
    if not run_id or any(
        character not in allowed_run_id_characters for character in run_id
    ):
        raise ValueError("run_id may contain only letters, digits, '-' and '_'.")
    run_dir = paths.dfl_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "config.json", asdict(config))
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # DFL updates use only the training split.  The validation split is fixed
    # and read-only for checkpoint selection; evaluate_stage alone sees test.
    feeder, observed_pool = _experiment_data(config, config.data.train_split)
    validation_feeder, validation_pool = _experiment_data(
        config, config.data.validation_split
    )
    if validation_feeder.buses != feeder.buses:
        raise ValueError("Training and validation splits use different feeders.")
    if paths.normalization.exists():
        codec = _load_codec(paths, feeder)
    elif config.dfl.initialize_from_pretrained:
        raise FileNotFoundError(
            f"Normalization file not found: {paths.normalization}. Pretrained "
            "initialization requires train_generator.py to run first."
        )
    else:
        codec = _fit_codec(config, observed_pool, feeder)
        _write_json(paths.normalization, codec.normalization_dict())
        print(
            f"Fitted training-split normalization: {paths.normalization}",
            flush=True,
        )
    _require_current_codec(config, codec)
    device = resolve_device(config.dfl.device)
    if config.dfl.initialize_from_pretrained:
        cvae = load_generator(paths, config.generator.kind, device)
        print("DFL initialization: pretrained generator checkpoint", flush=True)
    else:
        cvae = build_generator(
            config, codec.trajectory_dim, codec.context_dim
        ).to(device)
        print(
            "DFL initialization: random generator weights; joint CVAE-DFL training",
            flush=True,
        )
    # The training loop may run at a looser tolerance than the reported numbers.
    # evaluate_stage builds its own oracle from config.planning, so whatever is
    # set here never reaches a reported objective.
    training_planning = config.planning
    if config.dfl.training_relative_gap > 0.0:
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
    oracle = make_planning_oracle(
        feeder, training_planning, config.costs, config.data, config.data_center
    )
    validation_oracle = make_planning_oracle(
        feeder, config.planning, config.costs, config.data, config.data_center
    )
    writer = _summary_writer(
        run_dir / "tensorboard", tensorboard, config, timestamped=False
    )
    try:
        result = train_recourse_feasibility_cvae(
            cvae,
            codec,
            observed_pool,
            oracle,
            config.cvae,
            config.dfl,
            config.seed,
            validation_pool=validation_pool,
            validation_oracle=validation_oracle,
            writer=writer,
        )
        support_source_names = ["direct_cvae_prior"] * len(
            result.generated_scenarios
        )
    finally:
        if writer is not None:
            writer.close()

    checkpoint = {
        "run_id": run_id,
        "method": config.dfl.method,
        "generator": config.generator.kind,
        "support_count": config.dfl.num_support_scenarios,
        "latent_dim": cvae.latent_dim,
        "support_latent": torch.as_tensor(result.support_latent),
        "support_conditions": torch.as_tensor(result.support_conditions),
        "scenario_weights": torch.as_tensor(result.scenario_weights),
        "support_grid_available": torch.as_tensor(
            np.stack(
                [scenario.grid_available for scenario in result.generated_scenarios]
            ),
            dtype=torch.float32,
        ),
        "support_source_names": support_source_names,
        # Evaluation decodes the supports and replans. Persist the design chosen
        # during training so a near-degenerate replan that flips bus or capacity
        # is visible instead of looking like a change learned by DFL.
        "selected_training_design": result.planning_result.to_dict()["design"],
        "best_epoch": int(result.best_epoch),
        "initialize_from_pretrained": config.dfl.initialize_from_pretrained,
        "carbon_formulation": config.planning.carbon_formulation,
        "fixed_decision_anchors": config.dfl.fixed_decision_anchors,
        "dfl_statistical_weight": config.dfl.dfl_statistical_weight,
        "gradient_balance_ratio": config.dfl.gradient_balance_ratio,
        "gradient_balance_max_scale": config.dfl.gradient_balance_max_scale,
        "checkpoint_carbon_tolerance": config.dfl.checkpoint_carbon_tolerance,
    }
    checkpoint["fine_tuned_generator_state_dict"] = cvae.state_dict()
    checkpoint["generator_checkpoint"] = {
        "kind": cvae.kind,
        "state_dict": cvae.state_dict(),
        **cvae.checkpoint_payload(),
    }
    checkpoint["decision_regret"] = float(result.decision_regret)
    dfl_checkpoint = run_dir / "checkpoint.pt"
    torch.save(checkpoint, dfl_checkpoint)
    _write_json(
        run_dir / "history.json",
        result.history_as_dicts(),
    )
    payload = {
        "device": result.device,
        "training_split": config.data.train_split,
        "validation_split": config.data.validation_split,
        "training_scenarios": len(observed_pool.scenarios),
        "validation_scenarios": len(validation_pool.scenarios),
        "method": config.dfl.method,
        "generator": config.generator.kind,
        "epochs": len(result.history),
        "seed": config.seed,
        # Which tolerance the loop actually ran at. Two runs at different values
        # searched different landscapes even with everything else identical.
        "training_relative_gap": float(training_planning.solver_relative_gap),
        "planning_oracle": "single_pcc",
        "battery_capex_scale": float(config.costs.battery_capex_scale),
        "scenario_weights": list(result.scenario_weights),
        "support_source_names": checkpoint["support_source_names"],
        "planning": result.planning_result.to_dict(),
        "training_validation": result.full_validation_result.to_dict(),
        "best_epoch": int(result.best_epoch),
        "checkpoint": str(dfl_checkpoint),
        "run_id": run_id,
        "fixed_decision_anchors": config.dfl.fixed_decision_anchors,
        "dfl_statistical_weight": config.dfl.dfl_statistical_weight,
        "gradient_balance_ratio": config.dfl.gradient_balance_ratio,
        "gradient_balance_max_scale": config.dfl.gradient_balance_max_scale,
        "checkpoint_carbon_tolerance": config.dfl.checkpoint_carbon_tolerance,
    }
    payload["perfect_information_reference"] = result.reference_result.to_dict()
    payload["decision_regret"] = float(result.decision_regret)
    result_path = run_dir / "result.json"
    _write_json(result_path, payload)
    _write_json_atomic(
        paths.dfl_latest,
        {
            "run_id": run_id,
            "checkpoint": str(dfl_checkpoint.resolve()),
            "history": str((run_dir / "history.json").resolve()),
            "result": str(result_path.resolve()),
        },
    )
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
        carbon_ledger=weighted_carbon_ledger(results, normalized),
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
    aggregate, results = evaluate_fixed_design_recourse(
        oracle,
        scenarios,
        weights,
        design,
        allow_carbon_slack=True,
        use_cache=False,
        accepted_relative_gap=accepted_relative_gap,
    )
    wall_seconds = time.perf_counter() - started
    return aggregate, results, wall_seconds


@torch.no_grad()
def evaluate_stage(
    config_path: str | Path,
    *,
    method_override: str | None = None,
    generator_override: str | None = None,
    memory_limit_mb: float | None = None,
    scenarios: int | None = None,
    carbon_formulation_override: str | None = None,
    carbon_cap_scope_override: str | None = None,
    evaluation_carbon_formulation_override: str | None = None,
    checkpoint_override: str | Path | None = None,
    perfect_information: bool = True,
    normalization_override: str | Path | None = None,
    output_override: str | Path | None = None,
    solver_workers: int | None = None,
    solver_threads: int | None = None,
) -> dict:
    config = _apply_generator_override(load_config(config_path), generator_override)
    if carbon_formulation_override is not None:
        config = replace(
            config,
            planning=replace(
                config.planning, carbon_formulation=carbon_formulation_override
            ),
        )
    if carbon_cap_scope_override is not None:
        config = replace(
            config,
            planning=replace(
                config.planning, carbon_cap_scope=carbon_cap_scope_override
            ),
        )
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
    planning_parallel_overrides = {}
    if solver_workers is not None:
        if solver_workers <= 0:
            raise ValueError("solver_workers must be positive.")
        planning_parallel_overrides["solver_max_parallel_workers"] = solver_workers
    if solver_threads is not None:
        if solver_threads <= 0:
            raise ValueError("solver_threads must be positive.")
        planning_parallel_overrides["solver_threads"] = solver_threads
    if planning_parallel_overrides:
        config = replace(
            config,
            planning=replace(config.planning, **planning_parallel_overrides),
        )
    if method_override is not None:
        if method_override != "recourse_feasibility":
            raise ValueError("Only method_override='recourse_feasibility' is retained.")
        config = replace(config, dfl=replace(config.dfl, method=method_override))
    paths = ArtifactPaths(config.output_dir)
    feeder, observed_pool = _experiment_data(config, config.data.test_split)
    if normalization_override is None:
        normalization_path = paths.normalization
        codec = _load_codec(paths, feeder)
    else:
        normalization_path = Path(normalization_override).expanduser().resolve()
        if not normalization_path.exists():
            raise FileNotFoundError(
                f"Normalization file not found: {normalization_path}"
            )
        codec = ScenarioCodec.from_normalization_dict(
            _read_json(normalization_path), feeder
        )
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
    if checkpoint_override is not None:
        checkpoint_path = Path(checkpoint_override)
    else:
        latest_manifest = paths.dfl_latest
        if latest_manifest.exists():
            checkpoint_path = Path(_read_json(latest_manifest)["checkpoint"])
        else:
            tag = _method_tag(config)
            checkpoint_path = paths.dfl_checkpoint_for(tag)
            # Read old runs when no method-scoped checkpoint has been created yet.
            if not checkpoint_path.exists() and paths.dfl_checkpoint.exists():
                checkpoint_path = paths.dfl_checkpoint
    checkpoint = _load_torch(checkpoint_path, device)
    checkpoint_method = checkpoint.get("method", "recourse_feasibility")
    if checkpoint_method != config.dfl.method:
        raise RuntimeError(
            f"Requested DFL method {config.dfl.method!r}, but checkpoint "
            f"{checkpoint_path} contains {checkpoint_method!r}. Train that method first."
        )
    checkpoint_carbon = checkpoint.get("carbon_formulation", "layered_pcc")
    if checkpoint_carbon != config.planning.carbon_formulation:
        raise RuntimeError(
            f"Requested carbon formulation {config.planning.carbon_formulation!r}, "
            f"but checkpoint contains {checkpoint_carbon!r}."
        )
    evaluation_planning = config.planning
    if evaluation_carbon_formulation_override is not None:
        evaluation_planning = replace(
            evaluation_planning,
            carbon_formulation=evaluation_carbon_formulation_override,
        )
    evaluation_suffix = ""
    if evaluation_planning.carbon_formulation != config.planning.carbon_formulation:
        evaluation_suffix = f"_{evaluation_planning.carbon_formulation}"
    checkpoint_generator = str(checkpoint.get("generator", "cvae"))
    if checkpoint_generator != config.generator.kind:
        raise RuntimeError(
            f"Requested generator {config.generator.kind!r}, but checkpoint "
            f"{checkpoint_path} was trained with {checkpoint_generator!r}."
        )
    if "generator_checkpoint" in checkpoint:
        cvae = generator_from_checkpoint(checkpoint["generator_checkpoint"], device)
    elif "fine_tuned_generator_state_dict" in checkpoint:
        # Backward-compatible path for scratch/joint checkpoints written before
        # constructor metadata was embedded. Rebuild from the effective config
        # and codec; do not require a separately pretrained cvae.pt.
        cvae = build_generator(
            config, codec.trajectory_dim, codec.context_dim
        ).to(device)
        cvae.load_state_dict(checkpoint["fine_tuned_generator_state_dict"])
        cvae.freeze()
    else:
        cvae = load_generator(paths, config.generator.kind, device)
    latent = checkpoint["support_latent"].to(device=device, dtype=torch.float32)
    conditions = checkpoint["support_conditions"].to(device=device, dtype=torch.float32)
    weights = tuple(float(value) for value in checkpoint["scenario_weights"].cpu())
    decoded = cvae.decode(latent, conditions).cpu().numpy()
    generated = codec.decode_batch(
        decoded,
        conditions.cpu().numpy(),
        name_prefix="dfl_evaluation",
    )
    if "support_grid_available" in checkpoint:
        availability = checkpoint["support_grid_available"].cpu().numpy()
        generated = tuple(
            replace(
                scenario,
                grid_available=np.asarray(availability[index], dtype=float),
                annual_occurrences=None,
            )
            for index, scenario in enumerate(generated)
        )

    oracle = make_planning_oracle(
        feeder, evaluation_planning, config.costs, config.data, config.data_center
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
            evaluation_planning.solver_relative_gap,
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
        replace(evaluation_planning, max_storage_sites=0),
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
            evaluation_planning.solver_relative_gap,
        )
    )
    perfect_information = None
    exact_decision_regret = None
    if perfect_information and checkpoint.get("method") == "recourse_feasibility":
        print("Perfect-information reference: solving the true scenario MILP...", flush=True)
        perfect_information = oracle.solve(
            evaluation_pool.scenarios,
            weights=evaluation_weights,
            allow_carbon_slack=True,
            use_cache=True,
        )
        exact_decision_regret = normalized_decision_regret(
            validation, perfect_information
        )
    payload = {
        "method": checkpoint.get("method", "recourse_feasibility"),
        "generator": checkpoint_generator,
        "evaluated_checkpoint": str(checkpoint_path.resolve()),
        "normalization": str(normalization_path.resolve()),
        "training_run_id": checkpoint.get("run_id"),
        "device": str(device),
        "test_split": config.data.test_split,
        "test_scenarios_evaluated": len(evaluation_pool.scenarios),
        "evaluation_relative_gap": evaluation_planning.solver_relative_gap,
        "trained_carbon_formulation": config.planning.carbon_formulation,
        "evaluation_carbon_formulation": evaluation_planning.carbon_formulation,
        "planning_oracle": "single_pcc",
        "evaluation_selection_rule": config.dfl.evaluation_selection_rule,
        "evaluation_scenario_names": list(evaluation_names),
        "evaluation_scenario_weights": list(evaluation_weights),
        "no_storage_reference": reference.to_dict(),
        "no_storage_scenario_results": [
            result.to_dict() for result in reference_scenario_results
        ],
        "no_storage_wall_seconds": reference_wall_seconds,
        "objectives_comparable": (
            _bounded_result(reference, evaluation_planning.solver_relative_gap)
            and _bounded_result(validation, evaluation_planning.solver_relative_gap)
        ),
        "storage_value": (
            float(reference.objective) - float(validation.objective)
            if _bounded_result(reference, evaluation_planning.solver_relative_gap)
            and _bounded_result(validation, evaluation_planning.solver_relative_gap)
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
        "perfect_information_reference": (
            perfect_information.to_dict() if perfect_information is not None else None
        ),
        "decision_regret": exact_decision_regret,
    }
    if output_override is None:
        evaluation_path = checkpoint_path.parent / f"evaluation{evaluation_suffix}.json"
        trajectory_path = checkpoint_path.parent / f"trajectories{evaluation_suffix}.csv"
    else:
        evaluation_path = Path(output_override).expanduser().resolve()
        trajectory_path = evaluation_path.with_name(
            f"{evaluation_path.stem}_trajectories.csv"
        )
    _write_json(evaluation_path, payload)
    _write_trajectories(
        trajectory_path,
        evaluation_pool,
        generated,
        weights,
        feeder.buses,
        feeder.phases,
    )
    return payload
