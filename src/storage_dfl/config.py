from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    num_scenarios: int
    horizon: int
    delta_t_hours: float


@dataclass(frozen=True)
class CVAEConfig:
    latent_dim: int
    hidden_dim: int
    epochs: int
    learning_rate: float
    beta: float
    ramp_weight: float


@dataclass(frozen=True)
class DFLConfig:
    num_support_scenarios: int
    epochs: int
    validation_batch_size: int
    learning_rate: float
    baseline_momentum: float
    initial_exploration_std: float
    minimum_exploration_std: float
    exploration_decay: float
    diversity_margin: float
    diversity_weight: float
    weight_entropy_weight: float
    device: str


@dataclass(frozen=True)
class PlanningConfig:
    max_storage_sites: int
    initial_soc: float
    initial_carbon_intensity: float
    min_soc: float
    max_soc: float
    charge_efficiency: float
    discharge_efficiency: float
    self_discharge: float
    min_power_mw: float
    max_power_mw: float
    min_energy_mwh: float
    max_energy_mwh: float
    min_duration_hours: float
    max_duration_hours: float
    grid_limit_mw: float
    carbon_intensity_max: float
    dc_carbon_cap: float
    other_bus_carbon_cap: float
    solver_time_limit_seconds: float
    solver_relative_gap: float
    solver_threads: int
    verbose_solver: bool


@dataclass(frozen=True)
class CostConfig:
    capital_recovery_factor: float
    site_dollars: float
    power_dollars_per_mw: float
    energy_dollars_per_mwh: float
    degradation_dollars_per_mwh: float
    demand_dollars_per_mw_year: float
    delay_dollars_per_task_hour: float
    curtailment_dollars_per_mwh: float
    shedding_dollars_per_mwh: float
    validation_carbon_slack_dollars: float


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data: DataConfig
    cvae: CVAEConfig
    dfl: DFLConfig
    planning: PlanningConfig
    costs: CostConfig
    output_dir: Path


def _construct(section_type: type, raw: dict[str, Any]) -> Any:
    return section_type(**raw)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return ExperimentConfig(
        seed=int(raw["seed"]),
        data=_construct(DataConfig, raw["data"]),
        cvae=_construct(CVAEConfig, raw["cvae"]),
        dfl=_construct(DFLConfig, raw["dfl"]),
        planning=_construct(PlanningConfig, raw["planning"]),
        costs=_construct(CostConfig, raw["costs"]),
        output_dir=(config_path.parent.parent / raw["output_dir"]).resolve(),
    )
