from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset_path: Path
    train_split: str
    validation_split: str
    test_split: str
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
    batch_size: int = 64
    kl_warmup_epochs: int = 40
    active_load_weight: float = 0.20
    reactive_load_weight: float = 0.10
    pv_weight: float = 0.20
    workload_weight: float = 0.10
    pue_weight: float = 0.05
    price_weight: float = 0.25
    carbon_weight: float = 0.10
    net_load_weight: float = 0.15
    net_peak_weight: float = 0.20
    price_spread_weight: float = 0.20

    def field_weights(self) -> tuple[float, ...]:
        """Per-field reconstruction weights: P, Q, PV, workload, PUE, price, carbon.

        Every generator reads them through this accessor so a GAN, a diffusion
        model and the CVAE are all held to the same field balance.
        """

        return (
            self.active_load_weight,
            self.reactive_load_weight,
            self.pv_weight,
            self.workload_weight,
            self.pue_weight,
            self.price_weight,
            self.carbon_weight,
        )


@dataclass(frozen=True)
class GANConfig:
    """Conditional WGAN-GP settings.

    ``latent_dim``, ``hidden_dim``, ``epochs``, ``batch_size`` and the field
    weights are shared with the CVAE section so the two models differ only in
    their training objective.
    """

    critic_steps: int = 5
    gradient_penalty: float = 10.0
    critic_hidden_dim: int = 512
    critic_learning_rate: float = 2.0e-4
    generator_learning_rate: float = 2.0e-4
    adam_beta1: float = 0.5
    adam_beta2: float = 0.9
    # A GAN sample has no matching observation, so the decision-relevant
    # summaries the CVAE preserves per sample are imposed here as batch moments.
    moment_weight: float = 10.0
    # Post-hoc inversion phase; the generator is frozen while this runs.
    encoder_epochs: int = 200
    encoder_learning_rate: float = 1.0e-3
    latent_prior_weight: float = 0.01


@dataclass(frozen=True)
class DiffusionConfig:
    """Conditional DDPM settings with a DDIM sampler.

    ``latent_mode: projected`` confines the initial noise to a subspace of the
    shared ``cvae.latent_dim`` so ``DirectSupportPolicy`` can search it.
    ``full`` keeps the unrestricted noise and is only usable for
    generative-quality measurement, never for the DFL stage.
    """

    timesteps: int = 400
    sampling_steps: int = 50
    beta_schedule: str = "cosine"
    latent_mode: str = "projected"
    blocks: int = 3
    # Bound on the recovered x0, in standardized units. It stabilises both the
    # sampler and the auxiliary shape term; the data itself never leaves +-5.
    x_zero_clamp: float = 5.0
    # The shape term is a Huber loss on scaled summaries, so this multiplies a
    # quantity of order one against the field-weighted noise objective.
    physics_weight: float = 0.1


@dataclass(frozen=True)
class GeneratorConfig:
    kind: str = "cvae"
    gan: GANConfig = field(default_factory=GANConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


@dataclass(frozen=True)
class DFLConfig:
    num_support_scenarios: int
    epochs: int
    validation_batch_size: int
    final_validation_size: int
    learning_rate: float
    baseline_momentum: float
    initial_exploration_std: float
    minimum_exploration_std: float
    exploration_decay: float
    diversity_margin: float
    diversity_weight: float
    weight_entropy_weight: float
    device: str
    policy_samples_per_epoch: int = 1
    latent_prior_weight: float = 0.001
    method: str = "reinforce"
    candidate_pool_size: int = 12
    bo_initial_evaluations: int = 6
    bo_iterations: int = 6
    bo_parameter_bound: float = 3.0
    bo_candidate_draws: int = 512
    bo_kernel_length_scale: float = 1.0
    bo_exploration: float = 1.5
    bo_weight_floor: float = 0.02
    bo_finalists: int = 2
    decision_deadband_relative: float = 0.005
    # Number of distinct REINFORCE designs retained for a common final
    # validation. Keeping more than the single best noisy minibatch incumbent
    # prevents a late lucky sample from replacing a consistently good design.
    reinforce_finalists: int = 4
    # Optional decision-scale early stopping. A material improvement must exceed
    # decision_deadband_relative times the best validation objective; smaller
    # movements are below the resolution at which candidates are ranked.
    early_stopping_patience: int = 0
    early_stopping_min_epochs: int = 20


@dataclass(frozen=True)
class DataCenterConfig:
    """Facility power model at the data-center bus, in MW.

        power(t) = non_it_mw + PUE(t) * (it_base_mw + it_workload_mw * processed(t))

    ``workload_arrival`` and ``pue`` are dimensionless series in the dataset, so
    these three coefficients alone set the facility's scale.  Changing them
    rescales the data center without touching the dataset or any trained
    generator; only the planning results depend on them.

    The defaults give a 0.24-0.48 MW facility -- an edge data center.  Raising
    them past roughly 0.96 MW peak requires uprating the 671-692-675 branch,
    which is what an interconnection study for a larger facility would conclude.
    """

    non_it_mw: float = 0.03
    it_base_mw: float = 0.12
    it_workload_mw: float = 0.30

    def power_mw(self, pue: float, processed: float) -> float:
        return self.non_it_mw + pue * (self.it_base_mw + self.it_workload_mw * processed)


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
    carbon_formulation: str
    flow_limit_formulation: str
    solver_time_limit_seconds: float
    solver_relative_gap: float
    solver_threads: int
    verbose_solver: bool
    # Budget for the no-storage bootstrap that seeds each solve. Zero selects
    # max(60 s, 30% of solver_time_limit_seconds). The bootstrap is cached per
    # scenario set, so this is paid once per distinct scenario set, not per solve.
    warm_start_time_limit_seconds: float = 0.0
    # Backup generator rating at the generator bus. It scales with the facility
    # it backs up, so it belongs in configuration rather than in the model body.
    backup_generator_mw: float = 1.0
    backup_generator_mvar: float = 0.8
    # SCIP search strategy. These change how the solver searches, never what it
    # is solving, so the formulation is untouched. Decision-focused training
    # ranks samples by their incumbent objective, so reaching a good primal
    # solution quickly matters more than proving optimality; 'feasibility' plus
    # aggressive heuristics targets exactly that. Leave at the defaults when
    # producing final numbers that are reported as optima.
    solver_emphasis: str = "default"
    solver_aggressive_heuristics: bool = False
    # Planning solves that may run concurrently in isolated workers. Samples
    # within one DFL epoch are independent, so this is the main lever on
    # wall-clock time. Zero selects one worker, i.e. the original serial path.
    # Each worker holds a full copy of the model, so memory scales with this just
    # as much as speed does; a layered-carbon model at K=3 is large enough that
    # four concurrent workers can exhaust a 16 GB machine.
    solver_max_parallel_workers: int = 1
    # Which solver builds and solves the planning problem. Under the production
    # formulation the model is a pure MILP, which is where the gap between
    # solvers is widest, so this is worth having as a switch. 'gurobi' needs
    # gurobipy plus a license large enough for the model; the bundled restricted
    # license stops at 2000 variables and this model has tens of thousands.
    solver_backend: str = "scip"
    # Per-solve memory budget in MB; zero leaves SCIP unbounded. Without a budget
    # SCIP aborts the entire process on allocation failure and the run dies. With
    # one it stops at 'memorylimit' and returns its incumbent, which the pipeline
    # can still use. Budget roughly (usable RAM) / solver_max_parallel_workers.
    solver_memory_limit_mb: float = 0.0


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
    generator_dollars_per_mwh: float = 125.0


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data: DataConfig
    cvae: CVAEConfig
    dfl: DFLConfig
    planning: PlanningConfig
    costs: CostConfig
    output_dir: Path
    # Optional so configurations written before generators became pluggable keep
    # loading and keep selecting the CVAE.
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    # Optional so configurations written before the data-center scale became
    # tunable keep loading and keep the original edge-scale facility.
    data_center: DataCenterConfig = field(default_factory=DataCenterConfig)


def _construct(section_type: type, raw: dict[str, Any]) -> Any:
    return section_type(**raw)


def _generator_config(raw: dict[str, Any] | None) -> GeneratorConfig:
    payload = dict(raw or {})
    return GeneratorConfig(
        kind=str(payload.get("kind", "cvae")),
        gan=_construct(GANConfig, dict(payload.get("gan", {}))),
        diffusion=_construct(DiffusionConfig, dict(payload.get("diffusion", {}))),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    data_raw = dict(raw["data"])
    data_raw["dataset_path"] = (
        config_path.parent.parent / data_raw["dataset_path"]
    ).resolve()
    return ExperimentConfig(
        seed=int(raw["seed"]),
        data=_construct(DataConfig, data_raw),
        cvae=_construct(CVAEConfig, raw["cvae"]),
        dfl=_construct(DFLConfig, raw["dfl"]),
        planning=_construct(PlanningConfig, raw["planning"]),
        costs=_construct(CostConfig, raw["costs"]),
        output_dir=(config_path.parent.parent / raw["output_dir"]).resolve(),
        generator=_generator_config(raw.get("generator")),
        data_center=_construct(DataCenterConfig, dict(raw.get("data_center", {}))),
    )
