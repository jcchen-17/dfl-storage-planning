# Three-phase decision-focused storage planning

This repository trains directly on the processed IEEE13 × Smart-DS historical
dataset.  The production training path no longer calls the ten-scenario toy
generator.

```text
Smart-DS 2016 train windows -> phase-resolved CVAE
Smart-DS 2017 validation    -> black-box DFL support learning
Smart-DS 2018 test          -> fixed-design out-of-sample evaluation
```

## Dataset contract

The default configuration reads:

```text
data/processed/ieee13_smartds_dfl/dfl_training_windows_48h_price_regimes.npz
```

Each scenario contains 48 hourly intervals (two complete days) and the physical tensors:

- active load `[48, 13, 3]` in MW;
- reactive load `[48, 13, 3]` in Mvar;
- available PV `[48, 13, 3]` in MW;
- workload, PUE, grid price, grid carbon and grid availability `[48]`;
- calendar/weather/tariff-regime context `[5]`.

`scripts/build_dfl_windows.py` first cuts proper daily-stride 48-hour windows
from the continuous hourly Parquet table and recomputes each window's calendar
and temperature context. `scripts/augment_price_regimes.py` then creates the
price-regime file without overwriting the original fused dataset. Each physical window is paired with mean-preserving
price-spread factors `1.0, 1.25, 1.5, 1.75, 2.0`; the factor is included in the
CVAE context so tariff uncertainty is represented explicitly instead of being
averaged away.

The CVAE packs P, Q and PV for every bus phase plus four global generated
variables, giving `48 × (3 × 13 × 3 + 4) = 5,808` trajectory features.
Missing IEEE13 phases are masked to zero during both loading and decoding.

## Three-phase network

`ieee13_unbalanced_microgrid()` preserves the ABC, BC, AC and single-phase
branches of the official feeder.  The planning oracle creates P/Q flow and
voltage variables for every available line phase.  Industrial load, PV, the
three-phase data center, storage and generator are balanced at node-phase level.

The network is a decoupled multi-phase LinDistFlow approximation.  It represents
unbalanced phase injections, missing phases, phase limits and phase voltages,
but does not yet include the full 3×3 mutual-impedance matrices of OpenDSS.

`configs/demo.yaml` is the default 48-hour experiment. Repeated DFL solves retain
a distinct carbon vintage for every charging interval through the layered
`mccormick` formulation. The two-day horizon preserves cross-day storage and
carbon-inventory chronology while reducing the quadratic vintage-layer scale.
The `exact` option replaces the McCormick envelopes with nonconvex bilinear
equalities and is intended for final small-batch verification.
`configs/system_average.yaml` is the cheaper
system-average carbon baseline, and `configs/smoke.yaml` is the 12-hour wiring
check.

The CVAE loss is field-balanced: P, Q and PV are averaged over physically present
node-phase channels before being combined with workload, PUE, price and carbon.
It additionally preserves aggregate net load, net-load peaks and price spreads.
KL weight is warmed up during training. The default DFL stage freezes the CVAE
and uses a score-function/REINFORCE policy to learn one representative 48-hour
latent scenario. Its planning weight is fixed at one, while every resulting
design is ranked on a fixed multi-scenario validation set. The frozen CVAE
decoder and mixed-integer SCIP planner are treated as a black box; normalized
validation-cost advantages update the latent policy without differentiating the
solver. The Gaussian-process selector remains available with
`dfl.method: scenario_bo` for ablation experiments. You can also switch methods
from the command line without editing the configuration:

```powershell
python scripts/train_dfl.py --config configs/demo.yaml --method reinforce
python scripts/train_dfl.py --config configs/demo.yaml --method scenario_bo
python scripts/evaluate.py --config configs/demo.yaml --method reinforce
python scripts/evaluate.py --config configs/demo.yaml --method scenario_bo
```

Both methods share the trained CVAE, while their DFL checkpoints, histories,
results, TensorBoard logs, and evaluation trajectories use method-specific file
names so one run does not overwrite the other.

For numerical robustness, the proposed and aggregate models enforce line
apparent-power ratings with a conservative inner-octagon approximation.  The
smoke model keeps the original quadratic circle, and the pure-flow diagnostic
verifies that the generated scenarios remain well inside both limits.

## Interchangeable generators

The decision-focused stage treats the generator as a frozen black box and needs
only three things from it: a `latent_dim`, `decode(latent, context)` and
`encode(trajectory, context)`.  `ConditionalGenerator` in
`src/storage_dfl/models/base.py` states that contract, and three models
implement it:

| kind | objective | `encode` |
| --- | --- | --- |
| `cvae` | field-balanced ELBO with shape-preserving terms | amortised posterior mean |
| `gan` | conditional WGAN-GP plus batch moment matching | post-hoc inversion network fitted against the frozen generator |
| `diffusion` | field-weighted epsilon matching plus a Huber shape term | deterministic DDIM inversion |

All three read `cvae.latent_dim`, `cvae.hidden_dim`, `cvae.epochs`,
`cvae.batch_size` and the field weights from the same configuration section, so
they differ only in training objective and are compared at equal capacity.  The
`generator` section holds the model-specific settings.

A diffusion model has no learned latent: its free input is the initial noise
`x_T`, which has the full 5,808-dimensional trajectory size.  `DirectSupportPolicy`
perturbs the latent isotropically and updates it from a few REINFORCE samples per
epoch, so that dimension is unusable.  `latent_mode: projected` therefore fixes a
matrix `P` with orthonormal columns and sets `x_T = sqrt(D/d) P z` with
`z` in the CVAE's latent dimension; the scale keeps `E||x_T||^2 = D`, so the
sampler is unchanged.  `latent_mode: full` keeps the unrestricted noise for
generative-quality measurement only — `train_dfl_stage` rejects it.

Artifacts are scoped by generator, so one output directory holds all three
comparisons.  The CVAE keeps its historical file names (`cvae.pt`,
`dfl_support_reinforce.pt`) so existing runs stay readable; other generators get
a prefix (`gan.pt`, `dfl_support_gan_reinforce.pt`).

## Choosing a generator

The comparison runs in two stages against `configs/generator_compare.yaml`, a
copy of `configs/demo.yaml` whose only difference is
`output_dir: outputs/generator_comparison_48h`.  Use it rather than `demo.yaml`:
retraining a generator rewrites `cvae.pt` and `normalization.json` in the output
directory, which invalidates every DFL checkpoint already sitting there.

### Stage one: generative quality, no SCIP solves

```powershell
conda activate storage-dfl

# Train all three generators and score them (minutes on a GPU).
python scripts/compare_generators.py --config configs/generator_compare.yaml

# Re-score existing checkpoints without retraining, e.g. after a metric change.
python scripts/compare_generators.py --config configs/generator_compare.yaml --skip-training

# One generator only.
python scripts/compare_generators.py --config configs/generator_compare.yaml --generators gan
```

This writes `generator_comparison.json` and `generator_comparison.csv` to the
output directory: reconstruction error per field, 1-Wasserstein distance on the
statistics the planner actually prices (peak net load, price spread, carbon and
workload extremes), energy distance over whole trajectories, net-load
autocorrelation error, improved precision and recall, and the mass the codec had
to clip away.

Read `recall` first.  A collapsed GAN scores high precision with near-zero
recall — plausible samples with no diversity.  Then read the peak-net-load and
price-spread Wasserstein columns, because those two map directly onto the demand
charge and the arbitrage value.

Two metric caveats are baked into the implementation and worth knowing:

* Precision and recall are computed on the standardized decision statistics and
  on a ten-component PCA of the trajectories, never on the raw 5,808-dimensional
  vectors.  In the raw space every k-nearest-neighbour radius is far smaller
  than every cross-set distance, so both scores collapse to zero for every model
  and the metric carries no information at all.
* `reconstruction_all_fields` is not a fair head-to-head.  The GAN's inversion
  encoder is fitted for `encoder_epochs` against a frozen generator with no KL
  penalty, while the CVAE's reconstruction is held back by its KL term.  The
  distributional columns are fair, because they only use prior samples and never
  touch an encoder.

To separate a diffusion result from the projected-latent constraint imposed on
it, `configs/generator_compare_diffusion_full.yaml` re-runs it with
`latent_mode: full`, and `configs/generator_compare_diffusion_long.yaml` re-runs
it with six times the epoch budget.  Both write to their own output directories.

### Stage two: downstream decision quality

Only the survivors need to pay for solver time.  `--generator` and `--method`
are independent, so the two axes multiply: each combination gets its own
artifacts and none of them overwrite each other.  Omitting `--method` uses
`dfl.method` from the configuration, which is `reinforce`.

```powershell
python scripts/train_dfl.py --config configs/generator_compare.yaml --generator cvae --method reinforce
python scripts/evaluate.py  --config configs/generator_compare.yaml --generator cvae --method reinforce

python scripts/train_dfl.py --config configs/generator_compare.yaml --generator gan  --method reinforce
python scripts/evaluate.py  --config configs/generator_compare.yaml --generator gan  --method reinforce
```

Swap in `--method scenario_bo` for the Gaussian-process selector.  The artifact
tag is the method name for the CVAE and `<generator>_<method>` for every other
generator, so the CVAE keeps the file names it had before generators became
pluggable:

| generator | method | DFL checkpoint | evaluation result |
| --- | --- | --- | --- |
| `cvae` | `reinforce` | `dfl_support_reinforce.pt` | `result_reinforce.json` |
| `cvae` | `scenario_bo` | `dfl_support_scenario_bo.pt` | `result_scenario_bo.json` |
| `gan` | `reinforce` | `dfl_support_gan_reinforce.pt` | `result_gan_reinforce.json` |
| `gan` | `scenario_bo` | `dfl_support_gan_scenario_bo.pt` | `result_gan_scenario_bo.json` |

The number to compare is `out_of_sample_validation.objective` in each
`result_*.json`.  That is the verdict: generative fidelity and decision quality
do not have to agree, and a generator that wins stage one but loses stage two is
itself evidence for the decision-focused formulation, not a failed experiment.

Comparing generators and comparing DFL methods are separate questions, so vary
one axis at a time.  Pick the generator with `--method reinforce` held fixed,
then run `scenario_bo` on the winner as the method ablation.  Filling the whole
2x2 costs four full solver runs and answers neither question more sharply.

Any single run is one seed.  The REINFORCE variance is large enough to reorder
the two generators, so repeat with several `seed` values before reporting.

## Train

Activate the project environment and run from the repository root:

```powershell
conda activate storage-dfl
python scripts/train_generator.py --config configs/demo.yaml
python scripts/train_dfl.py --config configs/demo.yaml
python scripts/evaluate.py --config configs/demo.yaml
```

Any stage takes `--generator`:

```powershell
python scripts/train_generator.py --config configs/demo.yaml --generator gan
python scripts/train_dfl.py --config configs/demo.yaml --generator gan
python scripts/evaluate.py --config configs/demo.yaml --generator gan
```

`configs/demo.yaml` is now the historical three-phase experiment despite its
legacy filename.  `configs/smoke.yaml` performs a two-epoch CVAE and one-epoch
DFL wiring check; its numerical design is not an experimental result.

The exact batch-resolved carbon planning problem is a nonconvex MINLP.
The oracle first solves the no-storage operating case and injects that full
solution as a feasible SCIP warm start.  That bootstrap depends only on the
scenario set, the weights and the carbon-slack switch — never on the design
being evaluated — so it is cached and reused.  Because DFL validates every
candidate design against one fixed scenario set, the whole run pays for a single
validation bootstrap instead of one per solve.  `warm_start_time_limit_seconds`
sets its budget; `0` selects `max(60 s, 30% of solver_time_limit_seconds)`.
Lower it first when bootstraps rather than the planning solves dominate
wall-clock time.  On Windows, planning solves run in an isolated worker process
so conda NumPy and pip PyTorch do not load conflicting Intel OpenMP runtimes;
the cached bootstrap is passed into that worker and any newly computed one is
handed back.  The code deliberately does not use `KMP_DUPLICATE_LIB_OK`.

The demand charge is deliberately **not** weight-averaged.  `peak_grid` is one
variable bounded below by the grid draw of every interval of every scenario, so
`demand_dollars_per_mw_year` prices the worst-case peak across the scenario set
while the energy terms price a weighted expected annual cost.  With
`num_support_scenarios: 1` the planner therefore sees the peak of a single
generated scenario, whereas validation charges the maximum across the whole
validation set.  This is the conservative reading of a demand tariff, but it is
an asymmetry to state explicitly when reporting results, because at
`demand_dollars_per_mw_year: 150000` the term is a large share of total cost.

## Outputs

The default strong-storage run writes to
`outputs/ieee13_smartds_layered_storage_strong_48h/`. It uses a data-center
demand-charge case designed to make peak-shaving value material; retain the
earlier cost assumptions as an economic baseline when reporting sensitivity:

- `cvae.pt`, `normalization.json`, `cvae_history.json`;
- `dfl_support_reinforce.pt`, `dfl_history_reinforce.json`;
- `dfl_support_scenario_bo.pt`, `dfl_history_scenario_bo.json`;
- `result.json` and phase-resolved `scenario_trajectories.csv`;
- TensorBoard logs under `tensorboard/`.

DFL validation uses a fixed, decision-relevant subset because placing hundreds
of layered scenarios in one SCIP model is not computationally
practical. `validation_batch_size` controls the per-epoch decision loss for both
methods and `final_validation_size` controls finalist and test evaluation size.
Keeping the two equal lets the final comparison reuse cached validation solves
from training, which removes them from the run's cost entirely.

REINFORCE turns the per-epoch sample costs into advantages by rank, not by
standardized cost.  An infeasible sample carries a large sentinel loss, and
standardizing against it used to compress every feasible sample's advantage to
near zero and waste the epoch.

## Important modeling boundary

Normal historical windows have `grid_available = 1`.  The model is ready to
consume outages, but it cannot learn outage generation until extreme scenarios
with zeros are added.  Smart-DS, Azure Functions, EIA-930 DUK and the OPT-V
tariff are planning-compatible fused sources, not synchronized measurements of
one physical site.
