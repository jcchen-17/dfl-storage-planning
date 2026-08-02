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
data/processed/ieee13_smartds_dfl/dfl_training_windows_120h.npz
```

Each scenario contains 120 hourly intervals and the physical tensors:

- active load `[120, 13, 3]` in MW;
- reactive load `[120, 13, 3]` in Mvar;
- available PV `[120, 13, 3]` in MW;
- workload, PUE, grid price, grid carbon and grid availability `[120]`;
- calendar/weather context `[4]`.

The CVAE packs P, Q and PV for every bus phase plus four global generated
variables, giving `120 × (3 × 13 × 3 + 4) = 14,520` trajectory features.
Missing IEEE13 phases are masked to zero during both loading and decoding.

## Three-phase network

`ieee13_unbalanced_microgrid()` preserves the ABC, BC, AC and single-phase
branches of the official feeder.  The planning oracle creates P/Q flow and
voltage variables for every available line phase.  Industrial load, PV, the
three-phase data center, storage and generator are balanced at node-phase level.

The network is a decoupled multi-phase LinDistFlow approximation.  It represents
unbalanced phase injections, missing phases, phase limits and phase voltages,
but does not yet include the full 3×3 mutual-impedance matrices of OpenDSS.

`configs/demo.yaml` is the proposed 24-hour method: every charging interval
creates a distinct carbon vintage whose energy, carbon intensity, carbon mass,
and discharge are tracked explicitly.  Nodal proportional carbon flow is also
enforced exactly.  `configs/system_average.yaml` is a scalable 120-hour
aggregate-carbon baseline; it must not be reported as the proposed layered
method.  `configs/smoke.yaml` is the 12-hour wiring check.

For numerical robustness, the proposed and aggregate models enforce line
apparent-power ratings with a conservative inner-octagon approximation.  The
smoke model keeps the original quadratic circle, and the pure-flow diagnostic
verifies that the generated scenarios remain well inside both limits.

## Train

Activate the project environment and run from the repository root:

```powershell
conda activate storage-dfl
python scripts/train_cvae.py --config configs/demo.yaml
python scripts/train_dfl.py --config configs/demo.yaml
python scripts/evaluate.py --config configs/demo.yaml
```

`configs/demo.yaml` is now the historical three-phase experiment despite its
legacy filename.  `configs/smoke.yaml` performs a two-epoch CVAE and one-epoch
DFL wiring check; its numerical design is not an experimental result.

The 120-hour batch-resolved carbon planning problem is a large nonconvex MINLP.
The oracle first solves the no-storage operating case and injects that full
solution as a feasible SCIP warm start.  On Windows, planning solves run in an
isolated worker process so conda NumPy and pip PyTorch do not load conflicting
Intel OpenMP runtimes.  The code deliberately does not use
`KMP_DUPLICATE_LIB_OK`.

## Outputs

The default run writes to `outputs/ieee13_smartds_unbalanced/`:

- `cvae.pt`, `normalization.json`, `cvae_history.json`;
- `dfl_support.pt`, `dfl_history.json`;
- `result.json` and phase-resolved `scenario_trajectories.csv`;
- TensorBoard logs under `tensorboard/`.

DFL validation uses a configurable subset because placing hundreds of 120-hour
nonconvex scenarios in one SCIP model is not computationally practical.
`validation_batch_size` controls per-epoch feedback and
`final_validation_size` controls final validation/test size.

## Important modeling boundary

Normal historical windows have `grid_available = 1`.  The model is ready to
consume outages, but it cannot learn outage generation until extreme scenarios
with zeros are added.  Smart-DS, Azure Functions, EIA-930 DUK and the OPT-V
tariff are planning-compatible fused sources, not synchronized measurements of
one physical site.
