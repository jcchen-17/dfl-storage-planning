# Direct Decision-Focused Scenario Generation for Storage Planning

This repository is a small end-to-end prototype of the proposed paper workflow:

```text
10 toy days -> temporal-spatial CVAE -> DFL latent support scenarios + weights
            -> nonconvex shared-storage planning -> evaluation on the 10 toy days
```

It is deliberately a **pipeline smoke test**, not a claim that ten days are
enough to train a publishable scenario model.

## What changed from scenario selection

The implementation does not score or select days from a historical candidate
pool. The CVAE first learns a low-dimensional manifold of complete correlated
trajectories. A policy then learns latent support points and their probability
weights. Passing those points through the frozen CVAE decoder directly creates
the scenarios used by the planner.

The downstream problem is a nonconvex MINLP. DFL therefore uses a score-function
gradient estimator and treats SCIP as a black box; it does not use KKT-based
implicit differentiation. The feedback signal is the cost of the temporary
storage design when it is re-evaluated on observed toy days.

## Physical planning model

- balanced single-phase equivalent of the IEEE 13-node feeder;
- PV at buses 646, 680, and 675 and a flexible data center at bus 675;
- binary storage siting and continuous power/energy sizing;
- LinDistFlow voltage and line limits;
- workload arrival, processing, backlog, deadline, PUE, and shedding;
- one storage carbon vintage for every charging interval;
- vintage-specific energy, carbon intensity, and discharge;
- proportional-sharing nodal carbon constraints.

The original IEEE 13-node feeder is unbalanced. The demo retains the bus names
and radial topology but uses a balanced equivalent to match the paper's
LinDistFlow formulation.

## Run on `wsl-desktop`

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate copf
cd /home/jcchen17/code/storage_dfl_planning/storage_dfl_demo
python -m pip install -e .
```

Run the three stages independently:

```bash
python scripts/train_cvae.py --config configs/demo.yaml
python scripts/train_dfl.py --config configs/demo.yaml
python scripts/evaluate.py --config configs/demo.yaml
```

`train_dfl.py` loads the saved CVAE and normalization parameters; it does not
retrain the CVAE. `evaluate.py` loads both checkpoints and performs no learning.
The original one-command workflow remains available:

```bash
python scripts/run_demo.py --config configs/demo.yaml
```

The CVAE/policy use CUDA automatically when available; SCIP runs on the CPU.
The default six DFL iterations are intentionally small because every iteration
calls a nonconvex planner and a validation model.

## TensorBoard

Both training commands write logs by default:

```text
outputs/direct_demo/tensorboard/cvae/
outputs/direct_demo/tensorboard/dfl/
```

Start TensorBoard on the server:

```bash
tensorboard --logdir outputs/direct_demo/tensorboard --host 127.0.0.1 --port 6006
```

Create an SSH tunnel from the Mac in a second terminal:

```bash
ssh -N -L 6006:127.0.0.1:6006 wsl-desktop
```

Then open `http://localhost:6006`. Add `--no-tensorboard` to either training
command when event logging is not needed.

## Outputs

Files are written to `outputs/direct_demo/`:

- `result.json`: generated scenario weights, storage design, and validation;
- `scenario_trajectories.csv`: observed and generated nodal trajectories;
- `cvae_history.json`: CVAE pretraining losses;
- `dfl_history.json`: black-box decision loss and design at each iteration;
- `normalization.json`: reproducible trajectory/context scaling;
- `cvae.pt`: CVAE state and architecture metadata;
- `dfl_support.pt`: DFL policy, chosen latent supports, contexts, and weights;
- `tensorboard/`: CVAE and DFL event files.

## Package layout

```text
src/storage_dfl/
├── config.py
├── data/
│   ├── schema.py             physical scenario contract
│   ├── synthetic.py          ten correlated toy days
│   └── codec.py              physical <-> CVAE representation
├── models/
│   └── cvae.py               temporal-spatial conditional VAE
├── dfl/
│   ├── support.py            learnable latent points and weights
│   └── trainer.py            solver-agnostic score-function DFL
├── network/
│   └── ieee13.py             balanced IEEE-13 microgrid
├── planning/
│   ├── model.py              PySCIPOpt planning oracle
│   └── results.py
├── stages.py                 independent train/evaluate stages
└── pipeline.py               optional all-in-one composition
```

## Scale-up for the paper

Replace the toy generator with a Smart-DS loader, use separate train/validation/
test periods, increase the horizon to 24 hours, and compare against random
sampling, statistical scenario reduction, generation without DFL, and the full
historical set. Report distributional fidelity, planning cost/regret, carbon-cap
violations, siting stability, solver gaps, and ablations of batch-resolved carbon
accounting.
