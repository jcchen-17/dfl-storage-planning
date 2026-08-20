# Storage DFL — single-PCC data-centre microgrid

The main experiment reduces the microgrid to one point of common coupling:

- a calibrated 10 MW-class synthetic data-center demand, a three-year bus-675
  solar profile rescaled to 5 MW, a 3.5 MW backup generator and one battery;
- 48-hour historical scenarios with importance-weighted 1--4 hour grid outages;
- 16-dimensional CVAE with deterministic workload and full-weekend price restoration;
- direct CVAE fine-tuning with exact fixed-design recourse feedback;
- paper-inspired IPL/OPL losses balancing feasibility and preservation of the
  perfect-information design for load, PV and carbon constraint errors;
- disjoint train/validation/test use, fixed-validation best-checkpoint restore,
  independent reconstruction/decision batches and logged gradient balancing;
- exact MILP decision regret as an evaluation-only metric;
- hourly data-centre carbon cap using exact source-resolved storage vintages;
- Gurobi planning and out-of-sample evaluation.

The runtime planning path is single-PCC only. IEEE-13 structures remain solely
for rebuilding and auditing the source dataset.

The single configuration is
`configs/dataset_v2_dfl_hourly_layered_t1.yaml`.

## Commands

Rebuild and audit the dataset only when the processed input changes:

```powershell
python scripts/build_dfl_dataset_v2.py --force
python scripts/audit_dfl_dataset_v2.py
```

Optionally pretrain and validate a standalone CVAE:

```powershell
python scripts/train_generator.py
python scripts/verify_generator.py configs/dataset_v2_dfl_hourly_layered_t1.yaml
```

The active configuration uses joint training from random weights, so its normal
workflow needs only the DFL command; it fits `normalization.json` automatically
on the training split when absent. Then evaluate its storage design:

```powershell
python scripts/train_dfl.py
python scripts/evaluate.py
```

Artifacts are grouped by experiment config. The four hourly layered
configurations write to `outputs/hourly_layered/t1/` through `t4/`. Every DFL
launch creates one self-contained `runs/YYYYMMDD-HHMMSS/` directory containing
`config.json`, `checkpoint.pt`, `history.json`, `result.json`, and its own
`swanlab/` logs. Training metrics are synchronized to the
`carbon-aware-storage-planning` SwanLab project by default. Evaluation files are written beside that checkpoint,
while `latest.json` points to the most recently completed run.

Log in once before the first online run, or set `SWANLAB_MODE=offline` when a
machine has no network access:

```powershell
swanlab login
$env:SWANLAB_MODE = "offline"  # optional
```

Compare the learned supports against scenarios selected directly from the
observed training library:

```powershell
# One seed, K from the config, and the complete 182-scenario test split.
python scripts/run_scenario_selection_baselines.py

# A multi-seed Random/K-means/Aggregate comparison, with Farthest run once.
python scripts/run_scenario_selection_baselines.py --seeds 0 1 2 3 4 --parallel-runs 2
```

The runner computes one shared no-storage reference, plans with the same K for
`random`, `kmeans`, `farthest`, and `aggregate`, fixes each resulting storage
design, and solves exact operational recourse on the identical held-out set.
Each suite writes `summary.json`, `comparison.csv`, per-run JSON, and console
logs below `outputs/baselines/scenario_selection/`. A joint perfect-information
test MILP is deliberately opt-in with `--perfect-information`, since it can be
very large when all 182 test scenarios are used.

Evaluate a learned checkpoint on that same complete test split without starting
the very large joint perfect-information MILP:

```powershell
python scripts/evaluate.py --checkpoint PATH_TO_CHECKPOINT --scenarios 182 --skip-perfect-information
```

Planning JSON now includes an optional `carbon_ledger` with annualized PCC
imports, diesel/PV supply, storage charge/discharge, source emissions and carbon
delivered to the data centre. This makes the source, storage-vintage and
consumption layers directly auditable.

## Retained Python package

`src/storage_dfl/` contains the runtime library used by the commands above:
scenario PCC reduction and normalization, the single-PCC planning model, CVAE
components, the recourse-aware trainer, solver backends, and stage wiring.
Legacy REINFORCE, scenario-BO, GAN, diffusion and IEEE-13 planning experiment
files have been removed. The exact feasibility loss
and gradient paths are documented in `docs/recourse_feasibility_dfl.md`.
`tests/test_single_pcc.py` and `tests/test_recourse_dfl.py` cover the PCC model,
fixed-design recourse, directional losses, gradient flow and true-MILP final
evaluation.
