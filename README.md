# Storage DFL — current v2 workflow

This repository now keeps one supported experiment only:

- 48-hour non-overlapping v2 scenarios;
- 16-dimensional CVAE with deterministic workload and full-weekend price restoration;
- REINFORCE decision-focused training with one support scenario;
- hourly data-center carbon cap using feeder-wide carbon-attribute commodity
  flows and independently dispatchable storage carbon bins;
- Gurobi planning and out-of-sample evaluation.

The single configuration is
`configs/dataset_v2_dfl_hourly_layered.yaml`.

## Commands

Rebuild and audit the dataset only when the processed input changes:

```powershell
python scripts/build_dfl_dataset_v2.py --force
python scripts/audit_dfl_dataset_v2.py
```

Train and validate the CVAE:

```powershell
python scripts/train_generator.py
python scripts/verify_generator.py configs/dataset_v2_dfl_hourly_layered.yaml
```

Train REINFORCE and evaluate the selected storage design:

```powershell
python scripts/run_baselines.py
python scripts/train_dfl.py
python scripts/evaluate.py
```

The active artifacts are written under
`outputs/dataset_v2_dfl_hourly_layered/`.

## Retained Python package

`src/storage_dfl/` contains the runtime library used by the six commands above:
scenario schemas and normalization, the IEEE-13 feeder, CVAE components,
REINFORCE support policy, planning models, solver backends, and stage wiring.
`tests/test_core.py` is the retained automated test suite.
