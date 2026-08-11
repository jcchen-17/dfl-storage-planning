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

Train the recourse-aware CVAE and evaluate its storage design:

```powershell
python scripts/train_dfl.py
python scripts/evaluate.py
```

The active artifacts are written under
`outputs/dataset_v2_dfl_single_pcc_dc10mw_pv5mw_hourly_cap/`.

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
