# Active data audit

The active configuration uses
`data/processed/ieee13_smartds_dfl_v2_outage/dfl_scenarios_v2_48h.npz`.
The structural audit passes all checks, including split weights summing to one,
binary availability, non-overlapping windows and no exact cross-split leakage.

For the 182 training windows, after aggregation to the data-centre PCC:

| quantity | value |
|---|---:|
| mean facility load | 0.965 MW |
| 95th-percentile facility load | 1.193 MW |
| maximum facility load | 1.302 MW |
| mean available PV | 0.282 MW |
| maximum available PV | 1.467 MW |
| mean grid carbon intensity | 0.270 tCO2/MWh |
| 95th-percentile grid carbon intensity | 0.360 tCO2/MWh |
| outage windows / simulated training windows | 46 / 182 |
| outage hours | 106 |
| physical outage probability from `sample_weight` | about 0.71% |

The previous active data file had no outages. In addition, the previous 2 MW
diesel rating exceeded the observed 1.302 MW peak demand, so even an augmented
grid outage could not produce insufficient supply. The active case therefore
uses a 0.75 MW partial-firm diesel rating. With no storage this creates positive
deficit in 65 outage intervals in the training split, while normal grid-connected
hours remain supply adequate.

Outages are oversampled only to make small feasibility batches observe them.
Planning and evaluation use the dataset's importance-corrected `sample_weight`.
Grid availability is copied as a detached exogenous trajectory into the matched
generated scenario; gradients remain limited to generated load, PV and grid
carbon intensity.

The reproducible audit artifact is written to
`outputs/dataset_v2_dfl_single_pcc_outage_hourly_cap/data_audit.json`.
