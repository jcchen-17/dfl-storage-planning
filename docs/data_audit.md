# Active data audit

The active configuration uses
`data/processed/ieee13_smartds_dfl_v2_outage/dfl_scenarios_v2_48h.npz`.
The structural audit passes all checks, including split weights summing to one,
binary availability, non-overlapping windows and no exact cross-split leakage.

PCC demand is synthesized from the Azure workload proxy and temperature-derived
PUE, calibrated as a 10 MW-class continuously operating data center. Only the
three-year PV shape from original bus 675 is retained and its 0.65 MW nameplate
trajectory is rescaled to a 5 MW installation:

| quantity | value |
|---|---:|
| time resolution | 1 hour |
| all-split minimum facility load | 7.463 MW |
| all-split mean facility load | 8.477 MW |
| all-split 95th-percentile facility load | 9.416 MW |
| all-split maximum facility load | 10.224 MW |
| all-split load coefficient of variation | 0.070 |
| mean available PV | 0.821 MW |
| 95th-percentile available PV | 3.348 MW |
| maximum available PV | 4.533 MW |
| mean grid carbon intensity | 0.270 tCO2/MWh |
| 95th-percentile grid carbon intensity | 0.360 tCO2/MWh |
| outage windows / simulated training windows | 46 / 182 |
| outage hours | 106 |
| physical outage probability from `sample_weight` | about 0.71% |

The PCC reduction does not sum feeder demand or PV. A 3.5 MW partial-firm
diesel is paired with the calibrated facility. Across all 546
train/validation/test scenarios, the worst four-hour residual after local PV and
diesel is 20.325 MWh with a 5.817 MW peak. This supports battery bounds of
7.5 MW / 60 MWh when initial SOC is 50%, minimum SOC is 10%, and discharge
efficiency is 95%.

Grid-connected shedding is disabled. Carbon infeasibility is represented by the
explicit carbon-excess variable rather than economically curtailing served load.

Outages are oversampled only to make small feasibility batches observe them.
Planning and evaluation use the dataset's importance-corrected `sample_weight`.
Grid availability is copied as a detached exogenous trajectory into the matched
generated scenario; gradients remain limited to generated load, PV and grid
carbon intensity.

The reproducible audit artifact is written to
`outputs/dataset_v2_dfl_single_pcc_dc10mw_pv5mw_hourly_cap/data_audit.json`.
