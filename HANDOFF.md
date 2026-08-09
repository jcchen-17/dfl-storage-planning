# Handoff: decision-focused scenario selection for storage planning

Paste this whole file as context before asking for anything. Every number in it
was measured on this repository, at the settings the config currently carries.
Where something is uncertain it says so; do not treat the uncertain parts as
established.

> **2026-08-09 supersedes much of what follows.** The generative path was
> measured to have a degenerate action space, the contribution was redefined as
> selection, and the random-search control was finally run. Read
> "2026-08-09: measurements that changed the plan" at the end of this file
> first; where it conflicts with an older section, the newer section wins.
>
> **Start with "OPEN AND UNRESOLVED: the planning tolerance has been choosing
> the design".** It is the live problem and it puts a question mark over every
> result in this file that reports a plan with no storage. If you are picking
> this up to work on one thing, that is the thing.

## What the project is

A data centre on an IEEE13 unbalanced distribution microgrid decides where and
how large a battery to build. The planning model is a MILP over 48-hour
scenarios. Solving it over the whole scenario pool is intractable, so the
planner is given `K` representative scenarios, and the research question is how
to choose them.

The proposed method (`carbon_formulation` aside, this is the contribution)
learns `K` support scenarios by REINFORCE over the latent space of a
conditional VAE, using the resulting planning decision's out-of-sample cost as
the reward. It is compared against selection heuristics that never look at the
decision: `farthest`, `kmeans`, `aggregate`, `random`.

`num_support_scenarios: 1`, so all head-to-head numbers below are K=1.

## The measurement everything reduces to

    storage_value = no_storage_objective - out_of_sample_objective

Both terms come from the same 16-scenario test subset, chosen by
`codec.support_indices`. The reference is one constant, so `storage_value` is
an affine transform of the out-of-sample objective and ranking by either is
identical. Comparing values measured against different references is
meaningless, and several scripts refuse to run when they detect it.

Current reference: **1,723,797.62**.

## Current results, all at solver_relative_gap 1e-4

    perfect-information ceiling   [4,988, 6,024]    (the PI solve itself
                                                     timed out at gap 6.03e-04,
                                                     hence the interval)
    random best (seed 1)           4,783
    farthest                       4,252   <- the bar: best rule needing no
                                              learning and no luck
    DFL (farthest init, n=5)       4,170   sd   121
    DFL (kmeans init, n=1)         4,281   <- only one seed so far
    kmeans                         4,118
    random (n=10)                  3,967   sd   743
    aggregate                      2,764
    random worst (seed 3)          2,756

DFL captures 69-84% of the ceiling. It does not beat `farthest`: the gap is
-82 against a solver tolerance of 171 and a standard error of 54, so the two
are tied. Its spread is roughly one sixth of random's.

## The finding that matters most

**The REINFORCE training does not learn.** `scripts/dfl_training_report.py`
measures the total improvement in decision loss over each run against what the
solver can resolve:

    improvements over six runs: 125, 11, 74, 15, 73, 117
    solver resolution:          171
    0 of 6 runs improved by more than the solver can resolve

The best epoch lands at 9, 13, 17, 19, 20 with no pattern. `latent_shift`
reaches about 0.14 against an initial sampling width of 0.8, so the policy mean
never leaves the cloud it draws from.

So the good results do not come from the gradient. What the run actually does
is sample about 160 scenarios near its initialisation (20 epochs x 8 samples),
solve a planning model for each, score each design on a fixed validation set,
and keep the best of five finalists. **The value comes from sampling plus
validation-based selection, not from decision-focused learning.**

That is still a method, and its numbers are real -- sd 121 against random's 743
is a large reduction. But the claim has to be about selection, not learning.

### The open experiment that decides the paper's claim

Compare DFL against **random search in latent space at the same budget**: draw
the same ~160 latents around the same initialisation with no policy update,
score them the same way, keep the best. If that matches 4,170, the policy
contributes nothing and the method is "sample the generator and filter".
Nothing in the current results rules that out; it is the first thing a reviewer
will ask.

Also unfinished: seeds 1-4 of the kmeans-init arm. Seed 0 came in at 4,281,
inside the farthest-init range rather than near the kmeans baseline of 4,118,
which is early evidence that the result does not depend on where it started.
`farthest` ignores the seed, so all five farthest-init runs share one starting
scenario (`2017_006_48h`) and their sd measures the search's own variance, not
robustness to initialisation.

## How the setup got here, and why it cannot be rolled back casually

Five things were wrong at the start of this work. Each was measured, not
guessed, and the commit messages carry the evidence.

**1. The carbon target never bound.** `dc_carbon_cap` was 0.62 while grid
carbon intensity peaks at 0.409 tCO2/MWh, so it could not bite. Lowering it
changed nothing either: under `carbon_formulation: mccormick` the cap applies
to `nodal_carbon`, which the McCormick envelope bounds only as
`incoming_carbon / incoming_power_upper`, and that envelope's lower bound goes
negative whenever a flow sits below its own bound -- so a line may carry zero
carbon and any cap is satisfiable. Measured: the objective is bit-identical at
every cap from 0.62 down to 0.05. Three rounds of tightening the envelope
bounds did not change it; the looseness is structural. Fixed by moving to
`carbon_formulation: system_average`, where the cap is a linear constraint on
system intensity and binds exactly. Storage carbon vintages are still tracked,
so the layered accounting the project is about is retained.

**2. All storage value was synthetic.** `augment_price_regimes.py` multiplies
each window into five tariff regimes by scaling the price spread, and appends
the factor as a fifth context dimension. On the un-augmented windows the
perfect-information ceiling was **exactly 0** -- the optimal decision was not to
build, and every rule that built lost money. So every storage_value ever
recorded before this work came from that multiplier. Worse, the policy
optimises only the latent while the context stays fixed at the seed's, so the
seed's price regime was frozen for the whole run: with a 1.0x seed the largest
price spread any latent could reach was 13.57 $/MWh against the 87-139 that
good supports have. The dataset is now `dfl_training_windows_48h.npz`, context
is 4-dimensional, and a generator trained on the old path cannot be reused.

**3. Storage now earns against a carbon price.** A hard cap plus
`validation_carbon_slack_dollars` priced excess as a dimensionless slack scaled
by a big-M, so the implied price per tonne moved with that bound: storage was
worth 82,652 at cap 0.30 and 7,238,309 at 0.22, essentially all of the latter
avoided penalty, with no cap in between where the target binds and compliance is
still reachable. `carbon_price_dollars_per_t: 185` prices the excess in tonnes
directly. `dc_carbon_cap: 0.22` was chosen by measurement:

    cap    mean     sd   sd/mean   worst    best   rules below zero
    0.26  3,680    663      18%    2,222   4,584   0
    0.22  2,961    855      29%      862   3,948   0
    0.18    400    850     213%   -1,491   1,400   4

0.18's 213% is an artifact of a near-zero mean, not a usable signal. (Those
numbers are at 1e-3 and so are contaminated -- see below -- but the shape of
the trade-off is what chose 0.22.)

**4. The solver tolerance was larger than the effect.** At `1e-3` on a 1.72M
objective the tolerance leaves 1,724 unresolved while the values span 1,700 to
4,800. Holding everything fixed and changing only the tolerance moved every
rule up by 650-1,250 and swapped `kmeans` and `farthest` -- a ranking reversal
caused by nothing but where the solver stopped, reproduced identically on two
machines. Now `1e-4`, which leaves 171.

**5. `carbon_intensity_max` was 60% above anything physical.** It bounds every
carbon-intensity variable and so the envelopes built on them. 1.20 against a
real maximum of 0.72 (the generator; the grid peaks at 0.409). At cap 0.30 that
slack cost: gap 1.25e-03 in 922 s on the time limit, against 9.91e-04 in 259 s
optimal at 0.75. The old run was both unconverged and 50% high.

## Traps in this repository

Every one of these bit during this work, and each cost an hour or more. The
common shape is that **a failure looks exactly like a success**.

- A stale `normalization.json` **loads fine** with the wrong context width and
  only raises later, inside `encode_pool`, as a broadcast error. Check
  `codec.context_dim` against the data, do not rely on the load succeeding.
- A solve that stops on the time limit returns an unproven incumbent that
  prints exactly like a converged one. The scripts now flag it; the numbers it
  produced were 50% high in one case.
- `sweep_k.py` resumes by skipping `(rule, k, seed)` triples already present.
  It now writes and checks a settings fingerprint, because a tolerance change
  silently appended 1e-4 rows to nine 1e-3 rows in one file with nothing
  marking which was which.
- `--seeds` only repeats the `random` rule, and `random` is not in `--rules`'
  default. Passing `--seeds 10` without it produced a sweep with no random rows
  and no error. Now rejected.
- `early_stopping_min_epochs` equalled `epochs`, so the stop could first be
  considered on the final epoch and never fired -- configured, commented, and
  structurally dead. And `decision_deadband_relative` was 1,700 when a whole
  50-epoch run improves by about 506, so nothing the method achieves could have
  counted as material.
- Solver settings in configs were inherited from a different machine and said
  so in their own comments ("16 threads, the machine's physical core count",
  "28 GB of a 32 GB machine") while running on a 6-core / 16 GB box.
  `scripts/tune_solver.py` measures the machine; do not copy these numbers.

## Environment

- Windows. The project env is a conda env named `storage-dfl`; the base env has
  numpy but no torch, gurobipy, pyscipopt or highspy, so running from base
  silently reports "gurobipy NOT INSTALLED".
- Gurobi 13 with a NODE licence, which is **machine-locked**. A different
  machine needs its own. `scripts/check_gurobi.py` verifies licence, version
  match, and that the licence is not the size-limited pip one.
- `outputs/` and `data/` are gitignored. `data/processed/ieee13_smartds_dfl/
  dfl_training_windows_48h.npz` (5.2 MB) must be copied by hand to a new
  machine.
- PowerShell drops empty-string arguments before argparse sees them, so
  `--regime ""` fails with "expected one argument". Defaults were changed so
  the common call needs no flag.

## Tools, and what each is for

    scripts/check_gurobi.py            licence, version, and the 2000-variable
                                       restriction, tested by solving past it
    scripts/tune_solver.py             measures this machine and recommends
                                       solver_threads / workers / memory
    scripts/verify_generator.py        after training: dataset context width,
                                       saved normalization, an encode/decode
                                       round trip, and reconstruction quality
                                       on the carbon channel
    scripts/headroom_probe.py          is there room for a learned selector at
                                       all -- ceiling, rule spread, headroom.
                                       --skip-ceiling for the cheap version,
                                       --ceiling-only for the expensive half
    scripts/carbon_cap_scan.py         sweeps dc_carbon_cap and reports whether
                                       the target binds, whether compliance
                                       needs slack, and whether it converged
    scripts/sweep_k.py                 the baseline table
    scripts/run_dfl_seeds.py           one DFL run per seed into its own
                                       directory; --rule picks the init
    scripts/dfl_training_report.py     did training move the loss further than
                                       the solver can resolve
    scripts/compare_dfl_vs_baselines.py  the head-to-head

Typical order on a new machine: `check_gurobi` -> `tune_solver` -> copy the
config's solver numbers -> `train_generator` -> `verify_generator` ->
`headroom_probe --skip-ceiling` -> `sweep_k` -> `run_dfl_seeds` ->
`dfl_training_report` -> `compare_dfl_vs_baselines`.

## Known weaknesses not yet addressed

**The generator compresses the channel that drives value.** Reconstruction on
four validation windows:

    channel          error
    carbon mean      +0% to +2%     fine
    carbon swing     +4%, -15%, -45%, -26%   worse the larger the real swing
    peak net load    +2% to +5%, but -20% on the one extreme window
    price spread     -11% to -22%

Carbon swing is exactly what storage arbitrages against, and the largest real
swing (0.096) came back at 0.052. The loss is field-balanced with explicit
weights, and they still reflect the old objective: `price_weight: 0.25` and a
dedicated `price_spread_weight: 0.20` for a channel that is now nearly constant
(one value on weekdays, zero at weekends), against `carbon_weight: 0.10` and no
carbon-swing term at all. Rebalancing those weights, and adding a carbon-swing
auxiliary term symmetric with `price_spread_weight`, is the obvious next
generator change. It has not been tried.

**The K=3/5/8 baseline curve is incomplete.** K=1 is done. K=3 was in progress;
rows get much slower with K (farthest K=1 4.4 s, K=3 170 s) and
`--planning-time-limit 900` is worth passing. Note that shortening the limit is
only free for rows that would not have converged anyway -- for the rest it
biases the baseline down against a converged DFL result.

**One pathological instance.** `random` seed 6 (`2017_161_48h`) does not
converge at 1e-4 in 1800 s. Its row is flagged; excluding it moves the random
mean from 3,967 to 3,878 and the sd from 743 to 729, so it changes nothing.

## What to do next, in order

The old list below has been superseded by the dataset-v2 reset. The derived v1
regime datasets, generated scratch configs and incomplete K=1 run were removed;
raw inputs and provenance artifacts were retained.

1. Use `data/processed/ieee13_smartds_dfl_v2/dfl_scenarios_v2_48h.npz` and
   `configs/dataset_v2_baselines.yaml` as the supported baseline.
2. Treat `outputs/dataset_v2_baselines/exhaustive_k1_v2.json` as the K=1 oracle:
   182 training supports, 147 usable plans, 140 distinct engineering designs.
   The strict winner is `v2_2016_028_48h`, installing 0.337 MW / 1.303 MWh at
   bus 671. Its weighted storage value is $2,263.98 on validation and $8,534.10
   on test.
3. The channel audit shows that load/PV and price drive capacity. Current carbon
   timing is weak: flattening it changes the bus but only changes test value by
   about $7, while reversing it returns the original design.
4. Do not resume the current continuous CVAE+REINFORCE path. First implement a
   finite-library selector over observed training scenarios and require it to
   beat same-budget random search. At budgets 8--64, the generic GP-BO and CEM
   controls did not do so; see `search_controls_v2.json`.
5. Only after a selector passes that control should a generator be reconsidered.
   If it is, encode stochastic fields only (load, PV, temperature/carbon as
   justified), use a latent dimension near 8 rather than 64, and reconstruct
   deterministic workload/PUE/price outside the model.

## Carbon-accounting reset

The v2 channel ablation used `system_average`, which blends storage carbon and
does not test the paper's vintage-layer contribution. The model now supports
`layered_system`: it uses the identical linear system-boundary cap but retains
one fixed-carbon vintage per charging interval. This removes the invalid loose
nodal McCormick comparison and keeps the problem as a MILP.

`scripts/compare_carbon_accounting_v2.py` plans with both formulations and
evaluates both designs under `layered_system`. A four-representative screening
scan showed equal regret at caps 0.30/0.26, but layered regret fell from
$1,047.61 to $465.64 at 0.22 and from $2,135.78 to $295.05 at 0.18. Those caps
were soft targets priced at $185/tCO2. With price zero and slack forbidden, the
hourly cap is feasible at 0.30 and infeasible at 0.26 or below. Therefore the
next model change should be a 48-hour-total or rolling-24-hour carbon budget,
not artificial rescaling of the carbon data. Also decide whether the claimed
boundary is the whole feeder or requires explicit data-center power attribution.

That 48-hour-total change is now implemented as
`planning.carbon_cap_scope: horizon`; `hourly` preserves the original behavior.
Both scopes compute carbon from hourly power and carbon intensity. With a
positive `carbon_price_dollars_per_t`, hourly prices each interval's excess
tonnes separately, while horizon prices only net excess tonnes after summing
carbon mass and energy. With zero price and no slack, both are hard constraints.

The four-representative 2x2 scan is in
`carbon_accounting_scope_comparison_v2.json`. Every solve was optimal. Hourly
layered accounting beat blended accounting once caps tightened (regret 465.64
versus 1,047.61 at 0.22; 295.05 versus 2,135.78 at 0.18). Under the 48-hour
budget, both accounting methods had identical regret at all four caps. This is
expected under cyclic SOC and carbon-mass conservation: aggregation erases the
discharge-vintage timing that creates the contribution. Retain both results;
do not claim layered accounting improves a full-cycle total when the equations
show it cannot. Rolling 24-hour windows are the next plausible middle case.

## Runnable dataset-v2 DFL configuration

`configs/dataset_v2_dfl_hourly_layered.yaml` now provides an isolated exploratory
CVAE + REINFORCE run. ScenarioCodec layout v4 locks the one truly dataset-fixed
trajectory (workload) to its saved template after decoding; PUE and price remain
conditional generated fields because they vary across v2. The configuration
uses K=1, latent 8, 12 epochs x 8 policy samples, weighted k-means validation,
hourly `layered_system`, and a 1e-3 training / 1e-4 reporting solver gap. It does
not overwrite any v2 baseline output. Commands are in `DATASET_V2.md`.

Do not start by tuning REINFORCE hyperparameters. The gradient has been
measured not to move the decision loss further than the solver can resolve;
learning-rate and exploration changes cannot fix a signal that is below the
resolution of the thing being optimised.

Dataset-v2 details, rebuild commands and new empirical results are recorded in
`DATASET_V2.md`. (That file, `outputs/dataset_v2_baselines/exhaustive_k1_v2.json`
and `search_controls_v2.json` are **no longer present in this working copy**;
sections above that cite their numbers cannot currently be reproduced here.)

## 2026-08-09: measurements that changed the plan

### The action space was degenerate, and that is why there was no signal

The CAPEX sweep showed uniform designs and 0.3-0.6% cost differences at every
level. The cause is upstream of the solver and the loss: **every latent the
policy samples decodes to essentially the same scenario.**

    channel          generated at sigma=0.10      real validation pool
    peak net load    1.349 - 1.429 MW             0.793 - 3.466 MW
    carbon swing     0.067 - 0.078                0.034 - 0.162
    price spread     52.2 - 56.6                  0.0 - 69.7

Generated sd as a fraction of real sd, sampling the full prior: 0.30 with the
context held at one scenario (what REINFORCE actually does, since the policy
optimises only the latent), 0.46 with contexts drawn from the pool. So the
planner receives one scenario, decision loss is flat in the action, and the
REINFORCE gradient is exactly zero. `_rank_advantages` collapsing samples into
one tie group is a symptom, not the cause.

A six-variant capacity sweep settled that this is not a tuning problem:

    variant                       peak_net  carbon_sw   val_carbon_spread
    latent8-hidden256-beta0.01        0.55       0.65               0.895
    latent16-hidden256-beta0.01       0.46       0.53               0.948   <- current
    latent32-hidden256-beta0.01       0.38       0.43               1.099
    latent16-hidden512-beta0.01       0.55       0.58               0.984
    latent16-hidden256-beta0.001      0.38       0.49               0.979
    latent32-hidden512-beta0.001      0.42       0.42               1.079

Coverage falls as latent grows and as beta falls, opposite to the intuition that
more latent capacity helps; validation reconstruction moves the other way, so
**tuning this generator on reconstruction error systematically degrades
coverage.** The ceiling is 0.55, and none of it is reachable by the policy,
which is stuck near 0.30. There is no exploration std that both works and
covers: sigma 0.18-0.20 already left the exact COPF with no incumbent on 3 of 4
samples.

### The generator's carbon channel does not generalise at all

Held-out `carbon_spread` loss is `1 - R^2`. Training reaches 0.011 (R^2 0.99);
validation never drops below 0.948 at any epoch out of 300, i.e. **R^2 ~ 0 for
the whole run.** A loss of 1.0 is exactly what a constant predictor scores, and
that is what the decoder does: real swing 0.061 decodes to 0.081, real 0.121
decodes to 0.085. Early stopping cannot help a curve that was never better.
Other channels do generalise (net_peak 1.805 -> 0.240, price_spread 1.606 ->
0.242), so this is specific to carbon, whose blocks are permuted within split
and season and are therefore independent of everything else by construction.

`train_generator_stage` now records `train_eval_*` and `validation_*` curves
each epoch under one deterministic posterior-mean pass, plus
`validation_minima` in `cvae_result.json`. That is how the above was measured.

### Contribution redefined: selection, with REINFORCE retained

The claim is **decision-focused selection of K support scenarios from observed
data**, not generation of novel scenarios. This is a change of action space, not
of method: the policy becomes a Plackett-Luce distribution over the N observed
scenarios (sample K without replacement) instead of a Gaussian over the latent.
Reward, score-function estimator, rank advantages, baseline momentum, finalists
and early stopping are all unchanged; only `dfl/support.py`'s
`DirectSupportPolicy` is replaced. Prefer a scoring network over scenario
features to N free logits, so the policy generalises to a new pool and "what
kind of scenario matters" becomes reportable.

### The random-search control, finally run

`scripts/random_subset_control.py`, K=3, 24 draws from the 182 validation
scenarios, each planned then scored on a fixed 4-scenario test set against one
shared no-storage reference (1,611,042.07). Storage values:

    distribution        [-1612, 0)  6   quarter of draws chose not to build
                        [10k, 20k)  3
                        [20k, 25k)  3
                        [25k, 30k) 10
                        [30k, 32k)  2

    expected best of B draws (bootstrap)      B=1  18,961   p10 0
                                              B=2  24,979
                                              B=4  28,200
                                              B=8  29,604
                                              B=16 30,491
                                              B=24 30,874
    kmeans K=3                                     24,793
    farthest K=3                                   27,964

**Random search beats both heuristics by B=4.** The printed best-of-budget curve
in that script walks the realised draw order, so its B=1 entry is one lucky
draw; the bootstrap above is the honest bar.

The ceiling is low and flat: six times the budget (B=4 to B=24) buys 9.5%, and
no draw after the 16th improved on 31,527. Good subsets form a wide plateau. A
learned selector at B=8 must beat 29,604 against an observed maximum of 31,527,
so **the contestable band is about 1,900 dollars while fixed-design evaluation
uncertainty is 13,739-27,240.** The signal is an order of magnitude below the
measurement.

The defensible framing is single-run behaviour, not best-of-B: one draw has
expected value 18,961 with p10 = 0 and a 25% chance of not building at all, so a
selector that lands reliably in 25k-30k is a real and practically meaningful
result. Report both curves; a reviewer will ask for best-of-B.

### Solver resolution, measured properly

Free planning plateaus near 1.9-2%: 120s to 900s moved it only 2.06% -> 1.92%
(+/- $27,894). Fixed-design evaluation is bimodal. Stratified on peak net load
across the test pool, 3 of 5 hit the 900s limit at 0.9-1.5% while 2 solved to
optimality in 11s, and **difficulty does not track peak net load** (1.31 hard,
1.65 easy, 1.79 hard, 1.98 easy, 2.15 hard). The cause of the split is
unidentified and worth finding. No-storage references all solve to optimality
within 92s, so the reference term is clean.

An earlier probe that took `test_pool.scenarios[:2]` sampled two windows at the
98th percentile of peak net load and reported their tail difficulty (+/- $79,206)
as if it were typical. Stratify.

### Deadband is still on the wrong denominator

`decision_deadband_relative: 0.01` is applied to the raw decision loss (~1.7M),
giving a deadband near 17,000 against a 5,000-10,000 signal, in `trainer.py`
`_rank_advantages`, early stopping and finalist selection, and in
`scenario_bo.py`. Finalists therefore all tie and the winner is chosen by the
"prefer the smaller design" fallback rather than by the objective. Rebase it on
storage value against the shared no-storage reference -- **but only after the
action space is fixed**, or it will turn solver noise into gradient, which is
what the deadband exists to prevent.

### Things that were tried and did not work

- Rebalancing generator loss weights. `carbon_spread_weight` is already 0.50,
  tied with `price_spread_weight` for the highest in the config, and the term is
  minimised successfully on training data. Raising it further only trades other
  channels away.
- "Encode stochastic fields only" (item 5 of the older plan). The packed vector
  is 121 values per timestep and the deterministic ones (workload, pue, price)
  are 3 of them: **2.5%**. The other 97.5% is phase-resolved P/Q/PV, which is
  stochastic. There is nothing to remove.

### Data-side facts worth knowing before changing the dataset

- Splits are **entirely by year**: train 2016, validation 2017, test 2018, 182
  windows each. Carbon blocks are permuted within split, so no carbon trajectory
  is shared across splits. Any re-split mixes years and drops the temporal
  holdout, which is a research decision, not a mechanical one.
- `grid_available` is **1.0 in all 546 scenarios**. Outages are an entire
  degenerate channel; activating them would create genuinely different optimal
  designs.
- `workload_arrival` has **one distinct profile** across all 546 scenarios;
  `grid_price_per_mwh` has **7**, with only two distinct spreads (0 or 69.726),
  and is not determined by start weekday. Load, PV and carbon are the only
  genuinely varying fields (CV 0.25, 0.34, 0.16).
- Windows are non-overlapping and year-aligned, so 8760/48 = 182 per year is a
  hard cap at this horizon and stride.

### Outages were added, measured, and did not help

`scripts/add_outage_scenarios.py` builds
`data/processed/ieee13_smartds_dfl_v2_outage/` with a contiguous outage in 25%
of windows, importance weights undoing the oversampling, and per-scenario
`annual_occurrences`. `configs/dataset_v2_outage.yaml` runs it. Three model
changes were needed before an outage scenario would even solve, and all three
are real fixes worth keeping:

1. **PV and storage exchanged no reactive power at all.** The reactive balance
   had only the substation and the backup generator. `inverter_power_factor`
   (0.90, i.e. |Q| <= 0.484 of rating, per IEEE 1547-2018) now gives both a
   per-phase reactive variable, bounded by installed rating rather than
   instantaneous output so an inverter can support voltage at night.
2. **The backup generator split its output equally across phases.** On this
   unbalanced feeder that wasted the light phase's share; the IIS showed the
   conflict confined to phases B and C, at one hour, across nearly every bus.
   `per_phase_backup_generator` dispatches it per phase under the same totals.
3. **`grid_available` bounded only active import.** The substation kept
   supplying reactive power and regulating voltage straight through an outage.
   Now zeroed per phase.

`feeder_shedding_dollars_per_mwh` and `feeder_curtailable_fraction` add capped
feeder load shedding at a VOLL, which was expected to be the fix and was not:
with equal-phase generation the model needed 80% curtailable to solve, and with
the three fixes above it solves at 40%.

The result, at 1% evaluation gap:

    9 h outages, annual_occurrences absent (a window = 182.5 a year)
      outage supports  16.06 MWh mean, all three pinned to max_duration_hours
      normal supports   2.35 MWh mean

    2 h outages, annual_occurrences = 1.30 for outage windows
      outage supports   0.00 MWh -- all three chose not to build
      normal supports   1.13 MWh

**Outages do not create decision diversity; correctly weighted they remove it.**
A 0.71% event times any defensible cost cannot amortise the annualised
investment: one 2-hour outage is ~2.6 outage-hours a year, worth about $13,000
of avoided VOLL against $12,150 a year for the 1 MWh that would cover it. The
first table is what the model says when a rare window is priced as a common one,
and it is wrong by roughly 140x, not a usable alternative.

Making a rare event drive the design requires a reliability *constraint*, not an
expected cost. That is a fourth theme on top of layered carbon accounting and
decision-focused selection, so it was not pursued.

### The real obstacle is a flat optimum, not a thin dataset

Measured on the 24 K=3 control draws: **20 distinct designs**, spanning 0.271 to
1.566 MW and 0.898 to 7.246 MWh -- 5.8x and 8.1x -- across five different bus
choices, with five draws declining to build at all. The pool is not short of
diversity, and neither is the design space. But the best five out-of-sample
values were 31,527 / 30,149 / 29,137 / 28,805 / 28,293: **a 10% band**.

Many very different designs reach nearly the same out-of-sample value. That is
what a smooth objective does near an interior optimum -- the first-order
condition makes it flat by construction -- so no dataset change fixes it, and
the outage attempt above is a worked example of one that did not.

Two claims follow, one supportable and one not:

- **Not supportable**: "decision-focused selection finds the best support set."
  Best and fifth-best differ by 10%, and evaluation uncertainty at 1% of a 2.1M
  objective is about $21,000 -- the same order as the thing being ranked.
- **Supportable**: "decision-focused selection is reliable in a single run."
  One random draw has expected value 18,961, sd 11,979, p10 = 0, and a 25%
  chance of not building. A selector landing consistently in 28,000+ is a 50%
  improvement over drawing once, well outside the measurement error.

`scripts/carbon_cap_sharpness.py` tests whether a binding carbon cap sharpens
the top, using one set of subsets scored at several caps (paired, so the
comparison is not reading subset noise) and a separate no-storage reference per
cap. If `best - 5th` stays near 10% as the cap tightens, a learned selector has
no contest to win and the single-run framing is the only one left. The cap is
the right lever to try first because it is already part of the story: layered
accounting only beat blended once the cap bit (regret 465 vs 1,047 at 0.22, 295
vs 2,135 at 0.18), and the pool's mean carbon intensity is 0.283 against the
current cap of 0.28, so today the constraint barely binds.

### OPEN AND UNRESOLVED: the planning tolerance has been choosing the design

This is the live problem. Everything above that reports a plan with no storage
is suspect until it is rechecked.

Storage is worth about 30,000 against a 1.55M planning objective: **1.86%**. The
planning solves ran at a 2% relative gap. A relative gap of 2% means the solver
stops once the incumbent is provably within 31,000 of the bound, and "install
nothing" is already within 31,000 of optimal. So it stops without looking.

From the 24-draw K=3 control, sorted by value, the five zero-value rows:

    value   plan gap   status      sec   installed
        0      1.08%   optimal      34   []
        0      1.60%   optimal      37   []
        0      1.83%   optimal      41   []
        0      2.08%   timelimit   610   []
        0      2.78%   timelimit   638   []

Three returned **`optimal` in 34-41 seconds**, against a median of 911 seconds
for the 19 draws that did install something. That is not a solver deciding
storage is uneconomic; it is a solver meeting its tolerance before it starts.
The carbon scan at a 3% planning gap then returned zero on 7 of 7 draws, which
is the same effect one step worse.

**What this invalidates.** Any "the optimum is not to build" claim made at a 2-3%
planning gap, which includes: the bimodal shape of the K=3 control and the
"avoid the bad quarter" framing built on it; part of the outage conclusion (that
one has an independent reason in `annual_occurrences`, but the no-build rows
should still be rechecked); and the 130% CAPEX row in the original sweep.

**What does not fix it.** Two things were tried and reasoned through:

1. *Longer solves.* Planning moved 2.06% -> 1.92% between 120 s and 900 s.
   Measured here, and consistent with the maintainer's experience that this
   model's gap does not come down with time.
2. *Adding an absolute tolerance while leaving the relative one loose.*
   `solver_absolute_gap_dollars` now exists and maps to Gurobi `MIPGapAbs`,
   HiGHS `mip_abs_gap`, SCIP `limits/absgap`. But all three terminate when
   **either** criterion is met, so a 2% relative gap still stops the solve at
   31,000 and the absolute setting does nothing.

**What should fix it, untested at the time of writing.** Set the relative gap to
**zero** so the relative criterion can never be met, leaving `MIPGapAbs` as the
only stopping rule. The point is not to prove optimality -- this model will not
prove anything tight in the available time -- but to stop the solver quitting at
40 seconds. Under a tolerance it cannot meet it keeps searching to the time
limit and keeps improving the incumbent, which is the quantity the comparison
actually uses.

Note the distinction that a single gap number hides: **an unproven bound is not
the same as a bad incumbent.** The comparison needs good incumbents, not proofs.
What it cannot tolerate is a solver that stops before finding one.

Cost: every solve now runs to the time limit, so the cheap 40-second no-build
rows become full-length solves. Budget accordingly.

A second, harder option if that is not enough: subtract a valid design-independent
lower bound from the objective so the relative tolerance applies to the part that
the decision can actually move. The argmin is unchanged and the absolute gap is
unchanged, but the same relative tolerance becomes far tighter. The offset must
stay strictly below the true optimum or the reduced objective approaches zero and
the relative gap misbehaves.

### Do not do this

Uniformly rescaling the data cannot change generation quality. `ScenarioCodec`
normalises by mean and standard deviation, so a global factor cancels, and the
coverage ratio that matters is itself scale-free. Rescaling only moves the
economics relative to fixed storage costs, which is what the CAPEX sweep already
varies.
