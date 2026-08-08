# Handoff: decision-focused scenario selection for storage planning

Paste this whole file as context before asking for anything. Every number in it
was measured on this repository, at the settings the config currently carries.
Where something is uncertain it says so; do not treat the uncertain parts as
established.

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

1. Finish seeds 1-4 of `run_dfl_seeds.py --rule kmeans --tag kmeansinit`. If
   they land near 4,170 rather than near 4,118, the result is independent of
   initialisation.
2. Build the same-budget random-search control described above. This decides
   whether the policy contributes anything, and the current evidence suggests
   it may not.
3. Only if the answer to 2 is favourable is it worth returning to the generator
   weights or to larger K.

Do not start by tuning REINFORCE hyperparameters. The gradient has been
measured not to move the decision loss further than the solver can resolve;
learning-rate and exploration changes cannot fix a signal that is below the
resolution of the thing being optimised.
