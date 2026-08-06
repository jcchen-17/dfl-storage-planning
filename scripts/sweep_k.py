"""Sweep the planning scenario count K across several non-learned selection rules.

The DFL stage learns a support set of size ``dfl.num_support_scenarios``.  That
count, and the claim that learning the set beats choosing it, both have to be
established against a curve rather than asserted.  This script produces that
curve: for every (rule, K) pair it plans on K validation scenarios, fixes the
design, and re-solves on the held-out test set.

``out_of_sample_objective`` is the number to read.  A DFL result at the same K
has to beat every rule here to be worth its cost.

The rules deliberately include more than the farthest-point heuristic that seeds
the DFL stage.  Farthest-point maximises coverage, so at K=1 it returns the most
extreme day rather than a representative one, which makes it an unfairly weak
baseline on its own.  ``kmeans`` and ``aggregate`` are the conventional
scenario-reduction answers; ``aggregate`` at K=1 is exactly the expected-value
scenario.

Results are appended after every pair and completed pairs are skipped on a
re-run, so a long sweep survives interruption.

    python scripts/sweep_k.py --config configs/generator_compare.yaml
    python scripts/sweep_k.py --config configs/generator_compare.yaml \
        --rules farthest kmeans aggregate --k 1 2 3 --original-scale
    python scripts/sweep_k.py --config configs/generator_compare.yaml \
        --rules random --seeds 5 --k 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

from storage_dfl.config import DataCenterConfig, load_config
from storage_dfl.data import ScenarioPool
from storage_dfl.dfl import SELECTION_RULES, select_scenarios
from storage_dfl.planning import StoragePlanningOracle
from storage_dfl.stages import ArtifactPaths, _experiment_data, _load_codec

COLUMNS = (
    "rule",
    "k",
    "seed",
    "planning_status",
    "planning_gap",
    "planning_objective",
    "planning_seconds",
    "installed",
    "power_mw",
    "energy_mwh",
    "duration_h",
    "at_bound",
    "out_of_sample_objective",
    # Total system cost is the wrong denominator for comparing these rules. The
    # storage decision moves about 3.6% of it and every rule pays the same
    # untouchable remainder, so differences that are large relative to what the
    # decision is worth arrive looking like rounding. storage_value is the same
    # number measured against the no-storage case on the same scenarios: it is
    # what installing storage saved, and comparing how much of it each rule
    # captured is the comparison the sweep is actually for.
    "no_storage_objective",
    "storage_value",
    "out_of_sample_carbon_slack",
    "out_of_sample_status",
    "out_of_sample_seconds",
    "support_scenarios",
)


def _at_bound(power: float, energy: float, planning) -> str:
    if power <= 0.0:
        return "n/a"
    flags = []
    if power > 0.98 * planning.max_power_mw:
        flags.append("P")
    if energy > 0.98 * planning.max_energy_mwh:
        flags.append("E")
    if energy / power > 0.98 * planning.max_duration_hours:
        flags.append("duration")
    return "+".join(flags) if flags else "interior"


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write(rows: list[dict], output_dir: Path) -> None:
    with (output_dir / "k_sweep.json").open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2, ensure_ascii=False)
    with (output_dir / "k_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generator_compare.yaml")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--rules",
        nargs="+",
        choices=SELECTION_RULES,
        default=["farthest", "kmeans", "aggregate"],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Repetitions for the 'random' rule; other rules are deterministic.",
    )
    parser.add_argument(
        "--original-scale",
        action="store_true",
        help="Use the pre-scale-up facility and storage bounds.",
    )
    parser.add_argument(
        "--planning-time-limit",
        type=float,
        default=None,
        help=(
            "Seconds per solve; the config's solver_time_limit_seconds is used "
            "otherwise. These rows are the baseline a DFL result at the same K "
            "has to beat, so a shorter budget here is not a saving: it makes the "
            "baseline stop on the time limit while DFL converged, and the two "
            "objectives are then not comparable."
        ),
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=None,
        help=(
            "MB per solve, overriding solver_memory_limit_mb. That value is sized "
            "for solver_max_parallel_workers solves at once during training; this "
            "sweep runs one at a time over the larger final_validation_size "
            "scenario set. At 32 scenarios every out-of-sample solve stopped on "
            "memlimit, which makes the objectives incomparable."
        ),
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=None,
        help=(
            "Test scenarios per row, overriding final_validation_size. Must match "
            "what evaluate.py was given, or the DFL result and these baselines are "
            "measured on different sets and cannot be compared."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/k_sweep")
    args = parser.parse_args()

    config = load_config(args.config)
    planning = config.planning
    if args.planning_time_limit is not None:
        planning = replace(
            planning, solver_time_limit_seconds=args.planning_time_limit
        )
    if args.memory_limit is not None:
        planning = replace(planning, solver_memory_limit_mb=args.memory_limit)
    data_center = config.data_center
    if args.original_scale:
        data_center = DataCenterConfig(0.03, 0.12, 0.30)
        planning = replace(
            planning,
            max_power_mw=1.20,
            max_energy_mwh=3.00,
            max_duration_hours=4.0,
            backup_generator_mw=1.0,
            backup_generator_mvar=0.8,
        )

    paths = ArtifactPaths(config.output_dir)
    feeder, validation_pool = _experiment_data(config, config.data.validation_split)
    _, test_pool = _experiment_data(config, config.data.test_split)
    codec = _load_codec(paths, feeder)
    evaluation_scenarios = (
        args.scenarios if args.scenarios is not None else config.dfl.final_validation_size
    )
    test_subset = ScenarioPool(
        test_pool.subset(
            codec.support_indices(
                test_pool, min(evaluation_scenarios, len(test_pool.scenarios))
            ).tolist()
        )
    )
    oracle = StoragePlanningOracle(
        feeder, planning, config.costs, config.data, data_center
    )
    # One solve, shared by every row: the rules are all evaluated on this same
    # test subset, so what the system costs without storage is a constant here.
    # Subtracting it does not reorder anything -- it changes the denominator, so
    # a difference can be read against what the decision is worth rather than
    # against a total that is mostly untouchable.
    print("solving the no-storage reference on the test subset...", flush=True)
    reference = StoragePlanningOracle(
        feeder,
        replace(planning, max_storage_sites=0),
        config.costs,
        config.data,
        data_center,
    ).solve(test_subset.scenarios, allow_carbon_slack=True)
    if reference.status != "optimal" or not reference.feasible:
        raise RuntimeError(
            f"The no-storage reference stopped at {reference.status!r}; every "
            "storage_value would be measured against an unproven number. Raise "
            "--memory-limit or --planning-time-limit."
        )
    no_storage_objective = float(reference.objective)
    print(
        f"no-storage reference: {no_storage_objective:,.2f} "
        f"({reference.solve_time_seconds:.1f}s)\n",
        flush=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_existing(output_dir / "k_sweep.json")
    done = {(r["rule"], int(r["k"]), int(r["seed"])) for r in rows}

    print(f"facility      : {data_center}")
    print(f"storage bounds: P<={planning.max_power_mw} E<={planning.max_energy_mwh} "
          f"duration<={planning.max_duration_hours}")
    print(f"cost P/E      : {config.costs.power_dollars_per_mw} / "
          f"{config.costs.energy_dollars_per_mwh}")
    print(f"test set      : {len(test_subset.scenarios)} scenarios")
    if done:
        print(f"resuming      : {len(done)} pairs already complete")
    print(flush=True)

    for rule in args.rules:
        seeds = range(args.seeds) if rule == "random" else [0]
        for seed in seeds:
            for k in args.k:
                if (rule, k, seed) in done:
                    continue
                scenarios, weights, labels = select_scenarios(
                    rule, validation_pool, codec, k, seed=seed
                )
                started = time.perf_counter()
                design = oracle.solve(scenarios, weights=weights)
                planning_seconds = time.perf_counter() - started
                if not design.feasible:
                    print(f"{rule} K={k} seed={seed}: planning infeasible "
                          f"({design.status})", flush=True)
                    continue

                installed = list(design.design.installed_buses)
                power = sum(float(design.design.power_mw[b]) for b in installed)
                energy = sum(float(design.design.energy_mwh[b]) for b in installed)

                started = time.perf_counter()
                out_of_sample = oracle.solve(
                    test_subset.scenarios,
                    fixed_design=design.design,
                    allow_carbon_slack=True,
                )
                out_of_sample_seconds = time.perf_counter() - started

                row = {
                    "rule": rule,
                    "k": k,
                    "seed": seed,
                    "planning_status": design.status,
                    "planning_gap": round(min(float(design.relative_gap), 9.99), 5),
                    "planning_objective": round(float(design.objective), 2),
                    "planning_seconds": round(planning_seconds, 1),
                    "installed": "|".join(installed) if installed else "none",
                    "power_mw": round(power, 4),
                    "energy_mwh": round(energy, 4),
                    "duration_h": round(energy / power, 3) if power > 0 else 0.0,
                    "at_bound": _at_bound(power, energy, planning),
                    "out_of_sample_objective": round(float(out_of_sample.objective), 2),
                    "no_storage_objective": round(no_storage_objective, 2),
                    # Positive means storage paid for itself by this much.
                    "storage_value": round(
                        no_storage_objective - float(out_of_sample.objective), 2
                    ),
                    "out_of_sample_carbon_slack": round(
                        float(out_of_sample.carbon_slack_cost), 2
                    ),
                    "out_of_sample_status": out_of_sample.status,
                    "out_of_sample_seconds": round(out_of_sample_seconds, 1),
                    "support_scenarios": "|".join(labels),
                }
                rows.append(row)
                _write(rows, output_dir)
                print(
                    f"{rule:<9} K={k} seed={seed} | plan {row['planning_status']:<10} "
                    f"gap={row['planning_gap']:.4f} t={row['planning_seconds']:6.1f}s | "
                    f"{row['installed']:<8} {row['power_mw']:.3f} MW "
                    f"{row['energy_mwh']:.3f} MWh {row['duration_h']:.2f} h "
                    f"[{row['at_bound']}] | out-of-sample "
                    f"{row['out_of_sample_objective']:,.0f} | storage value "
                    f"{row['storage_value']:,.0f}"
                    # The planning status is printed above; without this the
                    # out-of-sample one was visible only in the CSV, and a row
                    # that stopped on memlimit read exactly like a converged one.
                    + (
                        ""
                        if row["out_of_sample_status"] == "optimal"
                        else f"  <-- {row['out_of_sample_status'].upper()}, NOT CONVERGED"
                    ),
                    flush=True,
                )

    if not rows:
        print("no results")
        return
    print()
    # out_of_sample_objective is the number the whole sweep exists to produce, so
    # its solve status is checked before any of it is reported as a ranking.
    unconverged = [r for r in rows if r["out_of_sample_status"] != "optimal"]
    if unconverged:
        statuses = sorted({r["out_of_sample_status"] for r in unconverged})
        print(
            f"WARNING: {len(unconverged)} of {len(rows)} out-of-sample solves stopped "
            f"at {statuses} instead of proving optimality. Their objectives are "
            "unproven incumbents, so the differences between them measure how far "
            "each solve got, not how good each design is. Re-run with a larger "
            "--memory-limit (these solves are serial, so they can have far more "
            "than solver_memory_limit_mb allows the concurrent training solves) "
            "before reading anything below as a ranking."
        )
    best = min(rows, key=lambda r: r["out_of_sample_objective"])
    print(f"lowest out-of-sample cost: {best['rule']} K={best['k']} "
          f"-> {best['out_of_sample_objective']:,.2f}")
    # The same spread against both denominators. The first is what the objective
    # reports, the second is what the decision is worth, and the gap between the
    # two percentages is the reason the rules looked indistinguishable.
    values = [r["storage_value"] for r in rows]
    objectives = [r["out_of_sample_objective"] for r in rows]
    spread = max(objectives) - min(objectives)
    mean_objective = sum(objectives) / len(objectives)
    print(f"\nno-storage reference {no_storage_objective:,.2f}")
    print(f"storage value: best {max(values):,.0f}, worst {min(values):,.0f}")
    print(
        f"spread between rules {spread:,.0f} = "
        f"{spread / mean_objective * 100:.3f}% of total cost"
    )
    if max(values) > 0:
        print(f"{' ':21s}= {spread / max(values) * 100:.1f}% of what storage is worth")
    else:
        print(
            "storage never paid for itself on this test set: every design cost "
            "more than installing nothing, so the rules differ only in how much "
            "they overbuilt."
        )
    stalled = sorted(
        {(r["rule"], r["k"]) for r in rows
         if r["planning_gap"] > config.planning.solver_relative_gap}
    )
    if stalled:
        print(f"NOTE: {stalled} did not reach the configured gap. Those designs are "
              "not proven optimal and must not be compared against converged ones.")
    print(f"wrote {output_dir / 'k_sweep.csv'}")


if __name__ == "__main__":
    main()
