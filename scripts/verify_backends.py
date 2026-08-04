"""Check that the Gurobi backend builds and solves the same model as SCIP.

Both backends run the identical 1,500-line builder, so any disagreement here is
a translation defect in the facade -- a wrong parameter unit, a mis-mapped
status, a warm start that matched nothing -- rather than a modelling difference.

A small scenario count keeps this quick; correctness of the translation does not
depend on problem size, and the sizes are compared explicitly so a silently
truncated model cannot pass.
"""

import sys
import time
from dataclasses import replace

from storage_dfl.config import load_config
from storage_dfl.dfl import select_scenarios
from storage_dfl.planning import StoragePlanningOracle
from storage_dfl.stages import ArtifactPaths, _experiment_data, _load_codec

K = int(sys.argv[1]) if len(sys.argv) > 1 else 1
TIME_LIMIT = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0

config = load_config("configs/generator_compare.yaml")
feeder, pool = _experiment_data(config, config.data.validation_split)
codec = _load_codec(ArtifactPaths(config.output_dir), feeder)
scenarios, weights, labels = select_scenarios("kmeans", pool, codec, K, seed=0)
print(f"K={K} scenarios: {labels}\n", flush=True)

results = {}
BACKENDS = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("scip", "gurobi")
for backend in BACKENDS:
    planning = replace(
        config.planning,
        solver_backend=backend,
        solver_time_limit_seconds=TIME_LIMIT,
        warm_start_time_limit_seconds=60.0,
        # Serial: this is a correctness check, not a timing run.
        solver_max_parallel_workers=1,
        # HiGHS has no memory-limit option and raises if one is set.
        solver_memory_limit_mb=0.0,
    )
    oracle = StoragePlanningOracle(
        feeder, planning, config.costs, config.data, config.data_center
    )
    try:
        built = oracle._build_model(
            scenarios, weights, fixed_design=None, allow_carbon_slack=False
        )
        size = (built.model.getNVars(), built.model.getNConss())
        built.model.freeProb()
    except Exception as exc:  # noqa: BLE001
        print(f"{backend}: BUILD FAILED {type(exc).__name__}: {exc}")
        continue

    started = time.perf_counter()
    try:
        result = oracle.solve(scenarios, weights=weights)
    except Exception as exc:  # noqa: BLE001
        print(f"{backend}: SOLVE FAILED {type(exc).__name__}: {exc}")
        continue
    seconds = time.perf_counter() - started

    installed = tuple(sorted(result.design.installed_buses))
    power = sum(float(result.design.power_mw[b]) for b in installed)
    energy = sum(float(result.design.energy_mwh[b]) for b in installed)
    results[backend] = {
        "size": size,
        "status": result.status,
        "objective": result.objective,
        "investment": result.investment_cost,
        "peak_grid": result.peak_grid_mw,
        "installed": installed,
        "power": round(power, 4),
        "energy": round(energy, 4),
        "seconds": seconds,
        "gap": min(float(result.relative_gap), 9.99),
        "design": result.design,
    }
    print(
        f"{backend:<7} | {size[0]:,} vars {size[1]:,} conss | {result.status:<12} "
        f"gap={results[backend]['gap']:.4f} | obj {result.objective:14,.2f} | "
        f"{installed or '()'} {power:.3f} MW {energy:.3f} MWh | {seconds:6.1f}s",
        flush=True,
    )

print()
if len(results) < len(BACKENDS):
    print("INCONCLUSIVE: one backend did not produce a result")
    raise SystemExit(1)

first, second = BACKENDS[0], BACKENDS[1]
scip, gurobi = results[first], results[second]
problems = []
if scip["size"] != gurobi["size"]:
    problems.append(f"model size differs: {scip['size']} vs {gurobi['size']}")
relative = abs(scip["objective"] - gurobi["objective"]) / max(abs(scip["objective"]), 1.0)
tolerance = 2 * config.planning.solver_relative_gap
if relative > tolerance:
    problems.append(
        f"objectives differ by {relative:.4%}, above the {tolerance:.2%} both solvers were allowed"
    )
print(f"objective difference: {relative:.5%} (tolerance {tolerance:.2%})")

if scip["installed"] != gurobi["installed"]:
    # Different siting is not by itself a defect. Both solves stop at a relative
    # gap, so when two sites cost almost the same the choice between them is
    # inside the tolerance and each solver may legitimately land on a different
    # one. What would be a defect is the two backends disagreeing about the cost
    # of the SAME design, so that is what gets checked.
    print(f"\nsiting differs ({scip['installed']} vs {gurobi['installed']}); "
          "cross-evaluating both designs with both backends")
    cross = {}
    for owner, source in ((first, results[first]), (second, results[second])):
        design = None
        for backend_name in BACKENDS:
            planning = replace(
                config.planning,
                solver_backend=backend_name,
                solver_time_limit_seconds=TIME_LIMIT,
                warm_start_time_limit_seconds=60.0,
                solver_max_parallel_workers=1,
                solver_memory_limit_mb=0.0,
            )
            oracle = StoragePlanningOracle(
                feeder, planning, config.costs, config.data, config.data_center
            )
            if design is None:
                design = source["design"]
            evaluated = oracle.solve(
                scenarios, weights=weights, fixed_design=design
            )
            cross[(backend_name, owner)] = float(evaluated.objective)
        left = cross[(BACKENDS[0], owner)]
        right = cross[(BACKENDS[1], owner)]
        gap = abs(left - right) / max(abs(left), 1.0)
        print(f"  {owner}'s design: {BACKENDS[0]} {left:,.2f} vs "
              f"{BACKENDS[1]} {right:,.2f} -> {gap:.5%}")
        if gap > 1e-6:
            problems.append(
                f"the backends value {owner}'s own design differently ({gap:.5%}); "
                "that is a translation defect, not degeneracy"
            )
    print("  (identical costs mean the models agree and the siting is degenerate)")
if problems:
    print("\nFAILED:")
    for problem in problems:
        print("  -", problem)
    raise SystemExit(1)
print("PASSED: the two backends agree")
speedup = scip["seconds"] / max(gurobi["seconds"], 1e-9)
print(
    f"speed: {first} {scip['seconds']:.1f}s -> {second} {gurobi['seconds']:.1f}s "
    f"= {speedup:.2f}x"
)
