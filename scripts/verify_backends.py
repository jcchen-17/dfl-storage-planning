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
for backend in ("scip", "gurobi"):
    planning = replace(
        config.planning,
        solver_backend=backend,
        solver_time_limit_seconds=TIME_LIMIT,
        warm_start_time_limit_seconds=60.0,
        # Serial: this is a correctness check, not a timing run.
        solver_max_parallel_workers=1,
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
    }
    print(
        f"{backend:<7} | {size[0]:,} vars {size[1]:,} conss | {result.status:<12} "
        f"gap={results[backend]['gap']:.4f} | obj {result.objective:14,.2f} | "
        f"{installed or '()'} {power:.3f} MW {energy:.3f} MWh | {seconds:6.1f}s",
        flush=True,
    )

print()
if len(results) < 2:
    print("INCONCLUSIVE: one backend did not produce a result")
    raise SystemExit(1)

scip, gurobi = results["scip"], results["gurobi"]
problems = []
if scip["size"] != gurobi["size"]:
    problems.append(f"model size differs: {scip['size']} vs {gurobi['size']}")
relative = abs(scip["objective"] - gurobi["objective"]) / max(abs(scip["objective"]), 1.0)
tolerance = 2 * config.planning.solver_relative_gap
if relative > tolerance:
    problems.append(
        f"objectives differ by {relative:.4%}, above the {tolerance:.2%} both solvers were allowed"
    )
if scip["installed"] != gurobi["installed"]:
    problems.append(f"siting differs: {scip['installed']} vs {gurobi['installed']}")

print(f"objective difference: {relative:.5%} (tolerance {tolerance:.2%})")
if problems:
    print("\nFAILED:")
    for problem in problems:
        print("  -", problem)
    raise SystemExit(1)
print("PASSED: the two backends agree")
speedup = scip["seconds"] / max(gurobi["seconds"], 1e-9)
print(f"speed: SCIP {scip['seconds']:.1f}s -> Gurobi {gurobi['seconds']:.1f}s = {speedup:.1f}x")
