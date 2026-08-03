from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

from storage_dfl.planning.model import StoragePlanningOracle


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated SCIP planning worker")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    os.environ["STORAGE_DFL_SOLVER_WORKER"] = "1"
    payload = pickle.loads(args.input.read_bytes())
    oracle = StoragePlanningOracle(
        payload["feeder"],
        payload["planning"],
        payload["costs"],
        payload["data"],
    )
    warm_start_cached = bool(payload.get("warm_start_cached", False))
    # A fresh worker starts with an empty warm-start cache. Adopt whatever the
    # parent already computed so the no-storage bootstrap is solved once per
    # scenario set rather than once per solve.
    oracle._injected_warm_start = (warm_start_cached, payload.get("warm_start_values"))
    result = oracle.solve(
        payload["scenarios"],
        weights=payload["weights"],
        fixed_design=payload["fixed_design"],
        allow_carbon_slack=payload["allow_carbon_slack"],
        use_cache=payload["use_cache"],
    )
    # solve() performs exactly one solve here, so the cache holds at most the
    # single entry belonging to this scenario set.
    computed = not warm_start_cached and bool(oracle._warm_start_cache)
    response = {
        "result": result,
        "warm_start_values": (
            next(iter(oracle._warm_start_cache.values())) if computed else None
        ),
        "warm_start_computed": computed,
    }
    args.output.write_bytes(pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    main()
