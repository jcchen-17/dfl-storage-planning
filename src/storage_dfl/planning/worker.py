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
    result = oracle.solve(
        payload["scenarios"],
        weights=payload["weights"],
        fixed_design=payload["fixed_design"],
        allow_carbon_slack=payload["allow_carbon_slack"],
        use_cache=payload["use_cache"],
    )
    args.output.write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    main()
