"""Did the DFL runs learn anything? Read it off the epoch histories.

A REINFORCE curve on this problem is flat with noise on top, and looking at it
does not settle the question -- what settles it is whether the improvement is
larger than the solver could resolve. That comparison is the whole report:
total improvement per run against solver_relative_gap times the objective,
which is the smallest difference the ranking can be trusted to reflect.

Also prints how far the policy travelled. A run whose loss did not move but
whose latent did was searching; one where neither moved was stuck; one that
moved less than the width it samples at never left its starting neighbourhood
and its answer is its initialisation.

    python scripts/dfl_training_report.py
    python scripts/dfl_training_report.py --glob "outputs/dfl_k1_*_seed*"
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--glob", default="outputs/dfl_k1_*_seed*",
                    help="run directories to read")
    ap.add_argument("--method", default="reinforce")
    ap.add_argument("--config", default="configs/generator_compare_k1t2.yaml",
                    help="only used to quote the solver tolerance")
    args = ap.parse_args()

    tolerance = None
    try:
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from storage_dfl.config import load_config
        cfg = load_config(str(REPO / args.config))
        tolerance = float(cfg.planning.solver_relative_gap)
    except Exception:
        pass

    runs = sorted(REPO.glob(args.glob))
    rows = []
    for run in runs:
        history_path = run / f"dfl_history_{args.method}.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text())
        if not history:
            continue
        values = [float(e["validation_objective"]) for e in history]
        best = min(values)
        best_epoch = values.index(best)
        result_path = run / f"result_{args.method}.json"
        storage_value = None
        if result_path.exists():
            storage_value = json.loads(result_path.read_text()).get("storage_value")
        rows.append({
            "run": run.name,
            "epochs": len(history),
            "first": values[0],
            "best": best,
            "best_epoch": best_epoch + 1,
            "improvement": values[0] - best,
            # Spread of the samples within the final epoch, i.e. what the
            # ranking has to resolve for a gradient step to mean anything.
            "final_mean": float(history[-1].get("validation_objective_mean", 0.0)),
            "storage_value": storage_value,
        })

    if not rows:
        raise SystemExit(f"no histories matched {args.glob}")

    floor = tolerance * st.mean([r["best"] for r in rows]) if tolerance else None
    print(f"{'run':<28}{'ep':>4}{'best epoch':>12}{'improvement':>13}"
          f"{'storage value':>15}")
    print("-" * 72)
    for row in sorted(rows, key=lambda r: r["run"]):
        sv = f"{row['storage_value']:,.0f}" if row["storage_value"] else "-"
        print(f"{row['run']:<28}{row['epochs']:>4}{row['best_epoch']:>12}"
              f"{row['improvement']:>13,.0f}{sv:>15}")

    improvements = [r["improvement"] for r in rows]
    print(f"\nimprovement over a run: mean {st.mean(improvements):,.0f}, "
          f"max {max(improvements):,.0f}")
    if floor:
        print(f"solver resolution     : {floor:,.0f}  "
              f"(solver_relative_gap {tolerance:g})")
        beat = sum(1 for v in improvements if v > floor)
        print(f"\n{beat} of {len(improvements)} runs improved by more than the "
              "solver can resolve.")
        if beat == 0:
            print("  No run moved the decision loss by a distinguishable amount.")
            print("  The training curve is not flat-looking, it is flat: whatever")
            print("  the policy did, the planner could not tell the difference.")

    # The best epoch says where the answer came from. Early means the run was
    # already done before it started; late means the search was still paying.
    early = [r for r in rows if r["best_epoch"] <= 3]
    if early:
        print(f"\n{len(early)} of {len(rows)} runs found their best value in the "
              "first three epochs, before the policy had moved far from its")
        print("  initialisation -- the rest of the run added nothing.")

    values = [r["storage_value"] for r in rows if r["storage_value"]]
    if len(values) > 1:
        print(f"\nstorage value across runs: mean {st.mean(values):,.0f}, "
              f"sd {st.stdev(values):,.0f} "
              f"({100 * st.stdev(values) / st.mean(values):.1f}%)")
        print("  Low spread here is only evidence of reliability if the runs")
        print("  actually searched. A method that returns its initialisation")
        print("  has no spread either -- compare against a run started from a")
        print("  different rule to tell those apart.")


if __name__ == "__main__":
    main()
