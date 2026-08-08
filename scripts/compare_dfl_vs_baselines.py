"""Head-to-head: a multi-seed DFL run against the K=1 sweep baselines.

One DFL run cannot be ranked against these baselines. Random selection at K=1
spans 2,762 to 4,804 on this case, so a single draw from a learned method says
almost nothing about the method -- what has to be compared is the distribution
it produces against the distribution random produces, and against the
deterministic rules, which have no distribution because they have no seed.

Refuses to report unless the runs are comparable. Every value here is
reference minus out-of-sample objective, so a difference is only meaningful
when both sides were measured against the same reference on the same test
subset; that is checked rather than assumed. Rows that did not converge are
reported separately instead of being averaged in, because an unproven
incumbent is not a measurement -- at the looser tolerance this project used
before, values came back 650-1,250 low and two rules swapped places.

    python scripts/compare_dfl_vs_baselines.py
    python scripts/compare_dfl_vs_baselines.py --dfl outputs/dfl_seeds_kmeans.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOLERANCE = 1.0  # dollars; references must agree to this


def load_baselines(path: Path, k: int) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing sweep csv: {path}")
    with open(path, newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if int(r["k"]) == k]
    if not rows:
        raise SystemExit(f"no rows with k={k} in {path}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sweep", default="outputs/k_sweep/k_sweep.csv")
    ap.add_argument("--dfl", default="outputs/dfl_seeds_farthest.json")
    ap.add_argument("--ceiling", default=None,
                    help="a headroom_probe json whose ceiling_interval to quote")
    ap.add_argument("--k", type=int, default=1)
    args = ap.parse_args()

    baselines = load_baselines(REPO / args.sweep, args.k)
    dfl_path = REPO / args.dfl
    if not dfl_path.exists():
        raise SystemExit(f"missing DFL summary: {dfl_path}")
    dfl_rows = json.loads(dfl_path.read_text())

    # One reference, or the differences are not differences of the same thing.
    references = {round(float(r["no_storage_objective"]), 2) for r in baselines}
    if len(references) != 1:
        raise SystemExit(
            f"the sweep rows carry {len(references)} no-storage references "
            f"({sorted(references)}); they were not all evaluated on one test "
            "subset and their values cannot be compared."
        )
    reference = references.pop()

    ok = [r for r in dfl_rows if r.get("status") == "ok"]
    failed = [r for r in dfl_rows if r.get("status") != "ok"]
    if not ok:
        raise SystemExit(f"no completed DFL runs in {dfl_path}")
    for row in ok:
        implied = float(row["out_of_sample_objective"]) + float(row["storage_value"])
        if abs(implied - reference) > TOLERANCE:
            raise SystemExit(
                f"DFL seed {row['seed']} was measured against a no-storage "
                f"reference of {implied:,.2f} but the sweep used "
                f"{reference:,.2f}. Different test subsets or settings; the "
                "comparison would be meaningless."
            )

    print(f"no-storage reference : {reference:,.2f}  (identical across "
          f"{len(baselines)} sweep rows and {len(ok)} DFL runs)")
    if args.ceiling:
        payload = json.loads((REPO / args.ceiling).read_text())
        interval = payload.get("ceiling_interval")
        if interval:
            lo, hi = interval
            print(f"storage-value ceiling: [{lo:,.0f}, {hi:,.0f}]")
    print()

    random_rows = [r for r in baselines if r["rule"] == "random"]
    deterministic = [r for r in baselines if r["rule"] != "random"]

    print(f"{'method':<24}{'n':>4}{'mean':>10}{'sd':>9}{'min':>10}{'max':>10}")
    print("-" * 67)
    values = [float(r["storage_value"]) for r in ok]
    mean = st.mean(values)
    sd = st.stdev(values) if len(values) > 1 else 0.0
    print(f"{'DFL (' + str(ok[0].get('rule', '?')) + ' init)':<24}{len(values):>4}"
          f"{mean:>10,.0f}{sd:>9,.0f}{min(values):>10,.0f}{max(values):>10,.0f}")
    if random_rows:
        rv = [float(r["storage_value"]) for r in random_rows]
        rsd = st.stdev(rv) if len(rv) > 1 else 0.0
        print(f"{'random':<24}{len(rv):>4}{st.mean(rv):>10,.0f}{rsd:>9,.0f}"
              f"{min(rv):>10,.0f}{max(rv):>10,.0f}")
    for row in sorted(deterministic, key=lambda r: -float(r["storage_value"])):
        v = float(row["storage_value"])
        print(f"{row['rule']:<24}{1:>4}{v:>10,.0f}{'':>9}{v:>10,.0f}{v:>10,.0f}")

    # The bar is the best rule that needs no learning, not the random mean.
    best_rule = max(deterministic, key=lambda r: float(r["storage_value"]))
    bar = float(best_rule["storage_value"])
    print(f"\nthe bar is {best_rule['rule']} at {bar:,.0f} -- the best rule that "
          "needs no learning.")
    beat = sum(1 for v in values if v > bar)
    print(f"DFL beats it in {beat} of {len(values)} runs "
          f"({100 * beat / len(values):.0f}%), mean margin {mean - bar:+,.0f}")
    if random_rows:
        rv = [float(r["storage_value"]) for r in random_rows]
        rmean = st.mean(rv)
        rsd = st.stdev(rv) if len(rv) > 1 else 0.0
        print(f"vs the random mean {rmean:,.0f}: {mean - rmean:+,.0f}")
        if sd and rsd:
            print(f"spread: DFL sd {sd:,.0f} vs random sd {rsd:,.0f} "
                  f"({'more' if sd > rsd else 'less'} variable)")

    print(f"\n{'seed':<6}{'value':>10}{'E MWh':>9}{'installed':>12}  support")
    for row in sorted(ok, key=lambda r: r["seed"]):
        print(f"{row['seed']:<6}{float(row['storage_value']):>10,.0f}"
              f"{float(row['energy_mwh']):>9.3f}"
              f"{','.join(row['installed']):>12}  "
              f"{','.join(row.get('support_source_names', []))}")
    if failed:
        print(f"\n{len(failed)} runs did not complete:")
        for row in failed:
            print(f"  seed {row['seed']}: {row['status']}")
    unconverged = [r for r in ok if r.get("out_of_sample_status") != "optimal"]
    if unconverged:
        print(f"\nWARNING: {len(unconverged)} completed runs have a "
              "non-optimal out-of-sample solve; their values are unproven "
              "incumbents and are included in the statistics above:")
        for row in unconverged:
            print(f"  seed {row['seed']}: {row['out_of_sample_status']}")


if __name__ == "__main__":
    main()
