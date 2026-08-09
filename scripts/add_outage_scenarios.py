"""Add grid outages to an existing v2 scenario file.

`build_dfl_dataset_v2.py` is the right place to generate outages, but it needs
the fused hourly parquet, which is not present in every working copy. This
applies the identical assignment to an already-built npz so the outage study
does not wait on re-fusing the source data. Rebuild from source when it is
available; this script and that path produce the same arrays for the same seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")

import numpy as np

from build_dfl_dataset_v2 import outage_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/ieee13_smartds_dfl_v2/dfl_scenarios_v2_48h.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/ieee13_smartds_dfl_v2_outage/dfl_scenarios_v2_48h.npz"
        ),
    )
    parser.add_argument("--outage-window-fraction", type=float, default=0.25)
    parser.add_argument("--outage-annual-frequency", type=float, default=1.3)
    # US distribution CAIDI is about 2 hours excluding major events, so 1-4
    # brackets the ordinary case. An earlier 2-10 range put 9-hour outages in
    # the pool, and a support scenario holding one drove the battery to
    # max_duration_hours.
    parser.add_argument("--outage-min-hours", type=int, default=1)
    parser.add_argument("--outage-max-hours", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {args.output}; pass --force.")
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("Refusing to overwrite the input in place.")

    with np.load(args.input, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}

    splits = payload["split"].astype(str)
    horizon = int(payload["grid_available"].shape[1])
    if not (payload["grid_available"] == 1.0).all():
        raise SystemExit(
            f"{args.input} already carries outages; add them to the all-available "
            "file instead of stacking two assignments."
        )

    grid_available, outage_hours = outage_assignment(
        splits,
        horizon,
        args.outage_window_fraction,
        args.outage_min_hours,
        args.outage_max_hours,
        args.seed,
    )
    has_outage = outage_hours > 0
    if not has_outage.any():
        raise SystemExit("--outage-window-fraction produced no outages.")

    # Same importance correction as the builder: outages are oversampled so a
    # split contains enough to choose among, and the weights undo it.
    weights = np.asarray(
        [1.0 / np.count_nonzero(splits == split) for split in splits], dtype=np.float32
    )
    true_probability = float(
        1.0 - np.exp(-args.outage_annual_frequency * horizon / 8760.0)
    )
    for split in np.unique(splits):
        in_split = splits == split
        outage = in_split & has_outage
        normal = in_split & ~has_outage
        simulated = outage.sum() / in_split.sum()
        weights[outage] *= true_probability / simulated
        if normal.any():
            weights[normal] *= (1.0 - true_probability) / (1.0 - simulated)

    # How often a year each window is taken to occur. The planning model scales
    # a window's operating cost by this; leaving it at the uniform 8760/horizon
    # priced one sampled outage as 182.5 outages a year, roughly 140 times the
    # rate the SAIFI assumption implies, and the design followed accordingly.
    # A representative scenario stands for its whole class, not for one window,
    # so these are not divided by how many windows the split happens to hold: an
    # outage support represents the ~1.3 outage windows a year, a normal support
    # the remaining ~181.2. A K=1 normal support therefore keeps essentially the
    # original 182.5 scaling, and only the rare class is repriced.
    blocks_per_year = 8760.0 / horizon
    expected_outage_windows = true_probability * blocks_per_year
    occurrences = np.where(
        has_outage,
        expected_outage_windows,
        blocks_per_year - expected_outage_windows,
    ).astype(np.float32)

    payload["grid_available"] = grid_available
    payload["outage_hours"] = outage_hours
    payload["sample_weight"] = weights
    payload["annual_occurrences"] = occurrences
    payload["deterministic_fields"] = np.asarray(
        [f for f in payload["deterministic_fields"].astype(str) if f != "grid_available"]
    )
    payload["stochastic_fields"] = np.asarray(
        list(payload["stochastic_fields"].astype(str)) + ["grid_available"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "derived_from": str(args.input),
        "dataset_sha256": digest,
        "seed": args.seed,
        "outages": {
            "window_fraction_simulated": args.outage_window_fraction,
            "windows_with_outage": int(has_outage.sum()),
            "duration_hours_range": [args.outage_min_hours, args.outage_max_hours],
            "mean_duration_hours": float(outage_hours[has_outage].mean()),
            "assumed_annual_frequency": args.outage_annual_frequency,
            "true_window_probability": true_probability,
            "importance_weight_ratio": float(
                weights[has_outage][0] / weights[~has_outage][0]
            ),
        },
        "known_limitations": [
            "Outage timing is independent of load, PV and carbon.",
            "Nothing in the training or evaluation pipeline reads sample_weight yet.",
        ],
    }
    manifest_path = args.output.parent / "manifest_outage.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    per_split = {
        str(split): int((has_outage & (splits == split)).sum())
        for split in np.unique(splits)
    }
    print(f"wrote {args.output}")
    print(f"outage windows: {int(has_outage.sum())}/{len(has_outage)} {per_split}")
    print(
        f"duration: {outage_hours[has_outage].min()}-{outage_hours[has_outage].max()} h "
        f"(mean {outage_hours[has_outage].mean():.1f})"
    )
    print(
        f"true window probability {true_probability:.4f}; "
        f"outage rows weighted {manifest['outages']['importance_weight_ratio']:.4f}x normal"
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
