"""Build horizon-specific DFL windows from the fused hourly Parquet table.

The source table is not modified. Windows never cross calendar-year/split
boundaries and start once per day, matching the historical 120-hour dataset.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


BUSES = ("650", "632", "633", "634", "645", "646", "671", "680", "684", "611", "652", "692", "675")
PHASES = ("A", "B", "C")
VECTOR_FIELDS = (
    "workload_arrival",
    "pue",
    "grid_price_per_mwh",
    "grid_carbon_t_per_mwh",
    "grid_available",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut the fused hourly IEEE13 data into daily-stride windows."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/ieee13_smartds_dfl/dfl_hourly_wide.parquet"),
    )
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon <= 0 or args.stride <= 0:
        raise ValueError("horizon and stride must be positive")
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.parent / f"dfl_training_windows_{args.horizon}h.npz"
    )
    if input_path == output_path:
        raise ValueError("Output must differ from the hourly source table.")

    table = pq.read_table(input_path)

    def column(name: str) -> np.ndarray:
        return np.asarray(table[name].combine_chunks().to_numpy(zero_copy_only=False))

    years = column("year").astype(int)
    splits = column("split").astype(str)
    day_of_year = column("day_of_year").astype(float)
    weekend = column("is_weekend").astype(float)
    temperature = column("temperature_c").astype(float)

    starts: list[int] = []
    window_splits: list[str] = []
    names: list[str] = []
    for year in np.unique(years):
        indices = np.flatnonzero(years == year)
        if indices.size < args.horizon or np.any(np.diff(indices) != 1):
            raise ValueError(f"Year {year} is not a sufficiently long contiguous block.")
        for local_start in range(0, indices.size - args.horizon + 1, args.stride):
            start = int(indices[0] + local_start)
            stop = start + args.horizon
            split = str(splits[start])
            if np.any(splits[start:stop] != split):
                raise ValueError("A proposed window crosses a dataset split boundary.")
            starts.append(start)
            window_splits.append(split)
            names.append(f"{year}_{local_start // 24:03d}_{args.horizon}h")
    starts_array = np.asarray(starts, dtype=np.int32)

    def windows(values: np.ndarray) -> np.ndarray:
        return np.stack(
            [values[start : start + args.horizon] for start in starts_array]
        ).astype(np.float32)

    bus_p = np.stack([column(f"active_load_mw__{bus}") for bus in BUSES], axis=1)
    bus_q = np.stack([column(f"reactive_load_mvar__{bus}") for bus in BUSES], axis=1)
    bus_pv = np.stack([column(f"pv_available_mw__{bus}") for bus in BUSES], axis=1)

    def phase_tensor(prefix: str) -> np.ndarray:
        return np.stack(
            [column(f"{prefix}__{bus}__{phase}") for bus in BUSES for phase in PHASES],
            axis=1,
        ).reshape(-1, len(BUSES), len(PHASES))

    context = np.empty((len(starts), 4), dtype=np.float32)
    for index, start in enumerate(starts):
        stop = start + args.horizon
        angle = 2.0 * math.pi * (day_of_year[start] - 1.0) / 365.0
        context[index] = (
            math.sin(angle),
            math.cos(angle),
            float(np.mean(weekend[start:stop])),
            float(np.mean(temperature[start:stop])),
        )

    payload = {
        "scenario_name": np.asarray(names),
        "split": np.asarray(window_splits),
        "context": context,
        "active_load_mw": windows(bus_p),
        "reactive_load_mvar": windows(bus_q),
        "pv_available_mw": windows(bus_pv),
        "active_load_phase_mw": windows(phase_tensor("active_load_mw")),
        "reactive_load_phase_mvar": windows(phase_tensor("reactive_load_mvar")),
        "pv_available_phase_mw": windows(phase_tensor("pv_available_mw")),
        "buses": np.asarray(BUSES),
        "phases": np.asarray(PHASES),
        "context_names": np.asarray(
            ["start_doy_sin", "start_doy_cos", "weekend_fraction", "mean_temperature_c"]
        ),
    }
    payload.update({field: windows(column(field)) for field in VECTOR_FIELDS})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    split_counts = {
        split: int(np.count_nonzero(np.asarray(window_splits) == split))
        for split in np.unique(window_splits)
    }
    print(f"hourly source: {input_path}")
    print(f"horizon/stride: {args.horizon}/{args.stride} h")
    print(f"windows: {len(starts)} {split_counts}")
    print(f"context shape: {context.shape}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
