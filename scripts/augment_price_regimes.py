"""Create tariff-regime scenarios without overwriting the source dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_factors(raw: str) -> np.ndarray:
    factors = np.asarray([float(item.strip()) for item in raw.split(",") if item.strip()])
    if factors.size == 0 or np.any(factors <= 0.0):
        raise ValueError("Price-spread factors must be positive.")
    if np.unique(factors).size != factors.size:
        raise ValueError("Price-spread factors must be unique.")
    return factors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand each historical window into several mean-preserving tariff regimes."
    )
    parser.add_argument(
        "--input",
        default="data/processed/ieee13_smartds_dfl/dfl_training_windows_48h.npz",
    )
    parser.add_argument(
        "--output",
        default=(
            "data/processed/ieee13_smartds_dfl/"
            "dfl_training_windows_48h_price_regimes.npz"
        ),
    )
    parser.add_argument("--factors", default="1.0,1.25,1.5,1.75,2.0")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if input_path == output_path:
        raise ValueError("Output must differ from input; the source dataset is never overwritten.")
    factors = _parse_factors(args.factors)

    with np.load(input_path, allow_pickle=False) as source:
        payload = {key: np.asarray(source[key]) for key in source.files}
    prices = np.asarray(payload["grid_price_per_mwh"], dtype=np.float64)
    contexts = np.asarray(payload["context"], dtype=np.float32)
    names = np.asarray(payload["scenario_name"]).astype(str)
    sample_count = prices.shape[0]
    if contexts.shape[0] != sample_count or names.shape[0] != sample_count:
        raise ValueError("Dataset scenario arrays have inconsistent first dimensions.")

    expanded: dict[str, np.ndarray] = {}
    for key, values in payload.items():
        if key in {"context", "context_names", "grid_price_per_mwh", "scenario_name"}:
            continue
        expanded[key] = (
            np.concatenate([values] * len(factors), axis=0)
            if values.ndim > 0 and values.shape[0] == sample_count
            else values
        )

    price_blocks = []
    context_blocks = []
    name_blocks = []
    regime_blocks = []
    means = prices.mean(axis=1, keepdims=True)
    for factor in factors:
        adjusted = means + factor * (prices - means)
        if np.any(adjusted < 0.0):
            raise ValueError(
                f"Factor {factor:g} creates a negative price; choose a smaller factor."
            )
        price_blocks.append(adjusted.astype(payload["grid_price_per_mwh"].dtype))
        context_blocks.append(
            np.concatenate(
                [contexts, np.full((sample_count, 1), factor, dtype=np.float32)],
                axis=1,
            )
        )
        suffix = str(factor).replace(".", "p")
        name_blocks.append(np.char.add(names, f"_price_{suffix}x"))
        regime_blocks.append(np.full(sample_count, factor, dtype=np.float32))

    expanded["grid_price_per_mwh"] = np.concatenate(price_blocks, axis=0)
    expanded["context"] = np.concatenate(context_blocks, axis=0)
    expanded["scenario_name"] = np.concatenate(name_blocks, axis=0)
    expanded["price_spread_factor"] = np.concatenate(regime_blocks, axis=0)
    context_names = np.asarray(payload.get("context_names", []), dtype=str)
    expanded["context_names"] = np.concatenate(
        [context_names, np.asarray(["price_spread_factor"])]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **expanded)
    original_spreads = np.ptp(prices, axis=1)
    augmented_spreads = np.ptp(expanded["grid_price_per_mwh"], axis=1)
    print(f"source scenarios: {sample_count}")
    print(f"augmented scenarios: {expanded['scenario_name'].shape[0]}")
    print(f"price-spread factors: {factors.tolist()}")
    print(
        f"price-spread range: {original_spreads.min():.3f}-{original_spreads.max():.3f} "
        f"-> {augmented_spreads.min():.3f}-{augmented_spreads.max():.3f} $/MWh"
    )
    print(f"context dimension: {contexts.shape[1]} -> {expanded['context'].shape[1]}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
