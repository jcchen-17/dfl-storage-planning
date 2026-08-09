from __future__ import annotations

import argparse
from pathlib import Path

from storage_dfl.models import GENERATOR_KINDS
from storage_dfl.stages import evaluate_stage, train_dfl_stage, train_generator_stage


def run(
    config_path: str | Path,
    *,
    tensorboard: bool = True,
    generator: str | None = None,
) -> dict:
    trained = train_generator_stage(
        config_path, tensorboard=tensorboard, generator_override=generator
    )
    print(
        f"{trained['generator']} complete: "
        f"loss {trained['initial_loss']:.6f} -> {trained['final_loss']:.6f}"
    )
    dfl = train_dfl_stage(
        config_path, tensorboard=tensorboard, generator_override=generator
    )
    print(f"DFL complete: weights {dfl['scenario_weights']}")
    result = evaluate_stage(config_path, generator_override=generator)
    site = result["planning"]["design"]["site"]
    installed = [bus for bus, value in site.items() if value > 0]
    print(f"{trained['generator']} + {dfl['method']} pipeline completed")
    print(f"installed buses: {installed}")
    print(f"planning objective: {result['planning']['objective']:.2f}")
    print(f"validation objective: {result['out_of_sample_validation']['objective']:.2f}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/dataset_v2_dfl_hourly_layered.yaml",
        help="YAML configuration path",
    )
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument(
        "--generator",
        choices=GENERATOR_KINDS,
        help="Override generator.kind for every stage of the run",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.config, tensorboard=not args.no_tensorboard, generator=args.generator)


if __name__ == "__main__":
    main()
