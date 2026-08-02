from __future__ import annotations

import argparse
from pathlib import Path

from storage_dfl.stages import evaluate_stage, train_cvae_stage, train_dfl_stage


def run(config_path: str | Path, *, tensorboard: bool = True) -> dict:
    cvae = train_cvae_stage(config_path, tensorboard=tensorboard)
    print(f"CVAE complete: loss {cvae['initial_loss']:.6f} -> {cvae['final_loss']:.6f}")
    dfl = train_dfl_stage(config_path, tensorboard=tensorboard)
    print(f"DFL complete: weights {dfl['scenario_weights']}")
    result = evaluate_stage(config_path)
    site = result["planning"]["design"]["site"]
    installed = [bus for bus, value in site.items() if value > 0]
    print("CVAE + direct-DFL storage-planning pipeline completed")
    print(f"installed buses: {installed}")
    print(f"planning objective: {result['planning']['objective']:.2f}")
    print(f"validation objective: {result['out_of_sample_validation']['objective']:.2f}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.yaml", help="YAML configuration path")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.config, tensorboard=not args.no_tensorboard)


if __name__ == "__main__":
    main()
