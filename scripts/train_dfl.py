import argparse

from storage_dfl.stages import train_dfl_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Train direct DFL scenario supports.")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    result = train_dfl_stage(args.config, tensorboard=not args.no_tensorboard)
    print(f"device: {result['device']}")
    print(f"epochs: {result['epochs']}")
    print(f"scenario weights: {result['scenario_weights']}")
    print(f"checkpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()
