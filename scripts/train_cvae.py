import argparse

from storage_dfl.stages import train_cvae_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the temporal-spatial CVAE.")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    result = train_cvae_stage(args.config, tensorboard=not args.no_tensorboard)
    print(f"device: {result['device']}")
    print(f"epochs: {result['epochs']}")
    print(f"loss: {result['initial_loss']:.6f} -> {result['final_loss']:.6f}")
    print(f"checkpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()
