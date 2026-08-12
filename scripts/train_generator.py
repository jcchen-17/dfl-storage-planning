import argparse
from pathlib import Path

from storage_dfl.stages import train_generator_stage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the current v2 CVAE scenario generator."
    )
    parser.add_argument("--config", default="configs/dataset_v2_dfl_hourly_layered_t1.yaml")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="resume weights and Adam state from a generator checkpoint",
    )
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=None,
        help="number of extra full-data epochs when resuming",
    )
    args = parser.parse_args()
    print(f"Starting generator training with {args.config}...", flush=True)
    result = train_generator_stage(
        args.config,
        tensorboard=not args.no_tensorboard,
        resume_checkpoint=args.resume_checkpoint,
        additional_epochs=args.additional_epochs,
    )
    print(f"generator: {result['generator']}")
    print(f"device: {result['device']}")
    print(f"epochs: {result['epochs']}")
    print(f"latent dim: {result['latent_dim']}")
    print(f"loss: {result['initial_loss']:.6f} -> {result['final_loss']:.6f}")
    print(f"checkpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()
