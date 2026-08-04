import argparse

from storage_dfl.models import GENERATOR_KINDS
from storage_dfl.stages import train_generator_stage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the temporal-spatial scenario generator (CVAE, GAN or diffusion)."
    )
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument(
        "--generator",
        choices=GENERATOR_KINDS,
        help="Override generator.kind without editing the YAML file.",
    )
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    print(f"Starting generator training with {args.config}...", flush=True)
    result = train_generator_stage(
        args.config,
        tensorboard=not args.no_tensorboard,
        generator_override=args.generator,
    )
    print(f"generator: {result['generator']}")
    print(f"device: {result['device']}")
    print(f"epochs: {result['epochs']}")
    print(f"latent dim: {result['latent_dim']}")
    print(f"loss: {result['initial_loss']:.6f} -> {result['final_loss']:.6f}")
    print(f"checkpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()
