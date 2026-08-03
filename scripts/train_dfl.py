import argparse

from storage_dfl.stages import train_dfl_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Train decision-focused scenario supports.")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument(
        "--method",
        choices=("reinforce", "scenario_bo"),
        help="Override dfl.method without editing the YAML file.",
    )
    parser.add_argument("--no-tensorboard", action="store_true")
    args = parser.parse_args()
    print(f"Starting DFL training with {args.config}...", flush=True)
    result = train_dfl_stage(
        args.config,
        tensorboard=not args.no_tensorboard,
        method_override=args.method,
    )
    print(f"method: {result['method']}")
    print(f"device: {result['device']}")
    print(f"epochs: {result['epochs']}")
    print(f"scenario weights: {result['scenario_weights']}")
    planning = result["planning"]
    installed = [
        bus for bus, value in planning["design"]["site"].items() if value > 0
    ]
    total_power = sum(
        planning["design"]["power_mw"][bus] for bus in installed
    )
    total_energy = sum(
        planning["design"]["energy_mwh"][bus] for bus in installed
    )
    print(
        f"final design: installed={installed}, P={total_power:.4f} MW, "
        f"E={total_energy:.4f} MWh"
    )
    print(f"checkpoint: {result['checkpoint']}")


if __name__ == "__main__":
    main()
