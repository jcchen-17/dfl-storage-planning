import argparse

from storage_dfl.models import GENERATOR_KINDS
from storage_dfl.stages import evaluate_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained supports in storage planning.")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument(
        "--method",
        choices=("reinforce", "scenario_bo"),
        help="Evaluate the checkpoint produced by this DFL method.",
    )
    parser.add_argument(
        "--generator",
        choices=GENERATOR_KINDS,
        help="Evaluate the checkpoint produced on top of this generator.",
    )
    args = parser.parse_args()
    print(f"Starting evaluation with {args.config}...", flush=True)
    result = evaluate_stage(
        args.config,
        method_override=args.method,
        generator_override=args.generator,
    )
    print(f"generator: {result['generator']}")
    print(f"method: {result['method']}")
    planning = result["planning"]
    validation = result["out_of_sample_validation"]
    installed = [bus for bus, value in planning["design"]["site"].items() if value > 0]
    print(f"installed buses: {installed}")
    print(f"power MW: {planning['design']['power_mw']}")
    print(f"energy MWh: {planning['design']['energy_mwh']}")
    print(f"planning objective: {planning['objective']:.2f}")
    print(f"validation objective: {validation['objective']:.2f}")


if __name__ == "__main__":
    main()
