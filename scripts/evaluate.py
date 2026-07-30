import argparse

from storage_dfl.stages import evaluate_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained supports in storage planning.")
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()
    result = evaluate_stage(args.config)
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
