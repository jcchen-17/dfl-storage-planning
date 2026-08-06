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
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=None,
        help=(
            "MB for this solve, overriding solver_memory_limit_mb. That value is "
            "sized for solver_max_parallel_workers solves at once during "
            "training, while evaluation runs one solve over the larger "
            "final_validation_size scenario set. A solve that stops on memlimit "
            "reports an unproven incumbent, so the objective it prints is not "
            "comparable with a converged one."
        ),
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=None,
        help=(
            "Test scenarios to evaluate on, overriding final_validation_size. "
            "That value is also used by the end-of-training finalist comparison, "
            "which runs under the training memory budget, so it often has to stay "
            "small; this reported number does not."
        ),
    )
    args = parser.parse_args()
    print(f"Starting evaluation with {args.config}...", flush=True)
    result = evaluate_stage(
        args.config,
        method_override=args.method,
        generator_override=args.generator,
        memory_limit_mb=args.memory_limit,
        scenarios=args.scenarios,
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
    reference = result["no_storage_reference"]
    print(f"no-storage reference: {reference['objective']:.2f}")
    if result["storage_value"] is not None:
        print(f"storage value: {result['storage_value']:.2f}")
    # An objective from a solve that stopped early is an incumbent nobody bounded,
    # so it must not be compared with a converged baseline without saying so.
    for label, entry in (
        ("planning", planning),
        ("out-of-sample", validation),
        ("no-storage reference", reference),
    ):
        if entry["status"] != "optimal":
            print(
                f"WARNING: the {label} solve stopped at {entry['status']!r}; its "
                "objective is an unproven incumbent, not a converged optimum."
            )


if __name__ == "__main__":
    main()
