import argparse

from storage_dfl.stages import train_dfl_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Train decision-focused scenario supports.")
    parser.add_argument("--config", default="configs/dataset_v2_dfl_hourly_layered_t1.yaml")
    parser.add_argument("--no-swanlab", action="store_true")
    parser.add_argument(
        "--carbon-formulation",
        choices=("layered_pcc", "average_pcc"),
        default=None,
        help="override only the PCC carbon-accounting formulation",
    )
    parser.add_argument(
        "--carbon-cap-scope",
        choices=("hourly", "horizon"),
        default=None,
        help="override only the temporal scope of the carbon constraint",
    )
    parser.add_argument(
        "--method",
        choices=("recourse_feasibility",),
        default=None,
        help="override dfl.method while retaining the same generator and MILP",
    )
    parser.add_argument(
        "--capex-scale",
        type=float,
        default=1.0,
        help="scale site, power and energy investment costs together",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "optional unique artifact name; by default a timestamp and process "
            "ID are used so concurrent runs cannot overwrite one another"
        ),
    )
    args = parser.parse_args()
    print(f"Starting DFL training with {args.config}...", flush=True)
    result = train_dfl_stage(
        args.config,
        swanlab_logging=not args.no_swanlab,
        method_override=args.method,
        battery_capex_scale=args.capex_scale,
        carbon_formulation_override=args.carbon_formulation,
        carbon_cap_scope_override=args.carbon_cap_scope,
        run_id=args.run_id,
    )
    print(f"generator: {result['generator']}")
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
    print(f"run id: {result['run_id']}")


if __name__ == "__main__":
    main()
