import argparse

from storage_dfl.stages import evaluate_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained supports in storage planning.")
    parser.add_argument("--config", default="configs/dataset_v2_dfl_hourly_layered.yaml")
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
        "--evaluation-carbon-formulation",
        choices=("layered_pcc", "average_pcc"),
        default=None,
        help=(
            "re-evaluate the saved design under a common carbon ledger; "
            "checkpoint selection still uses --carbon-formulation"
        ),
    )
    parser.add_argument(
        "--method",
        choices=("recourse_feasibility",),
        default=None,
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
            "Each test scenario is solved as a separate exact model and the "
            "results are aggregated, avoiding a multi-scenario memory spike."
        ),
    )
    args = parser.parse_args()
    print(f"Starting evaluation with {args.config}...", flush=True)
    result = evaluate_stage(
        args.config,
        method_override=args.method,
        memory_limit_mb=args.memory_limit,
        scenarios=args.scenarios,
        carbon_formulation_override=args.carbon_formulation,
        carbon_cap_scope_override=args.carbon_cap_scope,
        evaluation_carbon_formulation_override=(
            args.evaluation_carbon_formulation
        ),
    )
    print(f"generator: {result['generator']}")
    print(f"method: {result['method']}")
    print(f"trained carbon formulation: {result['trained_carbon_formulation']}")
    print(f"evaluation carbon formulation: {result['evaluation_carbon_formulation']}")
    planning = result["planning"]
    validation = result["out_of_sample_validation"]
    evaluated_design = validation["design"]
    installed = [
        bus for bus, value in evaluated_design["site"].items() if value > 0
    ]
    print(f"evaluated design source: {result['evaluated_design_source']}")
    print(f"installed buses: {installed}")
    print(f"power MW: {evaluated_design['power_mw']}")
    print(f"energy MWh: {evaluated_design['energy_mwh']}")
    print(f"planning objective: {planning['objective']:.2f}")
    print(f"validation objective: {validation['objective']:.2f}")
    reference = result["no_storage_reference"]
    print(f"no-storage reference: {reference['objective']:.2f}")
    if result["storage_value"] is not None:
        print(f"storage value: {result['storage_value']:.2f}")
    if result.get("decision_regret") is not None:
        print(f"normalized decision regret: {result['decision_regret']:.6f}")
    diagnostics = validation.get("recourse_diagnostics")
    if diagnostics is not None:
        print(f"load shedding MWh: {diagnostics['load_shedding_mwh']:.6f}")
        print(f"carbon excess tCO2: {diagnostics['carbon_excess_t']:.6f}")
        print(f"PV curtailment MWh: {diagnostics['pv_curtailment_mwh']:.6f}")
        print(f"served demand MWh: {diagnostics['served_demand_mwh']:.6f}")
        print(f"recourse operating cost: {diagnostics['operating_cost']:.2f}")
        print(
            f"recourse total planning cost: "
            f"{diagnostics['total_planning_cost']:.2f}"
        )
    accepted_gap = result.get("evaluation_relative_gap", 0.0)
    # Time/memory-limit incumbents are comparable when the solver has already
    # certified the requested relative gap; status text alone is insufficient.
    for label, entry in (
        ("planning", planning),
        ("out-of-sample", validation),
        ("no-storage reference", reference),
    ):
        bounded = (
            entry["status"] in {"optimal", "gaplimit"}
            or (
                accepted_gap > 0.0
                and entry["relative_gap"] <= accepted_gap
                and entry["objective"] != float("inf")
            )
        )
        if not bounded:
            print(
                f"WARNING: the {label} solve stopped at {entry['status']!r}; its "
                "objective is an unproven incumbent, not a converged optimum."
            )


if __name__ == "__main__":
    main()
