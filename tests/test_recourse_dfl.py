from dataclasses import replace

import numpy as np
import torch

from storage_dfl.config import load_config
from storage_dfl.data import (
    ScenarioCodec,
    ScenarioPool,
    TorchPhysicalTrajectories,
    make_toy_scenarios,
)
from storage_dfl.dfl.decision_loss import RecourseFeasibilityLoss
from storage_dfl.dfl.recourse_trainer import (
    _validation_checkpoint_is_better,
    _validation_checkpoint_score,
    train_recourse_feasibility_cvae,
)
from storage_dfl.models import ConditionalVAE
from storage_dfl.network import single_pcc_microgrid
from storage_dfl.planning import (
    PlanningResult,
    RecourseDiagnostics,
    SinglePCCPlanningOracle,
    StorageDesign,
    evaluate_fixed_design_recourse,
)
from storage_dfl.planning.results import scenario_content_key


def _result(
    *,
    shed=(0.0, 0.0),
    carbon=(0.0, 0.0),
    served=1.0,
) -> PlanningResult:
    design = StorageDesign(
        site={"PCC": 1}, power_mw={"PCC": 0.5}, energy_mwh={"PCC": 1.0}
    )
    diagnostics = RecourseDiagnostics(
        load_shedding_mwh=float(sum(shed)),
        carbon_excess_t=float(sum(carbon)),
        pv_curtailment_mwh=0.0,
        served_demand_mwh=served,
        operating_cost=10.0,
        total_planning_cost=20.0,
        hourly_load_shedding_mw=(tuple(shed),),
        hourly_carbon_excess_t_per_hour=(tuple(carbon),),
    )
    return PlanningResult(
        status="optimal",
        objective=20.0,
        investment_cost=10.0,
        operating_cost=10.0,
        carbon_slack_cost=0.0,
        peak_grid_mw=1.0,
        design=design,
        scenario_names=("real",),
        solve_time_seconds=0.0,
        relative_gap=0.0,
        recourse_diagnostics=diagnostics,
    )


def _physical(load: float, pv: float, carbon: float) -> TorchPhysicalTrajectories:
    active = torch.full((1, 2, 1, 3), load / 3.0)
    solar = torch.full((1, 2, 1, 3), pv / 3.0)
    intensity = torch.full((1, 2), carbon)
    return TorchPhysicalTrajectories(active, solar, intensity)


def test_directional_feasibility_losses_increase() -> None:
    truth = _physical(load=1.0, pv=0.2, carbon=0.6)
    shedding_result = _result(shed=(0.2, 0.2))
    carbon_result = _result(carbon=(0.1, 0.1))

    load_loss = RecourseFeasibilityLoss(
        load_weight=1.0, pv_weight=0.0, carbon_weight=0.0,
        carbon_cap_t_per_mwh=0.28,
    )
    assert load_loss(
        _physical(0.5, 0.2, 0.6), truth, shedding_result, [shedding_result]
    ).loss > load_loss(truth, truth, shedding_result, [shedding_result]).loss

    pv_loss = RecourseFeasibilityLoss(
        load_weight=0.0, pv_weight=1.0, carbon_weight=0.0,
        carbon_cap_t_per_mwh=0.28,
    )
    assert pv_loss(
        _physical(1.0, 0.8, 0.6), truth, shedding_result, [shedding_result]
    ).loss > pv_loss(truth, truth, shedding_result, [shedding_result]).loss

    carbon_loss = RecourseFeasibilityLoss(
        load_weight=0.0, pv_weight=0.0, carbon_weight=1.0,
        carbon_cap_t_per_mwh=0.28,
    )
    assert carbon_loss(
        _physical(1.0, 0.2, 0.2), truth, carbon_result, [carbon_result]
    ).loss > carbon_loss(truth, truth, carbon_result, [carbon_result]).loss


def test_directional_feasibility_gradient_signs() -> None:
    truth = _physical(load=1.0, pv=0.2, carbon=0.6)
    result = _result(shed=(0.2, 0.2), carbon=(0.1, 0.1))
    generated = TorchPhysicalTrajectories(
        active_load_mw=torch.full((1, 2, 1, 3), 0.5 / 3.0, requires_grad=True),
        pv_available_mw=torch.full((1, 2, 1, 3), 0.8 / 3.0, requires_grad=True),
        grid_carbon_t_per_mwh=torch.full((1, 2), 0.2, requires_grad=True),
    )
    output = RecourseFeasibilityLoss(
        load_weight=1.0,
        pv_weight=1.0,
        carbon_weight=1.0,
        carbon_cap_t_per_mwh=0.28,
    )(generated, truth, result, [result])
    output.loss.backward()
    # Gradient descent increases underestimated load/carbon and decreases
    # overestimated PV.
    assert torch.all(generated.active_load_mw.grad < 0.0)
    assert torch.all(generated.pv_available_mw.grad > 0.0)
    assert torch.all(generated.grid_carbon_t_per_mwh.grad < 0.0)


def test_horizon_carbon_excess_uses_average_intensity() -> None:
    truth = _physical(load=1.0, pv=0.2, carbon=0.6)
    result = _result(carbon=(0.2,), served=2.0)
    generated = TorchPhysicalTrajectories(
        active_load_mw=torch.full((1, 2, 1, 3), 1.0 / 3.0),
        pv_available_mw=torch.full((1, 2, 1, 3), 0.2 / 3.0),
        grid_carbon_t_per_mwh=torch.full((1, 2), 0.2, requires_grad=True),
    )
    output = RecourseFeasibilityLoss(
        load_weight=0.0,
        pv_weight=0.0,
        carbon_weight=1.0,
        carbon_cap_t_per_mwh=0.28,
    )(generated, truth, result, [result])
    output.loss.backward()
    assert output.carbon_underestimation > 0.0
    assert torch.all(generated.grid_carbon_t_per_mwh.grad < 0.0)


def test_soft_constraint_checkpoint_selection_prioritizes_regret() -> None:
    economically_better = _validation_checkpoint_score(0.20, 0.02, 0.01, 0.28)
    physically_better = _validation_checkpoint_score(0.30, 0.00, 0.00, 0.28)
    assert economically_better < physically_better


def test_carbon_first_checkpoint_selection_respects_reliability_guardrail() -> None:
    reliable_low_carbon = _validation_checkpoint_score(
        0.30, 0.00009, 0.001, 0.28,
        selection="carbon_first", shedding_tolerance=0.0001,
    )
    unreliable_zero_carbon = _validation_checkpoint_score(
        0.10, 0.00011, 0.0, 0.28,
        selection="carbon_first", shedding_tolerance=0.0001,
    )
    assert reliable_low_carbon < unreliable_zero_carbon


def test_carbon_tolerance_uses_regret_without_ratcheting() -> None:
    incumbent = _validation_checkpoint_score(
        0.020, 0.00005, 0.01200, 0.28,
        selection="carbon_first", shedding_tolerance=0.0001,
    )
    near_candidate = _validation_checkpoint_score(
        0.015, 0.00005, 0.01204, 0.28,
        selection="carbon_first", shedding_tolerance=0.0001,
    )
    better, floor = _validation_checkpoint_is_better(
        selection="carbon_first",
        candidate_score=near_candidate,
        incumbent_score=incumbent,
        candidate_shed=0.00005,
        candidate_carbon=0.01204,
        candidate_regret=0.015,
        incumbent_shed=0.00005,
        incumbent_carbon=0.01200,
        incumbent_regret=0.020,
        shedding_tolerance=0.0001,
        carbon_tolerance=0.00005,
        reliable_carbon_floor=0.01200,
    )
    assert better
    assert np.isclose(floor, 0.01200)

    outside_candidate = _validation_checkpoint_score(
        0.001, 0.00005, 0.01206, 0.28,
        selection="carbon_first", shedding_tolerance=0.0001,
    )
    better, floor = _validation_checkpoint_is_better(
        selection="carbon_first",
        candidate_score=outside_candidate,
        incumbent_score=near_candidate,
        candidate_shed=0.00005,
        candidate_carbon=0.01206,
        candidate_regret=0.001,
        incumbent_shed=0.00005,
        incumbent_carbon=0.01204,
        incumbent_regret=0.015,
        shedding_tolerance=0.0001,
        carbon_tolerance=0.00005,
        reliable_carbon_floor=floor,
    )
    assert not better
    assert np.isclose(floor, 0.01200)


def test_outage_hours_do_not_create_grid_carbon_gradient() -> None:
    truth = _physical(load=1.0, pv=0.0, carbon=0.7)
    result = _result(carbon=(0.2,), served=2.0)
    generated = TorchPhysicalTrajectories(
        active_load_mw=torch.full((1, 2, 1, 3), 1.0 / 3.0),
        pv_available_mw=torch.zeros((1, 2, 1, 3)),
        grid_carbon_t_per_mwh=torch.full((1, 2), 0.2, requires_grad=True),
    )
    output = RecourseFeasibilityLoss(
        load_weight=0.0, pv_weight=0.0, carbon_weight=1.0,
        carbon_cap_t_per_mwh=0.28,
    )(
        generated, truth, result, [result],
        grid_connected_mask=torch.zeros((1, 2)),
    )
    output.loss.backward()
    assert output.carbon_underestimation == 0.0
    assert torch.all(generated.grid_carbon_t_per_mwh.grad == 0.0)


def test_opl_pushes_overly_strict_predictions_toward_truth() -> None:
    truth = _physical(load=1.0, pv=0.4, carbon=0.3)
    preservation = _result(shed=(0.2, 0.2), carbon=(0.1, 0.1))
    generated = TorchPhysicalTrajectories(
        active_load_mw=torch.full((1, 2, 1, 3), 1.5 / 3.0, requires_grad=True),
        pv_available_mw=torch.full((1, 2, 1, 3), 0.1 / 3.0, requires_grad=True),
        grid_carbon_t_per_mwh=torch.full((1, 2), 0.7, requires_grad=True),
    )
    output = RecourseFeasibilityLoss(
        load_weight=1.0,
        pv_weight=1.0,
        carbon_weight=1.0,
        carbon_cap_t_per_mwh=0.28,
        infeasibility_aversion_alpha=0.0,
        margin=0.05,
    )(
        generated,
        truth,
        _result(),
        [_result()],
        preservation,
        [preservation],
    )
    output.loss.backward()
    # Gradient descent loosens predicted constraints: lower load/carbon and
    # higher PV, preserving the known-good real design.
    assert torch.all(generated.active_load_mw.grad > 0.0)
    assert torch.all(generated.pv_available_mw.grad < 0.0)
    assert torch.all(generated.grid_carbon_t_per_mwh.grad > 0.0)


def test_feasibility_surrogate_reaches_cvae_decoder() -> None:
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")
    feeder = single_pcc_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=4, horizon=4, seed=42)
    codec = ScenarioCodec.fit(pool, feeder)
    model = ConditionalVAE(codec.trajectory_dim, codec.context_dim, 2, 8)
    _, contexts = codec.encode_pool(pool)
    condition = torch.as_tensor(contexts[:1], dtype=torch.float32)
    generated = codec.physical_torch(model.decode(torch.zeros(1, 2), condition))
    truth = TorchPhysicalTrajectories(
        active_load_mw=generated.active_load_mw.detach() + 0.1,
        pv_available_mw=generated.pv_available_mw.detach(),
        grid_carbon_t_per_mwh=generated.grid_carbon_t_per_mwh.detach() + 0.1,
    )
    result = _result(
        shed=(0.2, 0.2, 0.2, 0.2),
        carbon=(0.1, 0.1, 0.1, 0.1),
    )
    loss = RecourseFeasibilityLoss(
        load_weight=1.0,
        pv_weight=0.0,
        carbon_weight=1.0,
        carbon_cap_t_per_mwh=config.planning.dc_carbon_cap,
    )(generated, truth, result, [result]).loss
    loss.backward()
    assert loss.item() > 0.0
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.decoder.parameters()
    )


def test_lambda_zero_trains_scratch_cvae_without_dfl_and_uses_true_milps(monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_DFL_SOLVER_WORKER", "1")
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")
    feeder = single_pcc_microgrid()
    original = make_toy_scenarios(feeder, num_scenarios=4, horizon=4, seed=43)
    scenarios = list(original.scenarios)
    scenarios[0] = replace(
        scenarios[0],
        grid_available=np.asarray((1.0, 0.0, 0.0, 1.0)),
        probability_weight=0.01,
    )
    scenarios[1:] = [
        replace(scenario, probability_weight=0.33)
        for scenario in scenarios[1:]
    ]
    pool = ScenarioPool(tuple(scenarios))
    codec = ScenarioCodec.fit(pool, feeder)
    model = ConditionalVAE(codec.trajectory_dim, codec.context_dim, 2, 8)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    planning = replace(
        config.planning,
        solver_backend="scip",
        solver_time_limit_seconds=20.0,
        solver_relative_gap=0.0,
        solver_threads=1,
        dc_carbon_cap=1.10,
    )
    oracle = SinglePCCPlanningOracle(
        feeder, planning, config.costs, config.data, config.data_center
    )
    dfl = replace(
        config.dfl,
        lambda_dfl=0.0,
        epochs=1,
        num_support_scenarios=2,
        decision_batch_size=1,
        final_validation_size=1,
        device="cpu",
    )
    result = train_recourse_feasibility_cvae(
        model, codec, pool, oracle, replace(config.cvae, epochs=1), dfl, 7
    )
    assert len(result.history) == 1
    assert not result.history[0].dfl_evaluated
    assert result.history[0].effective_lambda_dfl == 0.0
    assert result.history[0].statistical_loss_weight == 1.0
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )
    assert result.planning_result.status in {"optimal", "gaplimit"}
    assert result.full_validation_result.status in {"optimal", "gaplimit"}
    assert any(
        np.any(scenario.grid_available < 0.5)
        for scenario in result.generated_scenarios
    )
    assert min(result.scenario_weights) < 0.1
    generated_outage_mass = sum(
        weight
        for scenario, weight in zip(
            result.generated_scenarios, result.scenario_weights, strict=True
        )
        if np.any(scenario.grid_available < 0.5)
    )
    assert np.isclose(generated_outage_mass, 0.01)
    # The final planning model retains both the build and operating binaries.
    artifacts = oracle._build_model(
        result.generated_scenarios,
        result.scenario_weights,
        fixed_design=None,
        allow_carbon_slack=True,
    )
    variable_types = {
        variable.name: variable.vtype() for variable in artifacts.model.getVars()
    }
    assert variable_types["site[PCC]"] == "BINARY"
    assert any(
        name.startswith("charge_mode[") and kind == "BINARY"
        for name, kind in variable_types.items()
    )
    artifacts.model.freeProb()


def test_main_config_stabilizes_decision_feedback() -> None:
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")

    assert config.dfl.fixed_decision_anchors
    assert config.dfl.decision_batch_size == config.dfl.num_support_scenarios
    assert config.dfl.validation_interval == 1
    assert np.isclose(config.dfl.dfl_statistical_weight, 0.20)
    assert np.isclose(config.dfl.gradient_balance_ratio, 0.10)
    assert np.isclose(config.dfl.gradient_balance_max_scale, 1000.0)


def test_recourse_fixes_only_design_and_reoptimizes_operations(monkeypatch) -> None:
    monkeypatch.setenv("STORAGE_DFL_SOLVER_WORKER", "1")
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")
    feeder = single_pcc_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=4, horizon=4, seed=44)
    planning = replace(
        config.planning,
        solver_backend="scip",
        solver_time_limit_seconds=20.0,
        solver_relative_gap=0.0,
        solver_threads=1,
        dc_carbon_cap=1.10,
    )
    oracle = SinglePCCPlanningOracle(
        feeder, planning, config.costs, config.data, config.data_center
    )
    plan = oracle.solve(pool.scenarios[:1], allow_carbon_slack=True)
    aggregate, rows = evaluate_fixed_design_recourse(
        oracle, pool.scenarios[1:3], (0.5, 0.5), plan.design
    )
    assert aggregate.design == plan.design
    assert all(row.design == plan.design for row in rows)
    # Distinct realizations receive independent operating optimizations.
    assert rows[0].operating_cost != rows[1].operating_cost
    assert isinstance(aggregate.objective, float)
    assert isinstance(aggregate.recourse_diagnostics.load_shedding_mwh, float)
    artifacts = oracle._build_model(
        pool.scenarios[1:2],
        (1.0,),
        fixed_design=plan.design,
        allow_carbon_slack=True,
    )
    variables = {variable.name: variable for variable in artifacts.model.getVars()}
    for name in ("site[PCC]", "pcap[PCC]", "ecap[PCC]"):
        assert variables[name].getLbGlobal() == variables[name].getUbGlobal()
    charge_mode = next(
        variable
        for name, variable in variables.items()
        if name.startswith("charge_mode[")
    )
    assert charge_mode.getLbGlobal() == 0.0
    assert charge_mode.getUbGlobal() == 1.0
    grid_load = next(
        variable
        for name, variable in variables.items()
        if name.startswith("grid_load[")
    )
    assert grid_load.getLbGlobal() < grid_load.getUbGlobal()
    artifacts.model.freeProb()


def test_scenario_cache_identity_uses_contents() -> None:
    feeder = single_pcc_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=4, horizon=4, seed=45)
    first, second = pool.scenarios[:2]
    second_same_name = replace(second, name=first.name)
    assert scenario_content_key(first) != scenario_content_key(second_same_name)


def test_pcc_cache_does_not_round_distinct_designs_or_weights() -> None:
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")
    feeder = single_pcc_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=4, horizon=4, seed=47)
    oracle = SinglePCCPlanningOracle(
        feeder, config.planning, config.costs, config.data, config.data_center
    )
    first = StorageDesign(
        site={"PCC": 1},
        power_mw={"PCC": 0.5000001},
        energy_mwh={"PCC": 1.0},
    )
    second = replace(first, power_mw={"PCC": 0.5000002})
    scenarios = pool.scenarios[:2]
    assert oracle._cache_key(scenarios, (0.5, 0.5), first, True) != oracle._cache_key(
        scenarios, (0.5, 0.5), second, True
    )
    assert oracle._cache_key(
        scenarios, (0.500000001, 0.499999999), first, True
    ) != oracle._cache_key(scenarios, (0.5, 0.5), first, True)


def test_carbon_slack_switch_controls_constraint_semantics() -> None:
    config = load_config("configs/dataset_v2_dfl_hourly_layered_t1.yaml")
    feeder = single_pcc_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=4, horizon=4, seed=48)
    oracle = SinglePCCPlanningOracle(
        feeder, config.planning, config.costs, config.data, config.data_center
    )
    hard = oracle._build_model(
        pool.scenarios[:1], (1.0,), fixed_design=None, allow_carbon_slack=False
    )
    soft = oracle._build_model(
        pool.scenarios[:1], (1.0,), fixed_design=None, allow_carbon_slack=True
    )
    def variable_name(variable) -> str:
        return str(
            variable.name if hasattr(variable, "name") else variable.VarName
        )

    assert not any(
        variable_name(variable).startswith("carbon_excess[")
        for variable in hard.model.getVars()
    )
    assert any(
        variable_name(variable).startswith("carbon_excess[")
        for variable in soft.model.getVars()
    )
    hard.model.freeProb()
    soft.model.freeProb()


def test_torch_and_detached_milp_trajectories_match() -> None:
    feeder = single_pcc_microgrid()
    pool = make_toy_scenarios(feeder, num_scenarios=4, horizon=4, seed=46)
    codec = ScenarioCodec.fit(pool, feeder)
    encoded, contexts = codec.encode_pool(pool)
    normalized = torch.as_tensor(encoded[:2])
    physical = codec.physical_torch(normalized)
    detached = codec.decode_batch(
        normalized.detach().cpu().numpy(), contexts[:2], name_prefix="parity"
    )
    assert np.allclose(
        physical.active_load_mw.detach().cpu().numpy(),
        np.stack([scenario.active_load_mw for scenario in detached]),
    )
    assert np.allclose(
        physical.pv_available_mw.detach().cpu().numpy(),
        np.stack([scenario.pv_available_mw for scenario in detached]),
    )
    assert np.allclose(
        physical.grid_carbon_t_per_mwh.detach().cpu().numpy(),
        np.stack([scenario.grid_carbon_t_per_mwh for scenario in detached]),
    )
