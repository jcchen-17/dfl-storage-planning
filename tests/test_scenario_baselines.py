from __future__ import annotations

import pytest

from storage_dfl.baselines import comparison_metrics


def _result(objective: float) -> dict:
    return {
        "status": "optimal",
        "objective": objective,
        "relative_gap": 0.0,
        "design": {
            "site": {"PCC": 1},
            "power_mw": {"PCC": 2.0},
            "energy_mwh": {"PCC": 8.0},
        },
        "recourse_diagnostics": {
            "load_shedding_mwh": 0.25,
            "carbon_excess_t": 3.0,
        },
    }


def _baseline() -> dict:
    return {
        "evaluation_scenario_names": ["test-a", "test-b"],
        "evaluation_scenario_weights": [0.25, 0.75],
        "solver_relative_gap": 1.0e-3,
        "planning": _result(85.0),
        "out_of_sample_evaluation": _result(90.0),
        "risk": {
            "load_shedding_probability": 0.25,
            "carbon_excess_probability": 0.75,
        },
        "planning_wall_seconds": 2.0,
        "out_of_sample_wall_seconds": 4.0,
    }


def _reference() -> dict:
    return {
        "evaluation_scenario_names": ["test-a", "test-b"],
        "evaluation_scenario_weights": [0.25, 0.75],
        "no_storage_reference": _result(100.0),
        "perfect_information_reference": _result(80.0),
    }


def test_comparison_metrics_use_shared_reference() -> None:
    metrics = comparison_metrics(_baseline(), _reference())

    assert metrics["objectives_comparable"] is True
    assert metrics["planning_converged"] is True
    assert metrics["storage_value"] == pytest.approx(10.0)
    assert metrics["decision_regret"] == pytest.approx(10.0 / (80.0 + 1.0e-6))
    assert metrics["power_mw"] == 2.0
    assert metrics["energy_mwh"] == 8.0
    assert metrics["load_shedding_probability"] == 0.25


def test_comparison_metrics_reject_different_test_sets() -> None:
    reference = _reference()
    reference["evaluation_scenario_names"] = ["test-a", "different"]

    with pytest.raises(ValueError, match="different evaluation scenarios"):
        comparison_metrics(_baseline(), reference)
