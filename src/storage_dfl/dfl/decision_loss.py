"""Recourse-aware decision losses for constraint-parameter generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from storage_dfl.data import TorchPhysicalTrajectories
from storage_dfl.planning import PlanningResult


@dataclass(frozen=True)
class FeasibilityLossOutput:
    """Differentiable surrogate plus detached forward MILP diagnostics."""

    loss: torch.Tensor
    load_underestimation: torch.Tensor
    pv_overestimation: torch.Tensor
    carbon_underestimation: torch.Tensor
    forward_load_shedding_fraction: float
    forward_carbon_excess_t_per_mwh: float
    forward_pv_curtailment_mwh: float

    def diagnostics(self) -> dict[str, float]:
        return {
            "surrogate_feasibility": float(self.loss.detach().cpu()),
            "surrogate_load_underestimation": float(
                self.load_underestimation.detach().cpu()
            ),
            "surrogate_pv_overestimation": float(
                self.pv_overestimation.detach().cpu()
            ),
            "surrogate_carbon_underestimation": float(
                self.carbon_underestimation.detach().cpu()
            ),
            "forward_load_shedding_fraction": self.forward_load_shedding_fraction,
            "forward_carbon_excess_t_per_mwh": self.forward_carbon_excess_t_per_mwh,
            "forward_pv_curtailment_mwh": self.forward_pv_curtailment_mwh,
        }


class RecourseFeasibilityLoss:
    """ODECE-inspired directional constraint-parameter surrogate.

    Gurobi/SCIP outputs enter only as detached severity weights.  Gradients flow
    through generated load, PV and grid-carbon trajectories.  This is not the
    derivative of the MILP argmin map; it is a dense surrogate that moves a
    dangerous generated constraint parameter in the conservative direction.
    """

    def __init__(
        self,
        *,
        load_weight: float,
        pv_weight: float,
        carbon_weight: float,
        carbon_cap_t_per_mwh: float,
        epsilon: float = 1.0e-6,
    ) -> None:
        for name, value in (
            ("load_weight", load_weight),
            ("pv_weight", pv_weight),
            ("carbon_weight", carbon_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        self.load_weight = float(load_weight)
        self.pv_weight = float(pv_weight)
        self.carbon_weight = float(carbon_weight)
        self.carbon_cap = max(float(carbon_cap_t_per_mwh), epsilon)
        self.epsilon = float(epsilon)

    @staticmethod
    def _hourly_solver_tensor(
        results: list[PlanningResult],
        field: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        rows: list[tuple[float, ...]] = []
        for result in results:
            diagnostics = result.recourse_diagnostics
            if diagnostics is None:
                raise ValueError("The recourse result has no physical diagnostics.")
            values = getattr(diagnostics, field)
            if len(values) != 1:
                raise ValueError(
                    "Feasibility training expects one realized scenario per "
                    "fixed-design recourse solve."
                )
            rows.append(tuple(float(value) for value in values[0]))
        tensor = torch.as_tensor(
            rows, dtype=reference.dtype, device=reference.device
        )
        if tensor.shape != reference.shape:
            raise ValueError(
                f"Hourly recourse diagnostics {tuple(tensor.shape)} do not match "
                f"the generated trajectories {tuple(reference.shape)}."
            )
        return tensor.detach()

    def __call__(
        self,
        generated: TorchPhysicalTrajectories,
        truth: TorchPhysicalTrajectories,
        recourse_aggregate: PlanningResult,
        recourse_scenario_results: list[PlanningResult],
    ) -> FeasibilityLossOutput:
        generated_load = generated.aggregate_load_mw
        true_load = truth.aggregate_load_mw.detach()
        generated_pv = generated.aggregate_pv_mw
        true_pv = truth.aggregate_pv_mw.detach()
        generated_carbon = generated.grid_carbon_t_per_mwh
        true_carbon = truth.grid_carbon_t_per_mwh.detach()

        shed = self._hourly_solver_tensor(
            recourse_scenario_results,
            "hourly_load_shedding_mw",
            true_load,
        )
        carbon_excess = self._hourly_solver_tensor(
            recourse_scenario_results,
            "hourly_carbon_excess_t_per_hour",
            true_load,
        )

        power_scale = true_load.abs().clamp_min(0.05)
        shed_severity = shed / power_scale
        load_direction = torch.relu(true_load - generated_load) / power_scale
        pv_direction = torch.relu(generated_pv - true_pv) / power_scale

        served_power = (true_load - shed).clamp_min(0.05)
        carbon_severity = carbon_excess / served_power
        carbon_direction = torch.relu(true_carbon - generated_carbon) / self.carbon_cap

        load_term = torch.mean(shed_severity * load_direction)
        pv_term = torch.mean(shed_severity * pv_direction)
        carbon_term = torch.mean(carbon_severity * carbon_direction)
        loss = (
            self.load_weight * load_term
            + self.pv_weight * pv_term
            + self.carbon_weight * carbon_term
        )

        diagnostics = recourse_aggregate.recourse_diagnostics
        if diagnostics is None:
            raise ValueError("The aggregate recourse result has no diagnostics.")
        total_demand = diagnostics.served_demand_mwh + diagnostics.load_shedding_mwh
        shedding_fraction = diagnostics.load_shedding_mwh / max(
            total_demand, self.epsilon
        )
        carbon_rate = diagnostics.carbon_excess_t / max(
            diagnostics.served_demand_mwh, self.epsilon
        )
        return FeasibilityLossOutput(
            loss=loss,
            load_underestimation=load_term,
            pv_overestimation=pv_term,
            carbon_underestimation=carbon_term,
            forward_load_shedding_fraction=float(shedding_fraction),
            forward_carbon_excess_t_per_mwh=float(carbon_rate),
            forward_pv_curtailment_mwh=float(diagnostics.pv_curtailment_mwh),
        )

