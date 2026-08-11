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
    infeasibility_penalty: torch.Tensor
    optimality_preserving: torch.Tensor
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
            "surrogate_ipl": float(self.infeasibility_penalty.detach().cpu()),
            "surrogate_opl": float(self.optimality_preserving.detach().cpu()),
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
        infeasibility_aversion_alpha: float = 1.0,
        margin: float = 0.0,
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
        if not 0.0 <= infeasibility_aversion_alpha <= 1.0:
            raise ValueError("infeasibility_aversion_alpha must lie in [0, 1].")
        if margin < 0.0:
            raise ValueError("margin must be nonnegative.")
        self.alpha = float(infeasibility_aversion_alpha)
        self.margin = float(margin)
        self.epsilon = float(epsilon)

    @staticmethod
    def _solver_tensor(
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
        if tensor.ndim != 2 or tensor.shape[0] != reference.shape[0]:
            raise ValueError(
                f"Recourse diagnostics {tuple(tensor.shape)} do not match the "
                f"scenario count in {tuple(reference.shape)}."
            )
        return tensor.detach()

    def _carbon_terms(
        self,
        excess: torch.Tensor,
        generated_carbon: torch.Tensor,
        true_carbon: torch.Tensor,
        served_power: torch.Tensor,
        grid_connected: torch.Tensor,
        exposure: torch.Tensor | None,
        *,
        conservative: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return detached severity and a differentiable directional residual."""

        connected = grid_connected.detach().to(dtype=true_carbon.dtype)
        # A smooth cap-relative pressure preserves a useful gradient when an
        # untrained decoder predicts below the cap. A hard ReLU would be flat
        # there and could not correct a dangerously optimistic carbon forecast.
        pressure_scale = max(0.05, 0.1 * self.carbon_cap)
        true_pressure = pressure_scale * torch.nn.functional.softplus(
            (true_carbon - self.carbon_cap) / pressure_scale
        )
        generated_pressure = pressure_scale * torch.nn.functional.softplus(
            (generated_carbon - self.carbon_cap) / pressure_scale
        )
        if exposure is None:
            exposure = torch.diag_embed(served_power.detach().clamp_min(0.0))
        expected_shape = (
            true_carbon.shape[0], true_carbon.shape[1], true_carbon.shape[1]
        )
        if exposure.shape != expected_shape:
            raise ValueError(
                f"Grid-carbon exposure {tuple(exposure.shape)} does not match "
                f"the expected {expected_shape}."
            )
        exposure = exposure.detach() * connected.unsqueeze(1) * connected.unsqueeze(2)
        pressure_delta = (
            generated_pressure - true_pressure
            if conservative
            else true_pressure - generated_pressure
        )
        row_exposure = exposure.sum(dim=2).clamp_min(self.epsilon)
        row_direction = torch.relu(
            (exposure * pressure_delta.unsqueeze(1)).sum(dim=2)
        ) / (row_exposure * self.carbon_cap)
        if excess.shape[1] == true_carbon.shape[1]:
            severity = excess / served_power.clamp_min(0.05) * connected
            return severity, row_direction
        if excess.shape[1] != 1:
            raise ValueError(
                "Carbon recourse diagnostics must contain either one horizon "
                "value or one value per trajectory hour."
            )

        # A horizon cap is an energy-weighted average, so its single slack is
        # normalized by total served energy and paired with an average-carbon
        # directional residual rather than repeated across every hour.
        served_energy = (
            served_power.detach().clamp_min(0.0) * connected
        ).sum(dim=1, keepdim=True).clamp_min(0.05)
        total_exposure = exposure.sum(dim=1)
        exposure_energy = total_exposure.sum(dim=1, keepdim=True).clamp_min(
            self.epsilon
        )
        direction = torch.relu(
            (total_exposure * pressure_delta).sum(dim=1, keepdim=True)
        ) / (exposure_energy * self.carbon_cap)
        return excess / served_energy, direction

    @staticmethod
    def _solver_exposure_tensor(
        results: list[PlanningResult], reference: torch.Tensor
    ) -> torch.Tensor | None:
        matrices: list[tuple[tuple[float, ...], ...]] = []
        for result in results:
            diagnostics = result.recourse_diagnostics
            if diagnostics is None:
                raise ValueError("The recourse result has no physical diagnostics.")
            values = diagnostics.grid_carbon_exposure_mw
            if not values:
                return None
            if len(values) != 1:
                raise ValueError(
                    "Carbon exposure training expects one realized scenario per "
                    "fixed-design recourse solve."
                )
            matrices.append(values[0])
        return torch.as_tensor(
            matrices, dtype=reference.dtype, device=reference.device
        ).detach()

    def __call__(
        self,
        generated: TorchPhysicalTrajectories,
        truth: TorchPhysicalTrajectories,
        recourse_aggregate: PlanningResult,
        recourse_scenario_results: list[PlanningResult],
        preservation_aggregate: PlanningResult | None = None,
        preservation_scenario_results: list[PlanningResult] | None = None,
        grid_connected_mask: torch.Tensor | None = None,
    ) -> FeasibilityLossOutput:
        generated_load = generated.aggregate_load_mw
        true_load = truth.aggregate_load_mw.detach()
        generated_pv = generated.aggregate_pv_mw
        true_pv = truth.aggregate_pv_mw.detach()
        generated_carbon = generated.grid_carbon_t_per_mwh
        true_carbon = truth.grid_carbon_t_per_mwh.detach()
        if grid_connected_mask is None:
            grid_connected = torch.ones_like(true_carbon)
        else:
            grid_connected = grid_connected_mask.to(
                dtype=true_carbon.dtype, device=true_carbon.device
            )
            if grid_connected.shape != true_carbon.shape:
                raise ValueError(
                    f"Grid-connected mask {tuple(grid_connected.shape)} does not "
                    f"match carbon trajectories {tuple(true_carbon.shape)}."
                )

        shed = self._solver_tensor(
            recourse_scenario_results,
            "hourly_load_shedding_mw",
            true_load,
        )
        carbon_excess = self._solver_tensor(
            recourse_scenario_results,
            "hourly_carbon_excess_t_per_hour",
            true_load,
        )
        carbon_exposure = self._solver_exposure_tensor(
            recourse_scenario_results, true_carbon
        )

        power_scale = true_load.abs().clamp_min(0.05)
        shed_severity = shed / power_scale
        load_direction = torch.relu(true_load - generated_load) / power_scale
        pv_direction = torch.relu(generated_pv - true_pv) / power_scale

        served_power = (true_load - shed).clamp_min(0.05)
        carbon_severity, carbon_direction = self._carbon_terms(
            carbon_excess,
            generated_carbon,
            true_carbon,
            served_power,
            grid_connected,
            carbon_exposure,
            conservative=False,
        )

        # IPL: an optimistic predicted feasible set produced a design that
        # violates the real recourse constraints.  The ReLU direction is a
        # normalized constraint residual; softplus supplies the paper's smooth
        # margin while subtracting its value at zero keeps masked terms at zero.
        def smooth_margin(value: torch.Tensor) -> torch.Tensor:
            baseline = torch.nn.functional.softplus(
                torch.as_tensor(self.margin, dtype=value.dtype, device=value.device)
            )
            return torch.nn.functional.softplus(self.margin + value) - baseline

        load_term = torch.mean(shed_severity * smooth_margin(load_direction))
        pv_term = torch.mean(shed_severity * smooth_margin(pv_direction))
        carbon_term = torch.mean(carbon_severity * smooth_margin(carbon_direction))
        ipl = (
            self.load_weight * load_term
            + self.pv_weight * pv_term
            + self.carbon_weight * carbon_term
        )

        # OPL: hold the perfect-information *design* fixed and re-dispatch it on
        # the generated constraints.  If those constraints require shedding or
        # carbon slack, they have excluded a known-good first-stage decision.
        # This is the recourse analogue of evaluating x*(rho) under rho_hat.
        opl = torch.zeros((), dtype=ipl.dtype, device=ipl.device)
        if preservation_aggregate is not None:
            if preservation_scenario_results is None:
                raise ValueError("Preservation scenario results are required for OPL.")
            preserve_shed = self._solver_tensor(
                preservation_scenario_results,
                "hourly_load_shedding_mw",
                true_load,
            )
            preserve_carbon = self._solver_tensor(
                preservation_scenario_results,
                "hourly_carbon_excess_t_per_hour",
                true_load,
            )
            preserve_exposure = self._solver_exposure_tensor(
                preservation_scenario_results, true_carbon
            )
            preserve_shed_severity = preserve_shed / power_scale
            preserve_served = (generated_load - preserve_shed).clamp_min(0.05)
            preserve_carbon_severity, strict_carbon = self._carbon_terms(
                preserve_carbon,
                generated_carbon,
                true_carbon,
                preserve_served,
                grid_connected,
                preserve_exposure,
                conservative=True,
            )
            strict_load = torch.relu(generated_load - true_load) / power_scale
            strict_pv = torch.relu(true_pv - generated_pv) / power_scale
            opl = (
                self.load_weight
                * torch.mean(preserve_shed_severity * smooth_margin(strict_load))
                + self.pv_weight
                * torch.mean(preserve_shed_severity * smooth_margin(strict_pv))
                + self.carbon_weight
                * torch.mean(preserve_carbon_severity * smooth_margin(strict_carbon))
            )
        loss = self.alpha * ipl + (1.0 - self.alpha) * opl

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
            infeasibility_penalty=ipl,
            optimality_preserving=opl,
            forward_load_shedding_fraction=float(shedding_fraction),
            forward_carbon_excess_t_per_mwh=float(carbon_rate),
            forward_pv_curtailment_mwh=float(diagnostics.pv_curtailment_mwh),
        )
