"""Check pure three-phase network feasibility without carbon or storage models."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from pyscipopt import Model, quicksum

from diagnose_dfl_feasibility import generated_first_epoch
from storage_dfl.config import load_config
from storage_dfl.data import Scenario
from storage_dfl.network import Feeder, ieee13_unbalanced_microgrid


def solve_hour(
    feeder: Feeder,
    scenario: Scenario,
    hour: int,
    grid_limit_mw: float,
    time_limit_seconds: float,
    generator_max_mw: float,
) -> dict[str, object]:
    """Solve one convex LinDistFlow feasibility problem for one scenario hour."""
    buses = feeder.buses
    bus_phases = feeder.bus_phases
    root = feeder.root
    bus_index = feeder.bus_index
    phase_index = feeder.phase_index
    phase_edges = tuple(
        (line.parent, line.child, phase)
        for line in feeder.lines
        for phase in line.phases
    )
    line_data = {(line.parent, line.child): line for line in feeder.lines}
    node_phases = tuple(
        (bus, phase) for bus in buses for phase in bus_phases[bus]
    )
    outgoing = {
        (bus, phase): [
            edge for edge in phase_edges if edge[0] == bus and edge[2] == phase
        ]
        for bus, phase in node_phases
    }
    incoming = {
        (bus, phase): [
            edge for edge in phase_edges if edge[1] == bus and edge[2] == phase
        ]
        for bus, phase in node_phases
    }

    model = Model(f"pure_power_flow_{scenario.name}_{hour:03d}")
    model.hideOutput()
    model.setParam("limits/time", float(time_limit_seconds))

    grid_phase = {
        phase: model.addVar(lb=0.0, ub=grid_limit_mw, name=f"grid[{phase}]")
        for phase in bus_phases[root]
    }
    q_grid_phase = {
        phase: model.addVar(
            lb=-grid_limit_mw,
            ub=grid_limit_mw,
            name=f"qgrid[{phase}]",
        )
        for phase in bus_phases[root]
    }
    generator = model.addVar(lb=0.0, ub=generator_max_mw, name="generator")
    q_generator = model.addVar(lb=-0.8, ub=0.8, name="q_generator")
    pv = {
        (bus, phase): model.addVar(
            lb=0.0,
            ub=float(
                scenario.pv_available_mw[
                    hour,
                    bus_index[bus],
                    phase_index[phase],
                ]
            ),
            name=f"pv[{bus},{phase}]",
        )
        for bus, phase in node_phases
    }
    flow_p = {}
    flow_q = {}
    for edge in phase_edges:
        rating = line_data[edge[:2]].phase_rating_mva
        flow_p[edge] = model.addVar(
            lb=-rating,
            ub=rating,
            name=f"p[{edge[0]},{edge[1]},{edge[2]}]",
        )
        flow_q[edge] = model.addVar(
            lb=-rating,
            ub=rating,
            name=f"q[{edge[0]},{edge[1]},{edge[2]}]",
        )
    voltage = {
        (bus, phase): model.addVar(
            lb=0.95**2,
            ub=1.05**2,
            name=f"v[{bus},{phase}]",
        )
        for bus, phase in node_phases
    }

    model.addCons(quicksum(grid_phase.values()) <= grid_limit_mw)
    for edge in phase_edges:
        line = line_data[edge[:2]]
        model.addCons(
            flow_p[edge] * flow_p[edge] + flow_q[edge] * flow_q[edge]
            <= line.phase_rating_mva**2
        )
        model.addCons(
            voltage[edge[1], edge[2]]
            == voltage[edge[0], edge[2]]
            - 2.0
            * (
                line.resistance * flow_p[edge]
                + line.reactance * flow_q[edge]
            )
        )
    for phase in bus_phases[root]:
        model.addCons(voltage[root, phase] == 1.0)

    processed = float(scenario.workload_arrival[hour])
    data_center_power = 0.03 + float(scenario.pue[hour]) * (0.12 + 0.30 * processed)
    for bus, phase in node_phases:
        bi = bus_index[bus]
        pi = phase_index[phase]
        active_load = float(scenario.active_load_mw[hour, bi, pi])
        reactive_load = float(scenario.reactive_load_mvar[hour, bi, pi])
        phase_count = len(bus_phases[bus])
        if bus == feeder.data_center_bus:
            active_load += data_center_power / phase_count
        local_generation = (
            generator / phase_count if bus == feeder.generator_bus else 0.0
        )
        local_q_generation = (
            q_generator / phase_count if bus == feeder.generator_bus else 0.0
        )
        grid_injection = grid_phase[phase] if bus == root else 0.0
        q_grid_injection = q_grid_phase[phase] if bus == root else 0.0
        model.addCons(
            local_generation
            + pv[bus, phase]
            + grid_injection
            - active_load
            == quicksum(flow_p[edge] for edge in outgoing[bus, phase])
            - quicksum(flow_p[edge] for edge in incoming[bus, phase])
        )
        model.addCons(
            local_q_generation
            + q_grid_injection
            - reactive_load
            == quicksum(flow_q[edge] for edge in outgoing[bus, phase])
            - quicksum(flow_q[edge] for edge in incoming[bus, phase])
        )

    model.setObjective(quicksum(grid_phase.values()) + generator, "minimize")
    model.optimize()
    status = str(model.getStatus())
    solution = model.getBestSol()
    if solution is None:
        result = {
            "scenario": scenario.name,
            "hour": hour,
            "status": status,
            "feasible": False,
            "solve_seconds": float(model.getSolvingTime()),
            "total_load_mw": float(scenario.active_load_mw[hour].sum()),
            "total_pv_available_mw": float(scenario.pv_available_mw[hour].sum()),
            "data_center_power_mw": data_center_power,
            "generator_mw": None,
            "min_voltage_pu": None,
            "max_voltage_pu": None,
            "max_line_loading": None,
            "max_loaded_line": None,
        }
    else:
        voltage_values = {
            key: math.sqrt(max(0.0, float(model.getVal(variable))))
            for key, variable in voltage.items()
        }
        loading_values = {}
        for edge in phase_edges:
            p_value = float(model.getVal(flow_p[edge]))
            q_value = float(model.getVal(flow_q[edge]))
            rating = line_data[edge[:2]].phase_rating_mva
            loading_values[edge] = math.hypot(p_value, q_value) / rating
        max_loaded_edge = max(loading_values, key=loading_values.get)
        result = {
            "scenario": scenario.name,
            "hour": hour,
            "status": status,
            "feasible": True,
            "solve_seconds": float(model.getSolvingTime()),
            "total_load_mw": float(scenario.active_load_mw[hour].sum()),
            "total_pv_available_mw": float(scenario.pv_available_mw[hour].sum()),
            "data_center_power_mw": data_center_power,
            "generator_mw": float(model.getVal(generator)),
            "min_voltage_pu": min(voltage_values.values()),
            "max_voltage_pu": max(voltage_values.values()),
            "max_line_loading": loading_values[max_loaded_edge],
            "max_loaded_line": "-".join(max_loaded_edge),
        }
    model.freeProb()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--disable-generator", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    feeder = ieee13_unbalanced_microgrid()
    scenarios = generated_first_epoch(args.config)
    rows = [
        solve_hour(
            feeder,
            scenario,
            hour,
            config.planning.grid_limit_mw,
            args.time_limit,
            0.0 if args.disable_generator else 1.0,
        )
        for scenario in scenarios
        for hour in range(scenario.horizon)
    ]

    output_dir = config.output_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pure_power_flow_hourly.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    feasible_rows = [row for row in rows if row["feasible"]]
    failed_rows = [row for row in rows if not row["feasible"]]
    summary = {
        "config": str(args.config),
        "scenarios": len(scenarios),
        "hours_checked": len(rows),
        "feasible_hours": len(feasible_rows),
        "failed_hours": len(failed_rows),
        "minimum_voltage_pu": min(
            (float(row["min_voltage_pu"]) for row in feasible_rows),
            default=None,
        ),
        "maximum_voltage_pu": max(
            (float(row["max_voltage_pu"]) for row in feasible_rows),
            default=None,
        ),
        "maximum_line_loading": max(
            (float(row["max_line_loading"]) for row in feasible_rows),
            default=None,
        ),
        "failed_cases": [
            {
                "scenario": row["scenario"],
                "hour": row["hour"],
                "status": row["status"],
            }
            for row in failed_rows
        ],
        "hourly_csv": str(csv_path),
    }
    summary_path = output_dir / "pure_power_flow_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    if failed_rows:
        raise SystemExit(f"pure power flow failed for {len(failed_rows)} hour(s)")


if __name__ == "__main__":
    main()
