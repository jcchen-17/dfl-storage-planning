"""Export CSV tables and IEEE-style figures for Average vs. Layered DFL runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("Layered", "Average")
COLORS = {"Layered": "#0072B2", "Average": "#D55E00"}
MARKERS = {"Layered": "o", "Average": "s"}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, name))
    elif isinstance(value, list):
        flattened[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        flattened[prefix] = value
    return flattened


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 3.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(0.015, 0.97, label, transform=ax.transAxes, va="top", fontweight="bold")
    ax.grid(True, color="0.88", linewidth=0.45, linestyle="--")
    ax.set_axisbelow(True)


def _plot_history(
    histories: dict[str, list[dict[str, Any]]], output_dir: Path
) -> None:
    panels = (
        ("cvae_loss", "CVAE loss", False),
        ("infeasibility_penalty_loss", "IPL loss", True),
        ("optimality_preserving_loss", "OPL loss", True),
        ("effective_lambda_dfl", r"Effective $\lambda_{\mathrm{DFL}}$", False),
        ("decision_regret", "Training decision regret (%)", False),
        ("validation_carbon_excess_t_per_mwh", r"Validation carbon excess (tCO$_2$/MWh)", False),
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.25), constrained_layout=True)
    for index, (key, ylabel, log_scale) in enumerate(panels):
        ax = axes.flat[index]
        for method in METHODS:
            rows = histories[method]
            points = [row for row in rows if _finite(row.get(key))]
            x = [int(row["epoch"]) + 1 for row in points]
            scale = 100.0 if key == "decision_regret" else 1.0
            y = [scale * float(row[key]) for row in points]
            sparse = key.startswith("validation_")
            ax.plot(
                x,
                y,
                color=COLORS[method],
                marker=MARKERS[method] if sparse else None,
                markevery=1 if sparse else None,
                label=method,
            )
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("DFL epoch")
        ax.set_ylabel(ylabel)
        _panel(ax, f"({chr(97 + index)})")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.savefig(output_dir / "training_dynamics_ieee.pdf")
    fig.savefig(output_dir / "training_dynamics_ieee.png", dpi=600)
    plt.close(fig)


def _test_row(method: str, payload: dict[str, Any], source: Path) -> dict[str, Any]:
    result = payload["out_of_sample_validation"]
    diagnostics = result["recourse_diagnostics"]
    design = payload.get("selected_training_design") or result["design"]
    scenario_results = payload["out_of_sample_scenario_results"]
    statuses: dict[str, int] = {}
    for scenario in scenario_results:
        status = str(scenario["status"])
        statuses[status] = statuses.get(status, 0) + 1
    no_storage = payload["no_storage_reference"]
    raw_storage_value = float(no_storage["objective"]) - float(result["objective"])
    return {
        "method": method,
        "trained_carbon_formulation": payload["trained_carbon_formulation"],
        "evaluation_carbon_formulation": payload["evaluation_carbon_formulation"],
        "test_split": payload["test_split"],
        "test_scenarios": payload["test_scenarios_evaluated"],
        "power_mw": sum(float(x) for x in design["power_mw"].values()),
        "energy_mwh": sum(float(x) for x in design["energy_mwh"].values()),
        "duration_h": sum(float(x) for x in design["energy_mwh"].values())
        / max(sum(float(x) for x in design["power_mw"].values()), 1e-12),
        "total_cost_usd": result["objective"],
        "investment_cost_usd": result["investment_cost"],
        "operating_cost_usd": result["operating_cost"],
        "carbon_slack_cost_usd": result["carbon_slack_cost"],
        "load_shedding_mwh": diagnostics["load_shedding_mwh"],
        "carbon_excess_t": diagnostics["carbon_excess_t"],
        "served_demand_mwh": diagnostics["served_demand_mwh"],
        "pv_curtailment_mwh": diagnostics["pv_curtailment_mwh"],
        "status": result["status"],
        "relative_gap_pct": 100.0 * float(result["relative_gap"]),
        "best_bound_usd": result["best_bound"],
        "scenario_optimal_count": statuses.get("optimal", 0),
        "scenario_timelimit_count": statuses.get("timelimit", 0),
        "scenario_other_count": len(scenario_results)
        - statuses.get("optimal", 0)
        - statuses.get("timelimit", 0),
        "scenarios_above_target_gap": sum(
            float(x.get("relative_gap") or 0.0)
            > float(payload["evaluation_relative_gap"])
            for x in scenario_results
        ),
        "solver_time_sum_s": sum(
            float(x.get("solve_time_seconds") or 0.0) for x in scenario_results
        ),
        "wall_time_s": payload["out_of_sample_wall_seconds"],
        "no_storage_cost_usd": no_storage["objective"],
        "raw_storage_saving_usd": raw_storage_value,
        "certified_storage_value_usd": payload.get("storage_value"),
        "objectives_comparable": payload.get("objectives_comparable"),
        "source": str(source.resolve()),
    }


def _scenario_rows(method: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    names = payload["evaluation_scenario_names"]
    weights = payload["evaluation_scenario_weights"]
    for index, result in enumerate(payload["out_of_sample_scenario_results"]):
        diagnostics = result["recourse_diagnostics"]
        rows.append(
            {
                "method": method,
                "scenario_index": index,
                "scenario_name": names[index],
                "scenario_weight": weights[index],
                "status": result["status"],
                "objective_usd": result["objective"],
                "investment_cost_usd": result["investment_cost"],
                "operating_cost_usd": result["operating_cost"],
                "carbon_slack_cost_usd": result["carbon_slack_cost"],
                "load_shedding_mwh": diagnostics["load_shedding_mwh"],
                "carbon_excess_t": diagnostics["carbon_excess_t"],
                "served_demand_mwh": diagnostics["served_demand_mwh"],
                "pv_curtailment_mwh": diagnostics["pv_curtailment_mwh"],
                "relative_gap_pct": 100.0 * float(result.get("relative_gap") or 0.0),
                "best_bound_usd": result.get("best_bound"),
                "solve_time_s": result.get("solve_time_seconds"),
            }
        )
    return rows


def _bar(ax: plt.Axes, rows: list[dict[str, Any]], key: str, ylabel: str, panel: str,
         scale: float = 1.0) -> None:
    x = np.arange(len(rows))
    values = [float(row[key]) / scale for row in rows]
    bars = ax.bar(
        x,
        values,
        color=[COLORS[str(row["method"])] for row in rows],
        edgecolor="black",
        linewidth=0.55,
        width=0.62,
    )
    ax.set_xticks(x, [str(row["method"]) for row in rows])
    ax.set_ylabel(ylabel)
    for patch, value in zip(bars, values, strict=True):
        ax.annotate(
            f"{value:.3g}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    ax.margins(y=0.18)
    _panel(ax, panel)


def _plot_test(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.35), constrained_layout=True)
    ax = axes[0, 0]
    x = np.arange(len(rows))
    total_costs = np.array([float(row["total_cost_usd"]) / 1e6 for row in rows])
    ax.plot(x, total_costs, color="0.45", linewidth=0.8, zorder=1)
    for index, row in enumerate(rows):
        ax.scatter(index, total_costs[index], s=48, color=COLORS[str(row["method"])],
                   edgecolor="black", linewidth=0.55, zorder=2)
        ax.annotate(f"{total_costs[index]:.4f}", (index, total_costs[index]),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.5)
    span = max(float(np.ptp(total_costs)), 0.001)
    ax.set_ylim(float(total_costs.min() - 0.55 * span),
                float(total_costs.max() + 0.75 * span))
    ax.set_xticks(x, [str(row["method"]) for row in rows])
    ax.set_ylabel("Test total cost (M$)")
    _panel(ax, "(a)")

    ax = axes[0, 1]
    bottoms = np.zeros(len(rows))
    components = (
        ("investment_cost_usd", "Investment", "#4E79A7"),
        ("operating_cost_usd", "Operation", "#59A14F"),
        ("carbon_slack_cost_usd", "Carbon penalty", "#E15759"),
    )
    for key, label, color in components:
        values = np.array([float(row[key]) / 1e6 for row in rows])
        ax.bar(x, values, bottom=bottoms, label=label, color=color,
               edgecolor="black", linewidth=0.45, width=0.62)
        bottoms += values
    ax.set_xticks(x, [str(row["method"]) for row in rows])
    ax.set_ylabel("Cost components (M$)")
    ax.legend(frameon=False, loc="upper right")
    _panel(ax, "(b)")

    _bar(axes[0, 2], rows, "carbon_excess_t", r"Carbon excess (tCO$_2$)", "(c)")

    ax = axes[1, 0]
    width = 0.34
    power = [float(row["power_mw"]) for row in rows]
    energy = [float(row["energy_mwh"]) for row in rows]
    ax.bar(x - width / 2, power, width, label="Power (MW)", color="#4E79A7",
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, energy, width, label="Energy (MWh)", color="#F28E2B",
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x, [str(row["method"]) for row in rows])
    ax.set_ylabel("Installed capacity")
    ax.legend(frameon=False)
    _panel(ax, "(d)")

    _bar(axes[1, 1], rows, "wall_time_s", "Test wall time (s)", "(e)")
    _bar(axes[1, 2], rows, "relative_gap_pct", "Maximum scenario gap (%)", "(f)")
    axes[1, 2].axhline(0.1, color="0.25", linestyle="--", linewidth=0.9,
                       label="Target (0.1%)")
    axes[1, 2].legend(frameon=False, loc="upper left")

    fig.savefig(output_dir / "test_comparison_ieee.pdf")
    fig.savefig(output_dir / "test_comparison_ieee.png", dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layered-run", type=Path, required=True)
    parser.add_argument("--average-run", type=Path, required=True)
    parser.add_argument("--layered-evaluation", type=Path)
    parser.add_argument("--average-evaluation", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = {"Layered": args.layered_run, "Average": args.average_run}
    evaluation_paths = {
        "Layered": args.layered_evaluation
        or args.layered_run / "evaluation_test_native_32.json",
        "Average": args.average_evaluation
        or args.average_run / "evaluation_test_common_layered_32.json",
    }
    required = [run_dirs[m] / name for m in METHODS for name in ("history.json", "config.json", "result.json")]
    required.extend(evaluation_paths.values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    histories = {method: _read(run_dirs[method] / "history.json") for method in METHODS}
    history_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for source in histories[method]:
            row = {"method": method, "epoch": int(source["epoch"]) + 1, **source}
            row["epoch"] = int(source["epoch"]) + 1
            history_rows.append(row)
            key_rows.append(
                {
                    "method": method,
                    "epoch": row["epoch"],
                    "cvae_loss": row.get("cvae_loss"),
                    "total_loss": row.get("total_loss"),
                    "ipl_loss": row.get("infeasibility_penalty_loss"),
                    "opl_loss": row.get("optimality_preserving_loss"),
                    "effective_lambda_dfl": row.get("effective_lambda_dfl"),
                    "training_decision_regret_pct": 100.0 * float(row["decision_regret"])
                    if _finite(row.get("decision_regret")) else None,
                    "training_load_shedding_pct": 100.0 * float(row["load_shedding_fraction"])
                    if _finite(row.get("load_shedding_fraction")) else None,
                    "training_carbon_excess_t_per_mwh": row.get("carbon_excess_t_per_mwh"),
                    "validation_objective_usd": row.get("validation_objective"),
                    "validation_load_shedding_pct": 100.0 * float(row["validation_load_shedding_fraction"])
                    if _finite(row.get("validation_load_shedding_fraction")) else None,
                    "validation_carbon_excess_t_per_mwh": row.get("validation_carbon_excess_t_per_mwh"),
                    "is_best_checkpoint": row.get("is_best_checkpoint"),
                }
            )
    _write_csv(args.output_dir / "training_history_full.csv", history_rows)
    _write_csv(args.output_dir / "training_key_metrics.csv", key_rows)

    parameter_rows = []
    training_summary_rows = []
    for method in METHODS:
        config = _flatten(_read(run_dirs[method] / "config.json"))
        result = _read(run_dirs[method] / "result.json")
        parameter_rows.extend(
            {"method": method, "parameter": key, "value": value}
            for key, value in config.items()
        )
        training = result["training_validation"]
        diagnostics = training["recourse_diagnostics"]
        design = result["planning"]["design"]
        training_summary_rows.append(
            {
                "method": method,
                "run_id": result["run_id"],
                "carbon_formulation": result["carbon_formulation"],
                "epochs": result["epochs"],
                "best_epoch": int(result["best_epoch"]) + 1,
                "power_mw": sum(float(x) for x in design["power_mw"].values()),
                "energy_mwh": sum(float(x) for x in design["energy_mwh"].values()),
                "training_validation_objective_usd": training["objective"],
                "training_validation_load_shedding_mwh": diagnostics["load_shedding_mwh"],
                "training_validation_carbon_excess_t": diagnostics["carbon_excess_t"],
                "training_validation_status": training["status"],
                "training_validation_gap_pct": 100.0 * float(training["relative_gap"]),
                "source": str((run_dirs[method] / "result.json").resolve()),
            }
        )
    _write_csv(args.output_dir / "training_parameters.csv", parameter_rows)
    _write_csv(args.output_dir / "training_summary.csv", training_summary_rows)

    evaluations = {method: _read(evaluation_paths[method]) for method in METHODS}
    baseline = evaluations["Layered"]
    for method in METHODS:
        payload = evaluations[method]
        if payload["evaluation_scenario_names"] != baseline["evaluation_scenario_names"]:
            raise ValueError("Evaluation files use different test scenarios")
        if payload["evaluation_carbon_formulation"] != baseline["evaluation_carbon_formulation"]:
            raise ValueError("Evaluation files use different carbon formulations")
    test_rows = [
        _test_row(method, evaluations[method], evaluation_paths[method])
        for method in METHODS
    ]
    scenario_rows = [
        row
        for method in METHODS
        for row in _scenario_rows(method, evaluations[method])
    ]
    _write_csv(args.output_dir / "test_summary.csv", test_rows)
    _write_csv(args.output_dir / "test_scenario_results.csv", scenario_rows)

    _style()
    _plot_history(histories, args.output_dir)
    _plot_test(test_rows, args.output_dir)
    print(f"Wrote CSV tables and IEEE figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
