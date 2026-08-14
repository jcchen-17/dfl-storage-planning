"""Plot observed-scenario baseline trade-offs from one completed suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITES = (
    ROOT
    / "outputs"
    / "baselines"
    / "scenario_selection"
    / "hourly_layered_t1"
    / "suites"
)
BASE_METHODS = ("random", "aggregate", "kmeans", "farthest")
LABELS = {
    "random": "Random",
    "aggregate": "Aggregate",
    "kmeans": "K-means",
    "farthest": "Farthest",
}
COLORS = {
    "random": "#0072B2",
    "aggregate": "#CC79A7",
    "kmeans": "#D55E00",
    "farthest": "#009E73",
}
MARKERS = {"random": "o", "aggregate": "D", "kmeans": "s", "farthest": "^"}


def _latest_suite(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No scenario-selection suites found below {root}.")
    return candidates[-1]


def _bool_column(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().eq("true")


def _summary_table(frame: pd.DataFrame, reference_objective: float) -> pd.DataFrame:
    rows = []
    for method in frame["rule"].drop_duplicates():
        group = frame[frame["rule"] == method]
        if group.empty:
            continue
        row: dict[str, float | int | str] = {
            "method": LABELS[method],
            "runs": len(group),
            "fully_converged": int(group["fully_converged"].sum()),
        }
        for field in (
            "test_objective",
            "storage_value_plot",
            "load_shedding_mwh",
            "carbon_excess_t",
            "power_mw",
            "energy_mwh",
        ):
            values = pd.to_numeric(group[field], errors="coerce")
            row[f"{field}_mean"] = float(values.mean())
            row[f"{field}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row["storage_saving_percent"] = (
            100.0 * float(row["storage_value_plot_mean"]) / reference_objective
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _draw_method(
    ax: plt.Axes,
    group: pd.DataFrame,
    method: str,
    xfield: str,
    yfield: str,
) -> None:
    x = pd.to_numeric(group[xfield], errors="coerce").to_numpy(float)
    y = pd.to_numeric(group[yfield], errors="coerce").to_numpy(float)
    certified = group["fully_converged"].to_numpy(bool)
    color = COLORS[method]
    marker = MARKERS[method]
    if certified.any():
        ax.scatter(
            x[certified],
            y[certified],
            marker=marker,
            s=52,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.82,
            zorder=3,
        )
    if (~certified).any():
        ax.scatter(
            x[~certified],
            y[~certified],
            marker=marker,
            s=58,
            facecolor="none",
            edgecolor=color,
            linewidth=1.4,
            alpha=0.95,
            zorder=3,
        )

    mean_x, mean_y = float(np.mean(x)), float(np.mean(y))
    if len(x) > 1:
        sd_x = float(np.std(x, ddof=1))
        sd_y = float(np.std(y, ddof=1))
        ax.errorbar(
            mean_x,
            mean_y,
            xerr=sd_x,
            yerr=sd_y,
            fmt="X",
            markersize=10,
            markerfacecolor=color,
            markeredgecolor="#202020",
            markeredgewidth=0.8,
            ecolor=color,
            elinewidth=1.3,
            capsize=3,
            alpha=0.95,
            zorder=4,
        )
    label_offset = {
        "random": (8, 12),
        "aggregate": (-58, -18),
        "kmeans": (8, 10),
        "farthest": (8, 9),
    }.get(method, (8, 10))
    ax.annotate(
        LABELS[method],
        (mean_x, mean_y),
        xytext=label_offset,
        textcoords="offset points",
        color=color,
        fontsize=9,
        fontweight="bold",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot economic, reliability, carbon, and sizing trade-offs."
    )
    parser.add_argument(
        "--suite",
        default=None,
        help="suite directory; default uses the latest hourly_layered_t1 suite",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="default: <suite>/figures",
    )
    parser.add_argument(
        "--learned-evaluation",
        action="append",
        default=[],
        metavar="LABEL=JSON",
        help=(
            "append a learned-method evaluation; repeat for CVAE-only/DFL-full"
        ),
    )
    args = parser.parse_args()

    suite = Path(args.suite).resolve() if args.suite else _latest_suite(DEFAULT_SUITES)
    comparison_path = suite / "comparison.csv"
    reference_path = suite / "reference.json"
    if not comparison_path.exists() or not reference_path.exists():
        raise FileNotFoundError(
            f"{suite} must contain comparison.csv and reference.json."
        )
    frame = pd.read_csv(comparison_path)
    if frame.empty:
        raise ValueError(f"No completed runs are recorded in {comparison_path}.")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    no_storage = reference["no_storage_reference"]
    reference_objective = float(no_storage["objective"])
    reference_diagnostics = no_storage["recourse_diagnostics"]

    plot_methods = [method for method in BASE_METHODS if method in set(frame["rule"])]
    learned_colors = ("#E69F00", "#6A3D9A", "#56B4E9", "#8C564B")
    learned_markers = ("P", "*", "v", "h")
    reference_weight_map = dict(
        zip(
            reference["evaluation_scenario_names"],
            reference["evaluation_scenario_weights"],
            strict=True,
        )
    )
    learned_rows = []
    for index, specification in enumerate(args.learned_evaluation):
        if "=" not in specification:
            parser.error("--learned-evaluation must use LABEL=JSON syntax")
        label, raw_path = specification.split("=", 1)
        learned_path = Path(raw_path).expanduser().resolve()
        learned = json.loads(learned_path.read_text(encoding="utf-8"))
        learned_weight_map = dict(
            zip(
                learned["evaluation_scenario_names"],
                learned["evaluation_scenario_weights"],
                strict=True,
            )
        )
        if set(learned_weight_map) != set(reference_weight_map):
            raise ValueError(f"{label} uses a different held-out scenario set.")
        maximum_weight_delta = max(
            abs(learned_weight_map[name] - reference_weight_map[name])
            for name in reference_weight_map
        )
        if maximum_weight_delta > 1.0e-12:
            raise ValueError(
                f"{label} uses different held-out weights; maximum delta "
                f"{maximum_weight_delta:.3e}."
            )
        learned_reference = float(learned["no_storage_reference"]["objective"])
        if abs(learned_reference - reference_objective) > 1.0e-4:
            raise ValueError(f"{label} uses a different no-storage reference.")
        evaluated = learned["out_of_sample_validation"]
        planning = learned["planning"]
        accepted_gap = float(learned["evaluation_relative_gap"])
        planning_gap = float(planning["relative_gap"])
        planning_converged = planning["status"] in {"optimal", "gaplimit"} or (
            np.isfinite(planning_gap) and planning_gap <= accepted_gap
        )
        diagnostics = evaluated["recourse_diagnostics"]
        design = evaluated["design"]
        method = f"learned_{index}"
        LABELS[method] = label
        COLORS[method] = learned_colors[index % len(learned_colors)]
        MARKERS[method] = learned_markers[index % len(learned_markers)]
        plot_methods.append(method)
        learned_rows.append(
            {
                "rule": method,
                "seed": learned.get("training_run_id", index),
                "planning_converged": planning_converged,
                "objectives_comparable": bool(learned["objectives_comparable"]),
                "test_objective": float(evaluated["objective"]),
                "storage_value": learned.get("storage_value"),
                "power_mw": float(design["power_mw"]["PCC"]),
                "energy_mwh": float(design["energy_mwh"]["PCC"]),
                "load_shedding_mwh": float(diagnostics["load_shedding_mwh"]),
                "carbon_excess_t": float(diagnostics["carbon_excess_t"]),
            }
        )
    if learned_rows:
        frame = pd.concat([frame, pd.DataFrame(learned_rows)], ignore_index=True)

    frame["planning_converged"] = _bool_column(frame["planning_converged"])
    frame["objectives_comparable"] = _bool_column(frame["objectives_comparable"])
    frame["fully_converged"] = (
        frame["planning_converged"] & frame["objectives_comparable"]
    )
    # Plot the achieved incumbent for every method, while visually separating
    # rows that have not yet met the requested planning/evaluation gap.
    frame["storage_value_plot"] = reference_objective - pd.to_numeric(
        frame["test_objective"], errors="coerce"
    )
    frame["storage_value_k"] = frame["storage_value_plot"] / 1000.0

    output_dir = Path(args.output_dir).resolve() if args.output_dir else suite / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_table(frame, reference_objective)
    summary.to_csv(output_dir / "scenario_selection_plot_summary.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.7,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.29, wspace=0.28)

    panels = (
        (
            axes[0],
            "storage_value_k",
            "load_shedding_mwh",
            "A. Economic value vs reliability",
            "Annual storage value ($k/year)",
            "Expected load shedding (MWh)",
        ),
        (
            axes[1],
            "storage_value_k",
            "carbon_excess_t",
            "B. Economic value vs carbon compliance",
            "Annual storage value ($k/year)",
            "Expected carbon excess (tCO$_2$)",
        ),
        (
            axes[2],
            "power_mw",
            "energy_mwh",
            "C. Installed storage design",
            "Battery power (MW)",
            "Battery energy (MWh)",
        ),
    )
    for ax, xfield, yfield, title, xlabel, ylabel in panels:
        for method in plot_methods:
            group = frame[frame["rule"] == method]
            if not group.empty:
                _draw_method(ax, group, method, xfield, yfield)
        ax.set_title(title, loc="left", pad=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    axes[0].set_yscale("log")
    axes[0].scatter(
        [0.0],
        [float(reference_diagnostics["load_shedding_mwh"])],
        marker="*",
        s=110,
        color="#303030",
        zorder=5,
    )
    axes[0].annotate(
        "No storage",
        (0.0, float(reference_diagnostics["load_shedding_mwh"])),
        xytext=(7, -2),
        textcoords="offset points",
        fontsize=9,
    )
    axes[0].text(0.98, 0.96, "higher value →\nlower shedding ↓", transform=axes[0].transAxes, ha="right", va="top", color="#555555")

    axes[1].scatter(
        [0.0],
        [float(reference_diagnostics["carbon_excess_t"])],
        marker="*",
        s=110,
        color="#303030",
        zorder=5,
    )
    axes[1].annotate(
        "No storage",
        (0.0, float(reference_diagnostics["carbon_excess_t"])),
        xytext=(7, -2),
        textcoords="offset points",
        fontsize=9,
    )
    axes[1].text(0.98, 0.96, "higher value →\nlower excess ↓", transform=axes[1].transAxes, ha="right", va="top", color="#555555")

    axes[2].scatter([0.0], [0.0], marker="*", s=110, color="#303030", zorder=5)
    axes[2].annotate("No storage", (0.0, 0.0), xytext=(7, 5), textcoords="offset points", fontsize=9)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[method],
            color="none",
            markerfacecolor=COLORS[method],
            markeredgecolor=COLORS[method],
            markersize=7,
            label=LABELS[method],
        )
        for method in plot_methods
        if method in set(frame["rule"])
    ]
    status_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="white", markersize=7, label="gap certified"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#666666", markersize=7, label="provisional incumbent"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#666666", markeredgecolor="#202020", markersize=8, label="method mean ± SD"),
    ]
    fig.legend(
        handles=method_handles + status_handles,
        loc="outside lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.055),
    )
    comparison_title = (
        "Observed and learned scenario methods"
        if learned_rows
        else "Observed-scenario storage-planning baselines"
    )
    fig.suptitle(
        f"{comparison_title} — K=8, "
        f"N$_{{test}}$={int(reference['test_scenarios_evaluated'])}",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.018,
        "Open markers did not meet the requested 0.1% planning or evaluation gap; "
        "their achieved incumbent is shown for screening only.",
        ha="center",
        fontsize=8.5,
        color="#7A1F1F",
    )

    png = output_dir / "scenario_selection_tradeoffs.png"
    pdf = output_dir / "scenario_selection_tradeoffs.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {output_dir / 'scenario_selection_plot_summary.csv'}")


if __name__ == "__main__":
    main()
