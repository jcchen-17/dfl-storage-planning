"""Compile the K=8 main comparison and create a publication-ready figure.

The script combines the latest completed observed-scenario selection suite with
the latest learning-objective suite.  Learning evaluations must cover the full
test split and are expected beside each checkpoint as ``evaluation_182.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SELECTION_ROOT = (
    ROOT
    / "outputs"
    / "baselines"
    / "scenario_selection"
    / "hourly_layered_t1"
    / "suites"
)
DEFAULT_LEARNING_ROOT = (
    ROOT
    / "outputs"
    / "baselines"
    / "learning_objective"
    / "hourly_layered_k8"
    / "suites"
)
DEFAULT_OUTPUT = ROOT / "outputs" / "comparisons" / "main_k8"

METHOD_ORDER = (
    "random",
    "kmeans",
    "farthest",
    "aggregate",
    "cvae_only",
    "ipl_only",
    "opl_only",
    "dfl_full",
)
METHOD_LABELS = {
    "random": "Random",
    "kmeans": "K-means",
    "farthest": "Farthest",
    "aggregate": "Aggregate",
    "cvae_only": "CVAE-only",
    "ipl_only": "IPL-only",
    "opl_only": "OPL-only",
    "dfl_full": "Full DFL",
}
METHOD_GROUPS = {
    "random": "selection",
    "kmeans": "selection",
    "farthest": "selection",
    "aggregate": "selection",
    "cvae_only": "learning",
    "ipl_only": "learning",
    "opl_only": "learning",
    "dfl_full": "proposed",
}


def _latest_directory(root: Path, required: str) -> Path:
    candidates = sorted(
        path for path in root.glob("*") if path.is_dir() and (path / required).exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No suite containing {required!r} below {root}")
    return candidates[-1]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _nested_design_value(design: dict, field: str) -> float:
    values = list(design[field].values())
    if len(values) != 1:
        raise ValueError(f"Expected one storage site in {field}, got {design[field]}")
    return float(values[0])


def _sample_sd(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def _aggregate(method: str, records: list[dict], no_storage: float) -> dict:
    fields = (
        "annual_cost",
        "power_mw",
        "energy_mwh",
        "load_shedding_mwh",
        "carbon_excess_t",
    )
    row: dict[str, object] = {
        "method": method,
        "label": METHOD_LABELS[method],
        "group": METHOD_GROUPS[method],
        "runs": len(records),
        "certified_runs": sum(bool(record["objectives_comparable"]) for record in records),
        "test_scenarios": int(records[0]["test_scenarios"]),
    }
    for field in fields:
        values = [float(record[field]) for record in records]
        row[field] = mean(values)
        row[f"{field}_sd"] = _sample_sd(values)
    storage_values = [no_storage - float(record["annual_cost"]) for record in records]
    row["storage_value"] = mean(storage_values)
    row["storage_value_sd"] = _sample_sd(storage_values)
    row["storage_value_pct"] = 100.0 * float(row["storage_value"]) / no_storage
    row["storage_value_provisional"] = bool(
        int(row["certified_runs"]) < int(row["runs"])
    )
    row["mean_duration_h"] = (
        float(row["energy_mwh"]) / float(row["power_mw"])
        if float(row["power_mw"]) > 0
        else 0.0
    )
    return row


def _selection_records(suite: Path) -> tuple[dict[str, list[dict]], dict]:
    reference = _load_json(suite / "reference.json")
    no_storage = reference["no_storage_reference"]
    expected_names = reference["evaluation_scenario_names"]
    grouped: dict[str, list[dict]] = {name: [] for name in METHOD_ORDER[:4]}
    with (suite / "comparison.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            method = raw["rule"]
            if method not in grouped or int(raw["support_scenarios"]) != 8:
                continue
            payload = _load_json(Path(raw["result"]))
            if payload["evaluation_scenario_names"] != expected_names:
                raise RuntimeError(f"{method} uses a different test set")
            grouped[method].append(
                {
                    "annual_cost": float(raw["test_objective"]),
                    "power_mw": float(raw["power_mw"]),
                    "energy_mwh": float(raw["energy_mwh"]),
                    "load_shedding_mwh": float(raw["load_shedding_mwh"]),
                    "carbon_excess_t": float(raw["carbon_excess_t"]),
                    "objectives_comparable": raw["objectives_comparable"].lower()
                    == "true",
                    "test_scenarios": int(raw["test_scenarios"]),
                    "source": str(Path(raw["result"]).resolve()),
                }
            )
    missing = [method for method, records in grouped.items() if not records]
    if missing:
        raise RuntimeError(f"Selection suite is missing K=8 records for {missing}")
    return grouped, reference


def _learning_records(suite: Path, expected_names: list[str]) -> dict[str, list[dict]]:
    if len(expected_names) != len(set(expected_names)):
        raise RuntimeError("Reference test scenario list contains duplicate names")
    expected_name_set = set(expected_names)
    summary = _load_json(suite / "summary.json")
    grouped: dict[str, list[dict]] = {name: [] for name in METHOD_ORDER[4:]}
    for run in summary["runs"]:
        method = run["variant"]
        if method not in grouped or run.get("status") != "completed":
            continue
        latest = _load_json(Path(run["output_dir"]) / "latest.json")
        checkpoint = Path(latest["checkpoint"])
        candidates = sorted(checkpoint.parent.glob("evaluation_*.json"))
        evaluations = [
            path
            for path in candidates
            if "trajectories" not in path.stem
            and int(_load_json(path).get("test_scenarios_evaluated", 0)) == 182
        ]
        if not evaluations:
            raise FileNotFoundError(
                f"No 182-scenario evaluation beside {checkpoint}; run evaluate.py first"
            )
        evaluation_path = evaluations[-1]
        payload = _load_json(evaluation_path)
        evaluation_names = payload["evaluation_scenario_names"]
        if len(evaluation_names) != len(set(evaluation_names)):
            raise RuntimeError(f"{method} test scenario list contains duplicates")
        if set(evaluation_names) != expected_name_set:
            raise RuntimeError(f"{method} uses a different test set")
        result = payload["out_of_sample_validation"]
        diagnostics = result["recourse_diagnostics"]
        design = payload["selected_training_design"]
        grouped[method].append(
            {
                "annual_cost": float(result["objective"]),
                "power_mw": _nested_design_value(design, "power_mw"),
                "energy_mwh": _nested_design_value(design, "energy_mwh"),
                "load_shedding_mwh": float(diagnostics["load_shedding_mwh"]),
                "carbon_excess_t": float(diagnostics["carbon_excess_t"]),
                "objectives_comparable": bool(payload["objectives_comparable"]),
                "test_scenarios": int(payload["test_scenarios_evaluated"]),
                "source": str(evaluation_path.resolve()),
            }
        )
    missing = [method for method, records in grouped.items() if not records]
    if missing:
        raise RuntimeError(f"Learning suite is missing completed evaluations for {missing}")
    return grouped


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = (
        "method",
        "label",
        "group",
        "runs",
        "certified_runs",
        "test_scenarios",
        "power_mw",
        "power_mw_sd",
        "energy_mwh",
        "energy_mwh_sd",
        "mean_duration_h",
        "annual_cost",
        "annual_cost_sd",
        "storage_value",
        "storage_value_sd",
        "storage_value_pct",
        "storage_value_provisional",
        "load_shedding_mwh",
        "load_shedding_mwh_sd",
        "carbon_excess_t",
        "carbon_excess_t_sd",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)


def _write_latex(path: Path, rows: list[dict]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & $P^{\mathrm{ES}}$ & $E^{\mathrm{ES}}$ & Annual cost & Shedding & Carbon excess & Storage value \\",
        r" & (MW) & (MWh) & (\$) & (MWh) & (tCO$_2$) & (\$) \\",
        r"\midrule",
    ]
    for row in rows:
        marker = r"$^{\dagger}$" if row["storage_value_provisional"] else ""
        lines.append(
            f"{row['label']}{marker} & {row['power_mw']:.3f} & "
            f"{row['energy_mwh']:.3f} & {row['annual_cost']:,.0f} & "
            f"{row['load_shedding_mwh']:.3f} & {row['carbon_excess_t']:.1f} & "
            f"{row['storage_value']:,.0f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path_base: Path, rows: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "selection": "#777777",
        "learning": "#4477AA",
        "proposed": "#CC3311",
    }
    metrics = (
        ("annual_cost", "annual_cost_sd", 1.0e6, "(a) Annual system cost\n(M\$/yr)  $\downarrow$", ".3f"),
        ("load_shedding_mwh", "load_shedding_mwh_sd", 1.0, "(b) Load shedding\n(MWh/yr)  $\downarrow$", ".2f"),
        ("carbon_excess_t", "carbon_excess_t_sd", 1.0, "(c) Carbon-limit excess\n(tCO$_2$/yr)  $\downarrow$", ".0f"),
    )
    positions = np.arange(len(rows))[::-1]
    labels = [str(row["label"]) for row in rows]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 3.35),
        sharey=True,
        gridspec_kw={"wspace": 0.16},
    )

    for axis, (field, error_field, scale, title, value_format) in zip(axes, metrics):
        # The proposed method occupies the same highlighted row in every panel,
        # making its economic/reliability/carbon profile readable at a glance.
        axis.axhspan(-0.42, 0.42, color="#F7D9D2", zorder=0)
        axis.axhline(3.5, color="0.78", linewidth=0.65, zorder=0)
        axis.axhline(0.5, color="0.78", linewidth=0.65, zorder=0)

        plotted_values: list[float] = []
        plotted_errors: list[float] = []
        for position, row in zip(positions, rows):
            value = float(row[field]) / scale
            error = float(row[error_field]) / scale
            plotted_values.append(value)
            plotted_errors.append(error)
            provisional = bool(row["storage_value_provisional"])
            group_color = colors[str(row["group"])]
            axis.errorbar(
                value,
                position,
                xerr=error if error > 0 else None,
                fmt="s" if row["method"] == "dfl_full" else "o",
                markersize=5.8 if row["method"] == "dfl_full" else 4.5,
                markerfacecolor="white" if provisional else group_color,
                markeredgecolor=group_color,
                markeredgewidth=1.15 if provisional else 0.65,
                ecolor="0.45",
                elinewidth=0.75,
                capsize=2.0,
                zorder=3,
            )

        low = min(value - error for value, error in zip(plotted_values, plotted_errors))
        high = max(value + error for value, error in zip(plotted_values, plotted_errors))
        span = max(high - low, abs(high) * 0.025, 1.0e-9)
        axis.set_xlim(low - 0.10 * span, high + 0.31 * span)
        for position, value in zip(positions, plotted_values):
            axis.annotate(
                format(value, value_format),
                (value, position),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=5.8,
                color="0.20",
            )

        axis.set_title(title, pad=6)
        axis.grid(axis="x", color="0.88", linewidth=0.55, zorder=0)
        axis.tick_params(axis="y", length=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)

    axes[0].set_yticks(positions, labels)
    axes[0].tick_params(axis="y", labelleft=True, pad=4)
    axes[0].text(
        -0.34,
        0.78,
        "Scenario\nselection",
        transform=axes[0].transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color="0.35",
    )
    axes[0].text(
        -0.34,
        0.27,
        "Scenario\nlearning",
        transform=axes[0].transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color="0.35",
    )
    fig.text(
        0.51,
        0.015,
        "Lower is better in all panels. Filled markers: certified solution; "
        "open markers: time-limited incumbent. Error bars: $\pm$1 SD across seeds.",
        ha="center",
        va="bottom",
        fontsize=6.1,
        color="0.28",
    )
    fig.subplots_adjust(left=0.205, right=0.995, top=0.85, bottom=0.16)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_performance_matrix(path_base: Path, rows: list[dict]) -> None:
    """Show the three out-of-sample outcomes as one compact result matrix."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fields = ("annual_cost", "load_shedding_mwh", "carbon_excess_t")
    scales = np.array([1.0e6, 1.0, 1.0])
    values = np.array(
        [[float(row[field]) for field in fields] for row in rows], dtype=float
    ) / scales

    # Normalize each physical metric separately. A larger score always means a
    # smaller (better) outcome, while the printed cell retains the true value.
    minima = values.min(axis=0)
    spans = np.maximum(values.max(axis=0) - minima, 1.0e-12)
    scores = 1.0 - (values - minima) / spans

    fig, axis = plt.subplots(figsize=(5.65, 3.15))
    axis.imshow(scores, cmap="Blues", vmin=-0.15, vmax=1.15, aspect="auto")
    axis.set_xticks(
        np.arange(3),
        (
            "Expected annualized\ntotal cost (M\$/yr) $\downarrow$",
            "Load shedding\n(MWh/yr) $\downarrow$",
            "Carbon-limit excess\n(tCO$_2$/yr) $\downarrow$",
        ),
    )
    axis.xaxis.tick_top()
    axis.tick_params(axis="x", length=0, pad=7)
    axis.set_yticks(
        np.arange(len(rows)),
        [str(row["label"]) for row in rows],
    )
    axis.tick_params(axis="y", length=0, pad=7)

    formats = (".3f", ".2f", ".0f")
    for row_index, row in enumerate(rows):
        for column_index, value_format in enumerate(formats):
            score = scores[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                format(values[row_index, column_index], value_format),
                ha="center",
                va="center",
                color="white" if score > 0.62 else "0.12",
                fontsize=7.5,
                fontweight="bold" if row["method"] == "dfl_full" else "normal",
            )

    # White cell boundaries and two stronger separators expose the experiment
    # structure without introducing another legend or marker vocabulary.
    axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.25)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.axhline(3.5, color="white", linewidth=4.0)
    axis.axhline(6.5, color="white", linewidth=4.0)

    proposed_index = next(
        index for index, row in enumerate(rows) if row["method"] == "dfl_full"
    )
    from matplotlib.patches import Rectangle

    axis.add_patch(
        Rectangle(
            (-0.49, proposed_index - 0.48),
            2.98,
            0.96,
            fill=False,
            edgecolor="#B22222",
            linewidth=1.6,
            clip_on=False,
        )
    )
    axis.get_yticklabels()[proposed_index].set_color("#B22222")
    axis.get_yticklabels()[proposed_index].set_fontweight("bold")
    for spine in axis.spines.values():
        spine.set_visible(False)

    fig.text(
        0.5,
        0.025,
        "Darker cells indicate lower values within each column; exact test-set "
        "values are reported in the cells.",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color="0.32",
    )
    fig.subplots_adjust(left=0.225, right=0.99, top=0.78, bottom=0.12)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    positions = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    positions /= ordered_weights.sum()
    return np.interp(probabilities, positions, ordered_values)


def _plot_learning_distributions(
    path_base: Path, comparison_records: dict[str, list[dict]]
) -> None:
    """Compare two observed-scenario baselines with four learned methods."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    methods = (
        "random",
        "kmeans",
        "cvae_only",
        "ipl_only",
        "opl_only",
        "dfl_full",
    )
    labels = [METHOD_LABELS[method] for method in methods]
    colors = (
        "#A6A6A6",
        "#5F5F5F",
        "#76B7B2",
        "#F28E2B",
        "#4E79A7",
        "#E15759",
    )
    line_styles = ("--", ":", "-", "-", "-", "-")
    outcomes: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        value_parts: dict[str, list[np.ndarray]] = {
            "cost": [],
            "shedding": [],
            "carbon": [],
            "weight": [],
        }
        records = comparison_records[method]
        for record in records:
            payload = _load_json(Path(record["source"]))
            scenario_results = payload["out_of_sample_scenario_results"]
            run_weight = np.asarray(
                payload["evaluation_scenario_weights"], dtype=float
            ) / len(records)
            value_parts["weight"].append(run_weight)
            value_parts["cost"].append(
                np.asarray([result["objective"] for result in scenario_results])
                / 1.0e6
            )
            value_parts["shedding"].append(
                np.asarray(
                    [
                        result["recourse_diagnostics"]["load_shedding_mwh"]
                        for result in scenario_results
                    ]
                )
            )
            value_parts["carbon"].append(
                np.asarray(
                    [
                        result["recourse_diagnostics"]["carbon_excess_t"]
                        for result in scenario_results
                    ]
                )
            )
        outcomes[method] = {
            name: np.concatenate(parts) for name, parts in value_parts.items()
        }

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45))

    def draw_weighted_boxes(axis: plt.Axes, field: str) -> None:
        statistics = []
        for method, label in zip(methods, labels):
            values = outcomes[method][field]
            weights = outcomes[method]["weight"]
            q05, q25, q50, q75, q95 = _weighted_quantiles(
                values, weights, np.asarray([0.05, 0.25, 0.50, 0.75, 0.95])
            )
            statistics.append(
                {
                    "label": label,
                    "whislo": q05,
                    "q1": q25,
                    "med": q50,
                    "q3": q75,
                    "whishi": q95,
                    "fliers": [],
                }
            )
        artists = axis.bxp(
            statistics,
            showfliers=False,
            patch_artist=True,
            widths=0.52,
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"color": "0.25", "linewidth": 0.75},
            capprops={"color": "0.25", "linewidth": 0.75},
            boxprops={"edgecolor": "black", "linewidth": 0.7},
        )
        for box, color in zip(artists["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.82)
        axis.axvspan(0.5, 2.5, color="0.95", zorder=0)
        axis.axvline(2.5, color="0.68", linewidth=0.7, linestyle="--")
        axis.tick_params(axis="x", rotation=20)

    draw_weighted_boxes(axes[0], "cost")
    axes[0].set_title("(a) Total cost")
    axes[0].set_ylabel("Annualized two-stage cost (M\$/yr)")

    for method, label, color, line_style in zip(
        methods, labels, colors, line_styles
    ):
        shedding = outcomes[method]["shedding"]
        weights = outcomes[method]["weight"]
        positive = shedding > 1.0e-9
        ordered = np.argsort(shedding[positive])
        x = shedding[positive][ordered]
        ordered_weights = weights[positive][ordered]
        exceedance = np.cumsum(ordered_weights[::-1])[::-1] * 100.0
        x = np.concatenate(([0.0], x))
        exceedance = np.concatenate(([weights[positive].sum() * 100.0], exceedance))
        axes[1].step(
            x,
            exceedance,
            where="post",
            color=color,
            label=label,
            linewidth=1.45 if method == "dfl_full" else 1.05,
            linestyle=line_style,
            zorder=4 if method == "dfl_full" else 3,
        )
    axes[1].set_title("(b) Load-shedding risk")
    axes[1].set_xlabel("Annual unserved energy (MWh/yr)")
    axes[1].set_ylabel("Exceedance probability (\%)")
    axes[1].legend(
        frameon=False,
        loc="upper right",
        handlelength=1.7,
        ncol=2,
        columnspacing=0.8,
    )

    draw_weighted_boxes(axes[2], "carbon")
    axes[2].set_title("(c) Carbon excess")
    axes[2].set_ylabel("Carbon-limit excess (tCO$_2$/yr)")

    for axis in axes:
        axis.grid(axis="y", color="0.87", linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.87, bottom=0.27, wspace=0.37)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_all_method_distributions(
    path_base: Path,
    selection_records: dict[str, list[dict]],
    learning_records: dict[str, list[dict]],
) -> None:
    """Compare selection and learning methods in aligned, separate rows."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    groups = (
        (
            "Scenario selection",
            ("random", "kmeans", "farthest", "aggregate"),
            selection_records,
            ("#8A8A8A", "#6F6F6F", "#555555", "#A5A5A5"),
        ),
        (
            "Scenario learning",
            ("cvae_only", "ipl_only", "opl_only", "dfl_full"),
            learning_records,
            ("#7A7A7A", "#E69F00", "#4C78A8", "#C62828"),
        ),
    )

    def load_outcomes(
        methods: tuple[str, ...], records_by_method: dict[str, list[dict]]
    ) -> dict[str, dict[str, np.ndarray]]:
        loaded: dict[str, dict[str, np.ndarray]] = {}
        for method in methods:
            parts: dict[str, list[np.ndarray]] = {
                "cost": [],
                "shedding": [],
                "carbon": [],
                "weight": [],
            }
            records = records_by_method[method]
            for record in records:
                payload = _load_json(Path(record["source"]))
                scenario_results = payload["out_of_sample_scenario_results"]
                parts["weight"].append(
                    np.asarray(payload["evaluation_scenario_weights"], dtype=float)
                    / len(records)
                )
                parts["cost"].append(
                    np.asarray([result["objective"] for result in scenario_results])
                    / 1.0e6
                )
                parts["shedding"].append(
                    np.asarray(
                        [
                            result["recourse_diagnostics"]["load_shedding_mwh"]
                            for result in scenario_results
                        ]
                    )
                )
                parts["carbon"].append(
                    np.asarray(
                        [
                            result["recourse_diagnostics"]["carbon_excess_t"]
                            for result in scenario_results
                        ]
                    )
                )
            loaded[method] = {
                name: np.concatenate(arrays) for name, arrays in parts.items()
            }
        return loaded

    group_outcomes = [
        load_outcomes(methods, records) for _, methods, records, _ in groups
    ]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.16, 4.2),
        sharey="col",
        gridspec_kw={"hspace": 0.62, "wspace": 0.38},
    )

    def draw_boxes(
        axis: plt.Axes,
        methods: tuple[str, ...],
        outcomes: dict[str, dict[str, np.ndarray]],
        colors: tuple[str, ...],
        field: str,
    ) -> None:
        statistics = []
        for method in methods:
            values = outcomes[method][field]
            weights = outcomes[method]["weight"]
            q05, q25, q50, q75, q95 = _weighted_quantiles(
                values, weights, np.asarray([0.05, 0.25, 0.50, 0.75, 0.95])
            )
            statistics.append(
                {
                    "label": METHOD_LABELS[method],
                    "whislo": q05,
                    "q1": q25,
                    "med": q50,
                    "q3": q75,
                    "whishi": q95,
                    "fliers": [],
                }
            )
        artists = axis.bxp(
            statistics,
            showfliers=False,
            patch_artist=True,
            widths=0.57,
            medianprops={"color": "black", "linewidth": 0.95},
            whiskerprops={"color": "0.25", "linewidth": 0.7},
            capprops={"color": "0.25", "linewidth": 0.7},
            boxprops={"edgecolor": "black", "linewidth": 0.65},
        )
        for box, color in zip(artists["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.82)
        axis.tick_params(axis="x", rotation=16)

    panel_labels = (("a", "b", "c"), ("d", "e", "f"))
    for row_index, ((group_name, methods, _, colors), outcomes) in enumerate(
        zip(groups, group_outcomes)
    ):
        draw_boxes(axes[row_index, 0], methods, outcomes, colors, "cost")
        axes[row_index, 0].set_title(
            f"({panel_labels[row_index][0]}) Total cost"
        )
        axes[row_index, 0].set_ylabel("Annualized two-stage cost\n(M\$/yr)")

        for method, color in zip(methods, colors):
            shedding = outcomes[method]["shedding"]
            weights = outcomes[method]["weight"]
            positive = shedding > 1.0e-9
            order = np.argsort(shedding[positive])
            x = shedding[positive][order]
            ordered_weights = weights[positive][order]
            exceedance = np.cumsum(ordered_weights[::-1])[::-1] * 100.0
            x = np.concatenate(([0.0], x))
            exceedance = np.concatenate(
                ([weights[positive].sum() * 100.0], exceedance)
            )
            axes[row_index, 1].step(
                x,
                exceedance,
                where="post",
                color=color,
                label=METHOD_LABELS[method],
                linewidth=1.45 if method == "dfl_full" else 1.05,
            )
        axes[row_index, 1].set_title(
            f"({panel_labels[row_index][1]}) Load-shedding risk"
        )
        axes[row_index, 1].set_xlabel("Annual unserved energy (MWh/yr)")
        axes[row_index, 1].set_ylabel("Exceedance probability (\%)")
        axes[row_index, 1].legend(
            frameon=False, loc="upper right", handlelength=1.45
        )

        draw_boxes(axes[row_index, 2], methods, outcomes, colors, "carbon")
        axes[row_index, 2].set_title(
            f"({panel_labels[row_index][2]}) Carbon excess"
        )
        axes[row_index, 2].set_ylabel("Carbon-limit excess\n(tCO$_2$/yr)")

    for axis in axes.ravel():
        axis.grid(axis="y", color="0.87", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.95, bottom=0.12)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_pareto_frontiers(path_base: Path, rows: list[dict]) -> None:
    """Plot the cost--reliability and carbon--reliability frontiers."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    included = (
        "random",
        "kmeans",
        "cvae_only",
        "ipl_only",
        "opl_only",
        "dfl_full",
    )
    selected = {str(row["method"]): row for row in rows if row["method"] in included}
    colors = {
        "random": "#707070",
        "kmeans": "#707070",
        "cvae_only": "#4C78A8",
        "ipl_only": "#4C78A8",
        "opl_only": "#4C78A8",
        "dfl_full": "#C62828",
    }
    markers = {
        "random": "o",
        "kmeans": "o",
        "cvae_only": "s",
        "ipl_only": "s",
        "opl_only": "s",
        "dfl_full": "*",
    }
    offsets = {
        0: {
            "random": (5, 5),
            "kmeans": (-42, 7),
            "cvae_only": (5, 5),
            "ipl_only": (-45, -5),
            "opl_only": (5, -10),
            "dfl_full": (8, 7),
        },
        1: {
            "random": (5, 5),
            "kmeans": (7, 7),
            "cvae_only": (-45, 5),
            "ipl_only": (-48, 16),
            "opl_only": (-43, -10),
            "dfl_full": (9, 8),
        },
    }

    panels = (
        (
            "annual_cost",
            1.0e6,
            "(a) Cost--reliability frontier",
            "Expected annualized total cost (M\$/yr)",
        ),
        (
            "carbon_excess_t",
            1.0,
            "(b) Carbon--reliability frontier",
            "Expected carbon-limit excess (tCO$_2$/yr)",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
    for panel_index, (field, scale, title, xlabel) in enumerate(panels):
        axis = axes[panel_index]
        points: list[tuple[float, float, str]] = []
        for method in included:
            row = selected[method]
            x = float(row[field]) / scale
            y = float(row["load_shedding_mwh"])
            points.append((x, y, method))
            proposed = method == "dfl_full"
            axis.scatter(
                x,
                y,
                s=82 if proposed else 34,
                marker=markers[method],
                facecolor=colors[method] if proposed else "white",
                edgecolor=colors[method],
                linewidth=1.35 if proposed else 1.0,
                zorder=5 if proposed else 4,
            )
            axis.annotate(
                METHOD_LABELS[method],
                (x, y),
                xytext=offsets[panel_index][method],
                textcoords="offset points",
                fontsize=6.8,
                color=colors[method],
                fontweight="bold" if proposed else "normal",
                zorder=6,
            )

        # With both objectives minimized, a point belongs to the frontier when
        # no point to its left has an equal or smaller reliability loss.
        frontier: list[tuple[float, float]] = []
        best_y = float("inf")
        for x, y, _ in sorted(points):
            if y < best_y - 1.0e-12:
                frontier.append((x, y))
                best_y = y
        if len(frontier) > 1:
            axis.plot(
                [point[0] for point in frontier],
                [point[1] for point in frontier],
                color="0.38",
                linewidth=0.8,
                linestyle="--",
                zorder=2,
            )

        x_values = [point[0] for point in points]
        x_span = max(x_values) - min(x_values)
        axis.set_xlim(min(x_values) - 0.05 * x_span, max(x_values) + 0.12 * x_span)
        axis.set_ylim(0.0, max(point[1] for point in points) * 1.05)
        axis.text(
            0.025,
            0.035,
            "better $\swarrow$",
            transform=axis.transAxes,
            fontsize=6.3,
            color="0.35",
        )
        axis.set_title(title, pad=5)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Expected unserved energy (MWh/yr)")
        axis.grid(color="0.88", linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.09, right=0.995, top=0.89, bottom=0.19, wspace=0.28)
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-suite", type=Path, default=None)
    parser.add_argument("--learning-suite", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    selection_suite = (
        args.selection_suite.resolve()
        if args.selection_suite
        else _latest_directory(DEFAULT_SELECTION_ROOT, "comparison.csv")
    )
    learning_suite = (
        args.learning_suite.resolve()
        if args.learning_suite
        else _latest_directory(DEFAULT_LEARNING_ROOT, "summary.json")
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selection, reference = _selection_records(selection_suite)
    learning = _learning_records(
        learning_suite, reference["evaluation_scenario_names"]
    )
    no_storage = float(reference["no_storage_reference"]["objective"])
    grouped = {**selection, **learning}
    rows = [_aggregate(method, grouped[method], no_storage) for method in METHOD_ORDER]

    payload = {
        "support_scenarios": 8,
        "test_scenarios": len(reference["evaluation_scenario_names"]),
        "no_storage_objective": no_storage,
        "selection_suite": str(selection_suite),
        "learning_suite": str(learning_suite),
        "methods": rows,
    }
    (output_dir / "main_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(output_dir / "main_comparison.csv", rows)
    _write_latex(output_dir / "main_comparison_table.tex", rows)
    _plot_pareto_frontiers(output_dir / "fig_main_comparison", rows)
    print(f"Wrote main comparison to {output_dir}")


if __name__ == "__main__":
    main()
