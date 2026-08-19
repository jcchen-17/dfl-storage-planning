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
        "selection": "#7F7F7F",
        "learning": "#4C78A8",
        "proposed": "#D62728",
    }
    hatches = ("//", "\\\\", "xx", "..", "--", "++", "oo", "")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.05), constrained_layout=True)

    labels = [
        str(row["label"]) + ("$^{\dagger}$" if row["storage_value_provisional"] else "")
        for row in rows
    ]
    values = np.array([float(row["storage_value"]) / 1000.0 for row in rows])
    errors = np.array([float(row["storage_value_sd"]) / 1000.0 for row in rows])
    positions = np.arange(len(rows))
    bars = axes[0].bar(
        positions,
        values,
        yerr=errors,
        width=0.72,
        color=[colors[str(row["group"])] for row in rows],
        edgecolor="black",
        linewidth=0.55,
        capsize=2.0,
        error_kw={"elinewidth": 0.7, "capthick": 0.7},
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axes[0].axhline(0.0, color="black", linewidth=0.65)
    axes[0].set_ylabel("Annual cost reduction (k\$/yr)")
    axes[0].set_xticks(positions, labels, rotation=32, ha="right")
    axes[0].set_title("(a) Economic value")
    axes[0].grid(axis="y", color="0.88", linewidth=0.55)

    marker_map = {
        "random": "o",
        "kmeans": "s",
        "farthest": "^",
        "aggregate": "D",
        "cvae_only": "P",
        "ipl_only": "v",
        "opl_only": "X",
        "dfl_full": "*",
    }
    scatter_handles = []
    scatter_labels = []
    for row in rows:
        method = str(row["method"])
        x = float(row["carbon_excess_t"])
        y = float(row["load_shedding_mwh"])
        xerr = float(row["carbon_excess_t_sd"])
        yerr = float(row["load_shedding_mwh_sd"])
        artist = axes[1].errorbar(
            x,
            y,
            xerr=xerr if xerr > 0 else None,
            yerr=yerr if yerr > 0 else None,
            fmt=marker_map[method],
            markersize=8 if method == "dfl_full" else 5,
            markerfacecolor=colors[str(row["group"])],
            markeredgecolor="black",
            markeredgewidth=0.5,
            ecolor="0.45",
            elinewidth=0.65,
            capsize=2,
            zorder=4 if method == "dfl_full" else 3,
        )
        scatter_handles.append(artist[0])
        scatter_labels.append(str(row["label"]) + (
            r"$^{\dagger}$" if row["storage_value_provisional"] else ""
        ))
    axes[1].set_xlabel("Carbon-limit excess (tCO$_2$/yr)")
    axes[1].set_ylabel("Load shedding (MWh/yr)")
    axes[1].set_title("(b) Feasibility trade-off")
    axes[1].grid(color="0.88", linewidth=0.55)
    axes[1].legend(
        scatter_handles,
        scatter_labels,
        loc="upper left",
        ncol=2,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.35,
        fontsize=6.3,
    )
    axes[1].annotate(
        "Preferred",
        xy=(0.04, 0.05),
        xytext=(0.22, 0.24),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "linewidth": 0.7, "color": "0.25"},
        fontsize=7,
        color="0.25",
    )
    axes[1].text(
        0.985,
        0.018,
        r"$^{\dagger}$ time-limited incumbent(s)",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="0.3",
    )
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.savefig(path_base.with_suffix(".pdf"))
    fig.savefig(path_base.with_suffix(".png"), dpi=600)
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
    _plot(output_dir / "fig_main_comparison", rows)
    print(f"Wrote main comparison to {output_dir}")


if __name__ == "__main__":
    main()
