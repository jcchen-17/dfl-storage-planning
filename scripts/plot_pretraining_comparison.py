"""Create an IEEE-style comparison of DFL pretraining and CVAE-only results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(
    path: Path,
    label: str,
    reference_objective: float | None = None,
    reference_best_bound: float | None = None,
    reference_status: str | None = None,
    reference_gap: float | None = None,
) -> dict[str, float | str]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    validation = payload["out_of_sample_validation"]
    diagnostics = validation["recourse_diagnostics"]
    design = payload.get("selected_training_design") or validation["design"]
    power = sum(float(value) for value in design["power_mw"].values())
    energy = sum(float(value) for value in design["energy_mwh"].values())
    regret = payload.get("decision_regret")
    if regret is None and reference_objective is not None:
        regret = (
            float(validation["objective"]) - reference_objective
        ) / (abs(reference_objective) + 1.0e-8)
    regret_upper = float("nan")
    if reference_best_bound is not None:
        regret_upper = 100.0 * (
            float(validation["objective"]) - reference_best_bound
        ) / (abs(reference_best_bound) + 1.0e-8)
    return {
        "method": label,
        "oos_cost_musd": float(validation["objective"]) / 1e6,
        "regret_pct": 100.0 * float(regret),
        "regret_upper_bound_pct": regret_upper,
        "storage_value_kusd": float(payload["storage_value"]) / 1e3,
        "load_shedding_mwh": float(diagnostics["load_shedding_mwh"]),
        "carbon_excess_t": float(diagnostics["carbon_excess_t"]),
        "power_mw": power,
        "energy_mwh": energy,
        "status": str(validation["status"]),
        "relative_gap_pct": 100.0 * float(validation["relative_gap"]),
        "regret_reference_status": reference_status or "not_recorded",
        "regret_reference_gap_pct": (
            100.0 * reference_gap if reference_gap is not None else float("nan")
        ),
        "source": str(path.resolve()),
    }


def _bars(ax, values, colors, ylabel, panel, lower_is_better=True):
    x = np.arange(len(values))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.55)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.grid(axis="y", color="0.85", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.text(0.01, 0.97, panel, transform=ax.transAxes, va="top", fontweight="bold")
    span = max(values) - min(values)
    offset = max(span * 0.035, max(values) * 0.008, 1e-8)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset,
                f"{value:.4g}", ha="center", va="bottom", fontsize=6.5)
    ax.margins(y=0.16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dfl-pre1", type=Path)
    parser.add_argument("--dfl-pre50", type=Path)
    parser.add_argument("--dfl-pre100", type=Path)
    parser.add_argument("--cvae-only", type=Path)
    parser.add_argument(
        "--evaluation",
        action="append",
        default=[],
        metavar="LABEL=JSON",
        help="generic evaluation input; repeat once per method",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="pretraining_comparison")
    args = parser.parse_args()

    if args.evaluation:
        specifications = []
        for specification in args.evaluation:
            if "=" not in specification:
                parser.error("--evaluation must use LABEL=JSON syntax")
            label, raw_path = specification.split("=", 1)
            specifications.append((Path(raw_path), label))
    else:
        legacy = (
            args.dfl_pre1, args.dfl_pre50, args.dfl_pre100, args.cvae_only
        )
        if any(path is None for path in legacy):
            parser.error(
                "provide one or more --evaluation LABEL=JSON inputs, or all four "
                "legacy pretraining inputs"
            )
        specifications = [
            (args.dfl_pre1, "DFL\n(pretrain 1)"),
            (args.dfl_pre50, "DFL\n(pretrain 50)"),
            (args.dfl_pre100, "DFL\n(pretrain 100)"),
            (args.cvae_only, "CVAE-only\n(300 total)"),
        ]
    missing = [str(path) for path, _ in specifications if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing evaluation files:\n" + "\n".join(missing))
    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path, _ in specifications
    ]
    references = [
        reference
        for payload in payloads
        for reference in [payload.get("perfect_information_reference")]
        if reference is not None
    ]
    reference = references[0] if references else None
    reference_objective = float(reference["objective"]) if reference else None
    reference_best_bound = (
        float(reference["best_bound"])
        if reference and reference.get("best_bound") is not None
        else None
    )
    reference_status = str(reference["status"]) if reference else None
    reference_gap = float(reference["relative_gap"]) if reference else None
    if any(payload.get("decision_regret") is None for payload in payloads):
        if reference_objective is None:
            raise ValueError(
                "At least one evaluation has no decision_regret and none contains "
                "a perfect-information reference."
            )
    baseline = payloads[0]
    for payload in payloads[1:]:
        if payload["evaluation_scenario_names"] != baseline["evaluation_scenario_names"]:
            raise ValueError("Evaluation files use different held-out scenarios.")
        weight_delta = max(
            abs(float(left) - float(right))
            for left, right in zip(
                payload["evaluation_scenario_weights"],
                baseline["evaluation_scenario_weights"],
                strict=True,
            )
        )
        if weight_delta > 1.0e-12:
            raise ValueError("Evaluation files use different held-out weights.")
        if payload["evaluation_carbon_formulation"] != baseline[
            "evaluation_carbon_formulation"
        ]:
            raise ValueError("Evaluation files use different carbon formulations.")
    rows = [
        _load(
            path,
            label,
            reference_objective,
            reference_best_bound,
            reference_status,
            reference_gap,
        )
        for path, label in specifications
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / f"{args.stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7, "legend.fontsize": 6.7,
        "axes.linewidth": 0.65, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    palette = ["#0072B2", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]
    colors = [palette[index % len(palette)] for index in range(len(rows))]
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.45), constrained_layout=True)
    _bars(axes[0, 0], [r["oos_cost_musd"] for r in rows], colors,
          "OOS total cost (M$)", "(a)")
    regret_label = (
        "Decision regret (%)"
        if reference_status in {None, "optimal", "gaplimit"}
        else "Regret vs. PI incumbent (%)"
    )
    _bars(axes[0, 1], [r["regret_pct"] for r in rows], colors,
          regret_label, "(b)")
    _bars(axes[0, 2], [r["storage_value_kusd"] for r in rows], colors,
          "Storage value (k$)", "(c)", lower_is_better=False)
    _bars(axes[1, 0], [r["load_shedding_mwh"] for r in rows], colors,
          "Load shedding (MWh)", "(d)")
    _bars(axes[1, 1], [r["carbon_excess_t"] for r in rows], colors,
          r"Carbon excess (tCO$_2$)", "(e)")

    ax = axes[1, 2]
    x = np.arange(len(rows)); width = 0.36
    ax.bar(x - width / 2, [r["power_mw"] for r in rows], width,
           color="#4E79A7", edgecolor="black", linewidth=0.5, label="Power (MW)")
    ax.bar(x + width / 2, [r["energy_mwh"] for r in rows], width,
           color="#F28E2B", edgecolor="black", linewidth=0.5, label="Energy (MWh)")
    ax.set_ylabel("Installed capacity")
    ax.set_xticks(x); ax.set_xticklabels([])
    ax.grid(axis="y", color="0.85", linewidth=0.55); ax.set_axisbelow(True)
    ax.text(0.01, 0.97, "(f)", transform=ax.transAxes, va="top", fontweight="bold")
    ax.legend(frameon=False, ncol=1, loc="upper right")

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black", linewidth=.5)
               for c in colors]
    fig.legend(handles, [label.replace("\n", " ") for _, label in specifications],
               loc="outside lower center", ncol=4, frameon=False)
    fig.savefig(args.output_dir / f"{args.stem}_ieee.pdf", bbox_inches="tight")
    fig.savefig(
        args.output_dir / f"{args.stem}_ieee.png", dpi=600, bbox_inches="tight"
    )
    plt.close(fig)
    print(csv_path)


if __name__ == "__main__":
    main()
