"""Create an IEEE-style comparison of DFL pretraining and CVAE-only results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path, label: str) -> dict[str, float | str]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    validation = payload["out_of_sample_validation"]
    diagnostics = validation["recourse_diagnostics"]
    design = payload.get("selected_training_design") or validation["design"]
    power = sum(float(value) for value in design["power_mw"].values())
    energy = sum(float(value) for value in design["energy_mwh"].values())
    return {
        "method": label,
        "oos_cost_musd": float(validation["objective"]) / 1e6,
        "regret_pct": 100.0 * float(payload["decision_regret"]),
        "storage_value_kusd": float(payload["storage_value"]) / 1e3,
        "load_shedding_mwh": float(diagnostics["load_shedding_mwh"]),
        "carbon_excess_t": float(diagnostics["carbon_excess_t"]),
        "power_mw": power,
        "energy_mwh": energy,
        "status": str(validation["status"]),
        "relative_gap_pct": 100.0 * float(validation["relative_gap"]),
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
    parser.add_argument("--dfl-pre1", type=Path, required=True)
    parser.add_argument("--dfl-pre50", type=Path, required=True)
    parser.add_argument("--dfl-pre100", type=Path, required=True)
    parser.add_argument("--cvae-only", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    specifications = [
        (args.dfl_pre1, "DFL\n(pretrain 1)"),
        (args.dfl_pre50, "DFL\n(pretrain 50)"),
        (args.dfl_pre100, "DFL\n(pretrain 100)"),
        (args.cvae_only, "CVAE-only\n(300 total)"),
    ]
    missing = [str(path) for path, _ in specifications if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing evaluation files:\n" + "\n".join(missing))
    rows = [_load(path, label) for path, label in specifications]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "pretraining_comparison.csv"
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
    colors = ["#0072B2", "#56B4E9", "#009E73", "#D55E00"]
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.45), constrained_layout=True)
    _bars(axes[0, 0], [r["oos_cost_musd"] for r in rows], colors,
          "OOS total cost (M$)", "(a)")
    _bars(axes[0, 1], [r["regret_pct"] for r in rows], colors,
          "Decision regret (%)", "(b)")
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
    fig.savefig(args.output_dir / "pretraining_comparison_ieee.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "pretraining_comparison_ieee.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(csv_path)


if __name__ == "__main__":
    main()
