"""Summarize and plot the comparable K=1 storage-planning experiments.

The main K-means-initialized DFL result was produced on another machine and is
therefore recorded explicitly below from its evaluation console output.  All
other values are read from local JSON artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "k1_comparison"
RANDOM_CONTROL = ROOT / "outputs" / "random_subset_control_k1.json"
RANDOM_INIT_RESULT = (
    ROOT
    / "outputs"
    / "dataset_v2_dfl_hourly_layered_tou125_carbon200_random_init"
    / "result_reinforce.json"
)
RANDOM_INIT_HISTORY = RANDOM_INIT_RESULT.with_name("dfl_history_reinforce.json")

# User-provided evaluation output from the other machine.  Its no-storage
# reference and four test scenarios match the local K=1 control exactly, but
# its solver status/gap were not included in the pasted output.
MAIN_DFL = {
    "method": "DFL (K-means init)",
    "objective": 1_597_939.92,
    "reference": 1_635_614.95,
    "bus": "680",
    "power_mw": 0.6455567582714171,
    "energy_mwh": 2.721861985156063,
    "certification": "status/gap not reported",
    "source": "user-provided evaluation console (other machine)",
}


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def installed_design(design: dict) -> tuple[str, float, float]:
    buses = design.get("installed_buses")
    if buses is None:
        buses = [bus for bus, value in design["site"].items() if value > 0]
    power = sum(float(value) for value in design["power_mw"].values())
    energy = sum(float(value) for value in design["energy_mwh"].values())
    return ",".join(buses) if buses else "none", power, energy


def certified(status: str, gap: float | None, requested_gap: float = 0.01) -> bool:
    return status in {"optimal", "gaplimit"} and gap is not None and gap <= requested_gap


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    control = load_json(RANDOM_CONTROL)
    random_init = load_json(RANDOM_INIT_RESULT)
    history = load_json(RANDOM_INIT_HISTORY)
    assert isinstance(control, dict) and isinstance(random_init, dict)
    assert isinstance(history, list)

    reference = float(control["baselines"]["no_storage"]["objective"])
    evaluation_names = control["fingerprint"]["evaluation_names"]
    if random_init["evaluation_scenario_names"] != evaluation_names:
        raise RuntimeError("Random-init DFL and random control use different test scenarios.")
    if abs(float(random_init["no_storage_reference"]["objective"]) - reference) > 1e-4:
        raise RuntimeError("No-storage references do not match.")

    rows: list[dict] = []
    for name in ("kmeans", "farthest"):
        record = control["baselines"][name]
        bus, power, energy = installed_design(record["design"])
        status, gap = record["evaluation_status"], float(record["evaluation_gap"])
        rows.append(
            {
                "method": "K-means" if name == "kmeans" else "Farthest",
                "objective": float(record["decision_loss"]),
                "storage_value": reference - float(record["decision_loss"]),
                "relative_saving_pct": 100
                * (reference - float(record["decision_loss"]))
                / reference,
                "bus": bus,
                "power_mw": power,
                "energy_mwh": energy,
                "evaluation_status": status,
                "evaluation_gap": gap,
                "certified_at_1pct": certified(status, gap),
                "source": str(RANDOM_CONTROL.relative_to(ROOT)),
            }
        )

    draw_rows = []
    for index, record in enumerate(control["draws"], start=1):
        bus, power, energy = installed_design(record["design"])
        value = reference - float(record["decision_loss"])
        status, gap = record["evaluation_status"], float(record["evaluation_gap"])
        draw_rows.append(
            {
                "draw": index,
                "scenario": record["names"][0],
                "objective": float(record["decision_loss"]),
                "storage_value": value,
                "bus": bus,
                "power_mw": power,
                "energy_mwh": energy,
                "planning_status": record["planning_status"],
                "planning_gap": float(record["planning_gap"]),
                "evaluation_status": status,
                "evaluation_gap": gap,
                "certified_at_1pct": certified(status, gap),
                "seconds": float(record["seconds"]),
            }
        )
    best_draw = max(draw_rows, key=lambda row: row["storage_value"])
    rows.append(
        {
            "method": "Random best-of-20",
            "objective": best_draw["objective"],
            "storage_value": best_draw["storage_value"],
            "relative_saving_pct": 100 * best_draw["storage_value"] / reference,
            "bus": best_draw["bus"],
            "power_mw": best_draw["power_mw"],
            "energy_mwh": best_draw["energy_mwh"],
            "evaluation_status": best_draw["evaluation_status"],
            "evaluation_gap": best_draw["evaluation_gap"],
            "certified_at_1pct": best_draw["certified_at_1pct"],
            "source": f"{RANDOM_CONTROL.relative_to(ROOT)} draw {best_draw['draw']}",
        }
    )

    validation = random_init["out_of_sample_validation"]
    bus, power, energy = installed_design(validation["design"])
    value = reference - float(validation["objective"])
    rows.append(
        {
            "method": "DFL (random init)",
            "objective": float(validation["objective"]),
            "storage_value": value,
            "relative_saving_pct": 100 * value / reference,
            "bus": bus,
            "power_mw": power,
            "energy_mwh": energy,
            "evaluation_status": validation["status"],
            "evaluation_gap": float(validation["relative_gap"]),
            "certified_at_1pct": bool(random_init["objectives_comparable"]),
            "source": str(RANDOM_INIT_RESULT.relative_to(ROOT)),
        }
    )

    main_value = MAIN_DFL["reference"] - MAIN_DFL["objective"]
    rows.append(
        {
            "method": MAIN_DFL["method"],
            "objective": MAIN_DFL["objective"],
            "storage_value": main_value,
            "relative_saving_pct": 100 * main_value / MAIN_DFL["reference"],
            "bus": MAIN_DFL["bus"],
            "power_mw": MAIN_DFL["power_mw"],
            "energy_mwh": MAIN_DFL["energy_mwh"],
            "evaluation_status": "not reported",
            "evaluation_gap": "",
            "certified_at_1pct": False,
            "source": MAIN_DFL["source"],
        }
    )

    fieldnames = list(rows[0])
    with (OUTPUT / "k1_method_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "k1_random_draws.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(draw_rows[0]))
        writer.writeheader()
        writer.writerows(draw_rows)

    method_order = [
        "K-means",
        "Farthest",
        "DFL (random init)",
        "DFL (K-means init)",
        "Random best-of-20",
    ]
    by_method = {row["method"]: row for row in rows}
    plot_rows = [by_method[name] for name in method_order]
    colors = ["#9aa0a6", "#9aa0a6", "#f4a261", "#2a9d8f", "#457b9d"]

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    values = [row["storage_value"] / 1000 for row in plot_rows]
    bars = ax.bar(method_order, values, color=colors, edgecolor="#303030", linewidth=0.8)
    for bar, row, value_k in zip(bars, plot_rows, values, strict=True):
        if not row["certified_at_1pct"]:
            bar.set_hatch("///")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(value_k, 0) + 0.8,
            f"{value_k:.1f}k",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Storage value ($ thousands)")
    ax.set_title("A. Out-of-sample storage value (higher is better)")
    ax.tick_params(axis="x", rotation=18)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.text(
        0.01,
        0.97,
        "Hatched = provisional (1% gap not certified or status unavailable)",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        color="#7f1d1d",
    )

    ax = axes[0, 1]
    x = np.arange(1, len(draw_rows) + 1)
    draw_values = np.array([row["storage_value"] / 1000 for row in draw_rows])
    is_certified = np.array([row["certified_at_1pct"] for row in draw_rows])
    ax.scatter(x[is_certified], draw_values[is_certified], label="certified at 1%", color="#2a9d8f", s=42)
    ax.scatter(
        x[~is_certified],
        draw_values[~is_certified],
        label="scenario limit (provisional)",
        facecolors="none",
        edgecolors="#d1495b",
        s=52,
        linewidth=1.4,
    )
    ax.plot(x, np.maximum.accumulate(draw_values), color="#457b9d", linewidth=2, label="running best")
    ax.axhline(main_value / 1000, color="#2a9d8f", linestyle="--", linewidth=1.5, label="DFL K-means init")
    ax.set_xlabel("Random draw")
    ax.set_ylabel("Storage value ($ thousands)")
    ax.set_title("B. Random K=1 draws and best-of-budget")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1, 0]
    epochs = np.array([int(row["epoch"]) + 1 for row in history])
    objectives = np.array([float(row["validation_objective"]) / 1e6 for row in history])
    nonempty = np.array([bool(row["installed_buses"]) for row in history])
    ax.plot(epochs, objectives, color="#264653", linewidth=1.5, alpha=0.8)
    ax.scatter(epochs[nonempty], objectives[nonempty], color="#e76f51", s=40, label="storage design")
    ax.scatter(epochs[~nonempty], objectives[~nonempty], color="#9aa0a6", s=34, label="no storage")
    ax.axhline(1.4282, color="black", linestyle=":", label="training no-storage level")
    best_epoch = int(np.argmin(objectives))
    ax.annotate(
        f"best: epoch {epochs[best_epoch]}\n${objectives[best_epoch]:.5f}M",
        (epochs[best_epoch], objectives[best_epoch]),
        xytext=(epochs[best_epoch] + 2, objectives[best_epoch] - 0.004),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        fontsize=9,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training validation objective ($ millions)")
    ax.set_title("C. Random-init DFL training trajectory")
    ax.set_xticks([1, 5, 10, 15])
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    design_methods = ["DFL (random init)", "DFL (K-means init)", "Random best-of-20"]
    design_rows = [by_method[name] for name in design_methods]
    positions = np.arange(len(design_methods))
    width = 0.36
    p_bars = ax.bar(
        positions - width / 2,
        [row["power_mw"] for row in design_rows],
        width,
        label="Power (MW)",
        color="#e9c46a",
    )
    e_bars = ax.bar(
        positions + width / 2,
        [row["energy_mwh"] for row in design_rows],
        width,
        label="Energy (MWh)",
        color="#457b9d",
    )
    for bars_group in (p_bars, e_bars):
        for bar in bars_group:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{bar.get_height():.2f}",
                ha="center",
                fontsize=8.5,
            )
    ax.set_xticks(positions, [f"{row['method']}\nbus {row['bus']}" for row in design_rows], rotation=12)
    ax.set_ylabel("Installed capacity")
    ax.set_title("D. Non-zero K=1 designs")
    ax.legend(fontsize=9)

    fig.suptitle(
        "K=1 scenario-support comparison | shared test set: 4 v2-2018 representatives",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUTPUT / "k1_comprehensive_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUTPUT / "k1_comprehensive_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    meta = {
        "shared_no_storage_reference": reference,
        "evaluation_scenarios": evaluation_names,
        "random_draw_count": len(draw_rows),
        "random_positive_draws": sum(row["storage_value"] > 1e-8 for row in draw_rows),
        "random_certified_draws": sum(row["certified_at_1pct"] for row in draw_rows),
        "random_best_draw": best_draw,
        "random_mean_storage_value_provisional": float(
            np.mean([row["storage_value"] for row in draw_rows])
        ),
        "random_init_best_training_epoch": int(epochs[best_epoch]),
        "random_init_best_training_objective": float(objectives[best_epoch] * 1e6),
        "excluded_as_incomparable": [
            "outputs/dataset_v2_baselines/exhaustive_k1_v2.json (different 16-scenario test set)",
            "outputs/dfl_k1_*_seed*/result_reinforce.json (older dataset and different 16-scenario test set)",
        ],
    }
    (OUTPUT / "k1_comparison_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {OUTPUT / 'k1_comprehensive_comparison.png'}")
    print(f"wrote {OUTPUT / 'k1_method_summary.csv'}")


if __name__ == "__main__":
    main()
