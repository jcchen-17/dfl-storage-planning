"""Generate an SVG schematic of the proposed unbalanced IEEE 13-bus system.

This script has no third-party dependencies.  It distinguishes the physical
IEEE 13-node feeder from proposed planning devices:

* data center: bus 675;
* PV plants: buses 634 (0.25 MW), 675 (0.65 MW), and 680 (0.75 MW);
* BESS candidate buses: 632, 671, 675, and 680;
* existing dispatchable generator: bus 671.

Run from the repository root:

    python scripts/plot_ieee13_topology.py

The default output is ``outputs/ieee13_planning_topology.svg``.  SVG is a
browser-readable vector image and can be inserted into Word or converted to
PNG/PDF without losing resolution.
"""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Branch:
    source: str
    target: str
    phases: str
    length_ft: int
    linecode: str
    kind: str = "line"


BUS_PHASES = {
    "650": "ABC",
    "632": "ABC",
    "633": "ABC",
    "634": "ABC",
    "645": "BC",
    "646": "BC",
    "671": "ABC",
    "680": "ABC",
    "684": "AC",
    "611": "C",
    "652": "A",
    "692": "ABC",
    "675": "ABC",
}

# These are display coordinates, not geographic coordinates or line lengths.
POSITIONS = {
    "650": (160, 250),
    "632": (445, 250),
    "633": (730, 115),
    "634": (1035, 115),
    "645": (730, 305),
    "646": (1035, 305),
    "671": (730, 540),
    "680": (1120, 540),
    "684": (990, 725),
    "611": (1345, 675),
    "652": (1345, 805),
    "692": (730, 900),
    "675": (1120, 900),
}

BRANCHES = (
    Branch("650", "632", "ABC", 2000, "601", "regulator"),
    Branch("632", "633", "ABC", 500, "602"),
    Branch("633", "634", "ABC", 0, "XFM-1", "transformer"),
    Branch("632", "645", "BC", 500, "603"),
    Branch("645", "646", "BC", 300, "603"),
    Branch("632", "671", "ABC", 2000, "601", "distributed-load line"),
    Branch("671", "680", "ABC", 1000, "601"),
    Branch("671", "684", "AC", 300, "604"),
    Branch("684", "611", "C", 300, "605"),
    Branch("684", "652", "A", 800, "607", "underground"),
    Branch("671", "692", "ABC", 0, "switch", "closed switch"),
    Branch("692", "675", "ABC", 500, "606", "underground"),
)

STORAGE_CANDIDATES = {"632", "671", "675", "680"}
PV_MW = {"634": 0.25, "675": 0.65, "680": 0.75}
DATA_CENTER = {"bus": "675", "peak_mw": 0.60, "transformer_mva": 0.75}
GENERATOR = {"bus": "671", "capacity_mw": 1.00}

# Native IEEE loads.  For delta loads, the phase label is phase-to-phase.
LOAD_LABELS = {
    "634": "Y-PQ  400+j290 kVA",
    "645": "Y-PQ  170+j125 kVA (B)",
    "646": "Δ-Z  230+j132 kVA (BC)",
    "671": "Δ-PQ  1155+j660 kVA",
    "675": "Y-PQ  843+j462 kVA",
    "692": "Δ-I  170+j151 kVA (CA)",
    "611": "Y-I  170+j80 kVA (C)",
    "652": "Y-Z  128+j86 kVA (A)",
}

PHASE_COLORS = {
    "ABC": "#263238",
    "BC": "#e67e22",
    "AC": "#8e44ad",
    "A": "#c0392b",
    "C": "#16845b",
}


def validate_topology() -> None:
    """Fail early if the drawing data no longer describe a radial 13-bus feeder."""

    if len(BUS_PHASES) != 13:
        raise ValueError("IEEE 13 topology must contain exactly 13 named buses.")
    if len(BRANCHES) != 12:
        raise ValueError("A radial 13-bus feeder must contain 12 branches.")
    children = [branch.target for branch in BRANCHES]
    if len(children) != len(set(children)):
        raise ValueError("Every non-root bus must have exactly one parent.")
    if set(children) != set(BUS_PHASES) - {"650"}:
        raise ValueError("The feeder must connect every non-root bus to the root.")
    for bus in STORAGE_CANDIDATES | set(PV_MW) | {str(DATA_CENTER["bus"])}:
        if BUS_PHASES[bus] != "ABC":
            raise ValueError(f"Planning device at {bus} requires an ABC bus.")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(
    x: float,
    y: float,
    lines: str | tuple[str, ...],
    *,
    size: int = 14,
    color: str = "#263238",
    anchor: str = "middle",
    weight: str = "normal",
    line_height: int | None = None,
    css_class: str = "",
) -> str:
    if isinstance(lines, str):
        lines = (lines,)
    line_height = line_height or int(size * 1.25)
    class_attr = f' class="{esc(css_class)}"' if css_class else ""
    tspans = []
    start_y = y - (len(lines) - 1) * line_height / 2
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else line_height
        tspan_y = start_y if index == 0 else start_y + index * line_height
        tspans.append(f'<tspan x="{x}" y="{tspan_y}" dy="{dy * 0}">{esc(line)}</tspan>')
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}"{class_attr}>'
        + "".join(tspans)
        + "</text>"
    )


def label_box(
    x: float,
    y: float,
    lines: str | tuple[str, ...],
    *,
    width: float,
    height: float,
    size: int = 13,
    fill: str = "#ffffff",
    stroke: str = "none",
    color: str = "#263238",
    weight: str = "normal",
    radius: int = 7,
) -> str:
    if isinstance(lines, str):
        lines = (lines,)
    return (
        f'<g><rect x="{x - width / 2}" y="{y - height / 2}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.4" opacity="0.96"/>'
        + svg_text(x, y, lines, size=size, color=color, weight=weight, line_height=int(size * 1.25))
        + "</g>"
    )


def branch_description(branch: Branch) -> tuple[str, ...]:
    if branch.kind == "transformer":
        return ("XFM-1  500 kVA", "4.16/0.48 kV  ABC")
    if branch.kind == "closed switch":
        return ("闭合开关 / ABC",)
    base = f"{branch.length_ft} ft / {branch.linecode} / {branch.phases}"
    if branch.kind == "regulator":
        return ("三相调压器", base)
    if branch.kind == "distributed-load line":
        return (base, "沿线负荷 200+j116 kVA")
    if branch.kind == "underground":
        return (base + " / 地下",)
    return (base,)


def branch_label_position(branch: Branch) -> tuple[float, float, float, float]:
    x1, y1 = POSITIONS[branch.source]
    x2, y2 = POSITIONS[branch.target]
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    custom = {
        ("650", "632"): (mx, my - 45, 185, 46),
        ("632", "633"): (mx - 10, my - 32, 170, 30),
        ("633", "634"): (mx, my + 38, 190, 48),
        ("632", "645"): (mx + 10, my + 25, 170, 30),
        ("645", "646"): (mx, my - 28, 170, 30),
        ("632", "671"): (mx - 15, my, 205, 48),
        ("671", "680"): (mx, my - 30, 185, 30),
        ("671", "684"): (mx + 5, my + 5, 170, 30),
        ("684", "611"): (mx, my - 28, 170, 30),
        ("684", "652"): (mx, my + 30, 205, 30),
        ("671", "692"): (mx - 55, my, 130, 30),
        ("692", "675"): (mx, my - 28, 200, 30),
    }
    return custom[(branch.source, branch.target)]


def draw_branch(branch: Branch) -> str:
    x1, y1 = POSITIONS[branch.source]
    x2, y2 = POSITIONS[branch.target]
    color = PHASE_COLORS[branch.phases]
    dash = ' stroke-dasharray="11 7"' if branch.kind == "closed switch" else ""
    lx, ly, width, height = branch_label_position(branch)
    label = label_box(
        lx,
        ly,
        branch_description(branch),
        width=width,
        height=height,
        size=12,
        fill="#ffffff",
        color=color,
    )
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="5" stroke-linecap="round"{dash}/>'
        + label
    )


def draw_bus(bus: str) -> str:
    x, y = POSITIONS[bus]
    phase = BUS_PHASES[bus]
    pieces = ['<g class="bus">']
    if bus in STORAGE_CANDIDATES:
        pieces.append(
            f'<circle cx="{x}" cy="{y}" r="30" fill="none" stroke="#f39c12" stroke-width="7"/>'
        )
    pieces.extend(
        [
            f'<circle cx="{x}" cy="{y}" r="21" fill="#ffffff" stroke="{PHASE_COLORS[phase]}" stroke-width="4"/>',
            svg_text(x, y + 5, bus, size=14, color="#17202a", weight="bold"),
            svg_text(x, y + 46, phase, size=13, color=PHASE_COLORS[phase], weight="bold"),
            "</g>",
        ]
    )
    return "".join(pieces)


def draw_load(bus: str, text: str) -> str:
    x, y = POSITIONS[bus]
    centers = {
        "634": (x, y + 70),
        "645": (x, y - 48),
        "646": (x, y + 76),
        "671": (x - 142, y - 47),
        "675": (x, y - 50),
        "692": (x - 135, y + 45),
        "611": (x, y - 48),
        "652": (x, y + 50),
    }
    cx, cy = centers[bus]
    return label_box(
        cx,
        cy,
        text,
        width=196,
        height=28,
        size=11,
        fill="#f4f6f7",
        stroke="#bdc3c7",
        color="#455a64",
    )


def draw_pv(bus: str, capacity_mw: float) -> str:
    x, y = POSITIONS[bus]
    positions = {
        "634": (1165, 115, 72),
        "680": (x, y - 78, y - 118),
        "675": (1005, 973, 1017),
    }
    px, py, label_y = positions[bus]
    points = f"{px - 21},{py + 13} {px + 21},{py + 13} {px},{py - 22}"
    connectors = {
        "634": (x + 23, y, px - 23, py),
        "680": (x, y - 23, px, py + 15),
        "675": (x - 23, y + 2, px + 18, py - 20),
    }
    x1, y1, x2, y2 = connectors[bus]
    return (
        f'<g><line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#9a7d0a" stroke-width="2.5"/>'
        + f'<polygon points="{points}" fill="#f4d03f" stroke="#9a7d0a" stroke-width="2.5"/>'
        + svg_text(px, label_y, f"PV {capacity_mw:.2f} MW", size=13, color="#7d6608", weight="bold")
        + "</g>"
    )


def draw_special_devices() -> str:
    # Upstream grid and source transformer.
    pieces = [
        '<g><rect x="25" y="218" width="64" height="64" rx="8" fill="#d6eaf8" stroke="#2471a3" stroke-width="3"/>',
        svg_text(57, 244, ("上级", "电网"), size=13, color="#154360", weight="bold", line_height=16),
        '<line x1="89" y1="250" x2="139" y2="250" stroke="#263238" stroke-width="5"/>',
        label_box(112, 198, ("5 MVA", "115 kV Δ / 4.16 kV Yg"), width=185, height=47, size=11),
        "</g>",
    ]

    # Existing generator at 671.
    gx, gy = POSITIONS[str(GENERATOR["bus"])]
    pieces.extend(
        [
            f'<line x1="{gx}" y1="{gy + 22}" x2="{gx}" y2="{gy + 78}" stroke="#1e8449" stroke-width="3"/>',
            f'<rect x="{gx - 51}" y="{gy + 78}" width="102" height="36" rx="6" fill="#d5f5e3" stroke="#1e8449" stroke-width="2.5"/>',
            svg_text(gx, gy + 101, "G  1.00 MW", size=13, color="#145a32", weight="bold"),
        ]
    )

    # Proposed data center at 675.
    dx, dy = POSITIONS[str(DATA_CENTER["bus"])]
    pieces.extend(
        [
            f'<line x1="{dx + 22}" y1="{dy}" x2="{dx + 83}" y2="{dy}" stroke="#7d3c98" stroke-width="4"/>',
            f'<rect x="{dx + 83}" y="{dy - 55}" width="250" height="110" rx="12" fill="#e8daef" stroke="#7d3c98" stroke-width="3"/>',
            svg_text(
                dx + 208,
                dy,
                ("数据中心 DC（ABC）", "设施峰值 0.60 MW", "专用变压器 0.75 MVA"),
                size=13,
                color="#512e5f",
                weight="bold",
                line_height=19,
            ),
        ]
    )

    # Existing capacitor banks.
    x611, y611 = POSITIONS["611"]
    pieces.append(svg_text(x611 + 118, y611 + 8, "电容 100 kvar (C)", size=12, color="#117864"))
    x675, y675 = POSITIONS["675"]
    pieces.append(svg_text(x675 - 122, y675 - 86, "电容 600 kvar (ABC)", size=12, color="#117864"))
    return "".join(pieces)


def draw_legend() -> str:
    x, y = 42, 42
    rows = [
        ("ABC", "ABC 三相线路"),
        ("BC", "BC 两相线路"),
        ("AC", "AC 两相线路"),
        ("A", "A 单相线路"),
        ("C", "C 单相线路"),
    ]
    pieces = [
        f'<g><rect x="{x}" y="{y}" width="420" height="132" rx="10" fill="#ffffff" stroke="#bfc9ca" stroke-width="1.5" opacity="0.97"/>',
        svg_text(x + 20, y + 22, "图例", size=14, anchor="start", weight="bold"),
    ]
    for index, (phase, label) in enumerate(rows):
        col = index % 2
        row = index // 2
        px = x + 22 + col * 190
        py = y + 48 + row * 25
        pieces.append(f'<line x1="{px}" y1="{py}" x2="{px + 38}" y2="{py}" stroke="{PHASE_COLORS[phase]}" stroke-width="5"/>')
        pieces.append(svg_text(px + 48, py + 5, label, size=11, anchor="start"))

    py = y + 110
    pieces.extend(
        [
            f'<circle cx="{x + 218}" cy="{py}" r="12" fill="white" stroke="#f39c12" stroke-width="4"/>',
            svg_text(x + 237, py + 5, "BESS 候选（尚未建设）", size=11, anchor="start"),
            f'<polygon points="{x + 25},{py + 8} {x + 50},{py + 8} {x + 37.5},{py - 12}" fill="#f4d03f" stroke="#9a7d0a" stroke-width="2"/>',
            svg_text(x + 58, py + 5, "规划 PV", size=11, anchor="start"),
            "</g>",
        ]
    )
    return "".join(pieces)


def draw_planning_summary() -> str:
    return label_box(
        1505,
        330,
        (
            "规划设备位置",
            "数据中心：675",
            "PV：634 / 675 / 680",
            "BESS 候选：632 / 671 / 675 / 680",
            "主实验：最多建设 1 处 BESS",
        ),
        width=405,
        height=142,
        size=14,
        fill="#fef9e7",
        stroke="#d4ac0d",
        color="#2c3e50",
        weight="bold",
        radius=12,
    )


def build_svg() -> str:
    validate_topology()
    body = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1900" height="1080" viewBox="0 0 1900 1080">',
        "<style>text { font-family: 'Microsoft YaHei', 'Noto Sans CJK SC', 'SimHei', sans-serif; }</style>",
        '<rect width="1900" height="1080" fill="#fbfcfc"/>',
        svg_text(950, 33, "IEEE 13 节点三相不平衡系统与数据中心—光伏—储能规划拓扑", size=25, weight="bold"),
        svg_text(950, 62, "物理拓扑采用 IEEE 13 节点；橙色外环仅表示储能候选位置", size=14, color="#566573"),
        draw_legend(),
    ]
    body.extend(draw_branch(branch) for branch in BRANCHES)
    body.extend(draw_bus(bus) for bus in BUS_PHASES)
    body.extend(draw_load(bus, text) for bus, text in LOAD_LABELS.items())
    body.extend(draw_pv(bus, capacity) for bus, capacity in PV_MW.items())
    body.extend((draw_special_devices(), draw_planning_summary()))
    body.append(
        svg_text(
            950,
            1052,
            "原生负荷总计：3.466 MW + j2.102 Mvar；PV 总装机：1.65 MW；数据中心峰值：0.60 MW",
            size=14,
            color="#34495e",
            weight="bold",
        )
    )
    body.append("</svg>")
    return "\n".join(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ieee13_planning_topology.svg"),
        help="Output SVG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_svg(), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
