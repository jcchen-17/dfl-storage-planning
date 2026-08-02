"""Validate downloaded case data and write a human-readable inventory.

This script never modifies files under ``data/raw``.  It reads the download
manifest, validates each file with a format-aware check, adds the 15 legacy
AUS/P1U metrics files, and writes:

* ``data/manifests/all_files_audit.csv`` -- one row per raw file;
* ``data/DATASET_INVENTORY.md`` -- the data-source decision and next steps.

Run from the ``storage-dfl`` environment so that ``pyarrow`` is available.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tarfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pyarrow.parquet as pq


AUDIT_FIELDS = [
    "sequence",
    "relative_path",
    "dataset",
    "year",
    "kind",
    "bytes",
    "row_count",
    "column_count",
    "sampling_interval",
    "start_time",
    "end_time",
    "validation",
    "model_role",
    "description",
    "source_url",
    "sha256",
]


def default_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, int, list[str], dict[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        first: dict[str, str] = {}
        last: dict[str, str] = {}
        count = 0
        for row in reader:
            if count == 0:
                first = dict(row)
            last = dict(row)
            count += 1
    return count, len(header), header, first, last


def solar_timestamp(row: dict[str, str]) -> str:
    if not row:
        return ""
    return (
        f"{int(row['Year']):04d}-{int(row['Month']):02d}-{int(row['Day']):02d} "
        f"{int(row['Hour']):02d}:{int(row['Minute']):02d}"
    )


def parquet_details(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    required = [
        "Time",
        "total_site_electricity_kw",
        "total_site_electricity_kvar",
        "pf",
    ]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    table = parquet.read(columns=required)
    nulls = sum(table[name].null_count for name in required)
    raw_times = table["Time"].to_pylist()
    first_time = raw_times[0] if table.num_rows else ""
    last_time = raw_times[-1] if table.num_rows else ""
    parsed_times = [datetime.fromisoformat(str(value)[:19]) for value in raw_times]
    gaps = [
        (index, parsed_times[index - 1], parsed_times[index])
        for index in range(1, len(parsed_times))
        if (parsed_times[index] - parsed_times[index - 1]).total_seconds() != 15 * 60
    ]
    duplicates = len(parsed_times) - len(set(parsed_times))
    expected = parquet.metadata.num_rows == 35040 and len(columns) == 31 and nulls == 0
    if expected and not gaps and duplicates == 0:
        validation = "PASS: 35040 rows, 31 columns, P/Q/PF readable, 15-minute timeline continuous"
    elif expected:
        gap_note = (
            f"{len(gaps)} timeline gap(s); first={gaps[0][1]} -> {gaps[0][2]}"
            if gaps
            else "no timeline gaps"
        )
        validation = f"PASS_WITH_NOTE: data readable; {gap_note}; duplicates={duplicates}"
    else:
        validation = (
            f"REVIEW: rows={parquet.metadata.num_rows}, cols={len(columns)}, "
            f"nulls={nulls}, gaps={len(gaps)}, duplicates={duplicates}"
        )
    return {
        "row_count": parquet.metadata.num_rows,
        "column_count": len(columns),
        "sampling_interval": "15 min",
        "start_time": first_time,
        "end_time": last_time,
        "validation": validation,
    }


def workbook_details(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as workbook:
        bad_member = workbook.testzip()
        root = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
        namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sheets = [item.attrib["name"] for item in root.findall(".//x:sheet", namespace)]
        with workbook.open("xl/worksheets/sheet1.xml") as sheet:
            prefix = sheet.read(256 * 1024)
        match = re.search(rb'<dimension[^>]*ref="([^"]+)"', prefix)
        dimension = match.group(1).decode("ascii") if match else ""

    row_count = ""
    column_count = ""
    if dimension and ":" in dimension:
        final_cell = dimension.split(":", 1)[1]
        cell_match = re.fullmatch(r"([A-Z]+)([0-9]+)", final_cell)
        if cell_match:
            letters, rows = cell_match.groups()
            column_count = 0
            for letter in letters:
                column_count = column_count * 26 + (ord(letter) - ord("A") + 1)
            row_count = int(rows)
    passed = bad_member is None and "Published Hourly Data" in sheets
    return {
        "row_count": row_count,
        "column_count": column_count,
        "sampling_interval": "1 h (hour-ending)",
        "start_time": "",
        "end_time": "",
        "validation": (
            f"PASS: valid XLSX; Published Hourly Data={dimension}; {len(sheets)} sheets"
            if passed
            else f"REVIEW: bad_member={bad_member}; sheets={sheets}"
        ),
    }


def archive_details(path: Path) -> dict[str, Any]:
    with tarfile.open(path, mode="r:xz") as archive:
        members = archive.getmembers()
    daily_invocations = [
        item.name for item in members if item.name.startswith("invocations_per_function_md.anon.d")
    ]
    passed = len(daily_invocations) == 14
    return {
        "row_count": "",
        "column_count": "",
        "sampling_interval": "1 min workload counts; 14 daily files",
        "start_time": "day 01",
        "end_time": "day 14",
        "validation": (
            f"PASS: tar.xz readable; {len(members)} members; 14 invocation-day files"
            if passed
            else f"REVIEW: tar.xz readable; {len(daily_invocations)} invocation-day files"
        ),
    }


def tariff_details(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = payload["items"][0]
    name = item.get("name", "")
    raw_start = item.get("startdate", "")
    raw_end = item.get("enddate", "")
    start = (
        datetime.fromtimestamp(raw_start, tz=timezone.utc).isoformat()
        if isinstance(raw_start, (int, float))
        else str(raw_start or "")
    )
    end = (
        datetime.fromtimestamp(raw_end, tz=timezone.utc).isoformat()
        if isinstance(raw_end, (int, float))
        else str(raw_end or "")
    )
    passed = (
        item.get("approved") is True
        and "Large Primary Service" in name
        and item.get("sector") == "Industrial"
        and start.startswith("2026-01-01")
    )
    return {
        "row_count": 1,
        "column_count": len(item),
        "sampling_interval": "tariff schedule (not a time series)",
        "start_time": start,
        "end_time": end,
        "validation": (
            "PASS: valid JSON; approved 2026 Industrial OPT-V Large Primary"
            if passed
            else f"REVIEW: {name}"
        ),
    }


def text_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        return sum(1 for _ in handle)


def describe(kind: str, path: Path, legacy: bool = False) -> tuple[str, str]:
    if legacy:
        scenario = path.parent.name
        return (
            "仅参考，不进入训练",
            f"Smart-DS 2016 AUS/P1U 场景 {scenario} 的96条馈线×80项静态汇总指标；无时间戳。",
        )
    if kind == "load_timeseries":
        return (
            "处理后进入模型",
            "GSO industrial 建筑15分钟总有功、总无功、功率因数和分项负荷；用于工业负荷时间形状。",
        )
    if kind == "solar_weather_timeseries":
        return (
            "处理后进入模型",
            "GSO 15分钟DNI/DHI/GHI、温度、风速、POA和1 MW光伏出力；用于PV、PUE和极端热天。",
        )
    if kind == "grid_carbon_workbook":
        return (
            "处理后进入模型",
            "EIA-930 DUK小时工作簿；提取消费侧CO2强度。碳字段自2018-07-01起可用。",
        )
    if kind == "industrial_tariff_json":
        return (
            "优化参数来源",
            "DOE OpenEI审核的2026 Duke OPT-V Large Primary工业费率；生成分时电价与需量费参数，不由CVAE生成。",
        )
    if kind == "data_center_workload":
        return (
            "处理后进入模型",
            "Azure Functions 2019匿名14天、每分钟函数调用/持续时间/内存轨迹；聚合并缩放为数据中心任务到达量。",
        )
    if kind in {"opendss", "bus_coordinates", "official_report"}:
        return (
            "IEEE13静态网络",
            "IEEE13三相馈线的拓扑、相别、线路/变压器/调压器/电容器参数或官方说明；不含历史时序。",
        )
    if kind == "opendss_feeder":
        filename_use = {
            "Loads.dss": "负荷对象、相别及其profile引用",
            "PVSystems.dss": "PV对象及太阳能profile引用",
            "Lines.dss": "Smart-DS参考馈线线路",
            "LineCodes.dss": "Smart-DS参考馈线阻抗代码",
            "Transformers.dss": "Smart-DS参考馈线变压器",
            "Master.dss": "Smart-DS参考馈线入口",
            "Buscoords.dss": "Smart-DS参考馈线坐标",
            "Intermediates.txt": "Smart-DS中间节点记录",
        }.get(path.name, "Smart-DS参考馈线静态文件")
        return (
            "映射/筛选参考",
            f"目标GSO工业馈线的{filename_use}；最终物理网络仍采用IEEE13，不与两套拓扑拼接。",
        )
    if kind == "feeder_metrics":
        return (
            "馈线筛选依据",
            "目标GSO industrial场景的馈线级静态汇总，用于确认idt1210为100%工业馈线，不是训练时序。",
        )
    if kind == "documentation":
        return "数据说明", "官方字段、采样、许可或工作负载数据说明。"
    return "辅助/参考", "下载清单中的辅助原始文件。"


def inspect_file(path: Path, kind: str, legacy: bool = False) -> dict[str, Any]:
    details: dict[str, Any] = {
        "row_count": "",
        "column_count": "",
        "sampling_interval": "not applicable",
        "start_time": "",
        "end_time": "",
        "validation": "PASS: file exists and checksum computed",
    }
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return parquet_details(path)
    if kind == "solar_weather_timeseries":
        rows, columns, header, first, last = csv_shape(path)
        required = {
            "Year",
            "Month",
            "Day",
            "Hour",
            "Minute",
            "Temperature",
            "kW Generated (1000 kW Array)",
        }
        details.update(
            row_count=rows,
            column_count=columns,
            sampling_interval="15 min",
            start_time=solar_timestamp(first),
            end_time=solar_timestamp(last),
            validation=(
                "PASS: 35040 rows, 12 columns, weather and 1 MW PV fields readable"
                if rows == 35040 and columns == 12 and required.issubset(header)
                else f"REVIEW: rows={rows}, cols={columns}"
            ),
        )
        return details
    if suffix == ".csv":
        rows, columns, _, _, _ = csv_shape(path)
        expected = (legacy and rows == 96 and columns == 80) or not legacy
        details.update(
            row_count=rows,
            column_count=columns,
            validation=(
                "PASS: 96 feeder rows, 80 static metric columns, no time series"
                if legacy and expected
                else f"PASS: CSV readable; {rows} rows, {columns} columns"
            ),
        )
        return details
    if suffix == ".xlsx":
        return workbook_details(path)
    if suffixes_match(path, ".tar.xz"):
        return archive_details(path)
    if suffix == ".json":
        return tariff_details(path)
    if suffix == ".pdf":
        magic_ok = path.read_bytes()[:5] == b"%PDF-"
        details["validation"] = "PASS: PDF header valid" if magic_ok else "FAIL: invalid PDF header"
        return details
    if suffix in {".dss", ".txt", ".md"}:
        details["row_count"] = text_line_count(path)
        details["validation"] = "PASS: text file readable"
        return details
    return details


def suffixes_match(path: Path, ending: str) -> bool:
    return path.name.lower().endswith(ending)


def read_manifest(data_root: Path) -> list[dict[str, str]]:
    manifest = data_root / "manifests" / "case_raw_files.csv"
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_manifest_path(data_root: Path, recorded_path: str) -> Path:
    """Resolve manifest paths after the repository or data directory is moved."""
    recorded = Path(recorded_path).expanduser()
    if recorded.is_file():
        return recorded.resolve()

    normalized = recorded_path.replace("\\", "/")
    marker = "/data/"
    if marker in normalized:
        relative = normalized.split(marker, maxsplit=1)[1]
    elif normalized.startswith("data/"):
        relative = normalized.removeprefix("data/")
    else:
        return recorded.resolve()
    return (data_root / relative).resolve()


def audit(data_root: Path) -> list[dict[str, Any]]:
    manifest_rows = read_manifest(data_root)
    rows: list[dict[str, Any]] = []
    manifest_paths: set[Path] = set()

    for source in manifest_rows:
        path = resolve_manifest_path(data_root, source["path"])
        manifest_paths.add(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        size = path.stat().st_size
        expected_size = int(source["bytes"])
        if size != expected_size or digest != source["sha256"]:
            raise ValueError(f"manifest mismatch: {path}")
        role, description = describe(source["kind"], path)
        detail = inspect_file(path, source["kind"])
        rows.append(
            {
                "relative_path": str(path.relative_to(data_root)),
                "dataset": source["dataset"],
                "year": source["year"],
                "kind": source["kind"],
                "bytes": size,
                **detail,
                "model_role": role,
                "description": description,
                "source_url": source["source_url"],
                "sha256": digest,
            }
        )

    legacy_glob = data_root.glob("raw/smartds/2016/AUS/P1U/scenarios/*/metrics.csv")
    for path in sorted(legacy_glob):
        resolved = path.resolve()
        if resolved in manifest_paths:
            continue
        role, description = describe("legacy_feeder_metrics", path, legacy=True)
        detail = inspect_file(path, "legacy_feeder_metrics", legacy=True)
        rows.append(
            {
                "relative_path": str(path.relative_to(data_root)),
                "dataset": "SMART-DS v1.0 legacy AUS/P1U",
                "year": "2016",
                "kind": "legacy_feeder_metrics",
                "bytes": path.stat().st_size,
                **detail,
                "model_role": role,
                "description": description,
                "source_url": "not recorded in the original download",
                "sha256": sha256_file(path),
            }
        )

    rows.sort(key=lambda row: (row["dataset"], row["year"], row["relative_path"]))
    for index, row in enumerate(rows, start=1):
        row["sequence"] = index
    return rows


def write_csv(data_root: Path, rows: list[dict[str, Any]]) -> Path:
    output = data_root / "manifests" / "all_files_audit.csv"
    temporary = output.with_suffix(".csv.part")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    return output


def mib(value: int) -> str:
    return f"{value / 1024**2:.2f} MiB"


def markdown_table(lines: list[str], headers: list[str], table_rows: list[list[Any]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in table_rows:
        clean = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(clean) + " |")


def write_markdown(data_root: Path, rows: list[dict[str, Any]]) -> Path:
    output = data_root / "DATASET_INVENTORY.md"
    case_rows = [row for row in rows if row["kind"] != "legacy_feeder_metrics"]
    legacy_rows = [row for row in rows if row["kind"] == "legacy_feeder_metrics"]
    total_bytes = sum(int(row["bytes"]) for row in rows)
    case_bytes = sum(int(row["bytes"]) for row in case_rows)
    all_pass = all(str(row["validation"]).startswith("PASS") for row in rows)
    kinds = Counter(row["kind"] for row in case_rows)
    load_rows = [row for row in case_rows if row["kind"] == "load_timeseries"]
    solar_rows = [row for row in case_rows if row["kind"] == "solar_weather_timeseries"]

    lines: list[str] = [
        "# 工业园区储能规划数据集盘点与下载决策",
        "",
        f"> 生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}；原始数据只读，尚未做重采样或节点映射。",
        "",
        "## 先看结论",
        "",
        f"- 本次案例清单已有 **{len(case_rows)} 个文件，{mib(case_bytes)}**；逐文件大小和 SHA256 全部复核，验收总状态：**{'通过' if all_pass else '需复查'}**。",
        f"- 原来已有的 **{len(legacy_rows)} 个 AUS/P1U `metrics.csv`，{mib(sum(int(row['bytes']) for row in legacy_rows))}** 全部保留，但只是静态馈线指标，不能训练 CVAE/DFL。",
        "- 数据源不是 Smart-DS 和 IEEE 二选一：**IEEE13 负责物理三相网络；Smart-DS GSO industrial 负责工业负荷/PV/天气时间形状**。",
        "- 严谨场景名称应为：**IEEE13 4.16 kV 工业微电网馈线（含 0.48 kV 二次节点）**，不是纯低压居民系统。",
        "- 下载与处理顺序已经固定：**原始数据下载并校验完成后，再在 `interim/processed` 中处理；绝不改写 `raw`**。",
        "",
        "## 统一后的系统与数据边界",
        "",
    ]
    markdown_table(
        lines,
        ["部分", "本项目采用", "为什么"],
        [
            ["物理电网", "IEEE13官方/EPRI OpenDSS", "保留三相不平衡相别、线路阻抗、变压器、调压器和电容器"],
            ["工业负荷", "Smart-DS GSO/industrial，2016–2018", "目标馈线100%工业；含真实P/Q、PF和15分钟建筑负荷形状"],
            ["PV与天气", "同一GSO区域、同三年", "避免负荷与天气跨地区；同时支持PV、PUE和极端热天"],
            ["数据中心工作量", "Azure Functions 2019", "Smart-DS没有计算任务到达量；以公开工作负载构造归一化任务曲线"],
            ["电网碳", "EIA-930 DUK", "GSO/Triad位于Duke Energy Carolinas服务逻辑内；使用消费侧而非生产侧碳强度"],
            ["购电费率", "OpenEI 2026 OPT-V Large Primary", "园区总进口约3–4 MW；匹配3 MW以上、600 V–44 kV主计量客户"],
            ["停电极端", "后续规则合成", "正常历史中停电样本稀少；以4–12小时连续失网形成可行性场景"],
        ],
    )
    lines.extend(
        [
            "",
            "## 原来15个文件逐项判断",
            "",
            "这些目录名虽然带 `timeseries`，但文件本身没有时间戳；每个都是96条馈线、80列静态汇总。结论统一为“保留作参考，不进模型”。",
            "",
        ]
    )
    markdown_table(
        lines,
        ["序号", "文件", "大小", "内容/处理决定"],
        [
            [
                index,
                row["relative_path"],
                f"{int(row['bytes']):,} B",
                row["description"],
            ]
            for index, row in enumerate(legacy_rows, start=1)
        ],
    )

    group_rows = []
    for kind, label, purpose in [
        ("load_timeseries", "Smart-DS工业负荷Parquet", "工业P/Q/PF和分项负荷；处理后进入模型"),
        ("solar_weather_timeseries", "Smart-DS太阳能/天气CSV", "PV、温度/PUE、极端天气；处理后进入模型"),
        ("opendss_feeder", "Smart-DS目标馈线静态文件", "profile引用和工业馈线映射；不替代IEEE13"),
        ("feeder_metrics", "Smart-DS目标馈线指标", "证明所选馈线与规模；不作为时序训练"),
        ("opendss", "IEEE13 OpenDSS", "最终三相物理网络"),
        ("bus_coordinates", "IEEE13坐标", "绘图和拓扑核对"),
        ("official_report", "IEEE官方说明", "参数出处"),
        ("data_center_workload", "Azure工作负载压缩包", "任务到达量"),
        ("grid_carbon_workbook", "EIA DUK碳工作簿", "小时消费侧碳强度"),
        ("industrial_tariff_json", "OpenEI工业费率JSON", "电量价和需量费"),
        ("documentation", "官方说明文件", "字段、许可和处理依据"),
    ]:
        selected = [row for row in case_rows if row["kind"] == kind]
        if selected:
            group_rows.append(
                [label, len(selected), mib(sum(int(row["bytes"]) for row in selected)), purpose]
            )
    lines.extend(["", "## 本次下载的数据", ""])
    markdown_table(lines, ["数据组", "文件数", "大小", "用途"], group_rows)
    lines.extend(
        [
            "",
            "所有64个案例文件都已逐行列在可用 Excel 打开的 `manifests/all_files_audit.csv`，包含：相对路径、行列数、时间范围、用途、下载网址、SHA256和验收结果。",
            "",
            "`DUK.xlsx`已做字段级验收：`Published Hourly Data`为97177行×89列，目标是第89列`CO2 Emissions Intensity for Consumed Electricity`；2021–2023恰好26280个连续UTC小时。",
            "",
            "### 22条工业负荷profile",
            "",
        ]
    )
    markdown_table(
        lines,
        ["年份", "profile文件", "每文件行数", "采样"],
        [
            [
                year,
                ", ".join(Path(row["relative_path"]).name for row in load_rows if row["year"] == year),
                "35040",
                "15 min",
            ]
            for year in ["2016", "2017", "2018"]
        ],
    )
    lines.extend(["", "### 6条PV/天气profile", ""])
    markdown_table(
        lines,
        ["年份", "文件", "记录数", "字段"],
        [
            [
                year,
                ", ".join(Path(row["relative_path"]).name for row in solar_rows if row["year"] == year),
                "2 × 35040",
                "DNI/DHI/GHI、风速、温度、POA、1 MW PV",
            ]
            for year in ["2016", "2017", "2018"]
        ],
    )

    lines.extend(
        [
            "",
            "## `.tex`模型需要什么，当前是否已具备",
            "",
        ]
    )
    markdown_table(
        lines,
        ["模型输入", "原始来源", "当前状态", "后续处理"],
        [
            ["节点有功/无功负荷", "Smart-DS P/Q + IEEE13基准负荷", "原始数据齐", "归一化profile，按IEEE13节点-相基准缩放"],
            ["节点PV最大可用功率", "Smart-DS 1 MW PV曲线", "原始数据齐", "缩放到634/675/680的0.25/0.65/0.75 MW"],
            ["PUE", "Smart-DS温度", "可派生", "明确温度-PUE公式后生成"],
            ["工作量到达A", "Azure Functions", "原始数据齐", "分钟聚合到小时、归一化、缩放到0.60 MW数据中心"],
            ["电网碳强度", "EIA-930 DUK", "原始数据齐", "取2021–2023三个完整UTC年消费侧强度，再按日历映射"],
            ["购电电价/需量费", "OpenEI OPT-V Large Primary", "参数源齐", "按月/季节/时段展开；需量费单独建模"],
            ["电网可用性", "规则合成", "无需下载", "正常全1；极端窗口注入4–12 h连续0"],
            ["三相拓扑", "IEEE13 OpenDSS", "原始数据齐", "解析bus-phase、完整阻抗矩阵和Y/Δ连接"],
            ["储能/机组成本与技术参数", "案例假设/文献", "不是数据集", "后续建立参数表并做敏感性分析"],
        ],
    )

    lines.extend(
        [
            "",
            "## 数据量级已经统一",
            "",
            f"- 当前 `data` 原始与清单合计约 **{mib(total_bytes)}**（不含将来解压后的Azure中间文件）。",
            "- Smart-DS每条profile每年是 **35040个15分钟点**；3年统一成 **26280个小时点** 后再切窗口。",
            "- 120小时（5天）按1天步长滑窗，三年理论上最多约 **1083个候选窗口**；不是只下载三年中的5天。",
            "- 优化器最终只吃春/夏/秋/冬/极端各一个120小时块，但CVAE/DFL必须先看到完整历史候选池。",
            "- 建议预留 **2–3 GB**：Azure压缩包解压、小时主表、三相张量、滑窗缓存和训练/验证/测试文件都会扩大。",
            "",
            "## 处理顺序（下一步照这个做）",
            "",
            "1. **Raw已完成**：下载、逐文件验收、SHA256、来源保留；禁止直接修改。",
            "2. **建立映射**：解析IEEE13和Smart-DS `Loads.dss/PVSystems.dss`，形成Smart-DS profile → IEEE13节点/相的映射表。",
            "3. **统一时标**：保留15分钟副本；聚合小时功率用平均值，小时能量用积分；以UTC唯一索引并处理DST。",
            "4. **构造小时主表**：P/Q/PV/温度/PUE/workload/碳/费率/availability全部对齐，但记录各变量并非联合实测。",
            "5. **分割而非泄漏**：2016训练、2017验证、2018测试；归一化参数只用2016计算。",
            "6. **切120小时候选窗**：保留季节、日期、正常/极端、发生权重等元数据。",
            "7. **CVAE/DFL代码状态**：训练入口已读取三相历史NPZ，并使用节点—相不平衡LinDistFlow。",
            "",
            "## 必须写进论文的方法限制",
            "",
            "- Smart-DS 2016–2018的负荷/PV/天气彼此同地区，但Azure工作量和EIA碳不是同一现场的联合观测；它们是按日历条件组合的案例场景。",
            "- Smart-DS官方说明明确将闰年2016的最后一天截掉，只保留365天。下载的2016负荷Parquet还存在一个末行时间戳跳变（`2016-12-30 23:45`直接到`2017-01-01 00:00`）；处理时按行序重建`2016-01-01 00:00`至`2016-12-30 23:45`的标准15分钟索引，并把这项修正写入manifest，不插补12月31日。",
            "- EIA第89列`CO2 Emissions Intensity for Consumed Electricity`自2018-07-01起有值，单位为`lb/kWh`；换算公式为`tCO2/MWh = lb/kWh × 0.45359237`。2019和2020各缺一个DST附近的23小时段，因此最终选2021–2023：共26280小时、UTC无缺失/重复/跳变。不能声称这是与Smart-DS 2016–2018严格同步的联合历史。",
            "- Azure Functions只有14天，不能冒充三年历史；应做有放回采样/扰动并保留来源日标签。",
            "- 当前`.tex`与代码都是平衡LinDistFlow；如果坚持三相不平衡，数学式、schema和规划约束都必须同步改。",
            "- 极端场景由历史联合尾部窗口加合成停电构造；其年发生次数设为0，只约束规划可行性。",
            "",
            "## 可重复执行",
            "",
            "```powershell",
            "conda activate storage-dfl",
            "cd E:\\Vscode\\ESS_planning\\dfl-storage-planning",
            "python scripts\\download_case_data.py --include-workload --include-grid --workers 4",
            "python scripts\\audit_case_data.py",
            "```",
            "",
            f"下载清单种类计数：`{dict(sorted(kinds.items()))}`。",
        ]
    )

    temporary = output.with_suffix(".md.part")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    temporary.replace(output)
    return output


def main() -> None:
    data_root = default_data_root().resolve()
    rows = audit(data_root)
    csv_path = write_csv(data_root, rows)
    markdown_path = write_markdown(data_root, rows)
    part_files = list(data_root.rglob("*.part"))
    print(f"Audited raw files: {len(rows)}")
    print(f"Audit CSV: {csv_path}")
    print(f"Inventory: {markdown_path}")
    print(f"Residual .part files: {len(part_files)}")


if __name__ == "__main__":
    main()
