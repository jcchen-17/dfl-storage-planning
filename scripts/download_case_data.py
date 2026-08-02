"""Download the raw data required by the IEEE13 industrial planning case.

The script deliberately downloads a narrow, reproducible subset instead of
the 5.32 TB complete SMART-DS data lake:

* IEEE/EPRI IEEE13 static reference files;
* SMART-DS v1.0 GSO industrial feeder ``ihs1_1247--idt1210``;
* all load/PV/weather profiles referenced by that feeder in 2016--2018;
* optionally, the Microsoft Azure Functions 2019 workload trace.
* optionally, the EIA-930 DUK carbon workbook and OpenEI OPT-V tariff record.

Raw files are never rewritten.  Downloads use a ``.part`` file followed by an
atomic rename and a CSV manifest containing SHA-256 checksums.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SMARTDS_HTTP = "https://oedi-data-lake.s3.amazonaws.com/"
SMARTDS_PREFIX = "SMART-DS/v1.0/"
YEARS = (2016, 2017, 2018)
REGION = "GSO"
SUBREGION = "industrial"
SCENARIO = "solar_medium_batteries_none_timeseries"
SUBSTATION = "ihs1_1247"
FEEDER = "ihs1_1247--idt1210"
STATIC_FILES = (
    "Buscoords.dss",
    "Intermediates.txt",
    "LineCodes.dss",
    "Lines.dss",
    "Loads.dss",
    "Master.dss",
    "PVSystems.dss",
    "Transformers.dss",
)

AZURE_WORKLOAD_URL = (
    "https://github.com/Azure/AzurePublicDataset/releases/download/"
    "dataset-functions-2019/"
    "azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz"
)
AZURE_WORKLOAD_README_URL = (
    "https://raw.githubusercontent.com/Azure/AzurePublicDataset/"
    "master/AzureFunctionsDataset2019.md"
)
EIA_DUK_WORKBOOK_URL = (
    "https://www.eia.gov/electricity/gridmonitor/knownissues/xls/DUK.xlsx"
)
OPENEI_OPT_V_URL = (
    "https://api.openei.org/utility_rates?version=latest&format=json&"
    "detail=full&getpage=6998c5d6a746c90e550cb241&api_key=DEMO_KEY"
)


@dataclass(frozen=True)
class Download:
    dataset: str
    kind: str
    year: str
    url: str
    destination: Path
    source_key: str = ""


def default_data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def smartds_destination(data_root: Path, key: str) -> Path:
    relative = key.removeprefix(SMARTDS_PREFIX)
    return data_root / "raw" / "smartds" / Path(relative)


def smartds_download(data_root: Path, key: str, kind: str, year: str = "") -> Download:
    return Download(
        dataset="SMART-DS v1.0",
        kind=kind,
        year=year,
        url=SMARTDS_HTTP + key,
        destination=smartds_destination(data_root, key),
        source_key=key,
    )


def feeder_prefix(year: int) -> str:
    return (
        f"{SMARTDS_PREFIX}{year}/{REGION}/{SUBREGION}/scenarios/{SCENARIO}/"
        f"opendss_no_loadshapes/{SUBSTATION}/{FEEDER}/"
    )


def initial_downloads(data_root: Path) -> list[Download]:
    downloads = [
        smartds_download(
            data_root,
            f"{SMARTDS_PREFIX}User_Guide/Readme.md",
            "documentation",
        ),
        Download(
            dataset="IEEE 13-node",
            kind="official_report",
            year="",
            url="https://ewh.ieee.org/soc/pes/dsacom/testfeeders/testfeeders.pdf",
            destination=data_root / "raw" / "ieee13" / "official" / "testfeeders.pdf",
        ),
        Download(
            dataset="DSS-Extensions OpenDSS IEEE13 mirror",
            kind="opendss",
            year="",
            url=(
                "https://raw.githubusercontent.com/dss-extensions/electricdss-tst/"
                "master/Version8/Distrib/IEEETestCases/13Bus/IEEE13Nodeckt.dss"
            ),
            destination=data_root / "raw" / "ieee13" / "opendss" / "IEEE13Nodeckt.dss",
        ),
        Download(
            dataset="DSS-Extensions OpenDSS IEEE13 mirror",
            kind="opendss",
            year="",
            url=(
                "https://raw.githubusercontent.com/dss-extensions/electricdss-tst/"
                "master/Version8/Distrib/IEEETestCases/IEEELineCodes.DSS"
            ),
            destination=data_root / "raw" / "ieee13" / "opendss" / "IEEELineCodes.DSS",
        ),
        Download(
            dataset="DSS-Extensions OpenDSS IEEE13 mirror",
            kind="bus_coordinates",
            year="",
            url=(
                "https://raw.githubusercontent.com/dss-extensions/electricdss-tst/"
                "master/Version8/Distrib/IEEETestCases/13Bus/IEEE13Node_BusXY.csv"
            ),
            destination=data_root / "raw" / "ieee13" / "opendss" / "IEEE13Node_BusXY.csv",
        ),
    ]
    for year in YEARS:
        scenario_root = (
            f"{SMARTDS_PREFIX}{year}/{REGION}/{SUBREGION}/scenarios/{SCENARIO}/"
        )
        downloads.append(
            smartds_download(data_root, scenario_root + "metrics.csv", "feeder_metrics", str(year))
        )
        prefix = feeder_prefix(year)
        downloads.extend(
            smartds_download(data_root, prefix + name, "opendss_feeder", str(year))
            for name in STATIC_FILES
        )
    return downloads


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(item: Download, retries: int = 4) -> dict[str, str | int]:
    item.destination.parent.mkdir(parents=True, exist_ok=True)
    if item.destination.exists() and item.destination.stat().st_size > 0:
        size = item.destination.stat().st_size
        print(f"[keep] {item.destination} ({size:,} bytes)", flush=True)
        return manifest_row(item, size, sha256_file(item.destination), "existing")

    partial = item.destination.with_name(item.destination.name + ".part")
    if partial.exists():
        partial.unlink()

    request = urllib.request.Request(
        item.url,
        headers={
            "User-Agent": "Mozilla/5.0 storage-dfl-research-downloader/1.0",
            "Accept": "*/*",
        },
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
                expected = response.headers.get("Content-Length")
                expected_size = int(expected) if expected and expected.isdigit() else 0
                copied = 0
                next_report = 20 * 1024 * 1024
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    copied += len(chunk)
                    if expected_size >= 20 * 1024 * 1024 and copied >= next_report:
                        print(
                            f"[data] {item.destination.name}: {copied / 1024**2:.0f} MiB"
                            f" / {expected_size / 1024**2:.0f} MiB",
                            flush=True,
                        )
                        next_report += 20 * 1024 * 1024
            if expected_size and copied != expected_size:
                raise OSError(f"size mismatch: received {copied}, expected {expected_size}")
            partial.replace(item.destination)
            digest = sha256_file(item.destination)
            print(f"[done] {item.destination} ({copied:,} bytes)", flush=True)
            return manifest_row(item, copied, digest, "downloaded")
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            if partial.exists():
                partial.unlink()
            if attempt == retries:
                raise RuntimeError(f"failed to download {item.url}: {exc}") from exc
            delay = 2**attempt
            print(f"[retry {attempt}/{retries}] {item.url}: {exc}; waiting {delay}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def manifest_row(
    item: Download,
    size: int,
    digest: str,
    status: str,
) -> dict[str, str | int]:
    return {
        "dataset": item.dataset,
        "year": item.year,
        "kind": item.kind,
        "path": str(item.destination.resolve()),
        "source_url": item.url,
        "source_key": item.source_key,
        "bytes": size,
        "sha256": digest,
        "status": status,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_downloads(downloads: list[Download], workers: int) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, item): item for item in downloads}
        for future in as_completed(futures):
            rows.append(future.result())
    return rows


def profile_downloads(data_root: Path) -> list[Download]:
    downloads: list[Download] = []
    load_pattern = re.compile(r"!yearly=(?P<class>res|com)_kw_(?P<id>[^\s]+?)_pu\b")
    solar_pattern = re.compile(r"!yearly=(?P<profile>[^\s]+)")
    for year in YEARS:
        local_prefix = smartds_destination(data_root, feeder_prefix(year))
        loads_text = (local_prefix / "Loads.dss").read_text(encoding="utf-8")
        pv_text = (local_prefix / "PVSystems.dss").read_text(encoding="utf-8")

        load_names = sorted(
            {f"{match.group('class')}_{match.group('id')}.parquet" for match in load_pattern.finditer(loads_text)}
        )
        solar_names = sorted(
            {f"{match.group('profile')}_full.csv" for match in solar_pattern.finditer(pv_text)}
        )
        if not load_names:
            raise ValueError(f"No load profiles found in {local_prefix / 'Loads.dss'}")
        if not solar_names:
            raise ValueError(f"No solar profiles found in {local_prefix / 'PVSystems.dss'}")

        for name in load_names:
            key = f"{SMARTDS_PREFIX}{year}/{REGION}/{SUBREGION}/load_data/{name}"
            downloads.append(smartds_download(data_root, key, "load_timeseries", str(year)))
        for name in solar_names:
            key = f"{SMARTDS_PREFIX}{year}/{REGION}/{SUBREGION}/solar_data/{name}"
            downloads.append(smartds_download(data_root, key, "solar_weather_timeseries", str(year)))
    return downloads


def workload_downloads(data_root: Path) -> list[Download]:
    root = data_root / "raw" / "data_center" / "azure_functions_2019"
    return [
        Download(
            dataset="Microsoft Azure Functions Trace 2019",
            kind="data_center_workload",
            year="2019",
            url=AZURE_WORKLOAD_URL,
            destination=root / "azurefunctions_dataset2019.tar.xz",
        ),
        Download(
            dataset="Microsoft Azure Functions Trace 2019",
            kind="documentation",
            year="2019",
            url=AZURE_WORKLOAD_README_URL,
            destination=root / "README.md",
        ),
    ]


def grid_downloads(data_root: Path) -> list[Download]:
    """Return the regional carbon and retail-tariff source documents.

    The EIA workbook is DUK-specific and includes consumption-side hourly
    emissions intensity from July 2018 onward.  The DOE OpenEI OPT-V record
    is used instead of an ISO/RTO LMP because the case is a Duke Energy
    Carolinas retail customer.
    """

    return [
        Download(
            dataset="EIA-930 Duke Energy Carolinas (DUK)",
            kind="grid_carbon_workbook",
            year="2018-present",
            url=EIA_DUK_WORKBOOK_URL,
            destination=data_root / "raw" / "grid" / "carbon" / "eia930" / "DUK.xlsx",
        ),
        Download(
            dataset="DOE OpenEI USURDB - Duke Energy Carolinas OPT-V",
            kind="industrial_tariff_json",
            year="2026",
            url=OPENEI_OPT_V_URL,
            destination=(
                data_root
                / "raw"
                / "grid"
                / "price"
                / "openei_usurdb"
                / "OPT-V_2026.json"
            ),
        ),
    ]


def write_manifest(data_root: Path, rows: list[dict[str, str | int]]) -> Path:
    manifest_dir = data_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    destination = manifest_dir / "case_raw_files.csv"
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["year"]), str(row["path"])))
    fieldnames = [
        "dataset",
        "year",
        "kind",
        "path",
        "source_url",
        "source_key",
        "bytes",
        "sha256",
        "status",
        "checked_at_utc",
    ]
    temporary = destination.with_suffix(".csv.part")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--include-workload",
        action="store_true",
        help="Also download the 136 MiB Azure Functions 2019 trace.",
    )
    parser.add_argument(
        "--include-grid",
        action="store_true",
        help="Also download the EIA DUK carbon workbook and OpenEI OPT-V tariff record.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    if args.workers < 1 or args.workers > 8:
        raise ValueError("--workers must be between 1 and 8")
    print(f"Data root: {data_root}", flush=True)
    print("Phase 1/3: documentation and static feeder files", flush=True)
    rows = run_downloads(initial_downloads(data_root), args.workers)

    print("Phase 2/3: feeder-referenced load and solar/weather profiles", flush=True)
    rows.extend(run_downloads(profile_downloads(data_root), args.workers))

    if args.include_workload:
        print("Phase 3/3: data-center workload trace", flush=True)
        rows.extend(run_downloads(workload_downloads(data_root), min(args.workers, 2)))
    else:
        print("Phase 3/3: skipped workload (use --include-workload to download it)", flush=True)

    if args.include_grid:
        print("Additional phase: regional carbon and industrial tariff", flush=True)
        rows.extend(run_downloads(grid_downloads(data_root), min(args.workers, 2)))
    else:
        print("Additional phase: skipped grid data (use --include-grid to download it)", flush=True)

    manifest = write_manifest(data_root, rows)
    total = sum(int(row["bytes"]) for row in rows)
    print(f"Manifest: {manifest}", flush=True)
    print(f"Verified files: {len(rows)}; total bytes: {total:,}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; completed files are kept, partial files are not used.", file=sys.stderr)
        raise SystemExit(130)
