"""Measure this machine, then recommend the three solver knobs for it.

``solver_threads``, ``solver_max_parallel_workers`` and
``solver_memory_limit_mb`` are machine-specific, and getting them wrong is
expensive in both directions: too many workers and the box pages, which is
slower than running fewer; too few and cores sit idle. The committed values were
measured on a 6-core / 16 GB desktop and should not be carried to another
machine unread -- the config they came from was itself inherited from a 16-core
/ 32 GB machine, and on the smaller box that mistake cost roughly an order of
magnitude in memory pressure before it was caught.

Three measurements, in order of how much they matter:

1. Thread scaling. One planning solve at several thread counts. Past the point
   where the curve flattens, threads are worth more spent on concurrent solves.
2. Peak memory of one worker. This is the number multiplied by the worker count,
   and the one the old config admitted was "estimated, not measured".
3. Free memory right now, which is what actually bounds the worker count -- not
   installed RAM.

    python scripts/tune_solver.py
    python scripts/tune_solver.py --threads 1 2 4 8 16 --config configs/x.yaml
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import sys
import time
from ctypes import wintypes
from dataclasses import replace
from pathlib import Path

# Keep every solve in this process so its memory is ours to measure; the oracle
# otherwise forks a worker per solve on Windows and the peak lands out of reach.
os.environ.setdefault("STORAGE_DFL_SOLVER_WORKER", "1")

import torch  # noqa: F401

from storage_dfl.config import load_config
from storage_dfl.data import ScenarioCodec, ScenarioPool
from storage_dfl.dfl import select_scenarios
from storage_dfl.planning import StoragePlanningOracle
from storage_dfl.stages import ArtifactPaths, _experiment_data, _load_codec

REPO = Path(__file__).resolve().parent.parent
IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------- machine ---
class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


class _MEMSTATUS(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _peak_mb() -> float | None:
    """Peak working set of this process, in MB. None off Windows."""
    if not IS_WINDOWS:
        return None
    get = getattr(ctypes.windll.kernel32, "K32GetProcessMemoryInfo", None)
    if get is None:
        return None
    counters = _PMC()
    counters.cb = ctypes.sizeof(counters)
    get.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    get.restype = wintypes.BOOL
    # GetCurrentProcess returns a pseudo-handle of -1; without an explicit
    # restype ctypes truncates it on 64-bit and the call fails on a bad handle.
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    if not get(ctypes.windll.kernel32.GetCurrentProcess(),
               ctypes.byref(counters), counters.cb):
        return None
    return counters.PeakWorkingSetSize / 1024 ** 2


def _memory_gb() -> tuple[float | None, float | None]:
    """(total, available) physical memory in GB."""
    if IS_WINDOWS:
        status = _MEMSTATUS()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return (status.ullTotalPhys / 1024 ** 3,
                    status.ullAvailPhys / 1024 ** 3)
        return None, None
    try:  # Linux
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            info[key] = float(rest.strip().split()[0]) / 1024 ** 2
        return info.get("MemTotal"), info.get("MemAvailable")
    except OSError:
        return None, None


def _cores() -> tuple[int | None, int]:
    """(physical, logical) core counts. Physical is None when unknown."""
    logical = os.cpu_count() or 1
    physical = None
    try:
        import subprocess
        if IS_WINDOWS:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor | "
                 "Measure-Object -Property NumberOfCores -Sum).Sum"],
                capture_output=True, text=True, timeout=60,
            )
            physical = int(out.stdout.strip())
        else:
            text = Path("/proc/cpuinfo").read_text()
            physical = len({
                line.split(":")[1].strip()
                for line in text.splitlines() if line.startswith("core id")
            }) or None
    except Exception:
        physical = None
    return physical, logical


# ------------------------------------------------------------ measurement ---
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/generator_compare_k1t2.yaml")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="thread counts to time a single planning solve at")
    ap.add_argument("--time-limit", type=float, default=900.0)
    args = ap.parse_args()

    physical, logical = _cores()
    total_gb, free_gb = _memory_gb()
    print("=" * 68)
    print("machine")
    print("=" * 68)
    print(f"  cpu              : {platform.processor() or 'unknown'}")
    print(f"  cores            : {physical if physical else '?'} physical / "
          f"{logical} logical")
    if total_gb:
        print(f"  memory           : {total_gb:.1f} GB total, "
              f"{free_gb:.1f} GB available now")
        print("                     (the worker count is bounded by AVAILABLE, "
              "not installed)")
    print(f"  python           : {sys.executable}")

    config = load_config(str(REPO / args.config))
    feeder, pool = _experiment_data(config, config.data.validation_split)
    try:
        codec = _load_codec(ArtifactPaths(config.output_dir), feeder)
        if int(codec.context_dim) != int(pool.scenarios[0].context.size):
            raise ValueError("stale normalization")
    except (FileNotFoundError, ValueError):
        _, train_pool = _experiment_data(config, config.data.train_split)
        codec = ScenarioCodec.fit(train_pool, feeder)
        print("  (fitted a codec on the train split; saved one was "
              "missing or stale)")
    print(f"  backend          : {config.planning.solver_backend}")
    print(f"  carbon           : {config.planning.carbon_formulation}, "
          f"cap {config.planning.dc_carbon_cap}")

    support, weights, _ = select_scenarios("kmeans", pool, codec, 1, seed=0)

    print("\n" + "=" * 68)
    print("1. thread scaling, one K=1 planning solve")
    print("=" * 68)
    timings: dict[int, float] = {}
    for count in args.threads:
        if physical and count > 2 * physical:
            print(f"  {count:>3} threads : skipped (beyond 2x physical cores)")
            continue
        planning = replace(
            config.planning, solver_threads=count, solver_max_parallel_workers=1,
            solver_time_limit_seconds=args.time_limit, verbose_solver=False,
        )
        oracle = StoragePlanningOracle(
            feeder, planning, config.costs, config.data, config.data_center
        )
        started = time.perf_counter()
        result = oracle.solve(support, weights=weights)
        elapsed = time.perf_counter() - started
        timings[count] = elapsed
        best = min(timings.values())
        note = "  <-- best" if elapsed <= best else f"  ({elapsed / best:.2f}x best)"
        print(f"  {count:>3} threads : {elapsed:7.1f}s   {result.status}{note}",
              flush=True)

    if not timings:
        raise SystemExit("no thread counts were measured")
    best_time = min(timings.values())
    # The smallest count within 10% of the best: past it, extra threads buy
    # nothing on one solve and are worth more as concurrency.
    knee = min(c for c, t in timings.items() if t <= best_time * 1.10)

    print("\n" + "=" * 68)
    print("2. peak memory of one solve over the evaluation set")
    print("=" * 68)
    design = StoragePlanningOracle(
        feeder,
        replace(config.planning, solver_threads=knee,
                solver_max_parallel_workers=1,
                solver_time_limit_seconds=args.time_limit, verbose_solver=False),
        config.costs, config.data, config.data_center,
    ).solve(support, weights=weights)
    indices = codec.support_indices(pool, config.dfl.validation_batch_size)
    validation = ScenarioPool(pool.subset(indices.tolist())).scenarios
    oracle = StoragePlanningOracle(
        feeder,
        replace(config.planning, solver_threads=knee,
                solver_max_parallel_workers=1,
                solver_time_limit_seconds=args.time_limit, verbose_solver=False),
        config.costs, config.data, config.data_center,
    )
    started = time.perf_counter()
    oracle.solve(validation, fixed_design=design.design, allow_carbon_slack=True)
    elapsed = time.perf_counter() - started
    peak = _peak_mb()
    print(f"  {len(validation)} scenarios, fixed design : {elapsed:.1f}s")
    if peak is None:
        print("  peak memory      : not measurable on this platform")
        peak = 3000.0
        print(f"  assuming {peak:,.0f} MB per worker for the recommendation")
    else:
        print(f"  peak process RSS : {peak:,.0f} MB  "
              "<-- multiplied by the worker count")

    print("\n" + "=" * 68)
    print("3. recommended settings")
    print("=" * 68)
    # Reserve for the parent process: torch plus a CUDA context measured about
    # 500 MB here, and the OS needs slack it is not already using.
    budget_gb = max(1.0, (free_gb - 2.0)) if free_gb else 8.0
    per_worker_mb = peak * 1.15                       # headroom over the measurement
    by_memory = max(1, int((budget_gb * 1024) / per_worker_mb))
    # Against LOGICAL cores. Workers are separate processes and the solver does
    # not keep every thread busy, so holding the total to the physical count
    # leaves the machine idle; holding it to the logical count was what the
    # measured 3-worker setting came to on a 6-core / 12-thread box.
    by_cores = max(1, logical // knee)
    workers = max(1, min(by_memory, by_cores))
    samples = config.dfl.policy_samples_per_epoch
    # A wave that does not divide the sample count leaves the last one part
    # empty; prefer the nearest divisor at or below the cap.
    divisors = [d for d in range(1, samples + 1) if samples % d == 0 and d <= workers]
    if divisors:
        workers = max(divisors)

    print(f"  solver_threads              : {knee}")
    print(f"  solver_max_parallel_workers : {workers}")
    print(f"  solver_memory_limit_mb      : {int(per_worker_mb // 100 * 100)}")
    print()
    print(f"  bounded by memory to {by_memory} workers "
          f"({budget_gb:.1f} GB usable / {per_worker_mb:,.0f} MB each)")
    print(f"  bounded by cores  to {by_cores} workers "
          f"({logical} logical / {knee} threads each)")
    if workers != min(by_memory, by_cores):
        print(f"  then down to {workers}, the largest divisor of "
              f"policy_samples_per_epoch={samples} (a wave that does not divide "
              "it runs part-empty)")
    print(f"\n  free memory was {free_gb:.1f} GB when this ran. Close what you "
          "can\n  before a long run -- the worker count follows it directly.")
    print()
    print("  Memory is usually the binding one. If 'memlimit' shows up in a run,")
    print("  halve the workers before raising the per-worker budget: the product")
    print("  is what has to fit.")


if __name__ == "__main__":
    main()
