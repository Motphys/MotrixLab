# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Low-overhead host metrics sampled at training-panel refresh boundaries.

CPU samplers read Linux ``/proc`` interfaces and return ``None`` where they
are unavailable, so panels degrade to ``n/a`` fields; memory sampling also
supports Windows via ``GlobalMemoryStatusEx``. The GPU samplers use NVML,
which works on any platform with an NVIDIA driver.
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CpuLoad:
    """CPU execution time observed over one sampling window.

    Utilization excludes idle, I/O-wait, and stolen virtual-CPU time. The
    equivalent logical CPU count makes the normalized percentage unambiguous
    on SMT systems.
    """

    utilization_percent: float
    used_logical_cpus: float
    logical_cpu_count: int
    physical_core_count: int | None
    iowait_percent: float
    steal_percent: float


@dataclass(frozen=True)
class _CpuTimes:
    total: int
    executing: int
    iowait: int
    steal: int


class CpuLoadSampler:
    """Sample Linux CPU counters for the logical CPUs available to this process."""

    def __init__(
        self,
        *,
        stat_path: str | Path = "/proc/stat",
        topology_root: str | Path = "/sys/devices/system/cpu",
        cpu_ids: set[int] | None = None,
    ) -> None:
        self._stat_path = Path(stat_path)
        self._topology_root = Path(topology_root)
        self._cpu_ids = cpu_ids if cpu_ids is not None else self._available_cpu_ids()
        self._physical_core_count = self._read_physical_core_count()
        self._previous = self._read_times()

    def sample(self) -> CpuLoad | None:
        """Return utilization since the previous call, or ``None`` when unavailable."""
        current = self._read_times()
        common_ids = self._previous.keys() & current.keys()
        if not common_ids:
            self._previous = current
            return None

        total = sum(current[cpu].total - self._previous[cpu].total for cpu in common_ids)
        executing = sum(current[cpu].executing - self._previous[cpu].executing for cpu in common_ids)
        iowait = sum(current[cpu].iowait - self._previous[cpu].iowait for cpu in common_ids)
        steal = sum(current[cpu].steal - self._previous[cpu].steal for cpu in common_ids)
        self._previous = current
        if total <= 0:
            return None

        logical_cpu_count = len(common_ids)
        utilization_percent = 100.0 * executing / total
        return CpuLoad(
            utilization_percent=utilization_percent,
            used_logical_cpus=logical_cpu_count * utilization_percent / 100.0,
            logical_cpu_count=logical_cpu_count,
            physical_core_count=self._physical_core_count,
            iowait_percent=100.0 * iowait / total,
            steal_percent=100.0 * steal / total,
        )

    @staticmethod
    def _available_cpu_ids() -> set[int]:
        if hasattr(os, "sched_getaffinity"):
            return set(os.sched_getaffinity(0))
        return set(range(os.cpu_count() or 1))

    def _read_times(self) -> dict[int, _CpuTimes]:
        try:
            lines = self._stat_path.read_text().splitlines()
        except OSError:
            return {}

        times: dict[int, _CpuTimes] = {}
        for line in lines:
            label, *raw_fields = line.split()
            if not label.startswith("cpu") or not label[3:].isdigit():
                continue
            cpu_id = int(label[3:])
            if cpu_id not in self._cpu_ids:
                continue
            fields = [int(value) for value in raw_fields[:8]]
            fields.extend([0] * (8 - len(fields)))
            idle, iowait, steal = fields[3], fields[4], fields[7]
            total = sum(fields)
            times[cpu_id] = _CpuTimes(
                total=total,
                executing=total - idle - iowait - steal,
                iowait=iowait,
                steal=steal,
            )
        return times

    def _read_physical_core_count(self) -> int | None:
        cores: set[tuple[int, int]] = set()
        try:
            for cpu_id in self._cpu_ids:
                topology = self._topology_root / f"cpu{cpu_id}" / "topology"
                package_id = int((topology / "physical_package_id").read_text())
                core_id = int((topology / "core_id").read_text())
                cores.add((package_id, core_id))
        except (OSError, ValueError):
            return None
        return len(cores)


@dataclass(frozen=True)
class MemoryUsage:
    """Memory usage in bytes for a host or accelerator device."""

    used_bytes: int
    total_bytes: int


class _MemoryStatusEx(ctypes.Structure):
    """``MEMORYSTATUSEX`` layout for the Windows ``GlobalMemoryStatusEx`` call."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _windows_memory_status() -> MemoryUsage | None:
    """Physical memory usage via ``GlobalMemoryStatusEx``, mirroring the /proc semantics."""
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return MemoryUsage(used_bytes=status.ullTotalPhys - status.ullAvailPhys, total_bytes=status.ullTotalPhys)


class MemoryUsageSampler:
    """Read host memory usage from Linux ``/proc/meminfo`` or the Windows memory API."""

    def __init__(self, *, meminfo_path: str | Path = "/proc/meminfo") -> None:
        self._meminfo_path = Path(meminfo_path)

    def sample(self) -> MemoryUsage | None:
        if sys.platform == "win32":
            return _windows_memory_status()
        try:
            values: dict[str, int] = {}
            for line in self._meminfo_path.read_text().splitlines():
                key, value, *_ = line.split()
                if key in {"MemTotal:", "MemAvailable:"}:
                    values[key] = int(value) * 1024
        except (OSError, ValueError):
            return None
        total = values.get("MemTotal:")
        available = values.get("MemAvailable:")
        if total is None or available is None:
            return None
        return MemoryUsage(used_bytes=max(0, total - available), total_bytes=total)


# NVML reads the same counters nvidia-smi reports, but in-process at
# microsecond cost instead of a subprocess spawn per query.
_nvml_state: tuple[Any, list[Any]] | tuple[()] | None = None  # None: untried; (): unavailable


def _nvml() -> tuple[Any, list[Any]] | None:
    """Lazily initialize NVML and return ``(module, device_handles)``, or ``None``."""
    global _nvml_state
    if _nvml_state is None:
        try:
            import pynvml

            pynvml.nvmlInit()
            devices = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(pynvml.nvmlDeviceGetCount())]
            _nvml_state = (pynvml, devices)
        except Exception:  # ImportError (pynvml missing) or NVML init failure (no driver/GPU)
            _nvml_state = ()
    return _nvml_state or None


class GpuMemoryUsageSampler:
    """Read aggregate NVIDIA memory usage across all visible GPUs via NVML."""

    def sample(self) -> MemoryUsage | None:
        session = _nvml()
        if session is None:
            return None
        pynvml, devices = session
        used = total = 0
        try:
            for device in devices:
                info = pynvml.nvmlDeviceGetMemoryInfo(device)
                used += info.used
                total += info.total
        except pynvml.NVMLError:
            return None
        return MemoryUsage(used_bytes=used, total_bytes=total) if total > 0 else None


class GpuUtilizationSampler:
    """Read aggregate NVIDIA GPU utilization across all visible GPUs via NVML."""

    def sample(self) -> float | None:
        session = _nvml()
        if session is None:
            return None
        pynvml, devices = session
        values: list[int] = []
        try:
            for device in devices:
                values.append(pynvml.nvmlDeviceGetUtilizationRates(device).gpu)
        except pynvml.NVMLError:
            return None
        return sum(values) / len(values) if values else None
