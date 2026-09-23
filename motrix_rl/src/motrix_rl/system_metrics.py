# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Low-overhead host metrics sampled at training-panel refresh boundaries.

CPU samplers read Linux ``/proc`` interfaces and return ``None`` where they
are unavailable, so panels degrade to ``n/a`` fields; memory sampling also
supports Windows via ``GlobalMemoryStatusEx``. GPU samplers use NVML on
NVIDIA hosts and AMD SMI on ROCm hosts.
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
    on SMT systems. ``per_core_percent`` carries the same utilization broken
    down per logical CPU (sorted by cpu id) for the system view's per-core
    display; it comes free — the sampler already reads per-CPU counters to
    compute the aggregate.
    """

    utilization_percent: float
    used_logical_cpus: float
    logical_cpu_count: int
    physical_core_count: int | None
    iowait_percent: float
    steal_percent: float
    per_core_percent: tuple[float, ...] | None = None
    # The logical CPU ids aligned 1:1 with ``per_core_percent`` (sorted). Under
    # NUMA binding the sampler only sees the process's affinity, so positional
    # labels would misreport which physical cores are shown.
    per_core_ids: tuple[int, ...] | None = None
    # Static "model name" from /proc/cpuinfo (Linux); None where unavailable.
    model_name: str | None = None


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
        cpuinfo_path: str | Path = "/proc/cpuinfo",
        cpu_ids: set[int] | None = None,
    ) -> None:
        self._stat_path = Path(stat_path)
        self._topology_root = Path(topology_root)
        self._cpu_ids = cpu_ids if cpu_ids is not None else self._available_cpu_ids()
        self._physical_core_count = self._read_physical_core_count()
        self._model_name = self._read_model_name(Path(cpuinfo_path))
        self._previous = self._read_times()

    def sample(self) -> CpuLoad | None:
        """Return utilization since the previous call, or ``None`` when unavailable."""
        current = self._read_times()
        common_ids = self._previous.keys() & current.keys()
        if not common_ids:
            self._previous = current
            return None

        previous = self._previous
        total = sum(current[cpu].total - previous[cpu].total for cpu in common_ids)
        executing = sum(current[cpu].executing - previous[cpu].executing for cpu in common_ids)
        iowait = sum(current[cpu].iowait - previous[cpu].iowait for cpu in common_ids)
        steal = sum(current[cpu].steal - previous[cpu].steal for cpu in common_ids)
        self._previous = current
        if total <= 0:
            return None

        logical_cpu_count = len(common_ids)
        utilization_percent = 100.0 * executing / total
        per_core = tuple(
            100.0
            * (current[cpu].executing - previous[cpu].executing)
            / max(current[cpu].total - previous[cpu].total, 1)
            for cpu in sorted(common_ids)
        )
        return CpuLoad(
            utilization_percent=utilization_percent,
            used_logical_cpus=logical_cpu_count * utilization_percent / 100.0,
            logical_cpu_count=logical_cpu_count,
            physical_core_count=self._physical_core_count,
            iowait_percent=100.0 * iowait / total,
            steal_percent=100.0 * steal / total,
            per_core_percent=per_core,
            per_core_ids=tuple(sorted(common_ids)),
            model_name=self._model_name,
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

    def _read_model_name(self, cpuinfo_path: Path) -> str | None:
        """First ``model name`` entry from /proc/cpuinfo; None off-Linux or unreadable."""
        try:
            for line in cpuinfo_path.read_text().splitlines():
                if line.startswith("model name"):
                    _, _, value = line.partition(":")
                    return value.strip() or None
        except OSError:
            return None
        return None

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


@dataclass(frozen=True)
class GpuDeviceUsage:
    """One accelerator's utilization and memory, identified by device index."""

    index: int
    utilization_percent: float | None
    memory: MemoryUsage | None
    # Marketing/model name from NVML or AMD SMI; None when the backend or
    # driver does not expose one.
    name: str | None = None


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
_amd_smi_state: tuple[Any, list[Any]] | tuple[()] | None = None
_amd_activity_supported: bool | None = None


def _amd_smi() -> tuple[Any, list[Any]] | None:
    """Lazily initialize AMD SMI, returning ``(module, processor_handles)``."""
    global _amd_smi_state
    if _amd_smi_state is None:
        try:
            import amdsmi

            amdsmi.amdsmi_init()
            _amd_smi_state = (amdsmi, amdsmi.amdsmi_get_processor_handles())
        except Exception:
            _amd_smi_state = ()
    return _amd_smi_state or None


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
    """Read accelerator memory via AMD SMI or NVML, per device or summed."""

    def sample_per_device(self) -> list[MemoryUsage] | None:
        """One :class:`MemoryUsage` per device (backend enumeration order)."""
        session = _amd_smi()
        if session is not None:
            amdsmi, devices = session
            usages: list[MemoryUsage] = []
            try:
                for device in devices:
                    used = int(amdsmi.amdsmi_get_gpu_memory_usage(device, amdsmi.AmdSmiMemoryType.VRAM))
                    total = int(amdsmi.amdsmi_get_gpu_memory_total(device, amdsmi.AmdSmiMemoryType.VRAM))
                    usages.append(MemoryUsage(used_bytes=used, total_bytes=total))
            except Exception:
                return None
            return usages
        session = _nvml()
        if session is None:
            return None
        pynvml, devices = session
        try:
            return [
                MemoryUsage(used_bytes=info.used, total_bytes=info.total)
                for info in (pynvml.nvmlDeviceGetMemoryInfo(device) for device in devices)
            ]
        except pynvml.NVMLError:
            return None

    def sample(self) -> MemoryUsage | None:
        usages = self.sample_per_device()
        if not usages:
            return None
        used = sum(usage.used_bytes for usage in usages)
        total = sum(usage.total_bytes for usage in usages)
        return MemoryUsage(used_bytes=used, total_bytes=total) if total > 0 else None


class GpuUtilizationSampler:
    """Read accelerator utilization via AMD SMI or NVML, per device or averaged."""

    def sample_per_device(self) -> list[float | None] | None:
        """One utilization percentage per device, or ``None`` per unavailable device.

        Returns ``None`` when no per-device source exists (e.g. the AMD sysfs
        fallback reports unlabeled cards); callers fall back to the aggregate.
        """
        global _amd_activity_supported
        session = _amd_smi()
        if session is not None:
            amdsmi, devices = session
            if _amd_activity_supported is False:
                return None
            values: list[float | None] = []
            try:
                for device in devices:
                    activity = amdsmi.amdsmi_get_gpu_activity(device)
                    value = activity.get("gfx_activity", activity.get("gpu_busy_percent"))
                    values.append(float(value) if value is not None else None)
            except Exception:
                _amd_activity_supported = False
                return None
            return values
        session = _nvml()
        if session is None:
            return None
        pynvml, devices = session
        try:
            return [float(pynvml.nvmlDeviceGetUtilizationRates(device).gpu) for device in devices]
        except pynvml.NVMLError:
            return None

    def sample(self) -> float | None:
        amd_session = _amd_smi() is not None
        values = self.sample_per_device()
        if values is None:
            if not amd_session:
                return None
            # Some integrated AMD GPUs (including Radeon 890M) expose VRAM
            # through AMD SMI but return AMDSMI_STATUS_UNEXPECTED_DATA for
            # ``amdsmi_get_gpu_activity``. The kernel's DRM sysfs counter is
            # available on those devices and reports the same busy percentage
            # used by rocm-smi (per-card but unlabeled, hence aggregate-only).
            sysfs = _sysfs_gpu_busy_percent()
            return sum(sysfs) / len(sysfs) if sysfs else None
        known = [value for value in values if value is not None]
        return sum(known) / len(known) if known else None


def _gpu_device_names() -> list[str | None] | None:
    """One name per device (backend enumeration order), or ``None``.

    Names are static, so the query result is cached alongside the backend
    session state. AMD SMI product info varies by driver; unrecognized
    shapes degrade to a ``None`` entry rather than failing the whole list.
    """
    session = _amd_smi()
    if session is not None:
        amdsmi, devices = session
        names: list[str | None] = []
        try:
            for device in devices:
                info = amdsmi.amdsmi_get_processor_info(device)
                # Parenthesized: the isinstance guard selects the whole `or`
                # chain (conditional expressions bind loosest), so a non-dict
                # info degrades to None instead of touching .get().
                name = (info.get("market_name") or info.get("product_name")) if isinstance(info, dict) else None
                names.append(str(name) if name else None)
        except Exception:
            return None
        return names
    session = _nvml()
    if session is None:
        return None
    pynvml, devices = session
    try:
        # Older pynvml returns bytes; modern versions return str. Any failure
        # (unsupported handle, driver quirk) degrades to unnamed rows.
        names = [pynvml.nvmlDeviceGetName(device) for device in devices]
        return [name.decode() if isinstance(name, bytes) else name for name in names]
    except Exception:
        return None


def sample_gpu_devices(
    utilization: GpuUtilizationSampler,
    memory: GpuMemoryUsageSampler,
) -> list[GpuDeviceUsage] | None:
    """Combine the two samplers into one per-device usage list, or ``None``.

    Devices come from the same backend enumeration in both samplers, so
    indices pair up. A failing memory source degrades to utilization-only;
    a failing name source degrades to unnamed rows.
    """
    utils = utilization.sample_per_device()
    if utils is None:
        return None
    memories = memory.sample_per_device()
    names = _gpu_device_names()
    return [
        GpuDeviceUsage(
            index=index,
            utilization_percent=value,
            memory=memories[index] if memories is not None and index < len(memories) else None,
            name=names[index] if names is not None and index < len(names) else None,
        )
        for index, value in enumerate(utils)
    ]


def _sysfs_gpu_busy_percent() -> list[float]:
    """Read AMD DRM GPU busy counters as a fallback for unsupported SMI APIs."""
    if sys.platform == "win32":
        return []
    values: list[float] = []
    for path in Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"):
        try:
            value = float(path.read_text().strip())
        except (OSError, ValueError):
            continue
        if 0.0 <= value <= 100.0:
            values.append(value)
    return values
