# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import ctypes
import sys
import types

import motrix_rl.system_metrics as system_metrics
from motrix_rl.system_metrics import (
    CpuLoadSampler,
    GpuMemoryUsageSampler,
    GpuUtilizationSampler,
    MemoryUsage,
    MemoryUsageSampler,
)


def test_cpu_load_sampler_uses_counter_deltas_for_available_cpus(tmp_path) -> None:
    stat_path = tmp_path / "stat"
    stat_path.write_text(
        "cpu  200 0 0 1800 0 0 0 0\ncpu0 100 0 0 900 0 0 0 0\ncpu1 100 0 0 900 0 0 0 0\ncpu2 100 0 0 900 0 0 0 0\n"
    )
    sampler = CpuLoadSampler(stat_path=stat_path, topology_root=tmp_path, cpu_ids={0, 1})

    stat_path.write_text(
        "cpu  330 0 0 1850 20 0 0 0\ncpu0 150 0 0 950 0 0 0 0\ncpu1 180 0 0 900 20 0 0 0\ncpu2 300 0 0 900 0 0 0 0\n"
    )

    load = sampler.sample()

    assert load is not None
    assert load.utilization_percent == 65.0
    assert load.used_logical_cpus == 1.3
    assert load.logical_cpu_count == 2
    assert load.physical_core_count is None
    assert load.iowait_percent == 10.0
    assert load.steal_percent == 0.0


def test_cpu_load_sampler_returns_none_without_elapsed_cpu_time(tmp_path) -> None:
    stat_path = tmp_path / "stat"
    contents = "cpu0 100 0 0 900 0 0 0 0\n"
    stat_path.write_text(contents)
    sampler = CpuLoadSampler(stat_path=stat_path, topology_root=tmp_path, cpu_ids={0})
    stat_path.write_text(contents)

    assert sampler.sample() is None


def _fake_nvml(monkeypatch, handles, utilization, memory, error=None) -> None:
    """Install a fake ``(pynvml, handles)`` session with per-handle metric tables."""

    def utilization_rates(handle):
        if error is not None:
            raise error
        return types.SimpleNamespace(gpu=utilization[handle])

    def memory_info(handle):
        if error is not None:
            raise error
        return types.SimpleNamespace(used=memory[handle][0], total=memory[handle][1])

    fake_pynvml = types.SimpleNamespace(
        nvmlDeviceGetUtilizationRates=utilization_rates,
        nvmlDeviceGetMemoryInfo=memory_info,
        NVMLError=RuntimeError,
    )
    # The samplers prefer AMD SMI; pin it off so these tests exercise the
    # NVML backend on AMD hosts too, where a real session would win.
    monkeypatch.setattr(system_metrics, "_amd_smi_state", ())
    monkeypatch.setattr(system_metrics, "_nvml_state", (fake_pynvml, handles))


def test_gpu_samplers_aggregate_utilization_mean_and_memory_sum(monkeypatch) -> None:
    handles = ["gpu0", "gpu1"]
    _fake_nvml(
        monkeypatch,
        handles,
        utilization={"gpu0": 10, "gpu1": 30},
        memory={"gpu0": (100 * 1024**2, 200 * 1024**2), "gpu1": (300 * 1024**2, 400 * 1024**2)},
    )

    assert GpuUtilizationSampler().sample() == 20.0
    assert GpuMemoryUsageSampler().sample() == MemoryUsage(used_bytes=400 * 1024**2, total_bytes=600 * 1024**2)


def test_gpu_samplers_return_none_on_nvml_error(monkeypatch) -> None:
    handles = ["gpu0"]
    _fake_nvml(
        monkeypatch,
        handles,
        utilization={"gpu0": 10},
        memory={"gpu0": (1, 2)},
        error=RuntimeError("driver failure"),
    )

    assert GpuUtilizationSampler().sample() is None
    assert GpuMemoryUsageSampler().sample() is None


def _fake_amdsmi(monkeypatch, handles, utilization, memory) -> None:
    """Install a fake ``(amdsmi, handles)`` session with per-handle metric tables."""

    def gpu_activity(handle):
        return {"gfx_activity": utilization[handle]}

    def gpu_memory_usage(handle, memory_type):
        return memory[handle][0]

    def gpu_memory_total(handle, memory_type):
        return memory[handle][1]

    fake_amdsmi = types.SimpleNamespace(
        amdsmi_get_gpu_activity=gpu_activity,
        amdsmi_get_gpu_memory_usage=gpu_memory_usage,
        amdsmi_get_gpu_memory_total=gpu_memory_total,
        AmdSmiMemoryType=types.SimpleNamespace(VRAM="vram"),
    )
    # Pin NVML off so the AMD SMI backend is exercised on every host.
    monkeypatch.setattr(system_metrics, "_nvml_state", ())
    monkeypatch.setattr(system_metrics, "_amd_activity_supported", None)
    monkeypatch.setattr(system_metrics, "_amd_smi_state", (fake_amdsmi, handles))


def test_gpu_samplers_aggregate_via_amdsmi_backend(monkeypatch) -> None:
    handles = ["gpu0", "gpu1"]
    _fake_amdsmi(
        monkeypatch,
        handles,
        utilization={"gpu0": 10, "gpu1": 30},
        memory={"gpu0": (100 * 1024**2, 200 * 1024**2), "gpu1": (300 * 1024**2, 400 * 1024**2)},
    )

    assert GpuUtilizationSampler().sample() == 20.0
    assert GpuMemoryUsageSampler().sample() == MemoryUsage(used_bytes=400 * 1024**2, total_bytes=600 * 1024**2)


def test_gpu_samplers_return_none_without_nvml(monkeypatch) -> None:
    monkeypatch.setattr(system_metrics, "_amd_smi_state", ())
    monkeypatch.setattr(system_metrics, "_nvml_state", ())

    assert GpuUtilizationSampler().sample() is None
    assert GpuMemoryUsageSampler().sample() is None


def _patch_sysfs_gpu_busy(monkeypatch, entries) -> None:
    """Serve ``entries`` (text or Exception) as DRM ``gpu_busy_percent`` files."""

    class _SysfsFile:
        def __init__(self, payload) -> None:
            self._payload = payload

        def read_text(self) -> str:
            if isinstance(self._payload, Exception):
                raise self._payload
            return self._payload

    monkeypatch.setattr(system_metrics.Path, "glob", lambda self, pattern: iter(_SysfsFile(entry) for entry in entries))


def test_sysfs_gpu_busy_percent_reads_valid_counters(monkeypatch) -> None:
    _patch_sysfs_gpu_busy(monkeypatch, ["12\n", "34\n"])

    assert system_metrics._sysfs_gpu_busy_percent() == [12.0, 34.0]


def test_sysfs_gpu_busy_percent_skips_invalid_and_unreadable_counters(monkeypatch) -> None:
    _patch_sysfs_gpu_busy(monkeypatch, ["120\n", "-5\n", "n/a\n", OSError("denied"), "55\n"])

    assert system_metrics._sysfs_gpu_busy_percent() == [55.0]


def test_sysfs_gpu_busy_percent_returns_nothing_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    glob_called = False

    def fail_glob(pattern):
        nonlocal glob_called
        glob_called = True
        return iter(())

    monkeypatch.setattr(system_metrics.Path, "glob", fail_glob)

    assert system_metrics._sysfs_gpu_busy_percent() == []
    assert not glob_called


def test_gpu_utilization_sampler_falls_back_to_sysfs_when_amdsmi_activity_unsupported(monkeypatch) -> None:
    handles = ["gpu0", "gpu1"]

    def gpu_activity(handle):
        raise RuntimeError("AMDSMI_STATUS_UNEXPECTED_DATA")

    memory = {"gpu0": (1, 2), "gpu1": (3, 4)}
    fake_amdsmi = types.SimpleNamespace(
        amdsmi_get_gpu_activity=gpu_activity,
        amdsmi_get_gpu_memory_usage=lambda handle, memory_type: memory[handle][0],
        amdsmi_get_gpu_memory_total=lambda handle, memory_type: memory[handle][1],
        AmdSmiMemoryType=types.SimpleNamespace(VRAM="vram"),
    )
    monkeypatch.setattr(system_metrics, "_nvml_state", ())
    monkeypatch.setattr(system_metrics, "_amd_activity_supported", None)
    monkeypatch.setattr(system_metrics, "_amd_smi_state", (fake_amdsmi, handles))
    _patch_sysfs_gpu_busy(monkeypatch, ["20\n", "60\n"])

    assert GpuUtilizationSampler().sample() == 40.0

    # The failure is latched, so later samples keep reading sysfs instead of
    # retrying the unsupported AMD SMI activity API on every call.
    monkeypatch.setattr(system_metrics.Path, "glob", lambda self, pattern: iter([]))

    assert GpuUtilizationSampler().sample() is None


def test_memory_usage_sampler_reads_proc_meminfo(tmp_path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       32768000 kB\nMemFree:        1024000 kB\nCached:         8192000 kB\n"
        "MemAvailable:   16384000 kB\nSwapTotal:             0 kB\n"
    )
    sampler = MemoryUsageSampler(meminfo_path=meminfo)

    assert sampler.sample() == MemoryUsage(used_bytes=(32768000 - 16384000) * 1024, total_bytes=32768000 * 1024)


def test_memory_usage_sampler_returns_none_when_meminfo_missing(tmp_path) -> None:
    sampler = MemoryUsageSampler(meminfo_path=tmp_path / "missing")

    assert sampler.sample() is None


def test_memory_usage_sampler_dispatches_to_windows_api(monkeypatch) -> None:
    windows_usage = MemoryUsage(used_bytes=7, total_bytes=9)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(system_metrics, "_windows_memory_status", lambda: windows_usage)

    assert MemoryUsageSampler().sample() == windows_usage


def _patch_windows_memory_api(monkeypatch, *, succeed: bool) -> None:
    def global_memory_status_ex(pointer) -> int:
        status = ctypes.cast(pointer, ctypes.POINTER(system_metrics._MemoryStatusEx)).contents
        assert status.dwLength == ctypes.sizeof(system_metrics._MemoryStatusEx)
        if not succeed:
            return 0
        status.ullTotalPhys = 100
        status.ullAvailPhys = 25
        return 1

    kernel32 = types.SimpleNamespace(GlobalMemoryStatusEx=global_memory_status_ex)
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=kernel32), raising=False)


def test_windows_memory_status_maps_total_and_available_physical_memory(monkeypatch) -> None:
    _patch_windows_memory_api(monkeypatch, succeed=True)

    assert system_metrics._windows_memory_status() == MemoryUsage(used_bytes=75, total_bytes=100)


def test_windows_memory_status_returns_none_when_api_reports_failure(monkeypatch) -> None:
    _patch_windows_memory_api(monkeypatch, succeed=False)

    assert system_metrics._windows_memory_status() is None
