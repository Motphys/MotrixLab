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


def test_gpu_samplers_return_none_without_nvml(monkeypatch) -> None:
    monkeypatch.setattr(system_metrics, "_nvml_state", ())

    assert GpuUtilizationSampler().sample() is None
    assert GpuMemoryUsageSampler().sample() is None


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
