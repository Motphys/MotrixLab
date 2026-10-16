# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import ctypes
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from motrix_rl import numa


@pytest.fixture(autouse=True)
def clear_hip_cache():
    probe = numa._hip_pci_bus_ids
    probe.cache_clear()
    yield
    probe.cache_clear()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0000:07:00.0", "0000:07:00.0"),
        ("00000000:AF:00.1", "0000:af:00.1"),
        ("../../etc/passwd", None),
        ("bad", None),
        ("0000:07:00.9", None),
    ],
)
def test_normalize_pci_bus_id(raw, expected):
    assert numa._normalize_pci_bus_id(raw) == expected


def test_hip_probe_is_isolated_and_cached(monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(["00000000:07:00.0", "0000:0a:00.0"]))

    monkeypatch.setattr(numa.subprocess, "run", run)
    first = numa._hip_pci_bus_ids(None, None, None)
    assert first == ("0000:07:00.0", "0000:0a:00.0")
    assert numa._hip_pci_bus_ids(None, None, None) == first
    assert len(calls) == 1
    assert calls[0][0] == [sys.executable, "-c", numa._HIP_PCI_PROBE]
    assert calls[0][1]["timeout"] > 0
    for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        assert key not in calls[0][1]["env"]
    numa._hip_pci_bus_ids("1,0", "0,1", "1")
    assert len(calls) == 2
    assert calls[1][1]["env"]["HIP_VISIBLE_DEVICES"] == "1,0"
    assert calls[1][1]["env"]["ROCR_VISIBLE_DEVICES"] == "0,1"
    assert calls[1][1]["env"]["CUDA_VISIBLE_DEVICES"] == "1"


@pytest.mark.parametrize("output", ["garbage", "{}", "null", '["../../etc/passwd"]'])
def test_invalid_hip_probe_output_is_unknown(monkeypatch, output):
    monkeypatch.setattr(
        numa.subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=output)
    )
    result = numa._hip_pci_bus_ids(None, None, None)
    assert result is None or all(item is None for item in result)


def test_hip_probe_preserves_failed_ordinals(monkeypatch):
    monkeypatch.setattr(
        numa.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout='[null, "0000:0a:00.0"]'),
    )
    assert numa._hip_pci_bus_ids(None, None, None) == (None, "0000:0a:00.0")


@pytest.mark.parametrize(
    "error",
    [OSError("missing runtime"), subprocess.TimeoutExpired("probe", 10), subprocess.CalledProcessError(1, "probe")],
)
def test_hip_probe_failure_is_unknown(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(numa.subprocess, "run", fail)
    assert numa._hip_pci_bus_ids(None, None, None) is None


@pytest.mark.parametrize("library_name", ["libamdhip64.so.6", "libgalaxyhip.so.1"])
def test_child_probe_prefers_torch_wheel_versioned_library(monkeypatch, tmp_path, capsys, library_name):
    torch_dir = tmp_path / "torch"
    library = torch_dir / "lib" / library_name
    library.parent.mkdir(parents=True)
    library.touch()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__file__=str(torch_dir / "__init__.py")))
    loaded = []

    class GetCount:
        def __call__(self, pointer):
            pointer._obj.value = 2
            return 0

    class GetBusId:
        def __call__(self, buffer, size, index):
            if index == 0:
                return 1
            buffer.value = b"0000:0a:00.0"
            return 0

    def load(path):
        assert "torch" in sys.modules
        loaded.append(path)
        return SimpleNamespace(hipGetDeviceCount=GetCount(), hipDeviceGetPCIBusId=GetBusId())

    monkeypatch.setattr(ctypes, "CDLL", load)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: "system-" + name)
    exec(numa._HIP_PCI_PROBE, {})
    assert loaded == [str(library)]
    assert json.loads(capsys.readouterr().out) == [None, "0000:0a:00.0"]


@pytest.mark.parametrize("failure", ["load", "symbols", "enumeration"])
def test_child_probe_reports_runtime_failures(monkeypatch, tmp_path, capsys, failure):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__file__=str(tmp_path / "__init__.py")))
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)

    class GetCount:
        def __call__(self, pointer):
            return 1

    def load(path):
        if failure == "load":
            raise OSError("unavailable")
        if failure == "symbols":
            return SimpleNamespace()
        return SimpleNamespace(hipGetDeviceCount=GetCount(), hipDeviceGetPCIBusId=GetCount())

    monkeypatch.setattr(ctypes, "CDLL", load)
    exec(numa._HIP_PCI_PROBE, {})
    assert json.loads(capsys.readouterr().out) is None


def test_hip_ordinal_reads_sysfs_without_nvidia_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(numa, "_nvml", lambda: pytest.fail("HIP must not use NVML physical ordinals"))
    monkeypatch.setattr(numa, "_gpu_node_from_topology", lambda index: pytest.fail("HIP must not use NVIDIA topology"))
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,0")
    seen = []

    def probe(*visibility):
        seen.append(visibility)
        return ("0000:0a:00.0", "0000:07:00.0")

    monkeypatch.setattr(numa, "_hip_pci_bus_ids", probe)
    node = tmp_path / "0000:0a:00.0" / "numa_node"
    node.parent.mkdir()
    node.write_text("3\n")
    monkeypatch.setattr(numa, "Path", lambda value: tmp_path)
    assert numa.gpu_numa_node(0, runtime="hip") == 3
    assert seen[0][0] == "1,0"
    node.write_text("-1\n")
    assert numa.gpu_numa_node(0, runtime="hip") is None
    node.unlink()
    assert numa.gpu_numa_node(0, runtime="hip") is None
    assert numa.gpu_numa_node(9, runtime="hip") is None
    assert numa.gpu_numa_node(-1, runtime="hip") is None
    monkeypatch.setattr(numa, "_hip_pci_bus_ids", lambda *args: None)
    assert numa.gpu_numa_node(0, runtime="hip") is None


def test_nvidia_path_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(numa, "_nvml", lambda: (object(), ["0000:07:00.0"]))
    monkeypatch.setattr(numa, "_hip_pci_bus_ids", lambda *args: pytest.fail("NVIDIA must not probe HIP"))
    node = tmp_path / "0000:07:00.0" / "numa_node"
    node.parent.mkdir()
    node.write_text("2")
    monkeypatch.setattr(numa, "Path", lambda value: tmp_path)
    assert numa.gpu_numa_node(0, runtime="cuda") == 2
    node.unlink()
    monkeypatch.setattr(numa, "_gpu_node_from_topology", lambda index: 4)
    assert numa.gpu_numa_node(0, runtime="cuda") == 4


def test_gpu_runtime_must_be_explicit_and_supported():
    with pytest.raises(ValueError, match="Unknown GPU runtime"):
        numa.gpu_numa_node(0, runtime="unknown")
