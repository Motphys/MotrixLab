# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Lightweight Linux NUMA discovery and placement shared by training and benches.

Importing this module does not load NumPy, Torch, Numba, or the simulator;
the package initializer may load other framework dependencies. Memory policies
affect future allocations of the calling thread and are inherited by new threads
and spawned processes; apply placement before runtime initialization.
"""

from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import functools
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)
_NODE_SYSFS = Path("/sys/devices/system/node")


@dataclass(frozen=True)
class NodePlacement:
    node_id: int | None
    cpus: tuple[int, ...]


def parse_cpulist(text: str) -> list[int]:
    """Parse Linux CPU/node lists with sparse IDs and inclusive ranges."""
    result: set[int] = set()
    for part in text.strip().split(","):
        if not part:
            continue
        bounds = part.split("-")
        first = int(bounds[0])
        last = int(bounds[-1])
        if first < 0 or last < first or len(bounds) > 2:
            raise ValueError(f"Invalid ID range: {part!r}")
        result.update(range(first, last + 1))
    return sorted(result)


def available_numa_nodes(sysfs_root: str | Path = _NODE_SYSFS) -> list[int]:
    """Online node IDs, or an empty list when topology is unavailable."""
    try:
        return parse_cpulist((Path(sysfs_root) / "online").read_text())
    except OSError:
        return []


def numa_node_cpus(node: int, sysfs_root: str | Path = _NODE_SYSFS) -> list[int]:
    """CPU IDs belonging to a node; unknown nodes raise ValueError."""
    cpulist = Path(sysfs_root) / f"node{node}" / "cpulist"
    try:
        return parse_cpulist(cpulist.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"NUMA node {node} does not exist ({cpulist} missing)") from error


def discover_nodes(
    sysfs_root: str | Path = _NODE_SYSFS,
    status_path: str | Path = "/proc/self/status",
    affinity: set[int] | None = None,
) -> list[NodePlacement]:
    """Intersect online nodes with inherited CPU and allowed-memory constraints.

    Unavailable topology yields one inherited-placement entry. A known topology
    without a usable CPU/memory node raises rather than bypassing restrictions.
    """
    try:
        if affinity is None:
            affinity = set(os.sched_getaffinity(0))
        online = available_numa_nodes(sysfs_root)
        if not online:
            return [NodePlacement(None, ())]
        status = Path(status_path).read_text()
        allowed_line = next(line for line in status.splitlines() if line.startswith("Mems_allowed_list:"))
        allowed = set(parse_cpulist(allowed_line.split(":", 1)[1]))
        nodes = []
        for node_id in sorted(set(online) & allowed):
            usable = tuple(sorted(set(numa_node_cpus(node_id, sysfs_root)) & affinity))
            if usable:
                nodes.append(NodePlacement(node_id, usable))
    except (OSError, AttributeError, StopIteration):
        return [NodePlacement(None, ())]
    if not nodes:
        raise ValueError("No online NUMA nodes have both allowed memory and CPUs in the initial affinity")
    return nodes


def apply_cpu_affinity(cpus: list[int] | tuple[int, ...], role: str) -> list[int]:
    """Pin the caller without expanding inherited CPU affinity; return actual CPUs."""
    if not hasattr(os, "sched_getaffinity"):
        logger.warning("%s: CPU affinity unsupported; skipping binding", role)
        return []
    allowed = os.sched_getaffinity(0)
    selected = sorted(set(cpus) & allowed)
    if not selected:
        raise ValueError(f"{role}: none of the requested CPUs are allowed (affinity={sorted(allowed)})")
    os.sched_setaffinity(0, selected)
    if set(cpus) - allowed:
        logger.warning("%s: dropped non-allowed CPUs %s", role, sorted(set(cpus) - allowed))
    return sorted(os.sched_getaffinity(0))


@dataclass(frozen=True)
class MemoryPolicy:
    """Exact get_mempolicy snapshot, including mode flags and node mask words."""

    mode: int
    mask: tuple[int, ...]


def _load_libnuma():
    library = ctypes.util.find_library("numa")
    if library is None:
        raise OSError("libnuma not found")
    return ctypes.CDLL(library, use_errno=True)


def _mask_words(node: int) -> int:
    return node // (ctypes.sizeof(ctypes.c_ulong) * 8) + 1


def set_memory_policy(policy: MemoryPolicy) -> None:
    """Apply a policy with checked set_mempolicy return status and errno."""
    numa = _load_libnuma()
    set_policy = numa.set_mempolicy
    set_policy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
    set_policy.restype = ctypes.c_int
    mask = (ctypes.c_ulong * len(policy.mask))(*policy.mask)
    # MPOL_DEFAULT and MPOL_LOCAL require a null nodemask (flags occupy high bits).
    local_or_default = policy.mode & 0x3FFF in (0, 4)
    pointer = None if local_or_default else mask
    maxnode = 0 if local_or_default else len(mask) * ctypes.sizeof(ctypes.c_ulong) * 8
    ctypes.set_errno(0)
    if set_policy(policy.mode, pointer, maxnode) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def get_memory_policy() -> MemoryPolicy:
    """Snapshot the caller's mode and mask, raising when querying is unavailable."""
    numa = _load_libnuma()
    get_policy = numa.get_mempolicy
    get_policy.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    get_policy.restype = ctypes.c_int
    # The kernel requires room for all possible nodes, not merely online nodes.
    possible = parse_cpulist((_NODE_SYSFS / "possible").read_text())
    mask = (ctypes.c_ulong * _mask_words(max(possible)))()
    mode = ctypes.c_int()
    ctypes.set_errno(0)
    if get_policy(ctypes.byref(mode), mask, len(mask) * ctypes.sizeof(ctypes.c_ulong) * 8, None, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    return MemoryPolicy(mode.value, tuple(mask))


def bind_memory(node: int) -> None:
    """Request MPOL_BIND for future allocations; report failure via OSError."""
    bits = ctypes.sizeof(ctypes.c_ulong) * 8
    mask = [0] * _mask_words(node)
    mask[node // bits] = 1 << (node % bits)
    set_memory_policy(MemoryPolicy(2, tuple(mask)))


def apply_memory_policy(node: int, role: str) -> bool:
    """Best-effort memory binding for training callers, with honest success status."""
    try:
        bind_memory(node)
    except (OSError, AttributeError) as error:
        logger.warning("%s: memory binding to node %d failed: %s", role, node, error)
        return False
    return True


def apply_placement(node: NodePlacement) -> dict[str, object]:
    """Bind CPUs and future allocations, reporting effective affinity and errors."""
    result: dict[str, object] = {
        "node_id": node.node_id,
        "requested_cpus": list(node.cpus),
        "cpu_bound": False,
        "memory_bound": False,
        "memory_policy": "inherited",
    }
    if node.node_id is None:
        if hasattr(os, "sched_getaffinity"):
            result["actual_cpus"] = sorted(os.sched_getaffinity(0))
        result["placement_warning"] = "NUMA topology unavailable; worker is unbound"
        return result
    actual_cpus = apply_cpu_affinity(node.cpus, "worker")
    result["cpu_bound"] = bool(actual_cpus)
    result["actual_cpus"] = actual_cpus
    try:
        bind_memory(node.node_id)
        result["memory_bound"] = True
        result["memory_policy"] = "bind"
    except (OSError, AttributeError) as error:
        if isinstance(error, OSError) and error.errno is not None:
            result["memory_errno"] = error.errno
        result["memory_error"] = str(error)
        result["memory_policy"] = "cpu-only" if actual_cpus else "inherited"
    return result


@contextlib.contextmanager
def spawn_placement(node: int | None, role: str):
    """Pre-place a spawn child's import-time allocations, then restore the parent.

    CPU affinity and memory policy survive fork+exec. Temporarily binding the
    parent around ``Process.start()`` therefore places allocations made before
    the child's worker entry point, including runtime imports, on the node.
    """
    if node is None:
        yield
        return
    saved_affinity = os.sched_getaffinity(0)
    saved_policy = None
    memory_changed = False
    try:
        cpus = sorted(set(numa_node_cpus(node)) & saved_affinity)
        if not cpus:
            raise ValueError(f"{role}: no allowed CPUs on NUMA node {node}")
        apply_cpu_affinity(cpus, role)
        try:
            saved_policy = get_memory_policy()
        except (OSError, AttributeError) as error:
            logger.warning("%s: cannot snapshot memory policy; retaining inherited policy: %s", role, error)
        if saved_policy is not None:
            memory_changed = apply_memory_policy(node, role)
        yield
    finally:
        try:
            os.sched_setaffinity(0, saved_affinity)
        finally:
            if memory_changed:
                try:
                    set_memory_policy(saved_policy)
                except (OSError, AttributeError) as error:
                    logger.warning("%s: restoring inherited memory policy failed: %s", role, error)


def apply_binding(role: str, node: int | None, cpus: list[int]) -> None:
    """Apply precomputed CPU affinity and best-effort node memory binding.

    With no CPUs and no node this is a no-op. Placement decisions belong to
    the caller; memory binding only affects future allocations.
    """
    if cpus:
        apply_cpu_affinity(cpus, role)
    if node is not None:
        apply_memory_policy(node, role)


_nvml_state: tuple[Any, list[Any]] | tuple[()] | None = None  # None: untried; (): unavailable


def _nvml() -> tuple[Any, list[Any]] | None:
    """Lazily initialize NVML and return ``(pynvml, pci_bus_ids)``, or ``None``.

    Like ``system_metrics``, reuse an in-process session without initializing
    CUDA. PCI bus IDs connect device indices to sysfs NUMA locality.
    """
    global _nvml_state
    if _nvml_state is None:
        try:
            import pynvml

            pynvml.nvmlInit()
            bus_ids = [
                pynvml.nvmlDeviceGetPciInfo(pynvml.nvmlDeviceGetHandleByIndex(index)).busId
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
            _nvml_state = (pynvml, bus_ids)
        except Exception:  # ImportError (pynvml missing) or NVML init failure (no driver/GPU)
            _nvml_state = ()
    return _nvml_state or None


def _normalize_pci_bus_id(bus_id: str) -> str | None:
    """Validate a PCI address and reduce a zero-padded eight-digit domain for sysfs."""
    if not isinstance(bus_id, str):
        return None
    match = re.fullmatch(r"([0-9a-fA-F]{4}|[0-9a-fA-F]{8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", bus_id)
    if match is None:
        return None
    domain, bus, device, function = (int(part, 16) for part in match.groups())
    if domain > 0xFFFF or device > 0x1F:
        return None
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function}"


_HIP_PCI_PROBE = r"""
import ctypes
import ctypes.util
import json
from pathlib import Path

try:
    import torch

    # Torch prepares its wheel's shared-library dependencies before HIP loads.
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    candidates = []
    for name in ("amdhip64", "galaxyhip"):
        candidates.extend(str(path) for path in sorted(torch_lib.glob("lib" + name + ".so*")))
    for name in ("amdhip64", "galaxyhip"):
        candidates.append(ctypes.util.find_library(name) or ("lib" + name + ".so"))
    runtime = None
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            get_count = library.hipGetDeviceCount
            get_bus_id = library.hipDeviceGetPCIBusId
            runtime = library
            break
        except (OSError, AttributeError):
            continue
    if runtime is None:
        raise RuntimeError("HIP runtime unavailable")
    get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_count.restype = ctypes.c_int
    get_bus_id.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
    get_bus_id.restype = ctypes.c_int
    count = ctypes.c_int()
    if get_count(ctypes.byref(count)) != 0 or count.value < 0:
        raise RuntimeError("HIP device enumeration failed")
    bus_ids = []
    for index in range(count.value):
        buffer = ctypes.create_string_buffer(64)
        if get_bus_id(buffer, len(buffer), index) == 0:
            bus_ids.append(buffer.value.decode("ascii"))
        else:
            bus_ids.append(None)
    print(json.dumps(bus_ids))
except Exception:
    print("null")
"""


@functools.lru_cache(maxsize=8)
def _hip_pci_bus_ids(
    hip_visible_devices: str | None,
    rocr_visible_devices: str | None,
    cuda_visible_devices: str | None,
) -> tuple[str | None, ...] | None:
    """Query runtime ordinals in a child, caching each visibility mapping.

    Loading or initializing HIP in the pre-spawn parent is unsafe. The child
    inherits the environment, with visibility values also explicit in the cache
    key so changing an ordinal remapping cannot reuse an earlier probe.
    """
    env = os.environ.copy()
    for name, value in (
        ("HIP_VISIBLE_DEVICES", hip_visible_devices),
        ("ROCR_VISIBLE_DEVICES", rocr_visible_devices),
        ("CUDA_VISIBLE_DEVICES", cuda_visible_devices),
    ):
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _HIP_PCI_PROBE],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            env=env,
        )
        bus_ids = json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not isinstance(bus_ids, list):
        return None
    # Keep failed entries in place: dropping one would shift all later ordinals.
    return tuple(_normalize_pci_bus_id(bus_id) for bus_id in bus_ids)


def gpu_numa_node(device_index: int, runtime: Literal["cuda", "hip"]) -> int | None:
    """NUMA node hosting a GPU for the explicitly selected runtime, or ``None``.

    HIP queries visible runtime ordinals in an isolated subprocess, never
    initializing HIP in the parent. CUDA retains NVML and the NVIDIA topology
    fallback. Both read sysfs ``numa_node``; -1 means unknown. The caller owns
    runtime selection, so importing this module never imports Torch. CUDA uses
    physical NVML indices; CUDA_VISIBLE_DEVICES ordinal remapping is not handled.
    """
    if runtime not in ("cuda", "hip"):
        raise ValueError(f"Unknown GPU runtime: {runtime!r}")
    if device_index < 0:
        return None
    if runtime == "hip":
        bus_ids = _hip_pci_bus_ids(
            os.environ.get("HIP_VISIBLE_DEVICES"),
            os.environ.get("ROCR_VISIBLE_DEVICES"),
            os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    else:
        session = _nvml()
        bus_ids = session[1] if session is not None else None
    if bus_ids is None or device_index >= len(bus_ids):
        return None
    bus_id = _normalize_pci_bus_id(bus_ids[device_index])
    if bus_id is not None:
        try:
            node = int((Path("/sys/bus/pci/devices") / bus_id / "numa_node").read_text().strip())
            if node >= 0:
                return node
        except (OSError, ValueError):
            pass
    # Never use NVIDIA host ordinals to guess a HIP device's locality.
    return None if runtime == "hip" else _gpu_node_from_topology(device_index)


@functools.lru_cache(maxsize=1)
def _gpu_cpu_affinities() -> dict[int, frozenset[int]]:
    """GPU index -> CPU affinity set, parsed from ``nvidia-smi topo -m``."""
    try:
        proc = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return {}
    affinities: dict[int, frozenset[int]] = {}
    for line in proc.stdout.splitlines():
        cells = line.split()
        if not cells or not cells[0].lstrip("\x1b[4m").startswith("GPU"):
            continue
        try:
            index = int(cells[0].lstrip("\x1b[4m").removeprefix("GPU"))
        except ValueError:
            continue
        # Header/data rows differ in layout; find the multi-CPU list token.
        for cell in cells[1:]:
            if not re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)+|[0-9]+-[0-9]+", cell):
                continue
            cpus = parse_cpulist(cell)
            if len(cpus) > 1:
                affinities[index] = frozenset(cpus)
                break
    return affinities


def _gpu_node_from_topology(device_index: int) -> int | None:
    """Node whose CPU list overlaps the GPU's affinity the most."""
    affinity = _gpu_cpu_affinities().get(device_index)
    if not affinity:
        return None
    best_node, best_overlap = None, 0
    for node in available_numa_nodes():
        try:
            overlap = len(set(numa_node_cpus(node)) & affinity)
        except ValueError:
            continue
        if overlap > best_overlap:
            best_node, best_overlap = node, overlap
    return best_node
