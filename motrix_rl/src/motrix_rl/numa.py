# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Lightweight Linux NUMA discovery and placement shared by training and benches.

Importing this module does not load NumPy, Torch, Numba, or the simulator;
the package initializer may load other framework dependencies. Memory policies
affect future allocations of the calling thread and are inherited by new threads
and spawned processes; apply placement before runtime initialization.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path

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
