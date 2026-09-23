# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Best-effort NUMA binding for multi-collector async FastSAC training.

Each collector on a multi-NUMA-node server should run its env, staging buffers
and pinned host allocations with node-local memory, and the scheduler should not
migrate it across nodes. WHICH node each worker gets is decided by
``async_impl/topology.py`` (learners on their GPU's PCIe-local node, collectors
on their owning learner's node); this module applies the decided placement
(the ``numactl --cpunodebind= --membind=`` equivalent) from inside the
worker process:

* CPU affinity via ``os.sched_setaffinity`` (portable stdlib);
* memory policy via libnuma's ``set_membind`` (the same call ``numactl`` uses),
  loaded with :mod:`ctypes` — future allocations of the calling process come
  from the bound node.

Everything is best-effort: when libnuma is unavailable or the node is unknown
the binding logs a warning and continues with OS default placement, so single
NUMA machines and containers keep working unchanged.

Binding must happen at process start, before the env and any staging buffer is
allocated — ``set_membind`` only affects future allocations.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

_NODE_SYSFS = Path("/sys/devices/system/node")

logger = logging.getLogger(__name__)

_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")


def parse_cpulist(text: str) -> list[int]:
    """Parse a sysfs ``cpulist`` (e.g. ``"0-3,8,10-11"``) into CPU ids."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        match = _RANGE_RE.match(part)
        if match:
            cpus.extend(range(int(match.group(1)), int(match.group(2)) + 1))
        else:
            cpus.append(int(part))
    return cpus


def available_numa_nodes() -> list[int]:
    """NUMA node ids visible in sysfs; empty when the host is not NUMA-aware."""
    if not _NODE_SYSFS.is_dir():
        return []
    return sorted(int(p.name.removeprefix("node")) for p in _NODE_SYSFS.iterdir() if p.name.startswith("node"))


def numa_node_cpus(node: int) -> list[int]:
    """CPU ids of a NUMA node; raises ``ValueError`` for unknown nodes."""
    cpulist = _NODE_SYSFS / f"node{node}" / "cpulist"
    if not cpulist.is_file():
        raise ValueError(f"NUMA node {node} does not exist ({cpulist} missing)")
    return parse_cpulist(cpulist.read_text())


def select_collector_cpus(
    base: list[int],
    collector_id: int,
    cpus_per_collector: int | None,
) -> list[int]:
    """Pick this collector's CPU slice from ``base`` (its node's or the process's CPUs).

    ``collector_id`` indexes WITHIN ``base`` — callers holding a per-node base
    pass the node-local ordinal, not the global collector id. Without
    ``cpus_per_collector`` all collectors share the full ``base`` set and the
    OS load-balances them; with it, the set is split into contiguous chunks
    so collectors never compete for the same cores.
    """
    if not base:
        raise ValueError("empty CPU set for collector binding")
    if cpus_per_collector is None:
        return list(base)
    if cpus_per_collector <= 0:
        raise ValueError(f"cpus_per_collector must be positive, got {cpus_per_collector}")
    start = collector_id * cpus_per_collector
    end = min(start + cpus_per_collector, len(base))
    if start >= end:
        raise ValueError(
            f"cpus_per_collector={cpus_per_collector} leaves no CPUs for collector {collector_id} "
            f"(base set has {len(base)} CPUs)"
        )
    return base[start:end]


def apply_cpu_affinity(cpus: list[int], role: str) -> None:
    """Pin the current process to ``cpus``, intersected with allowed CPUs."""
    allowed = os.sched_getaffinity(0)
    selected = sorted(set(cpus) & allowed)
    if not selected:
        raise ValueError(
            f"{role}: none of the requested CPUs {sorted(cpus)} are allowed (process affinity is {sorted(allowed)})"
        )
    os.sched_setaffinity(0, selected)
    if set(cpus) - allowed:
        logger.warning("%s: dropped non-allowed CPUs %s from binding", role, sorted(set(cpus) - allowed))


def _load_libnuma():
    try:
        return ctypes.CDLL("libnuma.so.1", use_errno=True)
    except OSError:
        return None


def _membind(libnuma, bitmask_ptr, role: str, what: str, node: int | None = None) -> bool:
    if libnuma.numa_set_membind(ctypes.c_void_p(bitmask_ptr)) != 0:
        logger.warning("%s: numa_set_membind(%s) failed; %s not bound", role, node, what)
        return False
    return True


def _node_bitmask(node: int):
    libnuma = _load_libnuma()
    if libnuma is None:
        return None, None
    libnuma.numa_allocate_nodemask.restype = ctypes.c_void_p
    mask = libnuma.numa_allocate_nodemask()
    if not mask:
        return None, None
    libnuma.numa_bitmask_setbit(ctypes.c_void_p(mask), ctypes.c_uint(node))
    return libnuma, mask


def _free_bitmask(libnuma, mask) -> None:
    libnuma.numa_bitmask_free(ctypes.c_void_p(mask))


def apply_memory_policy(node: int, role: str) -> None:
    """Bind future allocations of this process to ``node`` (libnuma ``numa_set_membind``)."""
    libnuma, mask = _node_bitmask(node)
    if libnuma is None:
        logger.warning("%s: libnuma unavailable; memory policy left to the OS (node %d not bound)", role, node)
        return
    try:
        _membind(libnuma, mask, role, "memory policy", node)
    finally:
        _free_bitmask(libnuma, mask)


@contextlib.contextmanager
def spawn_placement(node: int | None, role: str):
    """Place a spawn-started child on ``node`` by pre-placing its parent.

    A ``spawn`` child re-imports torch / the simulator / glibc arenas before
    the worker entry function runs, so a bind executed inside the child only
    affects allocations made after those imports — simulator thread pools and
    malloc arenas created at import time land on a random node and physics
    stepping pays cross-node access forever. Affinity and memory policy both
    survive fork+exec, so briefly switching the *parent* onto ``node`` around
    ``Process.start()`` makes the child's import-time allocations node-local;
    the worker's own :func:`apply_binding` call then re-affirms the same
    binding (a no-op refinement).
    """
    if node is None:
        yield
        return
    saved_affinity = os.sched_getaffinity(0)
    try:
        cpus = sorted(set(numa_node_cpus(node)) & saved_affinity)
        if cpus:
            os.sched_setaffinity(0, cpus)
        libnuma, mask = _node_bitmask(node)
        if libnuma is not None and mask:
            try:
                _membind(libnuma, mask, role, "spawn placement", node)
            finally:
                _free_bitmask(libnuma, mask)
        yield
    finally:
        os.sched_setaffinity(0, saved_affinity)
        libnuma = _load_libnuma()
        if libnuma is not None:
            # restore the default "any node" policy for the parent
            try:
                all_nodes = ctypes.c_void_p.in_dll(libnuma, "numa_all_nodes_ptr")
                libnuma.numa_set_membind(all_nodes)
            except (ValueError, OSError, AttributeError):
                pass


def apply_binding(role: str, node: int | None, cpus: list[int]) -> None:
    """Apply a topology-decided binding: CPU affinity plus node memory policy.

    ``cpus`` comes pre-computed from the topology resolution (the worker
    never decides a binding itself). Both steps are best-effort; with no
    CPUs and no node this is a no-op.
    """
    if cpus:
        apply_cpu_affinity(cpus, role)
    if node is not None:
        apply_memory_policy(node, role)


_nvml_state: tuple[Any, list[Any]] | tuple[()] | None = None  # None: untried; (): unavailable


def _nvml() -> tuple[Any, list[Any]] | None:
    """Lazily initialize NVML and return ``(pynvml, pci_bus_ids)``, or ``None``.

    Same in-process NVML session pattern as ``system_metrics``: microsecond
    queries instead of an ``nvidia-smi`` subprocess spawn, and still no CUDA
    context. NVML maps a device index to its PCI bus id, from which the sysfs
    ``numa_node`` file gives the hosting NUMA node.
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


def gpu_numa_node(device_index: int) -> int | None:
    """NUMA node hosting a CUDA device; None when unknown.

    Primary source: NVML (pynvml, same session pattern as ``system_metrics``)
    for the index -> PCI bus id mapping and sysfs ``numa_node`` for the node
    lookup — no CUDA context is created, so the pre-spawn parent can call this
    freely. Fallback (containers often do not expose the PCI ``numa_node``
    sysfs attribute): the GPU's ``CPU Affinity`` column from
    ``nvidia-smi topo -m``, matched against each node's ``cpulist``. A sysfs
    value of -1 (unknown / single-node host) maps to None (no binding).
    """
    session = _nvml()
    if session is None or device_index >= len(session[1]):
        return None
    try:
        node = int((Path("/sys/bus/pci/devices") / session[1][device_index].lower() / "numa_node").read_text().strip())
        if node >= 0:
            return node
    except (OSError, ValueError):
        pass
    return _gpu_node_from_topology(device_index)


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
        # The header and data rows do not share a column layout (ANSI escapes,
        # padded diagonal cells, differing column counts), so positional
        # indexing is unreliable: take the token that IS a multi-cpu list.
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
