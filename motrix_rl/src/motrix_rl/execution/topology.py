# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Algorithm-independent worker layout; no transport or training policy.

Collectors form equal, contiguous blocks owned by learner ranks. Each collector
follows its owner's GPU locality and carries one equal environment shard. Device
resolution inspects runtime availability but never creates an accelerator context;
CPU and memory binding itself belongs to :mod:`motrix_rl.numa`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TypeVar

import torch

from motrix_rl import numa as system_numa

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


@dataclass(frozen=True)
class LearnerPlacement:
    """Device and host placement of one learner rank."""

    rank: int
    device: torch.device
    numa_node: int | None
    cpus: list[int]


@dataclass(frozen=True)
class CollectorPlacement:
    """Device, environment shard and host placement of an owned collector."""

    collector_id: int
    owner_rank: int
    num_envs: int
    device: torch.device
    numa_node: int | None
    cpus: list[int]


@dataclass(frozen=True)
class EndpointWiring:
    """Common producer-to-consumer placement and transport capability result."""

    collector_id: int
    owner_rank: int
    producer_device: torch.device
    consumer_device: torch.device
    transport: str


@dataclass(frozen=True)
class WorkerTopology:
    """Equal collector blocks, colocated with their owning learner's NUMA node."""

    learners: list[LearnerPlacement]
    collectors: list[CollectorPlacement]

    @property
    def num_learners(self) -> int:
        return len(self.learners)

    @property
    def num_collectors(self) -> int:
        return len(self.collectors)

    @property
    def per_learner(self) -> int:
        return self.num_collectors // self.num_learners

    @property
    def env_shards(self) -> list[int]:
        return [collector.num_envs for collector in self.collectors]

    def collector_owner(self, collector_id: int) -> int:
        """Return the rank owning this collector."""
        return self.collectors[collector_id].owner_rank

    def items_for_rank(self, items: list[_T], rank: int) -> list[_T]:
        """Select a learner's contiguous block from collector-indexed items."""
        return items[rank * self.per_learner : (rank + 1) * self.per_learner]

    def ring_slice_for_rank(self, rings: list[_T], rank: int) -> list[_T]:
        """Compatibility spelling for :meth:`items_for_rank`; no ring policy."""
        return self.items_for_rank(rings, rank)


def _learner_node(device: torch.device, multi_node: bool) -> int | None:
    if not multi_node or device.type != "cuda":
        return None
    index = device.index if device.index is not None else 0
    return system_numa.gpu_numa_node(index, runtime="hip" if torch.version.hip else "cuda")


def resolve_collector_inference_devices(
    collector_inference_device: str,
    learner_devices: list[torch.device],
    num_collectors: int,
    num_learners: int,
    default_device: torch.device,
) -> list[torch.device]:
    """Resolve generic CUDA to the owning learner's indexed GPU.

    Explicit CPU/CUDA/MPS specs pass through. Without an explicit learner layout,
    generic CUDA falls back to the trainer default when CUDA is unavailable.
    Workers retain responsibility for validating runtime device availability.
    """
    device = torch.device(collector_inference_device)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError(f"collector_inference_device must be cpu, cuda or mps, got '{collector_inference_device}'")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            f"collector_inference_device='{collector_inference_device}' requested MPS, but MPS is unavailable"
        )
    if device.type == "cuda" and device.index is None and not torch.cuda.is_available() and not learner_devices:
        logger.warning("collector_inference_device='cuda' but CUDA is unavailable; using %s", default_device)
        return [default_device] * num_collectors
    if (
        device.type == "cuda"
        and device.index is not None
        and torch.cuda.is_available()
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(
            f"collector_inference_device='{collector_inference_device}' selects CUDA device "
            f"{device.index}, but only {torch.cuda.device_count()} device(s) are available"
        )
    per_learner = num_collectors // num_learners
    devices = []
    for i in range(num_collectors):
        owner = learner_devices[i // per_learner] if learner_devices else default_device
        if device.type == "cuda" and device.index is None:
            owner_index = owner.index if owner.type == "cuda" and owner.index is not None else 0
            devices.append(torch.device("cuda", owner_index))
        else:
            devices.append(device)
    return devices


def resolve_learner_devices(
    learner_device_specs: list[str] | None,
    num_learners: int,
    default_device: torch.device,
) -> list[torch.device]:
    """Resolve distinct indexed CUDA ranks, or [] for an implicit single learner.

    An implicit single learner uses ``default_device`` in the topology. Multiple
    implicit ranks expand CUDA indices consecutively from the default device.
    """
    if num_learners == 1 and learner_device_specs is None:
        return []
    if learner_device_specs is None:
        base = default_device
        if base.type != "cuda":
            raise ValueError(f"num_learners={num_learners} requires CUDA learner devices, got {base}")
        start = base.index if base.index is not None else 0
        specs = [f"cuda:{start + i}" for i in range(num_learners)]
    else:
        if len(learner_device_specs) != num_learners:
            raise ValueError(
                f"learner_devices must have exactly num_learners={num_learners} entries, "
                f"got {len(learner_device_specs)}"
            )
        specs = learner_device_specs
    devices = [torch.device(spec) for spec in specs]
    indexes = [d.index for d in devices]
    if any(d.type != "cuda" or d.index is None for d in devices):
        raise ValueError(f"learner_devices must be explicit indexed CUDA devices, got {specs}")
    if len(set(indexes)) != len(indexes):
        raise ValueError(f"learner_devices reference a device more than once: {specs}")
    available = torch.cuda.device_count()
    for d in devices:
        if d.index >= available:
            raise ValueError(f"learner device {d} does not exist (only {available} CUDA device(s) available)")
    return devices


def split_num_envs(num_envs: int, num_collectors: int) -> list[int]:
    """Shard environments equally across collectors, requiring exact division."""
    if num_collectors < 1:
        raise ValueError(f"num_collectors must be >= 1, got {num_collectors}")
    if num_envs % num_collectors != 0:
        raise ValueError(
            f"num_envs={num_envs} must divide evenly across num_collectors={num_collectors} "
            f"({num_envs} % {num_collectors} != 0)"
        )
    return [num_envs // num_collectors] * num_collectors


def select_collector_cpus(base: list[int], collector_id: int, cpus_per_collector: int | None) -> list[int]:
    """Select a contiguous CPU chunk using the ordinal local to ``base``.

    Without a chunk size all collectors share the base. Exhausting the CPU set
    is a configuration error rather than a silent empty affinity.
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


def resolve_worker_topology(
    num_envs: int,
    num_collectors: int,
    num_learners: int,
    learner_device_specs: list[str] | None,
    collector_inference_device: str,
    default_device: torch.device,
    cpus_per_collector: int | None = None,
) -> WorkerTopology:
    """Compute devices, env shards, ownership and CPU/NUMA placement once.

    On multi-node hosts learners follow GPU PCI locality and collectors follow
    their owners. Otherwise CPU sets use inherited affinity. Missing topology
    degrades to no NUMA binding; no affinity or memory policy is applied here.
    """
    if num_learners < 1 or num_collectors < 1:
        raise ValueError(f"invalid worker counts: {num_collectors=} {num_learners=}")
    if num_collectors % num_learners != 0:
        raise ValueError(f"num_collectors={num_collectors} must divide evenly across num_learners={num_learners}")
    learner_devices = resolve_learner_devices(learner_device_specs, num_learners, default_device)
    collector_devices = resolve_collector_inference_devices(
        collector_inference_device, learner_devices, num_collectors, num_learners, default_device
    )
    env_shards = split_num_envs(num_envs, num_collectors)
    multi_node = len(system_numa.available_numa_nodes()) >= 2
    per_learner = num_collectors // num_learners

    def base_cpus(node: int | None) -> list[int]:
        if node is None:
            return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        try:
            return system_numa.numa_node_cpus(node)
        except ValueError:
            return []

    learners = []
    for rank in range(num_learners):
        device = learner_devices[rank] if learner_devices else default_device
        node = _learner_node(device, multi_node)
        learners.append(LearnerPlacement(rank, device, node, base_cpus(node)))
    collectors = []
    for i, (num_envs_i, device) in enumerate(zip(env_shards, collector_devices)):
        owner = i // per_learner
        node = learners[owner].numa_node
        base = base_cpus(node)
        cpus = select_collector_cpus(base, i % per_learner, cpus_per_collector) if base else []
        collectors.append(CollectorPlacement(i, owner, num_envs_i, device, node, cpus))
    return WorkerTopology(learners, collectors)


def same_cuda_device(learner_device: torch.device, collector_device: torch.device) -> bool:
    """Compare GPU ordinals without a CUDA context; unindexed CUDA means 0."""
    if learner_device.type != "cuda" or collector_device.type != "cuda":
        return False
    learner_index = learner_device.index if learner_device.index is not None else 0
    collector_index = collector_device.index if collector_device.index is not None else 0
    return learner_index == collector_index


def cuda_ipc_capable(producer_device: torch.device, consumer_device: torch.device) -> bool:
    """Return whether the common CUDA transport may be selected."""
    return same_cuda_device(producer_device, consumer_device)


def wire_endpoints(topology: WorkerTopology, capability: str = "auto") -> list[EndpointWiring]:
    """Describe one collector producer endpoint per common learner consumer."""
    if capability not in {"auto", "host", "cuda_ipc"}:
        raise ValueError(f"unknown endpoint capability {capability!r}")
    endpoints = []
    for collector in topology.collectors:
        consumer = topology.learners[collector.owner_rank]
        ipc = cuda_ipc_capable(collector.device, consumer.device)
        if capability == "cuda_ipc" and not ipc:
            raise ValueError("cuda_ipc requires colocated CUDA producer and consumer")
        endpoints.append(
            EndpointWiring(
                collector.collector_id,
                collector.owner_rank,
                collector.device,
                consumer.device,
                "cuda_ipc" if capability == "cuda_ipc" or (capability == "auto" and ipc) else "host",
            )
        )
    return endpoints
