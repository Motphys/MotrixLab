# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generic worker topology decisions for asynchronous training.

This module contains layout decisions that are independent of a particular RL
algorithm or transport implementation: device assignment, environment
sharding, NUMA co-location, and worker CPU affinity. Algorithm providers can
supply transport decisions when constructing the final topology.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

from motrix_rl import numa as system_numa

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelSpec:
    """Describe one logical channel between asynchronous workers.

    Args:
        channel_id: Collector ID associated with the channel.
        producer_id: Integer worker ID that publishes data.
        consumer_id: Integer worker ID that consumes data. Its role is
            determined by the channel collection containing this spec.
        use_cuda_ipc: Whether the channel uses CUDA IPC instead of host shared
            memory.
    """

    channel_id: int
    producer_id: int
    consumer_id: int
    use_cuda_ipc: bool = False


@dataclass(frozen=True)
class LearnerInfo:
    """Describe the device and CPU placement of one learner rank.

    Args:
        rank: Zero-based learner rank.
        device: Device used by the learner.
        numa_node: NUMA node assigned to the learner, or ``None`` when no
            NUMA placement is available or required.
        cpus: CPU IDs assigned to the learner.
    """

    rank: int
    device: torch.device
    numa_node: int | None
    cpus: tuple[int, ...]


@dataclass(frozen=True)
class CollectorInfo:
    """Describe one collector's workload and execution placement.

    Args:
        collector_id: Zero-based collector ID.
        num_envs: Number of environments assigned to the collector.
        infer_device: Device used for collector policy inference.
        numa_node: NUMA node assigned to the collector, or ``None``.
        cpus: CPU IDs assigned to the collector.
    """

    collector_id: int
    num_envs: int
    infer_device: torch.device
    numa_node: int | None
    cpus: tuple[int, ...]


@dataclass(frozen=True)
class AsyncTrainerTopology:
    """Describe workers and channel declarations for asynchronous training.

    Args:
        learners: Learner resources participating in the run.
        collectors: Collector workloads and resources participating in the run.
        transition_channels: SPSC channels carrying collector transitions.
        weight_channels: SPSC channels carrying learner weight snapshots.
    """

    learners: tuple[LearnerInfo, ...]
    collectors: tuple[CollectorInfo, ...]
    transition_channels: tuple[ChannelSpec, ...] = ()
    weight_channels: tuple[ChannelSpec, ...] = ()

    @staticmethod
    def channel_indices_for_consumer(channels: tuple[ChannelSpec, ...], consumer_id: int) -> tuple[int, ...]:
        """Return channel indexes addressed to one consumer."""
        return tuple(index for index, channel in enumerate(channels) if channel.consumer_id == consumer_id)

    @staticmethod
    def channel_indices_for_producer(channels: tuple[ChannelSpec, ...], producer_id: int) -> tuple[int, ...]:
        """Return channel indexes published by one producer."""
        return tuple(index for index, channel in enumerate(channels) if channel.producer_id == producer_id)

    @property
    def num_learners(self) -> int:
        """Return the number of learner ranks."""
        return len(self.learners)

    @property
    def num_collectors(self) -> int:
        """Return the number of collectors."""
        return len(self.collectors)

    @property
    def env_shards(self) -> tuple[int, ...]:
        """Return environment counts in collector order."""
        return tuple(collector.num_envs for collector in self.collectors)


def resolve_learner_numa_node(device: torch.device, multi_node: bool) -> int | None:
    """Resolve the NUMA node local to a learner's accelerator.

    Args:
        device: Learner device.
        multi_node: Whether the host exposes multiple NUMA nodes.

    Returns:
        The GPU-local NUMA node, or ``None`` for CPU, single-node, or unknown
        placements.
    """
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
    """Resolve one explicit inference device for every collector.

    Generic ``cuda`` is co-located with the owning learner. Explicit CPU,
    MPS, and indexed CUDA specifications are passed through. No CUDA context
    is created by this function.

    Args:
        collector_inference_device: Configured collector device string.
        learner_devices: Explicit learner devices, or an empty list for one
            learner resolved to ``default_device``.
        num_collectors: Number of collectors.
        num_learners: Number of learners.
        default_device: Device used by a single learner.

    Returns:
        Explicit device objects in collector order.

    Raises:
        ValueError: If the device type or indexed CUDA device is invalid.
        RuntimeError: If MPS is requested but unavailable.
    """
    device = torch.device(collector_inference_device)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError(f"collector_inference_device must be cpu, cuda or mps, got '{collector_inference_device}'")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            f"collector_inference_device='{collector_inference_device}' requested MPS, but MPS is unavailable"
        )
    if device.type == "cuda" and device.index is None and not torch.cuda.is_available() and not learner_devices:
        logger.warning(
            "collector_inference_device='cuda' but CUDA is unavailable; using %s",
            default_device,
        )
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

    def owner(rank: int) -> torch.device:
        return learner_devices[rank] if learner_devices else default_device

    devices: list[torch.device] = []
    for collector_id in range(num_collectors):
        owner_device = owner(collector_id // per_learner)
        if device.type == "cuda" and device.index is None:
            owner_index = owner_device.index if owner_device.type == "cuda" and owner_device.index is not None else 0
            devices.append(torch.device("cuda", owner_index))
        else:
            devices.append(device)
    return devices


def resolve_learner_devices(
    learner_device_specs: list[str] | None,
    num_learners: int,
    default_device: torch.device,
) -> list[torch.device]:
    """Resolve one explicit indexed CUDA device for each learner rank.

    Args:
        learner_device_specs: Explicit device strings, or ``None`` to derive
            devices from ``default_device``.
        num_learners: Number of learner ranks.
        default_device: Trainer device used by the single-learner path.

    Returns:
        Indexed CUDA devices, or an empty list when one learner resolves its
        device in-process.

    Raises:
        ValueError: If the requested learner layout is invalid.
    """
    if num_learners == 1 and learner_device_specs is None:
        return []
    if learner_device_specs is None:
        if default_device.type != "cuda":
            raise ValueError(f"num_learners={num_learners} requires CUDA learner devices, got {default_device}")
        start = default_device.index if default_device.index is not None else 0
        specs = [f"cuda:{start + i}" for i in range(num_learners)]
    else:
        if len(learner_device_specs) != num_learners:
            raise ValueError(
                f"learner_devices must have exactly num_learners={num_learners} entries, "
                f"got {len(learner_device_specs)}"
            )
        specs = learner_device_specs
    devices = [torch.device(spec) for spec in specs]
    indexes = [device.index for device in devices]
    if any(device.type != "cuda" or device.index is None for device in devices):
        raise ValueError(f"learner_devices must be explicit indexed CUDA devices, got {specs}")
    if len(set(indexes)) != len(indexes):
        raise ValueError(f"learner_devices reference a device more than once: {specs}")
    available = torch.cuda.device_count()
    for device in devices:
        if device.index >= available:
            raise ValueError(f"learner device {device} does not exist (only {available} CUDA device(s) available)")
    return devices


def split_num_envs(num_envs: int, num_collectors: int) -> list[int]:
    """Split environments evenly across collectors.

    Args:
        num_envs: Total number of environments.
        num_collectors: Number of collectors.

    Returns:
        Equal-sized environment shard counts in collector order.

    Raises:
        ValueError: If the collector count is invalid or the split is not
            exact.
    """
    if num_collectors < 1:
        raise ValueError(f"num_collectors must be >= 1, got {num_collectors}")
    if num_envs % num_collectors != 0:
        raise ValueError(
            f"num_envs={num_envs} must divide evenly across num_collectors={num_collectors} "
            f"({num_envs} % {num_collectors} != 0)"
        )
    return [num_envs // num_collectors] * num_collectors


def same_cuda_device(learner_device: torch.device, collector_device: torch.device) -> bool:
    """Return whether two devices refer to the same CUDA GPU.

    Args:
        learner_device: Learner device.
        collector_device: Collector inference device.

    Returns:
        ``True`` only when both devices are CUDA devices with the same index.
    """
    if learner_device.type != "cuda" or collector_device.type != "cuda":
        return False
    learner_index = learner_device.index if learner_device.index is not None else 0
    collector_index = collector_device.index if collector_device.index is not None else 0
    return learner_index == collector_index


def _select_collector_learner(
    collector_device: torch.device,
    collector_node: int | None,
    learners: list[LearnerInfo],
    assigned_counts: list[int],
    max_collectors_per_learner: int,
) -> int:
    """Select a learner with locality preference and a hard load quota."""
    available = {rank for rank, count in enumerate(assigned_counts) if count < max_collectors_per_learner}
    same_gpu = [
        learner.rank
        for learner in learners
        if learner.rank in available and same_cuda_device(learner.device, collector_device)
    ]
    same_numa = [
        learner.rank for learner in learners if learner.rank in available and learner.numa_node == collector_node
    ]
    candidates = same_gpu or same_numa or sorted(available)
    return min(candidates, key=lambda rank: (assigned_counts[rank], rank))


def _allocate_worker_cpus(
    available: list[int],
    cpus_per_worker: int | None,
) -> tuple[int, ...]:
    """Allocate logical CPU IDs from one shared placement domain.

    ``available`` is consumed in place when a fixed CPU count is requested.
    With ``None``, the complete domain is shared by the worker and remains
    available to other workers.
    """
    if not available:
        raise ValueError("empty CPU set for worker binding")
    if cpus_per_worker is None:
        return tuple(available)
    if cpus_per_worker <= 0:
        raise ValueError(f"cpus_per_worker must be positive, got {cpus_per_worker}")
    if len(available) < cpus_per_worker:
        raise ValueError(
            f"cpus_per_collector={cpus_per_worker} requires {cpus_per_worker} available CPUs, "
            f"but only {len(available)} remain"
        )
    allocated = tuple(available[:cpus_per_worker])
    del available[:cpus_per_worker]
    return allocated


def resolve_async_trainer_topology(
    num_envs: int,
    num_collectors: int,
    num_learners: int,
    learner_device_specs: list[str] | None,
    collector_inference_device: str,
    default_device: torch.device,
    *,
    cpus_per_collector: int | None = None,
) -> AsyncTrainerTopology:
    """Resolve the generic worker layout for an asynchronous trainer.

    Args:
        num_envs: Total number of environments.
        num_collectors: Number of collector workers.
        num_learners: Number of learner workers.
        learner_device_specs: Explicit learner device strings, or ``None``.
        collector_inference_device: Collector inference device specification.
        default_device: Device used by a single learner.
        cpus_per_collector: Optional number of CPUs per collector slice.

    Returns:
        A complete generic trainer topology.

    Raises:
        ValueError: If worker counts, device layout, or shard layout is
            invalid.
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

    inherited_cpus = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None

    def base_cpus(node: int | None) -> list[int]:
        if node is None:
            return sorted(inherited_cpus) if inherited_cpus is not None else []
        try:
            cpus = set(system_numa.numa_node_cpus(node))
        except ValueError:
            return []
        if inherited_cpus is not None:
            cpus &= inherited_cpus
        return sorted(cpus)

    def learner_node(device: torch.device) -> int | None:
        if not multi_node or device.type != "cuda":
            return None
        index = device.index if device.index is not None else 0
        return system_numa.gpu_numa_node(index, runtime="hip" if torch.version.hip else "cuda")

    reserved_cpus: set[int] = set()

    def allocate_collector_cpus(node: int | None) -> tuple[int, ...]:
        domain = base_cpus(node)
        if not domain:
            return ()
        if cpus_per_collector is None:
            return tuple(domain)
        available = [cpu for cpu in domain if cpu not in reserved_cpus]
        allocated = _allocate_worker_cpus(available, cpus_per_collector)
        reserved_cpus.update(allocated)
        return allocated

    def rank_device(rank: int) -> torch.device:
        return learner_devices[rank] if learner_devices else default_device

    learners: list[LearnerInfo] = []
    for rank in range(num_learners):
        device = rank_device(rank)
        node = learner_node(device)
        learners.append(LearnerInfo(rank=rank, device=device, numa_node=node, cpus=base_cpus(node)))

    collectors: list[CollectorInfo] = []
    collector_learner_ranks: list[int] = []
    assigned_counts = [0] * num_learners
    for collector_id, (num_envs_i, collector_device) in enumerate(zip(env_shards, collector_devices)):
        collector_node = learner_node(collector_device)
        learner_rank = _select_collector_learner(
            collector_device,
            collector_node,
            learners,
            assigned_counts,
            num_collectors // num_learners,
        )
        assigned_counts[learner_rank] += 1
        collector_learner_ranks.append(learner_rank)
        owner_node = learners[learner_rank].numa_node
        collectors.append(
            CollectorInfo(
                collector_id=collector_id,
                num_envs=num_envs_i,
                infer_device=collector_device,
                numa_node=owner_node,
                cpus=allocate_collector_cpus(owner_node),
            )
        )

    transition_channels = tuple(
        ChannelSpec(
            channel_id=collector.collector_id,
            producer_id=collector.collector_id,
            consumer_id=learner_rank,
        )
        for collector, learner_rank in zip(collectors, collector_learner_ranks)
    )
    weight_channels = tuple(
        ChannelSpec(
            channel_id=collector.collector_id,
            producer_id=learner_rank,
            consumer_id=collector.collector_id,
        )
        for collector, learner_rank in zip(collectors, collector_learner_ranks)
    )
    return AsyncTrainerTopology(
        learners=tuple(learners),
        collectors=tuple(collectors),
        transition_channels=transition_channels,
        weight_channels=weight_channels,
    )
