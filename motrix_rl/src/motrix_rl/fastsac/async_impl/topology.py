# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Worker topology decisions for the async FastSAC trainer.

Pure decision functions over ``(opts, devices, counts)`` — how envs shard
across collectors, which NUMA node each worker is placed on, and which
transport each ring uses. The policy is co-location, informed by two
measurements on dual-socket hosts:

1. a collector must never be placed on a different NUMA node than the learner
   that drains its ring — the transition ring, pinned staging and weight
   snapshots would all go cross-node;
2. a learner binds to its GPU's PCIe-local node, and each collector follows
   its OWNING learner (collector ``i`` belongs to learner
   ``i // (num_collectors // num_learners)``), so the pair shares one node's
   memory domain. Restricting a single collector to one node does NOT hurt:
   first-touch page locality inside the physics working set outweighs the
   halved aggregate bandwidth (measured +58% on a 2-socket/2-GPU host).

Everything degrades to "no binding" (all ``None``) on single-node hosts,
non-NUMA kernels/containers and CPU learners — the OS default placement is
already optimal there.

The binding itself (affinity + libnuma memory policy, spawn-time
pre-placement) lives in :mod:`motrix_rl.fastsac.async_impl.numa`, the ring
runtime objects in :mod:`motrix_rl.fastsac.async_impl.transport`; this module
only computes the decisions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

from motrix_rl.fastsac.async_impl import numa
from motrix_rl.fastsac.config import FastSacAsyncOptionsCfg


@dataclass(frozen=True)
class LearnerInfo:
    """Everything one learner rank needs to know about itself."""

    rank: int
    device: torch.device
    numa_node: int | None
    cpus: list[int]


@dataclass(frozen=True)
class CollectorInfo:
    """Everything one collector process needs to know about itself."""

    collector_id: int
    owner_rank: int
    num_envs: int
    device: torch.device
    numa_node: int | None
    cpus: list[int]
    ring_ipc: bool
    weight_ipc: bool


@dataclass(frozen=True)
class TrainerTopology:
    """The complete compute layout of one async trainer run.

    One structure describing how the distributed training network is wired:
    per-worker self descriptions (devices, NUMA placement, CPU bindings,
    transports, env shards) plus the collector→learner ownership relation.
    ``resolve_trainer_topology`` is the single computation API producing it.

    Invariants (by construction):

    * ``collectors[i].numa_node == learners[collectors[i].owner_rank].numa_node``
      — a collector/learner pair never straddles a NUMA boundary;
    * ``collectors[i].num_envs`` envs travel with collector ``i``;
    * ``sum(c.num_envs for c in collectors) == num_envs``.
    """

    learners: list[LearnerInfo]
    collectors: list[CollectorInfo]

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
        """The learner rank that owns (drains the rings of) this collector."""
        return self.collectors[collector_id].owner_rank

    def ring_slice_for_rank(self, rings: list, rank: int) -> list:
        """The rings this learner rank drains: contiguous collector block ``rank``.

        Each ring still has exactly one consumer, so the SPSC contract is
        unchanged.
        """
        return rings[rank * self.per_learner : (rank + 1) * self.per_learner]


def _learner_node(device: torch.device, multi_node: bool) -> int | None:
    """GPU-local node of one learner rank; ``None`` for CPU / unknown hosts."""
    if not multi_node:
        return None
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else 0
    return numa.gpu_numa_node(index)


def resolve_learner_devices(
    learner_device_specs: list[str] | None,
    num_learners: int,
    default_device: torch.device,
) -> list[torch.device]:
    """One indexed CUDA device per learner rank.

    ``None`` replicates the trainer device: for ``num_learners > 1`` an
    unindexed ``cuda`` expands to consecutive indexes starting at the trainer
    device's; anything else (an indexed single device or CPU) cannot back
    multiple ranks and is rejected. A single learner with ``None`` resolves
    in-process (empty list — the worker keeps its own resolution).
    """
    if num_learners == 1 and learner_device_specs is None:
        return []  # single-learner path resolves its device in-process
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
    """Shard ``num_envs`` evenly across collectors (requires exact divisibility).

    Topological complement of the collector→learner ownership encoded in
    :class:`TrainerTopology`: shard ``i`` (and so its envs) always travels
    with collector ``i``, which never straddles a NUMA boundary from its
    owning learner.
    """
    if num_collectors < 1:
        raise ValueError(f"num_collectors must be >= 1, got {num_collectors}")
    if num_envs % num_collectors != 0:
        raise ValueError(
            f"num_envs={num_envs} must divide evenly across num_collectors={num_collectors} "
            f"({num_envs} % {num_collectors} != 0)"
        )
    per_collector = num_envs // num_collectors
    return [per_collector] * num_collectors


def use_ipc_weight_channel(
    opts: FastSacAsyncOptionsCfg,
    learner_device: torch.device,
    collector_device: torch.device,
    actor_param_numel: int,
) -> bool:
    """Whether the weight channel for one collector should use CUDA-IPC.

    Like :func:`use_ipc_transition_ring`, plus a size threshold: IPC device
    slots only pay off when the actor parameters reach the configured
    ``weight_ipc_min_bytes``; host shared memory otherwise.
    """
    if not use_ipc_transition_ring(opts, learner_device, collector_device):
        return False
    return actor_param_numel * 4 >= opts.weight_ipc_min_bytes


def resolve_trainer_topology(
    num_envs: int,
    num_collectors: int,
    num_learners: int,
    learner_devices: list[torch.device],
    collector_devices: list[torch.device],
    default_device: torch.device,
    async_options: FastSacAsyncOptionsCfg,
    actor_param_numel: int,
    cpus_per_collector: int | None = None,
) -> TrainerTopology:
    """Single computation API: derive the full trainer topology in one pass.

    Combines env sharding (:func:`split_num_envs`), NUMA placement, CPU
    bindings, and the transport decisions for transition rings
    (:func:`ring_transport_is_ipc`) and weight channels
    (:func:`use_ipc_weight_channel`) into one :class:`TrainerTopology`.
    ``actor_param_numel`` is the parent-computed actor parameter count that
    sizes the weight-transport threshold; ``cpus_per_collector`` optionally
    chunks each binding base into per-collector slices. ``learner_devices``
    carries one indexed device per rank (empty for a single learner, whose
    device comes from ``default_device`` — an index-less ``cuda`` means
    device 0). ``collector_devices[i]`` is the inference device of collector
    ``i``.
    """
    if num_learners < 1 or num_collectors < 1:
        raise ValueError(f"invalid worker counts: {num_collectors=} {num_learners=}")
    if num_collectors % num_learners != 0:
        raise ValueError(f"num_collectors={num_collectors} must divide evenly across num_learners={num_learners}")
    if len(collector_devices) != num_collectors:
        raise ValueError(f"expected {num_collectors} collector devices, got {len(collector_devices)}")

    env_shards = split_num_envs(num_envs, num_collectors)
    multi_node = len(numa.available_numa_nodes()) >= 2
    per_learner = num_collectors // num_learners

    # Per-worker CPU bindings, decided here so workers only apply them. Base
    # is the NUMA node's CPUs when the worker is node-bound, else the
    # process's own affinity mask (children inherit it through spawn). An
    # unreadable node degrades to no binding, matching the numa module's
    # best-effort contract.
    def _base_cpus(node: int | None) -> list[int]:
        if node is None:
            return sorted(os.sched_getaffinity(0))
        try:
            return numa.numa_node_cpus(node)
        except ValueError:
            return []

    def _chunk(base: list[int], local_id: int) -> list[int]:
        if not base:
            return []
        return numa.select_collector_cpus(base, local_id, cpus_per_collector)

    # resolve_learner_devices' contract: learner_devices is empty iff
    # num_learners == 1, and the single learner runs on default_device.
    def _rank_device(rank: int) -> torch.device:
        return learner_devices[rank] if learner_devices else default_device

    ring_ipc = ring_transport_is_ipc(
        async_options, learner_devices, collector_devices, num_collectors, num_learners, default_device
    )

    learners: list[LearnerInfo] = []
    for rank in range(num_learners):
        device = _rank_device(rank)
        node = _learner_node(device, multi_node)
        learners.append(LearnerInfo(rank=rank, device=device, numa_node=node, cpus=_base_cpus(node)))

    collectors: list[CollectorInfo] = []
    for i, (num_envs_i, collector_dev) in enumerate(zip(env_shards, collector_devices)):
        owner = i // per_learner
        owner_node = learners[owner].numa_node
        collectors.append(
            CollectorInfo(
                collector_id=i,
                owner_rank=owner,
                num_envs=num_envs_i,
                device=collector_dev,
                numa_node=owner_node,
                # chunk by the NODE-LOCAL ordinal: the base list is the
                # owner node's own CPUs, so a global index would offset
                # rank>=1 collectors past their node's list (asymmetric
                # bindings, or "leaves no CPUs" on small nodes).
                cpus=_chunk(_base_cpus(owner_node), i % per_learner),
                ring_ipc=ring_ipc[i],
                weight_ipc=use_ipc_weight_channel(async_options, _rank_device(owner), collector_dev, actor_param_numel),
            )
        )
    return TrainerTopology(learners=learners, collectors=collectors)


def same_cuda_device(learner_device: torch.device, collector_device: torch.device) -> bool:
    """Whether the learner and the collector's inference device share one GPU.

    An index-less ``cuda`` means the default current device (index 0 — nothing
    in the trainer ever calls ``torch.cuda.set_device``), so it is resolved
    with 0 rather than treated as a wildcard matching any explicit index:
    ``learner=cuda`` (effectively cuda:0) with ``collector_inference_device:
    cuda:1`` is a cross-GPU setup and must NOT enable the device transports.
    Pure device arithmetic — no CUDA context is created, so the pre-spawn
    parent can call it as safely as the workers.
    """
    if learner_device.type != "cuda" or collector_device.type != "cuda":
        return False
    learner_index = learner_device.index if learner_device.index is not None else 0
    collector_index = collector_device.index if collector_device.index is not None else 0
    return learner_index == collector_index


def use_ipc_transition_ring(
    opts: FastSacAsyncOptionsCfg, learner_device: torch.device, collector_device: torch.device
) -> bool:
    """Whether the transition ring should use CUDA-IPC device slots.

    Requires learner and collector inference on the same GPU (see
    :func:`same_cuda_device`); otherwise the host shared-memory ring is used.
    Purely device-object arithmetic — no CUDA context is created here, so the
    parent can call it safely.
    """
    mode = opts.transition_ipc
    # YAML 1.1 parses unquoted ``on``/``off`` scalars as booleans; accept that
    # form so ``transition_ipc: on`` in a config behaves like the documented
    # string.
    if isinstance(mode, bool):
        mode = "on" if mode else "off"
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"async_options.transition_ipc must be auto, on or off, got {mode!r}")
    same_gpu = same_cuda_device(learner_device, collector_device)
    if mode == "off":
        return False
    if mode == "on" and not same_gpu:
        reason = (
            "collector inference device is not CUDA"
            if collector_device.type != "cuda"
            else f"learner device {learner_device} and collector device {collector_device} are different GPUs"
            if learner_device.type == "cuda"
            else "learner device is not CUDA"
        )
        logging.getLogger(__name__).warning(
            "async_options.transition_ipc=on requires learner and collector inference on the same GPU, "
            "but %s; falling back to the host shared-memory transition ring",
            reason,
        )
    return same_gpu


def ring_transport_is_ipc(
    opts: FastSacAsyncOptionsCfg,
    learner_devices: list[torch.device],
    collector_devices: list[torch.device],
    num_collectors: int,
    num_learners: int,
    default_owner: torch.device,
) -> list[bool]:
    """Per-collector CUDA-IPC ring decision, all-or-nothing per learner rank.

    A rank's rings share one transport so its drain path stays homogeneous
    (the generation merge would choke on mixed host/device views). Collector
    i belongs to rank i // (num_collectors // num_learners) and its ring goes
    IPC only when EVERY collector of that rank infers on the rank's GPU.
    """
    per_learner = num_collectors // num_learners
    decisions: list[bool] = []
    for rank in range(num_learners):
        owner = learner_devices[rank] if num_learners > 1 else default_owner
        rank_devs = collector_devices[rank * per_learner : (rank + 1) * per_learner]
        rank_ipc = all(use_ipc_transition_ring(opts, owner, dev) for dev in rank_devs)
        decisions.extend([rank_ipc] * per_learner)
    return decisions
