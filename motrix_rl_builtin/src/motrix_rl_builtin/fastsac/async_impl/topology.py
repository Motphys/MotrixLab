# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC transport policy layered on common worker placement.

Device resolution, env sharding, ownership and host placement are implemented
only in :mod:`motrix_rl.execution.topology`. Historical helper imports remain
available here for callers; ring/weight IPC decisions remain provider-owned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from motrix_rl.execution.topology import (
    CollectorPlacement,
    WorkerTopology,
    resolve_worker_topology,
    same_cuda_device,
)
from motrix_rl.execution.topology import (
    LearnerPlacement as LearnerInfo,
)
from motrix_rl.execution.topology import (
    resolve_collector_inference_devices as resolve_collector_inference_devices,
)
from motrix_rl.execution.topology import (
    resolve_learner_devices as resolve_learner_devices,
)
from motrix_rl.execution.topology import (
    split_num_envs as split_num_envs,
)
from motrix_rl_builtin.fastsac.config import FastSacAsyncOptionsCfg


@dataclass(frozen=True)
class CollectorInfo(CollectorPlacement):
    """Common collector placement plus FastSAC transport selections."""

    ring_ipc: bool
    weight_ipc: bool


@dataclass(frozen=True)
class TrainerTopology(WorkerTopology):
    """FastSAC worker layout with ring and weight transport decisions."""

    learners: list[LearnerInfo]
    collectors: list[CollectorInfo]


def use_ipc_weight_channel(
    opts: FastSacAsyncOptionsCfg,
    learner_device: torch.device,
    collector_device: torch.device,
    actor_param_numel: int,
) -> bool:
    """Use device weight slots only for colocated GPUs above the size threshold."""
    if not use_ipc_transition_ring(opts, learner_device, collector_device):
        return False
    return actor_param_numel * 4 >= opts.weight_ipc_min_bytes


def resolve_trainer_topology(
    num_envs: int,
    num_collectors: int,
    num_learners: int,
    learner_device_specs: list[str] | None,
    collector_inference_device: str,
    default_device: torch.device,
    async_options: FastSacAsyncOptionsCfg,
    actor_param_numel: int,
    cpus_per_collector: int | None = None,
) -> TrainerTopology:
    """Add provider-owned transport policy to the common worker topology."""
    topology = resolve_worker_topology(
        num_envs,
        num_collectors,
        num_learners,
        learner_device_specs,
        collector_inference_device,
        default_device,
        cpus_per_collector,
    )
    ring_ipc = ring_transport_is_ipc(
        async_options,
        [learner.device for learner in topology.learners],
        [collector.device for collector in topology.collectors],
        num_collectors,
        num_learners,
        topology.learners[0].device,
    )
    collectors = [
        CollectorInfo(
            collector_id=c.collector_id,
            owner_rank=c.owner_rank,
            num_envs=c.num_envs,
            device=c.device,
            numa_node=c.numa_node,
            cpus=c.cpus,
            ring_ipc=ring_ipc[c.collector_id],
            weight_ipc=use_ipc_weight_channel(
                async_options, topology.learners[c.owner_rank].device, c.device, actor_param_numel
            ),
        )
        for c in topology.collectors
    ]
    return TrainerTopology(learners=topology.learners, collectors=collectors)


def use_ipc_transition_ring(
    opts: FastSacAsyncOptionsCfg, learner_device: torch.device, collector_device: torch.device
) -> bool:
    """Choose CUDA IPC only when learner and collector share the same GPU."""
    mode = opts.transition_ipc
    # YAML 1.1 interprets unquoted on/off as booleans.
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
    """Choose homogeneous ring transport for each learner's collector block."""
    per_learner = num_collectors // num_learners
    decisions: list[bool] = []
    for rank in range(num_learners):
        owner = learner_devices[rank] if learner_devices else default_owner
        rank_devs = collector_devices[rank * per_learner : (rank + 1) * per_learner]
        rank_ipc = all(use_ipc_transition_ring(opts, owner, dev) for dev in rank_devs)
        decisions.extend([rank_ipc] * per_learner)
    return decisions
