# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC-specific transport policy layered on generic RL topology."""

from __future__ import annotations

import logging
from dataclasses import replace

import torch

from motrix_rl.topology import (
    AsyncTrainerTopology,
    ChannelSpec,
    CollectorInfo,
    LearnerInfo,
    resolve_collector_inference_devices,
    resolve_learner_devices,
    same_cuda_device,
    split_num_envs,
)
from motrix_rl.topology import (
    resolve_async_trainer_topology as resolve_generic_async_trainer_topology,
)
from motrix_rl_builtin.fastsac.config import FastSacAsyncOptionsCfg

logger = logging.getLogger(__name__)


# These names remain imported from this module for existing FastSAC callers.
__all__ = [
    "CollectorInfo",
    "LearnerInfo",
    "AsyncTrainerTopology",
    "ChannelSpec",
    "resolve_collector_inference_devices",
    "resolve_learner_devices",
    "resolve_fastsac_topology",
    "same_cuda_device",
    "split_num_envs",
    "use_ipc_transition_ring",
    "use_ipc_weight_channel",
]


def use_ipc_transition_ring(
    opts: FastSacAsyncOptionsCfg, learner_device: torch.device, collector_device: torch.device
) -> bool:
    """Return whether a collector's transition ring should use CUDA IPC.

    Args:
        opts: FastSAC asynchronous transport options.
        learner_device: Device used by the owning learner.
        collector_device: Collector inference device.

    Returns:
        ``True`` when both workers use the same CUDA GPU and the configured
        mode permits CUDA IPC.
    """
    mode = getattr(opts, "transition_ipc", "auto")
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
        logger.warning(
            "async_options.transition_ipc=on requires learner and collector inference on the same GPU, "
            "but %s; falling back to the host shared-memory transition ring",
            reason,
        )
    return same_gpu


def use_ipc_weight_channel(
    opts: FastSacAsyncOptionsCfg,
    learner_device: torch.device,
    collector_device: torch.device,
    actor_param_numel: int,
) -> bool:
    """Return whether a collector's weight channel should use CUDA IPC.

    Weight transport has an independent policy from transition transport.
    """
    mode = getattr(opts, "weight_ipc", "auto")
    if isinstance(mode, bool):
        mode = "on" if mode else "off"
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"async_options.weight_ipc must be auto, on or off, got {mode!r}")
    if mode == "off":
        return False
    same_gpu = same_cuda_device(learner_device, collector_device)
    if mode == "on" and not same_gpu:
        logger.warning(
            "async_options.weight_ipc=on requires learner and collector inference on the same GPU; "
            "falling back to host shared-memory weights"
        )
    if not same_gpu:
        return False
    return mode == "on" or actor_param_numel * 4 >= getattr(opts, "weight_ipc_min_bytes", 16 * 1024 * 1024)


def resolve_fastsac_topology(
    num_envs: int,
    num_collectors: int,
    num_learners: int,
    learner_device_specs: list[str] | None,
    collector_inference_device: str,
    default_device: torch.device,
    async_options: FastSacAsyncOptionsCfg,
    actor_param_numel: int,
    cpus_per_collector: int | None = None,
) -> AsyncTrainerTopology:
    """Resolve the complete FastSAC worker topology in one pass.

    Args:
        num_envs: Total number of environments.
        num_collectors: Number of collector workers.
        num_learners: Number of learner workers.
        learner_device_specs: Explicit learner device strings, or ``None``.
        collector_inference_device: Collector inference device specification.
        default_device: Device used by a single learner.
        async_options: FastSAC asynchronous transport options.
        actor_param_numel: Number of actor parameters.
        cpus_per_collector: Optional CPUs per collector slice.

    Returns:
        A generic topology populated with FastSAC transport decisions.
    """
    if num_learners < 1 or num_collectors < 1:
        raise ValueError(f"invalid worker counts: {num_collectors=} {num_learners=}")
    if num_collectors % num_learners != 0:
        raise ValueError(f"num_collectors={num_collectors} must divide evenly across num_learners={num_learners}")
    topology = resolve_generic_async_trainer_topology(
        num_envs,
        num_collectors,
        num_learners,
        learner_device_specs,
        collector_inference_device,
        default_device,
        cpus_per_collector=cpus_per_collector,
    )
    channels = tuple(
        replace(
            channel,
            use_cuda_ipc=use_ipc_transition_ring(
                async_options,
                topology.learners[channel.consumer_id].device,
                topology.collectors[index].infer_device,
            ),
        )
        for index, channel in enumerate(topology.transition_channels)
    )
    weight_channels = tuple(
        replace(
            channel,
            use_cuda_ipc=use_ipc_weight_channel(
                async_options,
                topology.learners[channel.producer_id].device,
                topology.collectors[index].infer_device,
                actor_param_numel,
            ),
        )
        for index, channel in enumerate(topology.weight_channels)
    )
    return replace(topology, transition_channels=channels, weight_channels=weight_channels)
