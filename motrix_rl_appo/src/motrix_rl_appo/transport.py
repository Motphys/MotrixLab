# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""One fixed rollout chunk per host/IPC ring slot; private ownership before release."""

from dataclasses import dataclass

import torch
from torch import nn

from motrix_rl.topology import same_cuda_device
from motrix_rl.transport.gpu_ipc_ring import GpuIpcTransitionRing
from motrix_rl.transport.inference_bundle import InferenceBundleSpec, ModuleSection, NormalizerSection
from motrix_rl.transport.ring import HostTransitionRing, RingCursors
from motrix_rl.transport.schema import TransitionField, TransitionSchema
from motrix_rl.transport.weight_channel import WeightChannelShared


@dataclass(frozen=True)
class CollectorMetadata:
    obs_dim: int
    critic_dim: int
    action_dim: int


@dataclass(frozen=True)
class GpuRingConnection:
    """IPC handles only; CUDA events belong to each process-local endpoint."""

    cursors: RingCursors
    slots: torch.Tensor
    schema: TransitionSchema
    capacity: int
    num_envs: int

    def endpoint(self) -> GpuIpcTransitionRing:
        return GpuIpcTransitionRing(self.cursors, self.slots, self.schema, self.capacity, self.num_envs)


@dataclass(frozen=True)
class CollectorConnection:
    ring: HostTransitionRing | GpuRingConnection
    weights: WeightChannelShared
    weight_params: list[torch.Tensor]
    chunk_device: str = "cpu"


def chunk_schema(steps: int, dims: CollectorMetadata) -> TransitionSchema:
    return TransitionSchema(
        (
            TransitionField("obs", (steps, dims.obs_dim), torch.float32),
            TransitionField("critic_obs", (steps, dims.critic_dim), torch.float32),
            TransitionField("actions", (steps, dims.action_dim), torch.float32),
            TransitionField("rewards", (steps,), torch.float32),
            TransitionField("dones", (steps,), torch.bool),
            TransitionField("behaviour_logp", (steps,), torch.float32),
            TransitionField("last_critic_obs", (dims.critic_dim,), torch.float32),
        )
    )


def inference_spec(actor: nn.Module, critic: nn.Module, dims: CollectorMetadata) -> InferenceBundleSpec:
    return InferenceBundleSpec(
        modules=tuple(
            ModuleSection(name, sum(p.numel() for p in module.parameters()))
            for name, module in (("actor", actor), ("critic", critic))
        ),
        normalizers=(NormalizerSection("actor_norm", dims.obs_dim), NormalizerSection("critic_norm", dims.critic_dim)),
    )


def resolve_transport(mode: str, learner: torch.device, collector: torch.device) -> str:
    """Resolve transition or weight transport independently with the same GPU rule."""
    if mode not in ("auto", "host", "cuda_ipc"):
        raise ValueError("transport must be auto, host or cuda_ipc")
    if mode == "host":
        return "host"
    available = same_cuda_device(learner, collector) and torch.cuda.is_available()
    if available:
        learner_index = learner.index if learner.index is not None else 0
        available = learner_index < torch.cuda.device_count()
    if mode == "cuda_ipc" and not available:
        raise ValueError(
            "cuda_ipc transport requires learner and collector inference on the same available CUDA device"
        )
    return "cuda_ipc" if available else "host"


def take_chunk(ring: HostTransitionRing | GpuIpcTransitionRing, device: torch.device) -> dict[str, torch.Tensor]:
    """Consume one slot into owned, typed, time-major contiguous tensors."""
    count, views = ring.read_span()
    if count == 0:
        raise RuntimeError("Cannot consume an empty rollout ring")
    batch = {}
    for field in ring.schema.fields:
        # IPC fields are flattened float32 views, including bool dones.
        # copy=True also prevents aliasing when consuming a host ring on CPU.
        value = views[field.name][0].to(device=device, dtype=field.dtype, copy=True)
        value = value.reshape(ring.schema.slot_shape(field, ring.num_envs))
        batch[field.name] = value if field.name == "last_critic_obs" else value.transpose(0, 1).contiguous()
    # All D2D reads must precede the IPC commit event on the current stream.
    ring.commit_reads(1)
    return batch
