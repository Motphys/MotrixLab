# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Cross-process transport primitives for the async FastSAC trainer.

This package is the established internal API surface; the facade keeps the
``motrix_rl.fastsac.async_impl.transport`` import path stable across the
worker/collector/learner modules and tests. Import from the defining
submodules for new internal uses; import from here only when relying on the
stable package namespace.

Submodules:

* :mod:`.common`         — the shared-memory allocator, shared scalars
  (:class:`Control`) and flat-parameter helpers.
* :mod:`.ring`           — the SPSC transition ring (collector -> learner):
  host-shm fields plus the shared :class:`~.ring.RingCursors` protocol.
* :mod:`.ipc_ring`        — CUDA-IPC device-fields variant of the ring, used
  when collector inference and the learner share one GPU.
* :mod:`.weight_channel` — the seqlock weight channel (learner -> collector)
  with host-shm and CUDA-IPC endpoint implementations.
"""

from motrix_rl.fastsac.async_impl.transport.common import (
    Control,
    bind_flat_params,
    flatten_params,
    load_flat_params,
)
from motrix_rl.fastsac.async_impl.transport.ipc_ring import IpcTransitionRing
from motrix_rl.fastsac.async_impl.transport.ring import RingCursors, SharedTransitionRing
from motrix_rl.fastsac.async_impl.transport.weight_channel import (
    GpuIpcWeightReceiver,
    GpuIpcWeightSender,
    HostWeightReceiver,
    HostWeightSender,
    WeightChannelShared,
    WeightReceiver,
    WeightSender,
    weight_receiver_for,
)

__all__ = [
    "Control",
    "GpuIpcWeightReceiver",
    "GpuIpcWeightSender",
    "HostWeightReceiver",
    "HostWeightSender",
    "IpcTransitionRing",
    "RingCursors",
    "SharedTransitionRing",
    "WeightChannelShared",
    "WeightReceiver",
    "WeightSender",
    "weight_receiver_for",
    "bind_flat_params",
    "flatten_params",
    "load_flat_params",
]
