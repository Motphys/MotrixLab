# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Transport surface for the async FastSAC trainer.

The generic primitives — transition schema, SPSC rings (host and CUDA-IPC),
shared-memory controls and flat-parameter helpers — live in
:mod:`motrix_rl.transport` and are schema-driven: the field set of a
transition slot is this trainer's declaration (see :mod:`.schema`), not a
transport decision. This package keeps the
``motrix_rl_builtin.fastsac.async_impl.transport`` import path stable for the
worker/collector/learner modules and tests; import from the defining
submodules (or ``motrix_rl.transport``) for new internal uses.

Local submodules:

* :mod:`.schema`         — the FastSAC :class:`~.schema.TransitionSchema`
  factory (six fields; next-obs derived from the successive slot).
* :mod:`.final_obs`      — the sparse per-slot final-observation store for
  auto-reset envs.
* :mod:`.handshake`      — one-shot startup handshake queues.
* :mod:`.weight_channel` — the seqlock weight channel (learner -> collector)
  with host-shm and CUDA-IPC endpoint implementations.
"""

from motrix_rl.transport.common import (
    Control,
    bind_flat_params,
    flatten_params,
    load_flat_params,
)
from motrix_rl.transport.gpu_ipc_ring import GpuIpcTransitionRing
from motrix_rl.transport.ring import HostTransitionRing, RingCursors
from motrix_rl.transport.schema import TransitionField, TransitionSchema
from motrix_rl_builtin.fastsac.async_impl.transport.final_obs import (
    FinalObsPayload,
    FinalObsStorage,
)
from motrix_rl_builtin.fastsac.async_impl.transport.handshake import StartupHandshake
from motrix_rl_builtin.fastsac.async_impl.transport.schema import fastsac_transition_schema
from motrix_rl_builtin.fastsac.async_impl.transport.weight_channel import (
    GpuIpcWeightReceiver,
    GpuIpcWeightSender,
    HostWeightReceiver,
    HostWeightSender,
    WeightChannelShared,
    WeightReceiver,
    WeightSender,
)

__all__ = [
    "Control",
    "FinalObsPayload",
    "FinalObsStorage",
    "GpuIpcWeightReceiver",
    "GpuIpcWeightSender",
    "HostWeightReceiver",
    "HostWeightSender",
    "GpuIpcTransitionRing",
    "RingCursors",
    "HostTransitionRing",
    "StartupHandshake",
    "TransitionField",
    "TransitionSchema",
    "WeightChannelShared",
    "WeightReceiver",
    "WeightSender",
    "bind_flat_params",
    "fastsac_transition_schema",
    "flatten_params",
    "load_flat_params",
]
