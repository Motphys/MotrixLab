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
* the seqlock weight channel (learner -> collector) lives in
  :mod:`motrix_rl.transport.weight_channel` with the other transport
  primitives; the FastSAC bundle layout (actor + obs normalizer) is declared
  in :mod:`.schema` via :func:`.fastsac_inference_bundle_spec`.
"""

from motrix_rl.transport.common import (
    Control,
    bind_flat_params,
    flatten_params,
    load_flat_params,
)
from motrix_rl.transport.gpu_ipc_ring import GpuIpcTransitionRing
from motrix_rl.transport.inference_bundle import (
    InferenceBundleSpec,
    ModuleSection,
    NormalizerSection,
)
from motrix_rl.transport.ring import HostTransitionRing, RingCursors
from motrix_rl.transport.schema import TransitionField, TransitionSchema
from motrix_rl.transport.weight_channel import (
    GpuIpcWeightReceiver,
    GpuIpcWeightSender,
    HostWeightReceiver,
    HostWeightSender,
    WeightChannelShared,
    WeightReceiver,
    WeightSender,
    weight_receiver_for,
)
from motrix_rl_builtin.fastsac.async_impl.transport.final_obs import (
    FinalObsPayload,
    FinalObsStorage,
)
from motrix_rl_builtin.fastsac.async_impl.transport.handshake import StartupHandshake
from motrix_rl_builtin.fastsac.async_impl.transport.schema import (
    fastsac_inference_bundle_spec,
    fastsac_transition_schema,
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
    "InferenceBundleSpec",
    "ModuleSection",
    "NormalizerSection",
    "StartupHandshake",
    "TransitionField",
    "TransitionSchema",
    "WeightChannelShared",
    "WeightReceiver",
    "WeightSender",
    "bind_flat_params",
    "fastsac_inference_bundle_spec",
    "fastsac_transition_schema",
    "flatten_params",
    "load_flat_params",
    "weight_receiver_for",
]
