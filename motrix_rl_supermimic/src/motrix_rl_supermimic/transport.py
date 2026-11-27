# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Host rings and CUDA storage descriptors for collector startup."""

from dataclasses import dataclass

import torch

from motrix_rl.transport.gpu_ipc_ring import GpuIpcTransitionRing
from motrix_rl.transport.ring import HostTransitionRing, RingCursors
from motrix_rl.transport.schema import TransitionSchema


@dataclass(frozen=True)
class GpuRingConnection:
    schema: TransitionSchema
    slots: torch.Tensor
    cursors: RingCursors


RingConnection = HostTransitionRing | GpuRingConnection


def connect_producer(connection: RingConnection) -> HostTransitionRing | GpuIpcTransitionRing:
    """Use the shared host ring or construct a process-local CUDA endpoint."""
    if isinstance(connection, HostTransitionRing):
        return connection
    return GpuIpcTransitionRing(
        connection.cursors,
        connection.slots,
        connection.schema,
        connection.slots.shape[0],
        connection.slots.shape[1],
    )
