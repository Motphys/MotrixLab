# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Serializable collector startup messages without process-local endpoints."""

from dataclasses import dataclass

import torch

from motrix_rl.transport.gpu_ipc_ring import GpuIpcTransitionRing
from motrix_rl.transport.ring import HostTransitionRing, RingCursors
from motrix_rl.transport.schema import TransitionSchema


@dataclass(frozen=True)
class HostRingConnection:
    schema: TransitionSchema
    capacity: int
    num_envs: int
    cursors: RingCursors
    data: dict[str, torch.Tensor]


@dataclass(frozen=True)
class GpuRingConnection:
    schema: TransitionSchema
    slots: torch.Tensor
    cursors: RingCursors


RingConnection = HostRingConnection | GpuRingConnection


def connect_producer(connection: RingConnection) -> HostTransitionRing | GpuIpcTransitionRing:
    """Attach shared storage and create only this process's endpoint state."""
    if isinstance(connection, HostRingConnection):
        return HostTransitionRing(
            connection.schema,
            connection.capacity,
            connection.num_envs,
            cursors=connection.cursors,
            data=connection.data,
        )
    return GpuIpcTransitionRing(
        connection.cursors,
        connection.slots,
        connection.schema,
        connection.slots.shape[0],
        connection.slots.shape[1],
    )
