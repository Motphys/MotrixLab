# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Bounded SPSC tensor slots backed by CPU shared memory.

Each endpoint has a single writer for its cursor. Slot data precedes publication;
consumers release views only after their reads complete. Aligned int64 cursor
loads/stores rely on x86-64 TSO: this is NOT a portable ARM memory-ordering
implementation and NOT an MPMC queue. CUDA endpoints additionally require
completion events (see cuda_ring).
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

import torch

_SUPPORTED_MACHINE = "x86_64"


def require_supported_host_transport() -> None:
    """Reject host cursor transport on platforms without its memory ordering."""
    machine = platform.machine().lower()
    if machine not in {_SUPPORTED_MACHINE, "amd64"}:
        raise RuntimeError(
            f"host tensor transport requires x86-64 TSO, got machine '{machine}'; "
            "ARM/weak-memory support is not implemented"
        )


def shared_tensor(shape, dtype) -> torch.Tensor:
    """Allocate zero-initialized CPU storage shared by spawn workers."""
    return torch.zeros(shape, dtype=dtype).share_memory_()


@dataclass(frozen=True)
class TensorField:
    """A named tensor's per-slot shape and storage dtype, excluding capacity."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32


class RingCursors:
    """Parent-created cumulative completed-slot counts, shared across endpoints."""

    def __init__(self):
        require_supported_host_transport()
        self._write = shared_tensor((1,), torch.int64)
        self._read = shared_tensor((1,), torch.int64)

    @property
    def write_completed_count(self) -> int:
        return int(self._write[0])

    @property
    def read_completed_count(self) -> int:
        return int(self._read[0])


class SharedTensorRing:
    """One producer, one consumer, fixed tensor layout and bounded ownership.

    Field order is the order supplied at construction. read_span returns views,
    not owned copies; commit_reads releases them to the producer. Host copies
    must be complete before commit_reads (in particular, asynchronous H2D reads
    of these views must have completed). No replay or sampling semantics live
    here. Callers own readiness, retry/backoff and cancellation.
    """

    def __init__(self, capacity: int, fields: tuple[TensorField, ...], cursors: RingCursors | None = None):
        require_supported_host_transport()
        if capacity < 1:
            raise ValueError(f"ring_capacity must be >= 1, got {capacity}")
        if not fields or len({field.name for field in fields}) != len(fields):
            raise ValueError("ring fields must be nonempty and uniquely named")
        self.capacity = capacity
        self.fields = fields
        self.tensors = tuple(shared_tensor((capacity, *field.shape), field.dtype) for field in fields)
        self.cursors = cursors if cursors is not None else RingCursors()

    @property
    def write_completed_count(self) -> int:
        return self.cursors.write_completed_count

    @property
    def read_completed_count(self) -> int:
        return self.cursors.read_completed_count

    @property
    def write_issued_count(self) -> int:
        return self.write_completed_count

    @property
    def read_issued_count(self) -> int:
        return self.read_completed_count

    def size(self) -> int:
        return self.write_completed_count - self.read_completed_count

    def is_full(self) -> bool:
        return self.size() >= self.capacity

    def push(self, *values: torch.Tensor) -> bool:
        if len(values) != len(self.tensors):
            raise ValueError("one tensor is required for each ring field")
        for field, value in zip(self.fields, values):
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != field.shape or value.dtype != field.dtype:
                raise ValueError(
                    f"field '{field.name}' has shape/dtype "
                    f"{getattr(value, 'shape', None)}/{getattr(value, 'dtype', None)}, "
                    f"expected {field.shape}/{field.dtype}"
                )
            if value.device.type != "cpu":
                raise ValueError(f"field '{field.name}' must be a CPU tensor")
        if self.is_full():
            return False
        slot = self.write_completed_count % self.capacity
        for storage, value in zip(self.tensors, values):
            storage[slot].copy_(value)
        self.publish_write()
        return True

    def publish_write(self) -> None:
        """Publish a fully written slot; adapter sidecars must be written first."""
        self.cursors._write[0] += 1

    def has_next(self) -> bool:
        return self.size() > 0

    def read_span(self) -> tuple[int, tuple[torch.Tensor, ...]]:
        size = self.size()
        if size <= 0:
            return 0, ()
        slot = self.read_completed_count % self.capacity
        count = min(size, self.capacity - slot)
        return count, tuple(storage[slot : slot + count] for storage in self.tensors)

    def commit_reads(self, n: int) -> None:
        """Release a consumed contiguous span after its reads have completed."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("commit count must be a nonnegative integer")
        readable = self.write_completed_count - self.read_completed_count
        if n > readable:
            raise ValueError(f"cannot commit {n} slots with only {readable} readable")
        self.cursors._read[0] += n
