# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SPSC CUDA tensor slots with event-proven publish/release cursors.

The learner owns and keeps slots alive; a one-shot spawn handshake transfers
CUDA IPC handles. Packing and optional sidecars belong to the caller. All reads
must be enqueued on the current slots-device stream before commit_reads.
"""

from collections import deque
from dataclasses import dataclass

import torch

from motrix_rl.transport.ring import RingCursors


@dataclass
class _PendingEvent:
    cursor: int
    event: torch.cuda.Event


class CudaTensorRing:
    """Bounded fixed-shape CUDA slots, one producer and one consumer only."""

    def __init__(self, cursors: RingCursors, slots: torch.Tensor):
        if not slots.is_cuda:
            raise ValueError(f"IPC ring requires CUDA slots, got {slots.device}")
        if slots.ndim < 1 or slots.shape[0] < 1:
            raise ValueError("CUDA ring requires a nonempty capacity dimension")
        self.cursors = cursors
        self.slots = slots
        self.capacity = slots.shape[0]
        self._write_issued_count = cursors.write_completed_count
        self._read_issued_count = cursors.read_completed_count
        self._write_events: deque[_PendingEvent] = deque()
        self._read_events: deque[_PendingEvent] = deque()

    # -------------------------------------------------------------- producer
    def _flush_writes(self) -> None:
        """Advance ``_write`` to the boundary of completed push events."""
        events = self._write_events
        while events and events[0].event.query():
            self.cursors._write[0] = events.popleft().cursor

    @property
    def write_completed_count(self) -> int:
        """Cumulative slots whose writes completed and were published."""
        self._flush_writes()
        return self.cursors.write_completed_count

    @property
    def read_completed_count(self) -> int:
        """Cumulative slots whose reads completed and were released."""
        self._flush_reads()
        return self.cursors.read_completed_count

    @property
    def write_issued_count(self) -> int:
        """Cumulative slots submitted for writing, including copies in flight."""
        return self._write_issued_count

    @property
    def read_issued_count(self) -> int:
        """Cumulative slots consumed, including acknowledgements still in flight."""
        return self._read_issued_count

    def size(self) -> int:
        """Published but not yet released slot count, including reads in flight."""
        return self.write_completed_count - self.read_completed_count

    def is_full(self) -> bool:
        # Backpressure must count the producer's issued (in-flight included)
        # writes, so a burst of pushes can never exceed capacity. The flush
        # publishes completed slots, mirroring the collector's every-step poll.
        self._flush_writes()
        completed_reads = self.read_completed_count
        if self._write_issued_count - completed_reads >= self.capacity:
            return True
        return False

    def push(self, value: torch.Tensor) -> bool:
        """Enqueue one slot copy; caller retains source until the copy completes."""
        if tuple(value.shape) != tuple(self.slots.shape[1:]) or value.dtype != self.slots.dtype:
            raise ValueError("CUDA slot source must match the slot shape and dtype")
        if self.is_full():
            return False
        self.write_slot(value)
        return True

    def write_slot(self, value: torch.Tensor) -> None:
        """Copy and publish after completion; caller has reserved capacity."""
        issued = self._write_issued_count
        self.slots[issued % self.capacity].copy_(value, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.slots.device))
        self._write_events.append(_PendingEvent(issued + 1, event))
        self._write_issued_count = issued + 1

    # -------------------------------------------------------------- consumer
    def _flush_reads(self) -> None:
        """Advance ``_read`` to the boundary of completed commit events."""
        events = self._read_events
        while events and events[0].event.query():
            self.cursors._read[0] = events.popleft().cursor

    def has_next(self) -> bool:
        # Flush this endpoint's commit events so the producer's is_full sees
        # freed slots (the learner's every-loop poll).
        self._flush_reads()
        return self.write_completed_count > self._read_issued_count

    def read_span(self) -> tuple[int, torch.Tensor | None]:
        """Return a contiguous slot view without releasing its ownership."""
        self._flush_reads()
        available = self.write_completed_count - self._read_issued_count
        if available <= 0:
            return 0, None
        slot = self._read_issued_count % self.capacity
        count = min(available, self.capacity - slot)
        return count, self.slots[slot : slot + count]

    def commit_reads(self, n: int) -> None:
        """Release ``n`` slots of a consumed run once their reads completed.

        Call AFTER the consumer's D2D copies reading the run have been
        enqueued on the current stream: the recorded event orders the cursor
        advance behind those copies, so the producer can never overwrite a
        slot the learner is still reading.
        """
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("commit count must be a nonnegative integer")
        self._flush_reads()
        available = self.write_completed_count - self._read_issued_count
        if n > available:
            raise ValueError(f"cannot commit {n} slots with only {available} readable")
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.slots.device))
        self._read_events.append(_PendingEvent(self._read_issued_count + n, event))
        self._read_issued_count += n
