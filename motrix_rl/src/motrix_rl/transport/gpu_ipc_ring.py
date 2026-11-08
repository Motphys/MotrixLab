# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""CUDA-IPC transition ring: SPSC ring with device-resident slots.

Used when collector inference and the learner share one GPU. All schema
fields are **fused** into a single ``(capacity, num_envs, feat)`` float32
device tensor (``feat = schema.feature_width``): integer and boolean fields
ride as floats (0/1 exactly representable; larger integers are converted
back losslessly by the consumer's ``copy_`` only while they stay
float32-exact — a schema carrying wide counters should use the host ring).
Fusion lets the collector publish one env-step batch with a **single** H2D
copy from one pinned staging buffer, and lets the learner ingest with pure
D2D strided copies — no host transfer stays on the learner's critical path.

Ownership and handoff
~~~~~~~~~~~~~~~~~~~~~
The owner (learner process, which owns the CUDA context) allocates the fused
slot tensor and must keep it alive for the process lifetime. The tensor is
shipped to the collector through a one-shot ``slot_queue`` handshake —
``torch.multiprocessing`` registers CUDA-IPC reducers for CUDA tensors, so
the queue transfer maps the same device memory into the collector process.
The receiver constructs its endpoint from the arrived tensor. Both sides
share the parent-created host :class:`~motrix_rl.transport.ring.RingCursors`.

Publish ordering (the correctness core)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Host cursors are bumped by CPU stores, but slot data is written by GPU copies
whose completion is NOT ordered by x86/TSO. A cursor may therefore only be
advanced once the corresponding device writes have completed, proven by CUDA
events:

* producer: after the slot's H2D is enqueued, record a per-slot event;
  ``_write`` advances only up to the boundary of completed events (lazily
  flushed by :meth:`GpuIpcTransitionRing.push` / :meth:`GpuIpcTransitionRing.is_full`).
  The single pinned staging buffer is likewise not reused until the previous
  H2D's event completed.
* consumer: ``commit_reads`` records an event after the run's D2D reads are
  enqueued; ``_read`` advances only up to the boundary of completed events
  (lazily flushed by :meth:`GpuIpcTransitionRing.read_span` /
  :meth:`GpuIpcTransitionRing.has_next` / :meth:`GpuIpcTransitionRing.size`).

Both cursor stores remain single-writer aligned int64 host stores, so the
host-side protocol of the shared-memory ring carries over unchanged.
"""

from __future__ import annotations

from collections import deque, namedtuple
from collections.abc import Mapping

import torch

from motrix_rl.transport.ring import RingCursors, SlotPayloadStore
from motrix_rl.transport.schema import TransitionSchema

# A slot-boundary event and the cursor value it certifies once complete:
# the write (or read) cursor value the entry advances the shared cursor to
# when its event is observed finished.
_PendingEvent = namedtuple("_PendingEvent", ["cursor", "event"])


class GpuIpcTransitionRing:
    """SPSC transition ring backed by one fused CUDA-IPC device tensor.

    The producer endpoint is built in the collector process from the shipped
    ``slots`` tensor; the consumer endpoint in the learner process may be the
    owner (it allocated ``slots``). The two roles use disjoint APIs:

    producer: :meth:`is_full`, :meth:`push`
    consumer: :meth:`has_next`, :meth:`read_span`, :meth:`commit_reads`

    Field layout along the fused last dimension follows ``schema.fields``
    order; vector fields keep their shape, multi-dim fields come back
    flattened (``(count, num_envs, prod)``).
    """

    def __init__(
        self,
        cursors: RingCursors,
        slots: torch.Tensor,
        schema: TransitionSchema,
        capacity: int,
        num_envs: int,
        payload_store: SlotPayloadStore | None = None,
    ):
        if not slots.is_cuda:
            raise ValueError(f"IPC transition ring requires CUDA slots, got device {slots.device}")
        expected = (capacity, num_envs, schema.feature_width)
        if tuple(slots.shape) != expected:
            raise ValueError(f"IPC slot tensor shape {tuple(slots.shape)} does not match {expected}")
        if slots.dtype != torch.float32:
            raise ValueError(f"IPC slots must be float32, got {slots.dtype}")
        self.cursors = cursors
        self.slots = slots
        self.schema = schema
        self.capacity = capacity
        self.num_envs = num_envs
        self.payload_store = payload_store
        # Field layout along the fused last dimension, in schema order.
        self._offsets: dict[str, tuple[int, int]] = {}
        offset = 0
        for field in schema.fields:
            width = field.per_env_numel
            self._offsets[field.name] = (offset, offset + width)
            offset += width
        # Endpoint-local issued counts. The shared cursors hold only the
        # event-proven (published) values, which LAG the issued counts while
        # copies are in flight — so slot indexing and publish targets must come
        # from the issued counts, never from the cursors.
        self._write_issued_count = self.cursors.write_completed_count
        self._read_issued_count = self.cursors.read_completed_count
        # Producer-side: events of pushed-but-not-yet-published slots, in
        # order. (published cursor value, event) — the write cursor value the
        # entry certifies once its event completes. Deques: flush pops from
        # the head, and list.pop(0) would shift the remaining entries.
        self._write_events: deque[_PendingEvent] = deque()
        # Consumer-side: events of committed-but-not-yet-advanced reads.
        self._read_events: deque[_PendingEvent] = deque()
        # Single pinned staging reused by every push; guarded by _staging_event.
        self._staging = torch.empty((num_envs, schema.feature_width), dtype=torch.float32, pin_memory=True)
        self._staging_event = torch.cuda.Event()
        self._staging_used = False

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
        store = self.payload_store
        return store is not None and not store.can_write_rows(self.num_envs, completed_reads)

    def push(self, batch: Mapping[str, torch.Tensor], payload=None) -> bool:
        """Publish one env-step batch into the next slot. Returns False if full.

        ``batch`` maps every schema field name to one of the collector's CPU
        tensors (integer / boolean fields may be any numeric dtype; they are
        stored as floats). One fused H2D lands the whole slot; the cursor
        publishes lazily once it completes.
        """
        if set(batch) != set(self._offsets):
            raise ValueError(f"transition batch fields {sorted(batch)} do not match the schema {list(self._offsets)}")
        if self.is_full():
            return False
        store = self.payload_store
        if payload is not None and store is None:
            raise ValueError("slot payload requires a payload store")
        if store is not None and not store.can_write(payload, self.cursors.read_completed_count):
            return False
        if self._staging_used and not self._staging_event.query():
            # The previous H2D still reads the staging buffer; it completes in
            # well under one env step, so this is normally a no-op wait.
            self._staging_event.synchronize()
        s = self._staging
        for name, value in batch.items():
            begin, end = self._offsets[name]
            s[:, begin:end].copy_(value.reshape(self.num_envs, -1))
        issued = self._write_issued_count
        slot = issued % self.capacity
        if store is not None:
            store.write(issued, payload)
        stream = torch.cuda.current_stream(self.slots.device)
        self.slots[slot].copy_(s, non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
        self._write_events.append(_PendingEvent(issued + 1, event))
        self._write_issued_count = issued + 1
        self._staging_event.record(stream)
        self._staging_used = True
        return True

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

    def read_span(self) -> tuple[int, dict[str, torch.Tensor]]:
        """Longest contiguous unread run, as ``(count, field_views)``.

        The views are strided slices of the fused slot tensor — device tensors
        keyed by field name. Vector fields have shape ``(count, num_envs,
        dim)``; scalar fields ``(count, num_envs)``; multi-dim fields come
        back flattened (see the class docstring). Their data is complete: the
        producer only publishes a slot after its H2D event completed. Returns
        ``(0, {})`` when empty.
        """
        self._flush_reads()
        available = self.write_completed_count - self._read_issued_count
        if available <= 0:
            return 0, {}
        slot = self._read_issued_count % self.capacity
        count = min(available, self.capacity - slot)
        run = self.slots[slot : slot + count]
        views = {}
        for field in self.schema.fields:
            begin, end = self._offsets[field.name]
            if field.per_env_shape == ():
                views[field.name] = run[:, :, begin]
            else:
                views[field.name] = run[:, :, begin:end]
        return count, views

    def read_payload_span(self, count: int) -> list:
        """Copy host payloads belonging to the next issued read span."""
        if self.payload_store is None:
            return [None] * count
        return self.payload_store.read(self._read_issued_count, count)

    def commit_reads(self, n: int) -> None:
        """Release ``n`` slots of a consumed run once their reads completed.

        Call AFTER the consumer's D2D copies reading the run have been
        enqueued on the current stream: the recorded event orders the cursor
        advance behind those copies, so the producer can never overwrite a
        slot the learner is still reading.
        """
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.slots.device))
        self._read_events.append(_PendingEvent(self._read_issued_count + n, event))
        self._read_issued_count += n
