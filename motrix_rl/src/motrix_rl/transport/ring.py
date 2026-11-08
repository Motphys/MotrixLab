# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SPSC transition ring between the collector and learner processes.

The ring protocol (cursor semantics, backpressure) is shared by two physical
transports:

* host shared-memory fields (this module, :class:`HostTransitionRing`) —
  the universal fallback;
* CUDA-IPC device fields (``gpu_ipc_ring.py``, :class:`GpuIpcTransitionRing`) —
  used when collector inference and the learner share one GPU.

Both transports advance the same parent-created :class:`RingCursors`, so the
producer/consumer contract below is transport-independent.

The slot layout is declared by a
:class:`~motrix_rl.transport.schema.TransitionSchema`; the ring allocates one
shared tensor per field with the schema's dtype and per-env shape, and
addresses batches **by field name** (``push``) and by name-keyed views
(``read_span``). The field set is the training backend's configuration-time
decision.

Producer (collector) calls :meth:`HostTransitionRing.push`; when the ring is
full it returns ``False`` and the caller must retry/backoff — that is the
backpressure that keeps the collector from outrunning the learner and
flooding memory.

Consumer (learner) calls :meth:`HostTransitionRing.read_span` to get
zero-copy views of the longest contiguous unread run of slots, moves them to
its replay buffer, then calls :meth:`HostTransitionRing.commit_reads`. The
read cursor only advances after the copy, so the producer can never clobber a
slot that is still being ingested.

Memory ordering
~~~~~~~~~~~~~~~
Only ``_write`` is read by the consumer, only ``_read`` is read by the
producer, and each cursor has a single writer — so an aligned int64 store
is enough to publish progress and no atomic RMW is needed. Correctness also
needs the consumer to not observe the ``_write`` bump before the slot's data
stores have landed (and symmetrically for ``_read``); on x86/TSO that
ordering is free, so no memory barrier is used and this path is x86-only
(see :mod:`motrix_rl.transport.common` for the full note). The CUDA-IPC
transport instead orders GPU writes behind per-slot events before bumping a
cursor (see ``gpu_ipc_ring.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import torch

from motrix_rl.transport.common import _shared
from motrix_rl.transport.schema import TransitionSchema


class RingCursors:
    """Host shared-memory read/write cursors shared by both ring transports.

    Counts are cumulative env-step batches (slots), not physical slot indices;
    they do not wrap at capacity. Completed writes are published to the consumer,
    and completed reads release slots to the producer.

    Created once by the parent process and inherited by both workers. Each
    cursor has a single writer (the producer owns ``_write``, the consumer
    owns ``_read``), so plain aligned int64 stores publish progress coherently
    on x86/TSO without barriers.
    """

    def __init__(self):
        self._write = _shared((1,), torch.int64)
        self._read = _shared((1,), torch.int64)

    @property
    def write_completed_count(self) -> int:
        # Consumer reads this; single-writer producer, so a plain aligned int64
        # load is coherent. On x86/TSO the data loads that follow cannot be
        # reordered ahead of it, so no acquire barrier is needed (x86-only).
        return int(self._write[0])

    @property
    def read_completed_count(self) -> int:
        # Producer reads this; single-writer consumer. On x86/TSO the following
        # is_full() decision is based on an up-to-date value without a barrier.
        return int(self._read[0])


class SlotPayloadStore(Protocol):
    """Duck contract of the optional sparse per-slot payload store.

    The generic ring treats the store as opaque: how payloads are shaped or
    which fields they carry is the owning trainer's business (see FastSAC's
    FinalObsStorage for the established example — sparse final-observation
    rows for auto-reset envs). Row reclamation is governed by the transition
    cursors, so the store needs no cursor of its own.
    """

    def can_write(self, payload, acknowledged: int) -> bool: ...

    def can_write_rows(self, count: int, acknowledged: int) -> bool: ...

    def write(self, index: int, payload) -> None: ...

    def read(self, index: int, count: int) -> list: ...


class HostTransitionRing:
    """SPSC ring of transition batches with bounded backpressure (host fields).

    Each slot holds one env-step batch: the per-field tensors declared by the
    schema, each with the leading dimension ``num_envs``. What a slot carries
    is entirely the schema's decision — e.g. one backend stores observation
    pairs, another stores privileged state vectors plus scalar episode
    metadata; the transport treats both identically.

    Fields live in host shared memory, usable by any collector/learner device
    combination. See ``gpu_ipc_ring.py`` for the CUDA-IPC device-fields variant
    sharing the same cursor protocol.
    """

    def __init__(
        self,
        schema: TransitionSchema,
        capacity: int,
        num_envs: int,
        cursors: RingCursors | None = None,
        payload_store: SlotPayloadStore | None = None,
    ):
        if capacity < 1:
            raise ValueError(f"ring_capacity must be >= 1, got {capacity}")
        self.schema = schema
        self.capacity = capacity
        self.num_envs = num_envs
        self.data = {
            field.name: _shared((capacity, *schema.slot_shape(field, num_envs)), field.dtype) for field in schema.fields
        }
        self.cursors = cursors if cursors is not None else RingCursors()
        self.payload_store = payload_store

    @property
    def write_completed_count(self) -> int:
        return self.cursors.write_completed_count

    @property
    def read_completed_count(self) -> int:
        return self.cursors.read_completed_count

    @property
    def write_issued_count(self) -> int:
        """Cumulative slots written; host writes complete synchronously."""
        return self.write_completed_count

    @property
    def read_issued_count(self) -> int:
        """Cumulative slots consumed; host acknowledgements complete synchronously."""
        return self.read_completed_count

    def size(self) -> int:
        """Published but not yet released slot count."""
        return self.write_completed_count - self.read_completed_count

    def is_full(self) -> bool:
        """Whether slots or worst-case payload rows prevent another env step."""
        if self.size() >= self.capacity:
            return True
        store = self.payload_store
        return store is not None and not store.can_write_rows(self.num_envs, self.read_completed_count)

    def push(self, batch: Mapping[str, torch.Tensor], payload=None) -> bool:
        """Copy one env-step batch into the next slot. Returns False if full.

        ``batch`` maps every schema field name to a CPU tensor of shape
        ``(num_envs,) + per_env_shape``; the key set must match the schema
        exactly. ``payload`` (optional) is handed to the payload store
        unchanged.
        """
        if set(batch) != set(self.data):
            raise ValueError(f"transition batch fields {sorted(batch)} do not match the schema {list(self.data)}")
        if self.is_full():
            return False
        store = self.payload_store
        if payload is not None and store is None:
            raise ValueError("slot payload requires a payload store")
        if store is not None and not store.can_write(payload, self.read_completed_count):
            return False
        index = self.write_completed_count
        slot = index % self.capacity
        for name, value in batch.items():
            self.data[name][slot].copy_(value)
        if store is not None:
            store.write(index, payload)
        # Publish the slot. On x86/TSO the field copies above are guaranteed
        # visible before this cursor bump, so a consumer that reads the new
        # write_completed_count also sees the data (x86-only; ARM would need a
        # release here).
        self.cursors._write[0] += 1
        return True

    def has_next(self) -> bool:
        """Whether at least one unread slot is committed (i.e. readable)."""
        return self.size() > 0

    def read_span(self) -> tuple[int, dict[str, torch.Tensor]]:
        """Longest contiguous unread run, as ``(count, field_views)``.

        ``field_views`` maps every schema field name to a tensor of shape
        ``(count, num_envs) + per_env_shape`` — contiguous in the slot
        (leading) dimension, so the consumer can move the whole run with one
        copy per field. Consumers that need a positional order use
        ``schema.fields`` order as the single authority. Returns ``(0, {})``
        when empty. Does NOT advance the read cursor; free the run with
        :meth:`commit_reads` once the data has been copied out.
        """
        size = self.size()
        if size <= 0:
            return 0, {}
        slot = self.read_completed_count % self.capacity
        count = min(size, self.capacity - slot)
        views = {name: field[slot : slot + count] for name, field in self.data.items()}
        return count, views

    def read_payload_span(self, count: int) -> list:
        """Owned payloads aligned with the current transition read span."""
        if self.payload_store is None:
            return [None] * count
        return self.payload_store.read(self.read_completed_count, count)

    def commit_reads(self, n: int) -> None:
        """Advance the read cursor by ``n`` slots of a consumed contiguous run.

        On x86/TSO the consumer's reads complete before this cursor bump, so
        the producer's is_full() cannot reuse a slot still being copied out.
        """
        self.cursors._read[0] += n
