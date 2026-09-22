# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""CUDA-IPC transition ring: SPSC ring with device-resident slots.

Used when collector inference and the learner share one GPU. The six
transition fields are fused into a single ``(capacity, num_envs, feat)``
float32 device tensor (``feat = obs_dim + critic_obs_dim + act_dim + 3``;
reward / done / truncation ride as floats, with 0/1 exactly representable and
converted back to int64 by the replay buffer's ``copy_``). This lets the
collector publish one env-step batch with a **single** H2D copy from one
pinned staging buffer, and lets the learner ingest with pure D2D strided
copies — no host transfer stays on the learner's critical path.

Ownership and handoff
~~~~~~~~~~~~~~~~~~~~~
The owner (learner process, which owns the CUDA context) allocates the fused
slot tensor and must keep it alive for the process lifetime. The tensor is
shipped to the collector through the existing one-shot ``slot_queue``
handshake — ``torch.multiprocessing`` registers CUDA-IPC reducers for CUDA
tensors, so the queue transfer maps the same device memory into the collector
process. The receiver constructs its endpoint from the arrived tensor. Both
sides share the parent-created host :class:`~.ring.RingCursors`.

Publish ordering (the correctness core)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Host cursors are bumped by CPU stores, but slot data is written by GPU copies
whose completion is NOT ordered by x86/TSO. A cursor may therefore only be
advanced once the corresponding device writes have completed, proven by CUDA
events:

* producer: after the slot's H2D is enqueued, record a per-slot event;
  ``_write`` advances only up to the boundary of completed events (lazily
  flushed by :meth:`push` / :meth:`is_full`). The single pinned staging
  buffer is likewise not reused until the previous H2D's event completed.
* consumer: ``commit_reads`` records an event after the run's D2D reads are
  enqueued; ``_read`` advances only up to the boundary of completed events
  (lazily flushed by :meth:`read_span` / :meth:`has_next` / :meth:`size`).

Both cursor stores remain single-writer aligned int64 host stores, so the
host-side protocol of the shared-memory ring carries over unchanged.
"""

from __future__ import annotations

from collections import deque, namedtuple

import torch

from motrix_rl.fastsac.async_impl.transport.ring import RingCursors

# A slot-boundary event and the cursor value it certifies once complete:
# the write (or read) cursor value the entry advances the shared cursor to
# when its event is observed finished.
_PendingEvent = namedtuple("_PendingEvent", ["cursor", "event"])


class IpcTransitionRing:
    """SPSC transition ring backed by one fused CUDA-IPC device tensor.

    The producer endpoint is built in the collector process from the shipped
    ``slots`` tensor; the consumer endpoint in the learner process may be the
    owner (it allocated ``slots``). The two roles use disjoint APIs:

    producer: :meth:`is_full`, :meth:`push`
    consumer: :meth:`has_next`, :meth:`read_span`, :meth:`commit_reads`
    """

    def __init__(
        self,
        cursors: RingCursors,
        slots: torch.Tensor,
        capacity: int,
        num_envs: int,
        obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
    ):
        if not slots.is_cuda:
            raise ValueError(f"IPC transition ring requires CUDA slots, got device {slots.device}")
        expected = (capacity, num_envs, obs_dim + critic_obs_dim + act_dim + 3)
        if tuple(slots.shape) != expected:
            raise ValueError(f"IPC slot tensor shape {tuple(slots.shape)} does not match {expected}")
        self.cursors = cursors
        self.slots = slots
        self.capacity = capacity
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.act_dim = act_dim
        # Field layout along the fused last dimension.
        self._o0 = 0
        self._o1 = obs_dim
        self._o2 = obs_dim + critic_obs_dim
        self._o3 = self._o2 + act_dim
        self._o4 = self._o3 + 1  # reward
        self._o5 = self._o4 + 1  # done
        # Endpoint-local issued counts. The shared cursors hold only the
        # event-proven (published) values, which LAG the issued counts while
        # copies are in flight — so slot indexing and publish targets must come
        # from the issued counts, never from the cursors.
        self._issued_writes = self.cursors.write_idx
        self._issued_reads = self.cursors.read_idx
        # Producer-side: events of pushed-but-not-yet-published slots, in
        # order. (published cursor value, event) — the write cursor value the
        # entry certifies once its event completes. Deques: flush pops from
        # the head, and list.pop(0) would shift the remaining entries.
        self._write_events: deque[_PendingEvent] = deque()
        # Consumer-side: events of committed-but-not-yet-advanced reads.
        self._read_events: deque[_PendingEvent] = deque()
        # Single pinned staging reused by every push; guarded by _staging_event.
        self._staging = torch.empty((num_envs, expected[2]), dtype=torch.float32, pin_memory=True)
        self._staging_event = torch.cuda.Event()
        self._staging_used = False

    # -------------------------------------------------------------- producer
    def _flush_writes(self) -> None:
        """Advance ``_write`` to the boundary of completed push events."""
        events = self._write_events
        while events and events[0].event.query():
            self.cursors._write[0] = events.popleft().cursor

    @property
    def write_idx(self) -> int:
        self._flush_writes()
        return self.cursors.write_idx

    @property
    def read_idx(self) -> int:
        self._flush_reads()
        return self.cursors.read_idx

    def size(self) -> int:
        """Published (event-proven) unread count — the observable ring fill."""
        return self.write_idx - self.read_idx

    def is_full(self) -> bool:
        # Backpressure must count the producer's issued (in-flight included)
        # writes, so a burst of pushes can never exceed capacity. The flush
        # publishes completed slots, mirroring the collector's every-step poll.
        self._flush_writes()
        return self._issued_writes - self.read_idx >= self.capacity

    def push(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncations: torch.Tensor,
    ) -> bool:
        """Publish one env-step batch into the next slot. Returns False if full.

        Inputs are the collector's CPU tensors (``dones``/``truncations`` may
        be any numeric dtype; they are stored as 0/1 floats). One fused H2D
        lands the whole slot; the cursor publishes lazily once it completes.
        """
        if self.is_full():
            return False
        if self._staging_used and not self._staging_event.query():
            # The previous H2D still reads the staging buffer; it completes in
            # well under one env step, so this is normally a no-op wait.
            self._staging_event.synchronize()
        s = self._staging
        s[:, self._o0 : self._o1].copy_(obs)
        s[:, self._o1 : self._o2].copy_(critic_obs)
        s[:, self._o2 : self._o3].copy_(actions)
        s[:, self._o3 : self._o4].copy_(rewards.reshape(-1, 1))
        s[:, self._o4 : self._o5].copy_(dones.reshape(-1, 1))
        s[:, self._o5 :].copy_(truncations.reshape(-1, 1))
        issued = self._issued_writes
        slot = issued % self.capacity
        stream = torch.cuda.current_stream(self.slots.device)
        self.slots[slot].copy_(s, non_blocking=True)
        event = torch.cuda.Event()
        event.record(stream)
        self._write_events.append(_PendingEvent(issued + 1, event))
        self._issued_writes = issued + 1
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
        return self.write_idx > self._issued_reads

    def read_span(self) -> tuple[int, tuple[torch.Tensor, ...]]:
        """Longest contiguous unread run, as ``(count, field_views)``.

        The views are strided slices of the fused slot tensor — device tensors
        of shape ``(count, num_envs, dim)`` (``(count, num_envs)`` for the
        scalar fields). Their data is complete: the producer only publishes a
        slot after its H2D event completed. Returns ``(0, ())`` when empty.
        """
        self._flush_reads()
        available = self.write_idx - self._issued_reads
        if available <= 0:
            return 0, ()
        slot = self._issued_reads % self.capacity
        count = min(available, self.capacity - slot)
        run = self.slots[slot : slot + count]
        return count, (
            run[:, :, self._o0 : self._o1],
            run[:, :, self._o1 : self._o2],
            run[:, :, self._o2 : self._o3],
            run[:, :, self._o3],
            run[:, :, self._o4],
            run[:, :, self._o5],
        )

    def commit_reads(self, n: int) -> None:
        """Release ``n`` slots of a consumed run once their reads completed.

        Call AFTER the consumer's D2D copies reading the run have been
        enqueued on the current stream: the recorded event orders the cursor
        advance behind those copies, so the producer can never overwrite a
        slot the learner is still reading.
        """
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.slots.device))
        self._read_events.append(_PendingEvent(self._issued_reads + n, event))
        self._issued_reads += n
