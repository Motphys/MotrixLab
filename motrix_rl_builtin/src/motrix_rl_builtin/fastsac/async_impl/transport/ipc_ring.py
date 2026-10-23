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

import torch

from motrix_rl.transport.cuda_ring import CudaTensorRing
from motrix_rl.transport.ring import RingCursors
from motrix_rl_builtin.fastsac.async_impl.transport.ring import FinalObsPayload, FinalObsStorage


class IpcTransitionRing(CudaTensorRing):
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
        final_obs_store: FinalObsStorage | None = None,
    ):
        if not slots.is_cuda:
            raise ValueError(f"IPC transition ring requires CUDA slots, got device {slots.device}")
        expected = (capacity, num_envs, obs_dim + critic_obs_dim + act_dim + 3)
        if tuple(slots.shape) != expected:
            raise ValueError(f"IPC slot tensor shape {tuple(slots.shape)} does not match {expected}")
        super().__init__(cursors, slots)
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.act_dim = act_dim
        self.final_obs_store = final_obs_store
        # Field layout along the fused last dimension.
        self._o0 = 0
        self._o1 = obs_dim
        self._o2 = obs_dim + critic_obs_dim
        self._o3 = self._o2 + act_dim
        self._o4 = self._o3 + 1  # reward
        self._o5 = self._o4 + 1  # done
        # Single pinned staging reused by every push; guarded by _staging_event.
        self._staging = torch.empty((num_envs, expected[2]), dtype=torch.float32, pin_memory=True)
        self._staging_event = torch.cuda.Event()
        self._staging_used = False

    def is_full(self) -> bool:
        # Backpressure must count the producer's issued (in-flight included)
        # writes, so a burst of pushes can never exceed capacity. The flush
        # publishes completed slots, mirroring the collector's every-step poll.
        self._flush_writes()
        completed_reads = self.read_completed_count
        if self._write_issued_count - completed_reads >= self.capacity:
            return True
        store = self.final_obs_store
        return store is not None and not store.can_write_rows(self.num_envs, completed_reads)

    def push(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncations: torch.Tensor,
        final_obs: FinalObsPayload | None = None,
    ) -> bool:
        """Publish one env-step batch into the next slot. Returns False if full.

        Inputs are the collector's CPU tensors (``dones``/``truncations`` may
        be any numeric dtype; they are stored as 0/1 floats). One fused H2D
        lands the whole slot; the cursor publishes lazily once it completes.
        """
        if self.is_full():
            return False
        store = self.final_obs_store
        if final_obs is not None and store is None:
            raise ValueError("final payload requires a slot final-observation store")
        if store is not None and not store.can_write(final_obs, self.cursors.read_completed_count):
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
        issued = self._write_issued_count
        if store is not None:
            store.write(issued, final_obs)
        stream = torch.cuda.current_stream(self.slots.device)
        self.write_slot(s)
        self._staging_event.record(stream)
        self._staging_used = True
        return True

    def read_span(self) -> tuple[int, tuple[torch.Tensor, ...]]:
        """Longest contiguous unread run, as ``(count, field_views)``.

        The views are strided slices of the fused slot tensor — device tensors
        of shape ``(count, num_envs, dim)`` (``(count, num_envs)`` for the
        scalar fields). Their data is complete: the producer only publishes a
        slot after its H2D event completed. Returns ``(0, ())`` when empty.
        """
        count, run = super().read_span()
        if count == 0:
            return 0, ()
        return count, (
            run[:, :, self._o0 : self._o1],
            run[:, :, self._o1 : self._o2],
            run[:, :, self._o2 : self._o3],
            run[:, :, self._o3],
            run[:, :, self._o4],
            run[:, :, self._o5],
        )

    def read_final_span(self, count: int) -> list[FinalObsPayload | None]:
        """Copy host payloads belonging to the next issued read span."""
        if self.final_obs_store is None:
            return [None] * count
        return self.final_obs_store.read(self._read_issued_count, count)
