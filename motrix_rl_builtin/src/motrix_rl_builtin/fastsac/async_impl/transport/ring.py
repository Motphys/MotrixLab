# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SPSC transition ring between the collector and learner processes.

The ring protocol (cursor semantics, backpressure) is shared by two physical
transports:

* host shared-memory fields (this module, :class:`SharedTransitionRing`) —
  the universal fallback;
* CUDA-IPC device fields (``ipc_ring.py``, :class:`IpcTransitionRing`) —
  used when collector inference and the learner share one GPU.

Both transports advance the same parent-created :class:`RingCursors`, so the
producer/consumer contract below is transport-independent.

Producer (collector) calls :meth:`push`; when the ring is full it returns
``False`` and the caller must retry/backoff — that is the backpressure that
keeps the collector from outrunning the learner and flooding memory.

Consumer (learner) calls :meth:`read_span` to get zero-copy views of the
longest contiguous unread run of slots, moves them to its replay buffer, then
calls :meth:`commit_reads`. The read cursor only advances after the copy, so
the producer can never clobber a slot that is still being ingested.

Memory ordering
~~~~~~~~~~~~~~~
Only ``_write`` is read by the consumer, only ``_read`` is read by the
producer, and each cursor has a single writer — so an aligned int64 store
is enough to publish progress and no atomic RMW is needed. Correctness also
needs the consumer to not observe the ``_write`` bump before the slot's data
stores have landed (and symmetrically for ``_read``); on x86/TSO that
ordering is free, so no memory barrier is used and this path is x86-only
(see the module "Memory ordering" note). The CUDA-IPC transport instead
orders GPU writes behind per-slot events before bumping a cursor (see
``ipc_ring.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from motrix_rl_builtin.fastsac.async_impl.transport.common import _shared


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


@dataclass
class FinalObsPayload:
    """Owned compact CPU observation rows belonging to one transition slot."""

    env_ids: torch.Tensor
    obs: torch.Tensor
    critic_obs: torch.Tensor


class FinalObsStorage:
    """Shared sparse rows whose lifetime is governed by the transition cursors.

    Only the producer manages row allocation. Every slot records an absolute
    row end, including slots without a payload. Observing an acknowledged slot
    releases its rows; no independently published or consumed cursor exists.
    """

    def __init__(self, capacity: int, num_envs: int, obs_dim: int, critic_obs_dim: int):
        self.slot_capacity = capacity
        self.capacity = 3 * num_envs
        self.env_ids = _shared((self.capacity,), torch.int64)
        self.obs = _shared((self.capacity, obs_dim), torch.float32)
        self.critic_obs = _shared((self.capacity, critic_obs_dim), torch.float32)
        self.starts = _shared((capacity,), torch.int64)
        self.counts = _shared((capacity,), torch.int64)
        self.ends = _shared((capacity,), torch.int64)
        self._row_write = 0
        self._row_read = 0
        self._reclaimed_slots = 0

    def can_write(self, payload: FinalObsPayload | None, acknowledged: int) -> bool:
        n = 0 if payload is None else len(payload.env_ids)
        if n > self.capacity:
            raise ValueError("final payload exceeds the sparse row store capacity")
        return self.can_write_rows(n, acknowledged)

    def can_write_rows(self, count: int, acknowledged: int) -> bool:
        """Reclaim completed slots and check space for a prospective row count."""
        # Reclaim before the producer can overwrite any slot metadata. The
        # main ring's capacity guarantees these ends still belong to the
        # acknowledged generation. IPC callers pass event-proven cursors.
        while self._reclaimed_slots < acknowledged:
            slot = self._reclaimed_slots % self.slot_capacity
            self._row_read = int(self.ends[slot])
            self._reclaimed_slots += 1
        return self._row_write - self._row_read + count <= self.capacity

    def write(self, index: int, payload: FinalObsPayload | None) -> None:
        """Write payload before the owning transition slot is published."""
        slot = index % self.slot_capacity
        n = 0 if payload is None else len(payload.env_ids)
        start = self._row_write
        if n:
            rows = (start + torch.arange(n, dtype=torch.int64)) % self.capacity
            self.env_ids[rows] = torch.as_tensor(payload.env_ids, dtype=torch.int64)
            self.obs[rows] = torch.as_tensor(payload.obs, dtype=torch.float32)
            self.critic_obs[rows] = torch.as_tensor(payload.critic_obs, dtype=torch.float32)
        self.starts[slot] = start
        self.counts[slot] = n
        self.ends[slot] = start + n
        self._row_write = start + n

    def read(self, index: int, count: int) -> list[FinalObsPayload | None]:
        """Copy all payload rows before the caller acknowledges their slots."""
        result = []
        for offset in range(count):
            slot = (index + offset) % self.slot_capacity
            n = int(self.counts[slot])
            if not n:
                result.append(None)
                continue
            rows = (int(self.starts[slot]) + torch.arange(n, dtype=torch.int64)) % self.capacity
            # Advanced indexing produces owned snapshots, including on wrap.
            result.append(FinalObsPayload(self.env_ids[rows], self.obs[rows], self.critic_obs[rows]))
        return result


# ---------------------------------------------------------------- transition ring
class SharedTransitionRing:
    """SPSC ring of transition batches with bounded backpressure (host fields).

    Each slot holds one env-step batch: the six tensors produced by one
    collector step, with the leading dimension being ``num_envs`` (so a slot
    is ``(num_envs, dim)``). ``next_obs``/``next_critic_obs`` are NOT stored:
    with auto-reset envs the observation returned by step ``t`` is exactly the
    stored observation of step ``t+1`` (at episode ends it is the reset
    observation), so the consumer's replay buffer derives them from the
    successive slot. This halves the per-slot copy volume and shared-memory
    footprint.

    Fields live in host shared memory, usable by any collector/learner device
    combination. See ``ipc_ring.py`` for the CUDA-IPC device-fields variant
    sharing the same cursor protocol.
    """

    FIELDS = (
        "obs",
        "critic_obs",
        "actions",
        "rewards",
        "dones",
        "truncations",
    )

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
        cursors: RingCursors | None = None,
        final_obs_store: FinalObsStorage | None = None,
    ):
        if capacity < 1:
            raise ValueError(f"ring_capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self.num_envs = num_envs
        f32, i64 = torch.float32, torch.int64
        self.obs = _shared((capacity, num_envs, obs_dim), f32)
        self.critic_obs = _shared((capacity, num_envs, critic_obs_dim), f32)
        self.actions = _shared((capacity, num_envs, act_dim), f32)
        self.rewards = _shared((capacity, num_envs), f32)
        self.dones = _shared((capacity, num_envs), i64)
        self.truncations = _shared((capacity, num_envs), i64)
        self.cursors = cursors if cursors is not None else RingCursors()
        self.final_obs_store = final_obs_store

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
        """Whether slots or worst-case final rows prevent another env step."""
        if self.size() >= self.capacity:
            return True
        store = self.final_obs_store
        return store is not None and not store.can_write_rows(self.num_envs, self.read_completed_count)

    def push(
        self, obs, critic_obs, actions, rewards, dones, truncations, final_obs: FinalObsPayload | None = None
    ) -> bool:
        """Copy one env-step batch into the next slot. Returns False if full.

        Inputs are CPU tensors shaped ``(num_envs, dim)``; ``dones``/``truncations``
        are int64 to match ``SimpleReplayBuffer.extend`` semantics.
        """
        if self.is_full():
            return False
        store = self.final_obs_store
        if final_obs is not None and store is None:
            raise ValueError("final payload requires a slot final-observation store")
        if store is not None and not store.can_write(final_obs, self.read_completed_count):
            return False
        index = self.write_completed_count
        slot = index % self.capacity
        self.obs[slot].copy_(obs)
        self.critic_obs[slot].copy_(critic_obs)
        self.actions[slot].copy_(actions)
        self.rewards[slot].copy_(rewards)
        self.dones[slot].copy_(dones)
        self.truncations[slot].copy_(truncations)
        if store is not None:
            store.write(index, final_obs)
        # Publish the slot. On x86/TSO the field copies above are guaranteed
        # visible before this cursor bump, so a consumer that reads the new
        # write_completed_count also sees the data (x86-only; ARM would need a release here).
        self.cursors._write[0] += 1
        return True

    def has_next(self) -> bool:
        """Whether at least one unread slot is committed (i.e. readable)."""
        return self.size() > 0

    def read_span(self) -> tuple[int, tuple]:
        """Longest contiguous unread run, as ``(count, field_views)``.

        ``field_views`` is the six per-field tensors of shape
        ``(count, num_envs, dim)`` (``(count, num_envs)`` for the scalar
        fields) — contiguous in the slot (leading) dimension, so the consumer
        can move the whole run with one copy per field. Returns ``(0, ())``
        when empty. Does NOT advance the read cursor; free the run with
        :meth:`commit_reads` once the data has been copied out.
        """
        size = self.size()
        if size <= 0:
            return 0, ()
        slot = self.read_completed_count % self.capacity
        count = min(size, self.capacity - slot)
        return count, (
            self.obs[slot : slot + count],
            self.critic_obs[slot : slot + count],
            self.actions[slot : slot + count],
            self.rewards[slot : slot + count],
            self.dones[slot : slot + count],
            self.truncations[slot : slot + count],
        )

    def read_final_span(self, count: int) -> list[FinalObsPayload | None]:
        """Owned final payloads aligned with the current transition read span."""
        if self.final_obs_store is None:
            return [None] * count
        return self.final_obs_store.read(self.read_completed_count, count)

    def commit_reads(self, n: int) -> None:
        """Advance the read cursor by ``n`` slots of a consumed contiguous run.

        On x86/TSO the consumer's reads complete before this cursor bump, so
        the producer's is_full() cannot reuse a slot still being copied out.
        """
        self.cursors._read[0] += n
