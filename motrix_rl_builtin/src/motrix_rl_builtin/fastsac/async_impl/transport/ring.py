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

from motrix_rl.transport.ring import RingCursors, SharedTensorRing, TensorField
from motrix_rl.transport.ring import shared_tensor as _shared


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
class SharedTransitionRing(SharedTensorRing):
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
        f32, i64 = torch.float32, torch.int64
        super().__init__(
            capacity,
            (
                TensorField("obs", (num_envs, obs_dim), f32),
                TensorField("critic_obs", (num_envs, critic_obs_dim), f32),
                TensorField("actions", (num_envs, act_dim), f32),
                TensorField("rewards", (num_envs,), f32),
                TensorField("dones", (num_envs,), i64),
                TensorField("truncations", (num_envs,), i64),
            ),
            cursors,
        )
        self.num_envs = num_envs
        self.obs, self.critic_obs, self.actions, self.rewards, self.dones, self.truncations = self.tensors
        self.final_obs_store = final_obs_store

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
        self.publish_write()
        return True

    def read_final_span(self, count: int) -> list[FinalObsPayload | None]:
        """Owned final payloads aligned with the current transition read span."""
        if self.final_obs_store is None:
            return [None] * count
        return self.final_obs_store.read(self.read_completed_count, count)
