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

import torch

from motrix_rl.fastsac.async_impl.transport.common import _shared


class RingCursors:
    """Host shared-memory read/write cursors shared by both ring transports.

    Created once by the parent process and inherited by both workers. Each
    cursor has a single writer (the producer owns ``_write``, the consumer
    owns ``_read``), so plain aligned int64 stores publish progress coherently
    on x86/TSO without barriers.
    """

    def __init__(self):
        self._write = _shared((1,), torch.int64)
        self._read = _shared((1,), torch.int64)

    @property
    def write_idx(self) -> int:
        # Consumer reads this; single-writer producer, so a plain aligned int64
        # load is coherent. On x86/TSO the data loads that follow cannot be
        # reordered ahead of it, so no acquire barrier is needed (x86-only).
        return int(self._write[0])

    @property
    def read_idx(self) -> int:
        # Producer reads this; single-writer consumer. On x86/TSO the following
        # is_full() decision is based on an up-to-date value without a barrier.
        return int(self._read[0])


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

    @property
    def write_idx(self) -> int:
        return self.cursors.write_idx

    @property
    def read_idx(self) -> int:
        return self.cursors.read_idx

    def size(self) -> int:
        """Number of unread slots currently buffered."""
        return self.write_idx - self.read_idx

    def is_full(self) -> bool:
        return self.size() >= self.capacity

    def push(self, obs, critic_obs, actions, rewards, dones, truncations) -> bool:
        """Copy one env-step batch into the next slot. Returns False if full.

        Inputs are CPU tensors shaped ``(num_envs, dim)``; ``dones``/``truncations``
        are int64 to match ``SimpleReplayBuffer.extend`` semantics.
        """
        if self.is_full():
            return False
        slot = self.write_idx % self.capacity
        self.obs[slot].copy_(obs)
        self.critic_obs[slot].copy_(critic_obs)
        self.actions[slot].copy_(actions)
        self.rewards[slot].copy_(rewards)
        self.dones[slot].copy_(dones)
        self.truncations[slot].copy_(truncations)
        # Publish the slot. On x86/TSO the field copies above are guaranteed
        # visible before this cursor bump, so a consumer that reads the new
        # write_idx also sees the data (x86-only; ARM would need a release here).
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
        slot = self.read_idx % self.capacity
        count = min(size, self.capacity - slot)
        return count, (
            self.obs[slot : slot + count],
            self.critic_obs[slot : slot + count],
            self.actions[slot : slot + count],
            self.rewards[slot : slot + count],
            self.dones[slot : slot + count],
            self.truncations[slot : slot + count],
        )

    def commit_reads(self, n: int) -> None:
        """Advance the read cursor by ``n`` slots of a consumed contiguous run.

        On x86/TSO the consumer's reads complete before this cursor bump, so
        the producer's is_full() cannot reuse a slot still being copied out.
        """
        self.cursors._read[0] += n
