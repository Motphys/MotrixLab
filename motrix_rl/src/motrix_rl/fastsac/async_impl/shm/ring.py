# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SPSC transition ring between the collector and learner processes."""

from __future__ import annotations

import torch

from motrix_rl.fastsac.async_impl.shm.common import _shared


# ---------------------------------------------------------------- transition ring
class SharedTransitionRing:
    """SPSC ring of transition batches with bounded backpressure.

    Each slot holds one env-step batch: the six tensors produced by one
    collector step, with the leading dimension being ``num_envs`` (so a slot
    is ``(num_envs, dim)``). ``next_obs``/``next_critic_obs`` are NOT stored:
    with auto-reset envs the observation returned by step ``t`` is exactly the
    stored observation of step ``t+1`` (at episode ends it is the reset
    observation), so the consumer derives them from the successive slot via
    :meth:`has_next`/:meth:`peek_next`. This halves the per-slot copy volume
    and shared-memory footprint.

    Producer (collector) calls :meth:`push`; when the ring is full it returns
    ``False`` and the caller must retry/backoff — that is the backpressure that
    keeps the collector from outrunning the learner and flooding memory.

    Consumer (learner) calls :meth:`read_slot` to get zero-copy CPU views of the
    oldest unread slot plus :meth:`peek_next` for the derived next-observation
    views, moves them to its device, then calls :meth:`commit_read`. The read
    cursor only advances after the copy, so the producer can never clobber a
    slot that is still being ingested (``push`` blocks while the ring is full).

    Memory ordering
    ~~~~~~~~~~~~~~~
    Only ``_write`` is read by the consumer, only ``_read`` is read by the
    producer, and each cursor has a single writer — so an aligned int64 store
    is enough to publish progress and no atomic RMW is needed. Correctness also
    needs the consumer to not observe the ``_write`` bump before the slot's data
    stores have landed (and symmetrically for ``_read``); on x86/TSO that
    ordering is free, so no memory barrier is used and this path is x86-only
    (see the module "Memory ordering" note). The same guarantee covers
    :meth:`peek_next`: the producer wrote the successor slot's data before
    publishing it, which is a precondition of the consumer seeing it committed.
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
    ):
        # The consumer derives next_obs from the successor slot, so capacity 1
        # can never satisfy has_next() and would deadlock the pipeline.
        if capacity < 2:
            raise ValueError(f"ring_capacity must be >= 2 (consumer reads the successor slot), got {capacity}")
        self.capacity = capacity
        self.num_envs = num_envs
        f32, i64 = torch.float32, torch.int64
        self.obs = _shared((capacity, num_envs, obs_dim), f32)
        self.critic_obs = _shared((capacity, num_envs, critic_obs_dim), f32)
        self.actions = _shared((capacity, num_envs, act_dim), f32)
        self.rewards = _shared((capacity, num_envs), f32)
        self.dones = _shared((capacity, num_envs), i64)
        self.truncations = _shared((capacity, num_envs), i64)
        # cursors are shared so the two processes see each other's progress.
        self._write = _shared((1,), i64)
        self._read = _shared((1,), i64)

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
        self._write[0] += 1
        return True

    def read_slot(self):
        """Return CPU views of the oldest unread slot, or ``None`` if empty.

        Does NOT advance the read cursor; call :meth:`commit_read` after the
        consumer has finished copying the data elsewhere.
        """
        if self.size() <= 0:
            return None
        slot = self.read_idx % self.capacity
        return (
            self.obs[slot],
            self.critic_obs[slot],
            self.actions[slot],
            self.rewards[slot],
            self.dones[slot],
            self.truncations[slot],
        )

    def has_next(self) -> bool:
        """Whether the successor of the oldest unread slot is already committed.

        The consumer needs it to derive ``next_obs``/``next_critic_obs`` for the
        oldest slot (see :meth:`peek_next`), so it must wait for one extra
        committed slot before ingesting.
        """
        return self.size() > 1

    def peek_next(self):
        """Return CPU views of the successor slot's obs/critic_obs.

        Valid only when :meth:`has_next` is true; the views alias ring memory
        that stays untouched until the consumer's own ``commit_read`` calls
        advance past it.
        """
        slot = (self.read_idx + 1) % self.capacity
        return self.obs[slot], self.critic_obs[slot]

    def commit_read(self) -> None:
        # Free the slot. On x86/TSO our reads above complete before this cursor
        # bump, so the producer's is_full() cannot reuse a slot we are still
        # copying out (x86-only; ARM would need a release here).
        self._read[0] += 1
