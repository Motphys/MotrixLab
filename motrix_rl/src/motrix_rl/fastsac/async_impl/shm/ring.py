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
    observation), so the consumer's replay buffer derives them from the
    successive slot. This halves the per-slot copy volume and shared-memory
    footprint.

    Producer (collector) calls :meth:`push`; when the ring is full it returns
    ``False`` and the caller must retry/backoff — that is the backpressure that
    keeps the collector from outrunning the learner and flooding memory.

    Consumer (learner) calls :meth:`read_slot` to get zero-copy views of the
    oldest unread slot, moves them to its device, then calls :meth:`commit_read`.
    The read cursor only advances after the copy, so the producer can never
    clobber a slot that is still being ingested (``push`` blocks while the ring
    is full).

    Transport
    ~~~~~~~~~
    By default every field lives in host shared memory. When learner and
    collector share one GPU, the two large fields (``obs``/``critic_obs``) can
    instead live in CUDA-IPC device slots allocated by the collector
    (:meth:`init_device_slots`) and bound by the learner
    (:meth:`bind_device_slots`): the collector pushes device-side copies of
    observations it already uploaded for inference (one D2D copy per field)
    and the learner drains straight into its device replay buffer without any
    host crossing. The small fields stay in host shared memory either way —
    they are a few dozen kilobytes per slot.

    Memory ordering
    ~~~~~~~~~~~~~~~
    Only ``_write`` is read by the consumer, only ``_read`` is read by the
    producer, and each cursor has a single writer — so an aligned int64 store
    is enough to publish progress and no atomic RMW is needed. Correctness also
    needs the consumer to not observe the ``_write`` bump before the slot's data
    stores have landed (and symmetrically for ``_read``); on x86/TSO that
    ordering is free, so no memory barrier is used and this path is x86-only
    (see the module "Memory ordering" note).

    With device slots the data stores are asynchronous GPU copies, so the
    producer records and synchronizes a CUDA event before bumping ``_write``
    (device-globally visible once complete — both processes alias the same
    device memory). Symmetrically the consumer must NOT call
    :meth:`commit_read` right after enqueueing its asynchronous reads; it keeps
    per-slot events and retires commits only once the copies have completed
    (see the learner's device drain). The read cursor therefore lags by at most
    a few in-flight slots, which only makes backpressure fire slightly early.
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
        # cursors are shared so the two processes see each other's progress.
        self._write = _shared((1,), i64)
        self._read = _shared((1,), i64)
        # CUDA-IPC device slots for the two large fields (None -> host path).
        self._device_obs: list[torch.Tensor] | None = None
        self._device_critic_obs: list[torch.Tensor] | None = None
        self._push_event: torch.cuda.Event | None = None

    # ------------------------------------------------------------ device transport
    @property
    def device_slots_bound(self) -> bool:
        """Whether the large fields are served from CUDA-IPC device slots."""
        return self._device_obs is not None

    def init_device_slots(self, device: torch.device) -> tuple[list, list]:
        """Collector: allocate device-side slots for obs/critic_obs.

        Returns the slot tensor lists so the caller can ship them to the
        learner through a multiprocessing queue (reduction turns them into
        cudaIpcMemHandles, so both processes alias the same device memory).
        The exporter (this process) must keep the ring alive for the process
        lifetime — it owns the allocation.
        """
        if device.type != "cuda":
            raise ValueError(f"device ring slots require a CUDA device, got {device}")
        self._device_obs = [torch.empty_like(self.obs[0], device=device) for _ in range(self.capacity)]
        self._device_critic_obs = [torch.empty_like(self.critic_obs[0], device=device) for _ in range(self.capacity)]
        self._push_event = torch.cuda.Event()
        return self._device_obs, self._device_critic_obs

    def bind_device_slots(self, device_obs: list, device_critic_obs: list) -> None:
        """Learner: bind the collector's device slots received via IPC handles."""
        self._device_obs = device_obs
        self._device_critic_obs = device_critic_obs

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

        ``obs``/``critic_obs`` are either CPU tensors (host transport: every
        field is memcpy'd into host shared memory) or CUDA tensors on the
        ring's device (device transport: the two large fields are copied
        device-side and only the small fields touch host shared memory).
        ``dones``/``truncations`` are int64 to match
        ``SimpleReplayBuffer.extend`` semantics.
        """
        if self.is_full():
            return False
        slot = self.write_idx % self.capacity
        if obs.is_cuda:
            assert self._device_obs is not None, "device tensors pushed but no device slots initialized"
            self._device_obs[slot].copy_(obs)
            self._device_critic_obs[slot].copy_(critic_obs)
            # The cursor bump below is a CPU store, but the slot writes above
            # are asynchronous GPU copies: complete (and thus device-globally
            # visible to the learner's alias of this memory) before publishing
            # the slot.
            self._push_event.record()
            self._push_event.synchronize()
        else:
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
        """Return views of the oldest unread slot, or ``None`` if empty.

        The obs/critic_obs views alias device memory when device slots are
        bound (learner side) and host shared memory otherwise. Does NOT
        advance the read cursor; call :meth:`commit_read` after the consumer
        has finished copying the data elsewhere — with device slots that means
        after the enqueued reads have *completed*, not merely been enqueued.
        """
        if self.size() <= 0:
            return None
        slot = self.read_idx % self.capacity
        obs = self._device_obs[slot] if self._device_obs is not None else self.obs[slot]
        critic_obs = (
            self._device_critic_obs[slot] if self._device_critic_obs is not None else self.critic_obs[slot]
        )
        return (
            obs,
            critic_obs,
            self.actions[slot],
            self.rewards[slot],
            self.dones[slot],
            self.truncations[slot],
        )

    def has_next(self) -> bool:
        """Whether at least one unread slot is committed (i.e. readable)."""
        return self.size() > 0

    def commit_read(self) -> None:
        # Free the slot. With host slots our reads above complete before this
        # cursor bump on x86/TSO, so the producer's is_full() cannot reuse a
        # slot we are still copying out. With device slots the caller is
        # responsible for proving its async reads retired first.
        self._read[0] += 1
