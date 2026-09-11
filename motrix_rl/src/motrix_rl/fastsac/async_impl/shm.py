# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared-memory primitives for the async FastSAC trainer.

All tensors are CPU tensors marked ``share_memory_()`` so they can be handed to a
``torch.multiprocessing`` ``spawn`` child in M1 unchanged. In M0 everything runs
in one process; the same objects work in-process, which lets us validate the
collector/learner decomposition before introducing real concurrency.

Three primitives:

* :class:`SharedTransitionRing` — single-producer / single-consumer ring of raw
  transition batches (collector -> learner) with bounded backpressure.
* :class:`WeightSnapshot`  — double-buffered actor weights + obs-normalizer stats
  (learner -> collector) guarded by a seqlock so readers always see a complete,
  consistent snapshot even when the writer publishes twice during a read.
* :class:`Control`         — a few shared scalars (stop flag, global_step, ...).

Memory ordering
---------------
All cross-process safety here rests on two invariants:

1. **8-byte aligned loads/stores of int64 are hardware-atomic** on every ISA we
   target (x86-64, ARM64). Each cursor/counter below has a *single writer*, so
   an aligned 8-byte store is enough to publish a coherent value — no atomic
   RMW primitive (CAS / fetch_add) is needed, and a reader can never see a
   torn counter.

2. **Ordering between the counter and the data holds on strong-memory ISAs.**
   The "publish data, then bump counter" pattern is free on x86-64 (TSO: stores
   are not reordered with stores, loads not with loads), so a consumer that
   reads a bumped cursor is guaranteed to also see the slot's data stores.

   .. warning::
      x86-64 is the **only** platform class currently supported. On weak-memory
      ISAs (ARM64 — Jetson, Grace, Apple Silicon) the CPU *may* reorder a data
      store after the counter store, letting a consumer observe a bumped cursor
      while the slot's data stores are still in flight — a torn snapshot. This
      code inserts no memory barriers (there is no portable standalone fence in
      the Python stdlib, and the ``atomics`` package offers only ordered
      load/store on an ``atomicview``, with no x86 wheel). Supporting ARM would
      mean routing every cursor / ``_seq`` store and load through an
      ``atomicview`` with ``MemoryOrder.RELEASE`` / ``.ACQUIRE``.
"""

from __future__ import annotations

import time

import torch
from torch import nn


def _shared(shape, dtype) -> torch.Tensor:
    """Allocate a zero CPU tensor in shared memory."""
    return torch.zeros(shape, dtype=dtype).share_memory_()


# ---------------------------------------------------------------- transition ring
class SharedTransitionRing:
    """SPSC ring of transition batches with bounded backpressure.

    Each slot holds one env-step batch: the eight tensors that
    :meth:`motrix_rl.fastsac.buffer.SimpleReplayBuffer.extend` consumes, with the
    leading dimension being ``num_envs`` (so a slot is ``(num_envs, dim)``).

    Producer (collector) calls :meth:`push`; when the ring is full it returns
    ``False`` and the caller must retry/backoff — that is the backpressure that
    keeps the collector from outrunning the learner and flooding memory.

    Consumer (learner) calls :meth:`read_slot` to get zero-copy CPU views of the
    oldest unread slot, moves them to its device, then calls :meth:`commit_read`.
    The read cursor only advances after the copy, so the producer can never
    clobber a slot that is still being ingested (``push`` blocks while the ring
    is full).

    Memory ordering
    ~~~~~~~~~~~~~~~
    Only ``_write`` is read by the consumer, only ``_read`` is read by the
    producer, and each cursor has a single writer — so an aligned int64 store
    is enough to publish progress and no atomic RMW is needed. Correctness also
    needs the consumer to not observe the ``_write`` bump before the slot's data
    stores have landed (and symmetrically for ``_read``); on x86/TSO that
    ordering is free, so no memory barrier is used and this path is x86-only
    (see the module "Memory ordering" note).
    """

    FIELDS = (
        "obs",
        "critic_obs",
        "actions",
        "rewards",
        "dones",
        "truncations",
        "next_obs",
        "next_critic_obs",
    )

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
    ):
        self.capacity = capacity
        self.num_envs = num_envs
        f32, i64 = torch.float32, torch.int64
        self.obs = _shared((capacity, num_envs, obs_dim), f32)
        self.critic_obs = _shared((capacity, num_envs, critic_obs_dim), f32)
        self.actions = _shared((capacity, num_envs, act_dim), f32)
        self.rewards = _shared((capacity, num_envs), f32)
        self.dones = _shared((capacity, num_envs), i64)
        self.truncations = _shared((capacity, num_envs), i64)
        self.next_obs = _shared((capacity, num_envs, obs_dim), f32)
        self.next_critic_obs = _shared((capacity, num_envs, critic_obs_dim), f32)
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

    def push(self, obs, critic_obs, actions, rewards, dones, truncations, next_obs, next_critic_obs) -> bool:
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
        self.next_obs[slot].copy_(next_obs)
        self.next_critic_obs[slot].copy_(next_critic_obs)
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
            self.next_obs[slot],
            self.next_critic_obs[slot],
        )

    def commit_read(self) -> None:
        # Free the slot. On x86/TSO our reads above complete before this cursor
        # bump, so the producer's is_full() cannot reuse a slot we are still
        # copying out (x86-only; ARM would need a release here).
        self._read[0] += 1


# ---------------------------------------------------------------- weight snapshot
def flatten_params(module: nn.Module) -> torch.Tensor:
    """Flatten a module's parameters into a single CPU float vector (in order)."""
    return torch.cat([p.detach().reshape(-1).float() for p in module.parameters()]).cpu()


def flatten_buffers(module: nn.Module) -> torch.Tensor:
    """Flatten persistent non-empty module buffers."""
    values = [buffer.detach().reshape(-1).float().cpu() for buffer in module.buffers() if buffer.numel()]
    return torch.cat(values) if values else torch.empty(0)


def load_flat_params(module: nn.Module, flat: torch.Tensor) -> None:
    """Inverse of :func:`flatten_params`; copies a flat vector into the params."""
    offset = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat[offset : offset + n].view_as(p).to(p.device))
        offset += n


def load_flat_buffers(module: nn.Module, flat: torch.Tensor) -> None:
    """Copy a flat snapshot into persistent non-empty module buffers."""
    offset = 0
    for buffer in module.buffers():
        if not buffer.numel():
            continue
        size = buffer.numel()
        buffer.data.copy_(flat[offset : offset + size].view_as(buffer).to(buffer.device))
        offset += size


def bind_flat_params(module: nn.Module) -> torch.Tensor:
    """Bind all module parameters to views of one device-contiguous flat tensor.

    The collector owns an inference-only actor, so its parameters do not need
    optimizer storage. Binding them once lets every later CPU snapshot version
    reach CUDA through one H2D copy instead of one transfer per parameter.
    """
    params = list(module.parameters())
    flat = torch.empty(sum(p.numel() for p in params), dtype=params[0].dtype, device=params[0].device)
    offset = 0
    with torch.no_grad():
        for param in params:
            view = flat[offset : offset + param.numel()].view_as(param)
            view.copy_(param)
            param.data = view
            offset += param.numel()
    return flat


# obs-normalizer stat buffers that the collector needs (read-only) to reproduce
# the sync ``act()`` path (normalize with update=False, see agent.act).
_NORM_KEYS = ("_mean", "_std", "_var", "count")


class WeightSnapshot:
    """Double-buffered actor weights + obs-normalizer stats, learner -> collector.

    The naive double buffer ("writer publishes to the other slot, reader reads
    the current one") is **not** race-free on its own: if the learner publishes
    twice while a collector is mid-``maybe_load``, the second publish reuses the
    slot the collector is still copying from — producing a torn snapshot that
    then drives the policy for thousands of steps. Hard to reproduce, expensive
    to debug.

    We guard the buffer with a **seqlock**: a shared counter ``_seq`` that the
    writer bumps to *odd* while writing and back to *even* when done. Readers
    spin until they observe a stable even value that did not change across the
    read, retrying on any inconsistency. Because the only writer is the learner
    (one process), seqlock gives us:

    * **Lock-free, CAS-free reads.** Plain aligned int64 loads/stores suffice;
      no atomic RMW primitive is needed. Readers almost never retry.
    * **No collision on the common path.** A single publish during a read
      targets the *other* slot, so the seq value doesn't even change and the
      reader's read is consistent on the first try.
    * **Safe detection of the rare race.** A second publish during a read bumps
      ``_seq`` twice; the reader notices (``s1 != s2``) and retries from
      scratch, never exposing a torn snapshot to the actor.

    Memory ordering
    ~~~~~~~~~~~~~~~
    Correctness needs the writer's slot-data stores visible before the seq
    change, and the reader's data loads not hoisted above the seq read. On
    x86/TSO that holds for free, so no memory barrier is used and this path is
    x86-only (see the module "Memory ordering" note); ARM would need real
    release/acquire around the seq bumps and reads.
    """

    def __init__(self, param_numel: int, obs_dim: int, buffer_numel: int = 0):
        self.param_numel = param_numel
        self.buffer_numel = buffer_numel
        # Two slots for params + normalizer stats (double buffer).
        self._params = [_shared((param_numel,), torch.float32) for _ in range(2)]
        self._buffers = [_shared((buffer_numel,), torch.float32) for _ in range(2)]
        self._mean = [_shared((1, obs_dim), torch.float32) for _ in range(2)]
        self._std = [_shared((1, obs_dim), torch.float32) for _ in range(2)]
        self._var = [_shared((1, obs_dim), torch.float32) for _ in range(2)]
        self._count = [_shared((1,), torch.int64) for _ in range(2)]
        # Seqlock counter. Even = quiescent, odd = write in progress.
        # One writer (learner) → aligned int64 stores suffice; the seq value
        # itself can never tear. Publish/observe ordering relies on x86/TSO (see
        # the module "Memory ordering" note; no barriers, x86-only).
        # Public version = seq // 2; starts at 0 (matching the collector's
        # ``_local_version = 0`` initial state so the first publish is seen).
        self._seq = _shared((1,), torch.int64)

    @property
    def version(self) -> int:
        """Number of completed publishes (= seq // 2). Best-effort; readers
        that need a consistent snapshot must go through :meth:`maybe_load`."""
        return int(self._seq[0]) // 2

    def publish(self, actor: nn.Module, obs_normalizer: nn.Module) -> None:
        """Write current actor params + normalizer stats to the inactive slot
        and flip the active pointer via the seqlock."""
        # Materialize the learner's CUDA state on CPU before making the seqlock
        # odd. Collector readers may keep using the previous complete version
        # while these device-to-host copies finish.
        params = flatten_params(actor)
        buffers = flatten_buffers(actor)
        has_stats = all(hasattr(obs_normalizer, k) for k in _NORM_KEYS)
        if has_stats:
            mean = obs_normalizer._mean.detach().cpu()
            std = obs_normalizer._std.detach().cpu()
            var = obs_normalizer._var.detach().cpu()
            count = int(obs_normalizer.count)

        prev = int(self._seq[0])
        # 1) Mark "write in progress" (odd). Readers seeing this retry.
        self._seq[0] = prev + 1
        # 2) Write to the slot opposite the previously-active one
        #    (active slot before this publish was (prev // 2) % 2). Two
        #    publishes in a row therefore alternate slots, so a single
        #    publish during a read targets the *other* slot — no collision
        #    even without the seqlock guard. The guard exists for the case
        #    of two publishes during one read.
        slot = ((prev // 2) + 1) % 2
        self._params[slot].copy_(params)
        if self.buffer_numel:
            self._buffers[slot].copy_(buffers)
        if has_stats:
            self._mean[slot].copy_(mean)
            self._std[slot].copy_(std)
            self._var[slot].copy_(var)
            self._count[slot][0] = count
        # 3) Bump seq back to even. On x86/TSO the data stores above are visible
        #    before this store, so a reader that observes the new even value
        #    sees all writes above. Readers that observed the odd value retry.
        self._seq[0] = prev + 2

    def maybe_load(
        self,
        actor: nn.Module,
        obs_normalizer: nn.Module,
        local_version: int,
        *,
        param_staging: torch.Tensor,
        normalizer_staging: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        buffer_staging: torch.Tensor | None = None,
        flat_params: torch.Tensor | None = None,
    ) -> tuple[int, float, float, float]:
        """Load the latest snapshot into ``actor``/``obs_normalizer`` if newer.

        Returns the version actually loaded (== ``local_version`` if nothing
        new), followed by wall-clock seconds spent waiting for an in-progress
        writer, copying a stable host snapshot, and loading it onto the actor's
        device. The breakdown lets the collector distinguish shared-memory
        publication contention from actor-load enqueue work.

        Implements the seqlock read loop: read ``_seq``, ensure it is even and
        stable, copy the data out, then re-check ``_seq``. If the writer
        published in between we discard the (possibly torn) copy and retry.
        In practice retries are extremely rare — publish cadence is bounded by
        ``weight_publish_interval`` gradient steps, while a read is one CPU
        memcpy.
        """
        wait_writer_s = 0.0
        host_snapshot_s = 0.0
        while True:
            s1 = int(self._seq[0])
            if s1 & 1:  # writer is mid-publish; wait for a completed version.
                wait_start = time.perf_counter()
                while s1 & 1:
                    s1 = int(self._seq[0])
                wait_writer_s += time.perf_counter() - wait_start
            version = s1 // 2
            if version <= local_version:  # nothing new to load
                return local_version, wait_writer_s, host_snapshot_s, 0.0
            slot = version % 2  # active slot at this seq
            # Snapshot the slot into local tensors first. We do not load
            # directly into the actor because load_flat_params performs
            # many small per-param copies that could each see a different
            # seq state; we want one consistent view of the whole slot.
            snapshot_start = time.perf_counter()
            param_staging.copy_(self._params[slot])
            if self.buffer_numel:
                if buffer_staging is None:
                    raise ValueError("buffer_staging is required for actor buffers")
                buffer_staging.copy_(self._buffers[slot])
            has_stats = all(hasattr(obs_normalizer, k) for k in _NORM_KEYS)
            if has_stats:
                mean, std, var = normalizer_staging
                mean.copy_(self._mean[slot])
                std.copy_(self._std[slot])
                var.copy_(self._var[slot])
                count = int(self._count[slot][0])
            # Re-check seq. If it changed, the writer published (at least
            # once) during our copy and the slot may have been overwritten
            # — discard and retry. This is the line that closes the race
            # the old double-buffer had.
            s2 = int(self._seq[0])
            host_snapshot_s += time.perf_counter() - snapshot_start
            if s1 != s2:
                continue

            actor_load_start = time.perf_counter()
            if flat_params is None:
                load_flat_params(actor, param_staging)
            else:
                flat_params.copy_(param_staging, non_blocking=True)
            if self.buffer_numel:
                load_flat_buffers(actor, buffer_staging)
            if has_stats:
                obs_normalizer._mean.copy_(mean, non_blocking=True)
                obs_normalizer._std.copy_(std, non_blocking=True)
                obs_normalizer._var.copy_(var, non_blocking=True)
                obs_normalizer.count.fill_(count)
            # CUDA collectors keep the host staging buffers alive. The copies
            # above are ordered before the next inference on the same stream;
            # that inference already synchronizes after returning actions to
            # the CPU environment, so a separate weight-load barrier only
            # serializes collector and learner work unnecessarily.
            actor_load_s = time.perf_counter() - actor_load_start
            return version, wait_writer_s, host_snapshot_s, actor_load_s


# ---------------------------------------------------------------- control block
class Control:
    """A handful of shared scalar controls / counters."""

    def __init__(self):
        self._stop = _shared((1,), torch.int64)
        self._global_step = _shared((1,), torch.int64)  # learner iteration counter
        self._collector_steps = _shared((1,), torch.int64)  # env-step batches produced

    @property
    def stop(self) -> bool:
        return bool(self._stop[0])

    def set_stop(self) -> None:
        self._stop[0] = 1

    @property
    def global_step(self) -> int:
        return int(self._global_step[0])

    @global_step.setter
    def global_step(self, v: int) -> None:
        self._global_step[0] = v

    @property
    def collector_steps(self) -> int:
        return int(self._collector_steps[0])

    @collector_steps.setter
    def collector_steps(self, v: int) -> None:
        self._collector_steps[0] = v

    def inc_collector_steps(self) -> None:
        self._collector_steps[0] += 1
