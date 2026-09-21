# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Single-producer / single-consumer weight channel (learner -> collector).

The learner publishes actor weights + obs-normalizer stats; the collector
loads the latest published version into its inference-only actor copy. The
protocol state (normalizer-stat double buffers + seqlock counter) is created
once by the parent process in :class:`WeightChannelShared` and pickled into
both worker processes, where the concrete endpoints are constructed:

* the learner builds a :class:`WeightSender` subclass — host shared-memory
  slots anywhere, or CUDA-IPC device slots in the learner process (which owns
  the CUDA context and must keep the IPC tensors alive) — and the orchestration
  ships the slot tensors to the collector over a one-shot queue;
* the collector builds the matching :class:`WeightReceiver` subclass from the
  arrived tensors (:func:`weight_receiver_for` dispatches on their device).

The endpoints themselves know nothing about queues.

Seqlock protocol
----------------
The naive double buffer ("writer publishes to the other slot, reader reads
the current one") is **not** race-free on its own: if the learner publishes
twice while the collector is mid-read, the second publish reuses the slot the
collector is still copying from — producing a torn snapshot that then drives
the policy for thousands of steps. The buffer is guarded by a **seqlock**: a
shared counter that the writer bumps to *odd* while writing and back to
*even* when done; readers retry on any inconsistency. Single writer + single
reader means plain aligned int64 stores suffice and no atomic RMW is needed.
On x86/TSO the data-before-counter ordering is free; ARM would need real
release/acquire (see the shm module's "Memory ordering" note).

Transport choice
----------------
Host shared memory is the default: it is sub-millisecond for small actors and
avoids any GPU synchronization. CUDA-IPC device slots win only for large
actors — both directions pay a stream/event synchronization to preserve the
seqlock's "data visible before version bump" ordering, which is a net win
once the avoided host transfer dominates that barrier (the orchestration
gates this on async_options.weight_ipc and the actor parameter size).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import torch
import torch.multiprocessing  # noqa: F401  registers CUDA-IPC reducers in every importing process
from torch import nn

from motrix_rl.fastsac.async_impl.shm.common import _NORM_KEYS, _shared, flatten_params, load_flat_params

# obs-normalizer stat buffers that the collector needs (read-only) to reproduce
# the sync ``act()`` path (normalize with update=False, see agent.act).


class WeightChannelShared:
    """Parent-created protocol state shared by both channel endpoints."""

    def __init__(self, obs_dim: int):
        # Two slots for normalizer stats (double buffer) + the seqlock counter.
        self.mean = [_shared((1, obs_dim), torch.float32) for _ in range(2)]
        self.std = [_shared((1, obs_dim), torch.float32) for _ in range(2)]
        self.var = [_shared((1, obs_dim), torch.float32) for _ in range(2)]
        self.count = [_shared((1,), torch.int64) for _ in range(2)]
        # Even = quiescent, odd = write in progress. Public version = seq // 2;
        # starts at 0 (matching the receiver's ``loaded_version = 0`` initial
        # state so the first publish is seen).
        self.seq = _shared((1,), torch.int64)


class WeightSender(ABC):
    """Learner-side endpoint; publishes snapshots under the seqlock."""

    def __init__(self, shared: WeightChannelShared):
        self.shared = shared

    @property
    def version(self) -> int:
        """Number of completed publishes (= seq // 2)."""
        return int(self.shared.seq[0]) // 2

    #: The double-buffered slot tensors. Construction ships nothing by itself;
    #: the orchestration sends these to the collector to build its receiver.
    params: list[torch.Tensor]

    @abstractmethod
    def _write_slot(self, slot: int, device_flat: torch.Tensor) -> None:
        """Start writing the flattened params into a slot (may be asynchronous)."""

    @abstractmethod
    def _complete_slot_write(self) -> None:
        """Block until the last :meth:`_write_slot` has fully landed.

        Splitting the write lets the GPU copy overlap with the normalizer-stat
        CPU writes in between; completion must still precede the seq-even bump
        (a published version implies landed data — that is the seqlock's
        publish contract).
        """

    def publish(self, actor: nn.Module, obs_normalizer: nn.Module) -> None:
        """Write current actor params + normalizer stats to the inactive slot
        and flip the active pointer via the seqlock."""
        device_flat = flatten_params(actor)
        has_stats = all(hasattr(obs_normalizer, k) for k in _NORM_KEYS)
        if has_stats:
            mean = obs_normalizer._mean.detach().cpu()
            std = obs_normalizer._std.detach().cpu()
            var = obs_normalizer._var.detach().cpu()
            count = int(obs_normalizer.count)

        prev = int(self.shared.seq[0])
        # 1) Mark "write in progress" (odd). Readers seeing this retry.
        self.shared.seq[0] = prev + 1
        # 2) Write to the slot opposite the previously-active one
        #    (active slot before this publish was (prev // 2) % 2). Two
        #    publishes in a row therefore alternate slots, so a single
        #    publish during a read targets the *other* slot — no collision
        #    even without the seqlock guard. The guard exists for the case
        #    of two publishes during one read.
        slot = ((prev // 2) + 1) % 2
        # Start the slot write; on GPU transports this only enqueues, letting
        # the copy run under the normalizer-stat CPU writes below.
        self._write_slot(slot, device_flat)
        if has_stats:
            self.shared.mean[slot].copy_(mean)
            self.shared.std[slot].copy_(std)
            self.shared.var[slot].copy_(var)
            self.shared.count[slot][0] = count
        # Land the slot write before announcing the version: a reader that
        # observes the new even seq value must be guaranteed to see the data.
        self._complete_slot_write()
        # 3) Bump seq back to even. On x86/TSO the data stores above are visible
        #    before this store, so a reader that observes the new even value
        #    sees all writes above. Readers that observed the odd value retry.
        self.shared.seq[0] = prev + 2


class HostWeightSender(WeightSender):
    """Host shared-memory slots; one fused pinned device->host transfer per publish."""

    def __init__(self, shared: WeightChannelShared, param_numel: int):
        super().__init__(shared)
        self.params = [_shared((param_numel,), torch.float32) for _ in range(2)]
        # Writer-private pinned staging, allocated lazily and keyed by size.
        self._staging_cache: tuple[int, torch.Tensor] | None = None

    def _staging(self, device_flat: torch.Tensor) -> torch.Tensor:
        size = device_flat.numel()
        if self._staging_cache is None or self._staging_cache[0] != size:
            pinned = device_flat.is_cuda
            self._staging_cache = (size, torch.empty(size, dtype=torch.float32, pin_memory=pinned))
        return self._staging_cache[1]

    def _write_slot(self, slot: int, device_flat: torch.Tensor) -> None:
        staging = self._staging(device_flat)
        # Keep the source device (not just cuda-ness) so the completion sync
        # below targets the stream the D2H copy was actually enqueued on.
        self._pending = (slot, staging, device_flat.device if device_flat.is_cuda else None)
        staging.copy_(device_flat, non_blocking=True)

    def _complete_slot_write(self) -> None:
        # The shared-memory copy reads the staging buffer on the CPU, so a CUDA
        # source's D2H transfer must land first. (Pinned-ness of the staging
        # buffer is not the discriminator: staging is always a CPU tensor.)
        # The device is passed explicitly: the process's current device may
        # differ from the device the copy ran on.
        slot, staging, source_device = self._pending
        if source_device is not None:
            torch.cuda.current_stream(source_device).synchronize()
        self.params[slot].copy_(staging)


class GpuIpcWeightSender(WeightSender):
    """CUDA-IPC device slots; publish is a device-to-device copy.

    Must be constructed in the learner process: the exporter of IPC handles
    needs a CUDA context and must keep the slot tensors alive for the process
    lifetime. The endpoint itself is not shippable — it holds a CUDA event;
    only the ``params`` tensors cross processes.
    """

    def __init__(self, shared: WeightChannelShared, param_numel: int, device: torch.device):
        super().__init__(shared)
        if device.type != "cuda":
            raise ValueError(f"GPU weight slots require a CUDA device, got {device}")
        self.params = [torch.zeros(param_numel, device=device) for _ in range(2)]
        # The event binds to the device current at creation time; create it
        # under the slot device so record/synchronize always target the right
        # context even when the learner device is not the process default.
        with torch.cuda.device(device):
            self._event = torch.cuda.Event()

    def _write_slot(self, slot: int, device_flat: torch.Tensor) -> None:
        # Explicit device: the current device may differ from the slot device.
        stream = torch.cuda.current_stream(self.params[slot].device)
        self.params[slot].copy_(device_flat)
        # Mark the copy; completion is awaited in _complete_slot_write so the
        # copy can overlap with the normalizer-stat CPU writes in between.
        self._event.record(stream)

    def _complete_slot_write(self) -> None:
        # The seq-even bump is a CPU store, but the slot write is an
        # asynchronous GPU copy: complete (and thus device-globally visible) it
        # here so a reader that observes the new version sees the data.
        self._event.synchronize()


class WeightReceiver(ABC):
    """Collector-side endpoint; owns the reader's load cursor.

    maybe_load implements the seqlock read loop; the transport-specific
    snapshot/load steps are the abstract hooks.
    """

    def __init__(self, shared: WeightChannelShared):
        self.shared = shared
        self._loaded_version = 0
        # Reader-private staging for the normalizer stats (they always travel
        # through host shared memory); pinned when the actor lives on CUDA so
        # the stat copies into it can be non-blocking. Allocated lazily.
        self._norm_staging: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @property
    def version(self) -> int:
        """Number of completed publishes (= seq // 2). Best-effort; readers
        that need a consistent snapshot must go through :meth:`maybe_load`."""
        return int(self.shared.seq[0]) // 2

    @property
    def loaded_version(self) -> int:
        """The snapshot version this receiver last loaded (0 = nothing yet)."""
        return self._loaded_version

    @property
    def lag(self) -> int:
        """How many published versions behind the loaded policy is."""
        return max(0, self.version - self._loaded_version)

    def _ensure_norm_staging(self, actor: nn.Module) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._norm_staging is None:
            obs_dim = self.shared.mean[0].shape[1]
            pinned = next(actor.parameters()).is_cuda
            self._norm_staging = tuple(
                torch.empty((1, obs_dim), dtype=torch.float32, pin_memory=pinned) for _ in range(3)
            )
        return self._norm_staging

    @abstractmethod
    def _snapshot(self, slot: int, flat_params: torch.Tensor | None, actor: nn.Module) -> None:
        """Make a private, completed copy of the slot for this reader."""

    @abstractmethod
    def _load(self, slot: int, flat_params: torch.Tensor | None, actor: nn.Module) -> None:
        """Update the actor from the snapshot taken by :meth:`_snapshot`."""

    def maybe_load(
        self,
        actor: nn.Module,
        obs_normalizer: nn.Module,
        flat_params: torch.Tensor | None = None,
    ) -> tuple[int, float, float, float]:
        """Load the latest snapshot into ``actor``/``obs_normalizer`` if newer.

        Args:
            actor: The collector's inference-only actor copy. Only touched
                when a new version is loaded; the host transport without
                ``flat_params`` copies per-parameter into it, otherwise it is
                updated through the bound flat-parameter tensor.
            obs_normalizer: The collector's read-only observation normalizer
                (``EmpiricalNormalization`` or ``Identity``). Its ``_mean`` /
                ``_std`` / ``_var`` / ``count`` stats are refreshed alongside
                the weights when the learner publishes them.
            flat_params: The collector actor's parameters bound into one
                contiguous CUDA tensor (see :func:`bind_flat_params`), letting
                the load be a single copy. Required by the CUDA-IPC transport;
                ``None`` falls back to per-parameter copies on the host path.
                Host-path staging buffers are an internal detail of the
                receiver, allocated lazily on first use.

        Returns:
            A ``(version, wait_writer_s, host_snapshot_s, actor_load_s)``
            tuple: the
            version actually loaded (== the receiver's previous
            ``loaded_version`` if nothing new was available this poll),
            followed by wall-clock seconds spent waiting for an in-progress
            writer (always 0 — a mid-publish poll returns immediately), copying
            a stable snapshot, and loading it onto the actor's device. The
            breakdown lets the collector distinguish shared-memory publication
            contention from actor-load work.

        Implements the seqlock read loop — non-blocking on the writer: read
        ``seq``; if it is odd (writer mid-publish) return the current version
        immediately and retry on the next poll. Otherwise ensure the value is
        stable, copy the data out, then re-check ``seq``: if the writer
        published in between we discard the (possibly torn) copy and retry.
        In practice retries are extremely rare — publish cadence is bounded by
        ``weight_publish_interval`` gradient steps, while a read is one CPU
        memcpy.
        """
        wait_writer_s = 0.0
        host_snapshot_s = 0.0
        local_version = self._loaded_version
        while True:
            s1 = int(self.shared.seq[0])
            if s1 & 1:
                # Writer is mid-publish. Weights are eventually consistent —
                # keep running the current policy and pick up the new version
                # on the next poll. Busy-waiting here would burn a core on the
                # collector's critical path: the odd window spans the learner's
                # in-flight gradient kernels (publish runs right after the
                # async update), so it can last tens of milliseconds. This
                # also covers retries: a torn read re-reads seq fresh instead
                # of spinning on the stale value.
                return local_version, wait_writer_s, host_snapshot_s, 0.0
            version = s1 // 2
            if version <= local_version:  # nothing new to load
                return local_version, wait_writer_s, host_snapshot_s, 0.0
            slot = version % 2  # active slot at this seq
            has_stats = all(hasattr(obs_normalizer, k) for k in _NORM_KEYS)
            # Phase 1 — snapshot: the transport makes a private, completed copy
            # of the slot (host staging, or the bound flat params on the device
            # path — each implementation owns its ordering constraints).
            snapshot_start = time.perf_counter()
            self._snapshot(slot, flat_params, actor)
            if has_stats:
                mean, std, var = self._ensure_norm_staging(actor)
                mean.copy_(self.shared.mean[slot])
                std.copy_(self.shared.std[slot])
                var.copy_(self.shared.var[slot])
                count = int(self.shared.count[slot][0])
            # Re-check seq. If it changed, the writer published (at least
            # once) during our copy and the slot may have been overwritten
            # — discard and retry. This is the line that closes the race
            # the old double-buffer had.
            s2 = int(self.shared.seq[0])
            host_snapshot_s += time.perf_counter() - snapshot_start
            if s1 != s2:
                continue

            # Phase 2 — load: update the actor from the stable snapshot.
            actor_load_start = time.perf_counter()
            self._load(slot, flat_params, actor)
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
            self._loaded_version = version
            return version, wait_writer_s, host_snapshot_s, actor_load_s


class HostWeightReceiver(WeightReceiver):
    """Host shared-memory slots; stages through CPU pinned memory.

    The actor copy is deliberately deferred to :meth:`_load` so a seqlock
    retry never reuses staging memory that an in-flight H2D still reads.
    """

    def __init__(self, shared: WeightChannelShared, params: list[torch.Tensor]):
        super().__init__(shared)
        self.params = params
        self._param_staging: torch.Tensor | None = None

    def _ensure_param_staging(self, actor: nn.Module) -> torch.Tensor:
        if self._param_staging is None or self._param_staging.numel() != self.params[0].numel():
            pinned = next(actor.parameters()).is_cuda
            self._param_staging = torch.empty(self.params[0].numel(), dtype=torch.float32, pin_memory=pinned)
        return self._param_staging

    def _snapshot(self, slot: int, flat_params: torch.Tensor | None, actor: nn.Module) -> None:
        del flat_params
        self._ensure_param_staging(actor).copy_(self.params[slot])

    def _load(self, slot: int, flat_params: torch.Tensor | None, actor: nn.Module) -> None:
        del slot  # the snapshot is already staged
        staging = self._ensure_param_staging(actor)
        if flat_params is not None:
            flat_params.copy_(staging, non_blocking=True)
        else:
            load_flat_params(actor, staging)


class GpuIpcWeightReceiver(WeightReceiver):
    """CUDA-IPC device slots; reads device-to-device into the bound flat params.

    The copy must COMPLETE before the seq re-check (a republish during an
    in-flight copy would tear the read), so it is synchronized rather than
    left async.
    """

    def __init__(self, shared: WeightChannelShared, params: list[torch.Tensor]):
        super().__init__(shared)
        self.params = params

    def _snapshot(self, slot: int, flat_params: torch.Tensor | None, actor: nn.Module) -> None:
        del actor
        if flat_params is None:
            raise ValueError("the CUDA-IPC weight path requires bound flat_params")
        flat_params.copy_(self.params[slot])
        # Sync the stream(s) the copy was enqueued on. Same-device (the gated
        # design) it is one stream; a cross-device edge case may use both.
        torch.cuda.current_stream(self.params[slot].device).synchronize()
        if flat_params.device != self.params[slot].device:
            torch.cuda.current_stream(flat_params.device).synchronize()

    def _load(self, slot: int, flat_params: torch.Tensor | None, actor: nn.Module) -> None:
        del slot, flat_params, actor  # no-op: _snapshot already wrote the actor


def weight_receiver_for(shared: WeightChannelShared, params: list[torch.Tensor]) -> WeightReceiver:
    """Build the receiver matching a shipped slot pair (dispatch on device)."""
    if params[0].is_cuda:
        return GpuIpcWeightReceiver(shared, params)
    return HostWeightReceiver(shared, params)
