# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Versioned inference tensor snapshots over host memory or CUDA IPC.

A single writer publishes model parameters and named auxiliary tensors under
one seqlock. Readers skip in-progress publishes and retry torn snapshots.
The host protocol is x86/TSO-only; CUDA copies preserve explicit stream/event
completion before publishing or checking versions. No queue is owned here.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.multiprocessing  # noqa: F401  registers CUDA-IPC reducers in every importing process
from torch import nn


def _shared(shape, dtype) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype).share_memory_()


def flatten_params(module: nn.Module) -> torch.Tensor:
    """Flatten module parameters into one contiguous float vector on the module's device."""
    return torch.cat([p.detach().reshape(-1).float() for p in module.parameters()])


def load_flat_params(module: nn.Module, flat: torch.Tensor) -> None:
    """Inverse of :func:`flatten_params`; copies a flat vector into the params."""
    offset = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat[offset : offset + n].view_as(p).to(p.device))
        offset += n


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


@dataclass(frozen=True)
class SnapshotTensorSpec:
    """One named auxiliary tensor carried atomically with model parameters.

    host_scalar preserves scalar integer state through Python int/fill_, without
    introducing a device tensor copy. It requires a single-element int64 tensor.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    host_scalar: bool = False


class TensorSnapshotShared:
    """Parent-created double-buffered host auxiliary state and seqlock.

    Single writer/reader; host ordering requires x86/TSO. CUDA parameter slots
    are separately owned by the sender for its entire process lifetime.
    """

    def __init__(self, auxiliary: Mapping[str, SnapshotTensorSpec]):
        self.specs = dict(auxiliary)
        for spec in self.specs.values():
            if spec.host_scalar and (spec.dtype != torch.int64 or torch.Size(spec.shape).numel() != 1):
                raise ValueError("host_scalar requires a single-element int64 tensor")
        self.auxiliary = {
            name: [_shared(spec.shape, spec.dtype) for _ in range(2)] for name, spec in self.specs.items()
        }
        self.seq = _shared((1,), torch.int64)


class SnapshotSender(ABC):
    """Learner-side endpoint; publishes snapshots under the seqlock."""

    def __init__(self, shared: TensorSnapshotShared):
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

        Splitting the write lets the GPU copy overlap with the auxiliary-state
        CPU writes in between; completion must still precede the seq-even bump
        (a published version implies landed data — that is the seqlock's
        publish contract).
        """

    def publish(self, actor: nn.Module, auxiliary: Mapping[str, torch.Tensor]) -> None:
        """Publish model parameters and named auxiliary state as one version."""
        self.publish_flat(flatten_params(actor), auxiliary)

    def publish_flat(self, device_flat: torch.Tensor, auxiliary: Mapping[str, torch.Tensor]) -> None:
        """Publish a prepared flat parameter tensor and named auxiliary state.

        The caller must supply the same auxiliary keys on every publish (or an
        empty mapping consistently). GPU reads finish before opening the seqlock.
        """
        prepared = {
            name: int(value) if self.shared.specs[name].host_scalar else value.detach().cpu()
            for name, value in auxiliary.items()
        }

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
        # the copy run under the auxiliary-state CPU writes below.
        self._write_slot(slot, device_flat)
        for name, value in prepared.items():
            target = self.shared.auxiliary[name][slot]
            if self.shared.specs[name].host_scalar:
                target.fill_(value)
            else:
                target.copy_(value)
        # Land the slot write before announcing the version: a reader that
        # observes the new even seq value must be guaranteed to see the data.
        self._complete_slot_write()
        # 3) Bump seq back to even. On x86/TSO the data stores above are visible
        #    before this store, so a reader that observes the new even value
        #    sees all writes above. Readers that observed the odd value retry.
        self.shared.seq[0] = prev + 2


class HostSnapshotSender(SnapshotSender):
    """Host shared-memory slots; one fused pinned device->host transfer per publish."""

    def __init__(self, shared: TensorSnapshotShared, param_numel: int):
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


class GpuIpcSnapshotSender(SnapshotSender):
    """CUDA-IPC device slots; publish is a device-to-device copy.

    Must be constructed in the learner process: the exporter of IPC handles
    needs a CUDA context and must keep the slot tensors alive for the process
    lifetime. The endpoint itself is not shippable — it holds a CUDA event;
    only the ``params`` tensors cross processes.
    """

    def __init__(self, shared: TensorSnapshotShared, param_numel: int, device: torch.device):
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
        # copy can overlap with the auxiliary-state CPU writes in between.
        self._event.record(stream)

    def _complete_slot_write(self) -> None:
        # The seq-even bump is a CPU store, but the slot write is an
        # asynchronous GPU copy: complete (and thus device-globally visible) it
        # here so a reader that observes the new version sees the data.
        self._event.synchronize()


class SnapshotReceiver(ABC):
    """Collector-side endpoint; owns the reader's load cursor.

    maybe_load implements the seqlock read loop; the transport-specific
    snapshot/load steps are the abstract hooks.
    """

    def __init__(self, shared: TensorSnapshotShared):
        self.shared = shared
        self._loaded_version = 0
        # Reused private host snapshots; pinned only for CUDA destinations.
        self._aux_staging: dict[str, torch.Tensor] | None = None

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

    def _ensure_aux_staging(self, actor: nn.Module) -> dict[str, torch.Tensor]:
        if self._aux_staging is None:
            pinned = next(actor.parameters()).is_cuda
            self._aux_staging = {
                name: torch.empty(spec.shape, dtype=spec.dtype, pin_memory=pinned)
                for name, spec in self.shared.specs.items()
                if not spec.host_scalar
            }
        return self._aux_staging

    @abstractmethod
    def _snapshot(self, slot: int, flat_params: torch.Tensor | None, actor: nn.Module) -> None:
        """Make a private, completed copy of the slot for this reader."""

    @abstractmethod
    def _load(self, slot: int, flat_params: torch.Tensor | None, actor: nn.Module) -> None:
        """Update the actor from the snapshot taken by :meth:`_snapshot`."""

    def maybe_load(
        self,
        actor: nn.Module,
        auxiliary: Mapping[str, torch.Tensor],
        flat_params: torch.Tensor | None = None,
    ) -> tuple[int, float, float, float]:
        """Load a coherent snapshot, returning version and timing components.

        Returns (loaded_version, writer_wait_s, snapshot_s, load_s). Odd writer
        sequences return immediately. Named targets must match the publisher's
        layout; an empty mapping supports models with no auxiliary state.
        CUDA callers must not reuse host staging before prior inference consumes
        its asynchronous copies (the collector's action handoff provides this).
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
            # Phase 1 — snapshot: the transport makes a private, completed copy
            # of the slot (host staging, or the bound flat params on the device
            # path — each implementation owns its ordering constraints).
            snapshot_start = time.perf_counter()
            self._snapshot(slot, flat_params, actor)
            staged = self._ensure_aux_staging(actor) if auxiliary else {}
            scalars = {}
            for name in auxiliary:
                source = self.shared.auxiliary[name][slot]
                if self.shared.specs[name].host_scalar:
                    scalars[name] = int(source.reshape(-1)[0])
                else:
                    staged[name].copy_(source)
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
            for name, target in auxiliary.items():
                if self.shared.specs[name].host_scalar:
                    target.fill_(scalars[name])
                else:
                    target.copy_(staged[name], non_blocking=True)
            # CUDA collectors keep the host staging buffers alive. The copies
            # above are ordered before the next inference on the same stream;
            # that inference already synchronizes after returning actions to
            # the CPU environment, so a separate weight-load barrier only
            # serializes collector and learner work unnecessarily.
            actor_load_s = time.perf_counter() - actor_load_start
            self._loaded_version = version
            return version, wait_writer_s, host_snapshot_s, actor_load_s


class HostSnapshotReceiver(SnapshotReceiver):
    """Host shared-memory slots; stages through CPU pinned memory.

    The actor copy is deliberately deferred to :meth:`_load` so a seqlock
    retry never reuses staging memory that an in-flight H2D still reads.
    """

    def __init__(self, shared: TensorSnapshotShared, params: list[torch.Tensor]):
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


class GpuIpcSnapshotReceiver(SnapshotReceiver):
    """CUDA-IPC device slots; reads device-to-device into the bound flat params.

    The copy must COMPLETE before the seq re-check (a republish during an
    in-flight copy would tear the read), so it is synchronized rather than
    left async.
    """

    def __init__(self, shared: TensorSnapshotShared, params: list[torch.Tensor]):
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


def snapshot_receiver_for(shared: TensorSnapshotShared, params: list[torch.Tensor]) -> SnapshotReceiver:
    """Build the receiver matching a shipped slot pair (dispatch on device)."""
    if params[0].is_cuda:
        return GpuIpcSnapshotReceiver(shared, params)
    return HostSnapshotReceiver(shared, params)
