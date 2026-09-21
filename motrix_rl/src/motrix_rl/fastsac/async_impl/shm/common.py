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
* the weight channel (see :mod:`motrix_rl.fastsac.async_impl.shm.weight_channel`)
  — double-buffered actor weights + obs-normalizer stats (learner ->
  collector) guarded by a seqlock so readers always see a complete,
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

import torch
from torch import nn


def _shared(shape, dtype) -> torch.Tensor:
    """Allocate a zero-initialized CPU tensor in shared memory (usable across spawn children)."""
    return torch.zeros(shape, dtype=dtype).share_memory_()


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


# ---------------------------------------------------------------- flat-param helpers
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


# obs-normalizer stat buffers that the collector needs (read-only) to reproduce
# the sync ``act()`` path (normalize with update=False, see agent.act); consumed
# by the weight channel (weight_channel.py).
_NORM_KEYS = ("_mean", "_std", "_var", "count")
