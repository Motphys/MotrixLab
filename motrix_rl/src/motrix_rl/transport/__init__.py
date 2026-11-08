# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Cross-process transport primitives shared by asynchronous trainers.

Submodules define the contracts; import from the defining submodule rather
than from this initializer (the package is not an aggregation facade):

* :mod:`.schema`   — :class:`~.schema.TransitionSchema`, the declared field
  set of one transition slot (provider-declared, not transport-hardcoded).
* :mod:`.common`   — the shared-memory allocator, shared scalar controls
  (:class:`~.common.Control`) and flat-parameter helpers.
* :mod:`.ring`     — the SPSC transition ring (collector -> learner) over
  host shared-memory fields, plus the shared :class:`~.ring.RingCursors`.
* :mod:`.gpu_ipc_ring` — the CUDA-IPC device-fields variant of the ring, used
  when collector inference and the learner share one GPU.
"""
