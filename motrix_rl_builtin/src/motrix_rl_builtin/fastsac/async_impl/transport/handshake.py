# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Algorithm slot delivery for async FastSAC workers.

Learners ship weight and CUDA-IPC transition-ring slots to their collectors.
The public execution runtime owns readiness, release and queue cleanup.
"""

from __future__ import annotations

from multiprocessing.queues import Queue

from motrix_rl.execution.building import BuildRuntime
from motrix_rl.execution.startup import await_startup_resource


class StartupHandshake:
    """Parent-allocated, one-shot slot queues shared by algorithm workers."""

    SLOT_TIMEOUT_S = 60.0

    def __init__(self, runtime: BuildRuntime, num_collectors: int):
        # Ring messages carry None for host shared-memory rings.
        self.slot_queues: list[Queue] = [runtime.queue(maxsize=1) for _ in range(num_collectors)]
        self.ring_slot_queues: list[Queue] = [runtime.queue(maxsize=1) for _ in range(num_collectors)]

    def ship_weight_slots(self, collector_id: int, params) -> None:
        """Ship the learner-built weight-slot pair for one collector."""
        self.slot_queues[collector_id].put(params)

    def ship_ring_slots(self, collector_id: int, ring_slots) -> None:
        """Ship CUDA-IPC transition-ring slots (None for host rings)."""
        self.ring_slot_queues[collector_id].put(ring_slots)

    def await_weight_slots(self, collector_id: int, control=None):
        """Wait cancellably for the publishing learner's weight slots."""
        return self._await(self.slot_queues[collector_id], "weight-slot", control)

    def await_ring_slots(self, collector_id: int, control=None):
        """Wait cancellably for ring slots; None denotes a host ring."""
        return self._await(self.ring_slot_queues[collector_id], "ring-slot", control)

    def _await(self, queue: Queue, kind: str, control):
        return await_startup_resource(
            queue,
            should_stop=lambda: control is not None and control.stop,
            timeout=self.SLOT_TIMEOUT_S,
            description=f"learner {kind} tensors (check learner startup logs)",
        )
