# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Startup rendezvous for the async trainer: slot delivery + readiness barrier.

One structured object owns every piece of the boot handshake that used to be
scattered across per-worker queues:

* one-shot slot delivery — learners ship CUDA-IPC handle bundles (weight
  slots, transition-ring device slots) to the collectors they feed, through
  per-collector one-shot queues;
* the readiness barrier — every worker reports readiness and then holds on
  the start event until the parent releases everyone at once, so stepping
  begins in lockstep (warmup counters, weight generations and generation
  merges align from step one).

Created in the parent with the spawn context and inherited by the children
through the process args; the parent is the only ``release()`` caller.
With ``barrier=False`` the readiness half is inert (direct callers that
drive a single collector by hand).
"""

from __future__ import annotations

from multiprocessing.queues import Queue


class StartupHandshake:
    """Parent-allocated boot rendezvous shared by every trainer worker."""

    SLOT_TIMEOUT_S = 60.0

    def __init__(self, ctx, num_collectors: int, barrier: bool = True):
        self.barrier = barrier
        self.ready_queue = ctx.Queue()
        self.start_event = ctx.Event()
        # One-shot slot queue per collector (maxsize 1: exactly one message
        # per direction is ever sent). Ring-slot messages carry ``None`` for
        # the host shared-memory ring.
        self.slot_queues: list[Queue] = [ctx.Queue(maxsize=1) for _ in range(num_collectors)]
        self.ring_slot_queues: list[Queue] = [ctx.Queue(maxsize=1) for _ in range(num_collectors)]

    # ------------------------------------------------------- learner side
    def ship_weight_slots(self, collector_id: int, params) -> None:
        """Ship the learner-built weight-slot pair for one collector."""
        self.slot_queues[collector_id].put(params)

    def ship_ring_slots(self, collector_id: int, ring_slots) -> None:
        """Ship the CUDA-IPC transition-ring slots (``None`` for host rings)."""
        self.ring_slot_queues[collector_id].put(ring_slots)

    # ------------------------------------------------------ collector side
    def await_weight_slots(self, collector_id: int):
        """Block until the publishing learner ships this collector's weight slots."""
        return self._await(self.slot_queues[collector_id], "weight-slot")

    def await_ring_slots(self, collector_id: int):
        """Block until the draining learner ships this collector's ring slots.

        The message is ``None`` when the collector's ring is host shared
        memory (no device slots to map).
        """
        return self._await(self.ring_slot_queues[collector_id], "ring-slot")

    def _await(self, queue: Queue, kind: str):
        import queue as queue_mod

        try:
            return queue.get(timeout=self.SLOT_TIMEOUT_S)
        except queue_mod.Empty as exc:
            raise RuntimeError(
                f"timed out after {self.SLOT_TIMEOUT_S:.0f}s waiting for the learner to ship the {kind} tensors "
                "(learner startup — agent build / checkpoint load / CUDA warmup — "
                "likely failed; check the learner process's error queue/log)"
            ) from exc

    # ------------------------------------------------------------- barrier
    def report_ready(self, role: str, worker_id: int) -> None:
        """Report this worker as booted (no-op when the barrier is disabled)."""
        if self.barrier:
            self.ready_queue.put((role, worker_id))

    def wait_for_start(self, control) -> None:
        """Hold until the parent releases every worker (no-op without barrier)."""
        if not self.barrier:
            return
        while not self.start_event.is_set() and not control.stop:
            self.start_event.wait(timeout=0.2)

    def drain_ready(self) -> set:
        """Pop every readiness report delivered so far (parent only)."""
        import queue as queue_mod

        ready: set = set()
        try:
            while True:
                ready.add(self.ready_queue.get_nowait())
        except queue_mod.Empty:
            pass
        return ready

    def release(self) -> None:
        """Release the barrier; also used to free workers on an aborted boot."""
        self.start_event.set()

    def close(self) -> None:
        """Drain and close the one-shot queues so feeder threads can shut down."""
        for one_shot in (*self.slot_queues, *self.ring_slot_queues):
            try:
                while True:
                    one_shot.get_nowait()
            except Exception:
                pass
            one_shot.close()
        self.ready_queue.close()
