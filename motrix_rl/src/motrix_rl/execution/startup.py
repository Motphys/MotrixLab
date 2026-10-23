# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Parent-controlled readiness gate and cancellable startup resource waits.

Resource contents and expected worker identities belong to the trainer adapter.
The parent must check process exits while waiting for readiness and own queue
cleanup after all workers have been reaped.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from queue import Empty
from typing import Any

from motrix_rl.execution.contracts import ExecutionCancelled


class StartupBarrier:
    """Spawn-shared readiness reports and a parent-owned start event."""

    def __init__(self, ctx, *, enabled: bool = True):
        self.enabled = enabled
        self.ready_queue = ctx.Queue()
        self.start_event = ctx.Event()

    def report_ready(self, role: str, worker_id: int) -> None:
        if self.enabled:
            self.ready_queue.put((role, worker_id))

    def wait_for_start(self, should_stop: Callable[[], bool]) -> None:
        if not self.enabled:
            return
        while not self.start_event.is_set():
            if should_stop():
                raise ExecutionCancelled("worker startup cancelled")
            self.start_event.wait(timeout=0.05)
        if should_stop():
            raise ExecutionCancelled("worker startup cancelled")

    def drain_ready(self) -> set[tuple[str, int]]:
        ready = set()
        while True:
            try:
                ready.add(self.ready_queue.get_nowait())
            except Empty:
                return ready

    def release(self) -> None:
        self.start_event.set()

    def close(self) -> None:
        # Only call after peers exit; never join a feeder whose receiver failed.
        self.ready_queue.cancel_join_thread()
        self.ready_queue.close()


def await_startup_resource(
    queue,
    *,
    should_stop: Callable[[], bool],
    timeout: float,
    description: str,
) -> Any:
    """Receive one opaque resource, with finite waits and cooperative cancellation."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("startup resource timeout must be positive and finite")
    deadline = time.monotonic() + timeout
    while True:
        if should_stop():
            raise ExecutionCancelled(f"startup cancelled while waiting for {description}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out after {timeout:g}s waiting for {description}")
        try:
            return queue.get(timeout=min(0.05, remaining))
        except Empty:
            continue
