# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Parent-side ownership of low-frequency worker message delivery.

Tensor training data continues to use algorithm-selected transports; this
module never forwards samples or inference snapshots through Python queues.
"""

from __future__ import annotations

from collections import defaultdict
from queue import Empty
from typing import Any


class MessageInbox:
    """Drain a worker control queue into state and opaque report batches.

    The protocol requires per-worker ready/finish/error messages; algorithms
    interpret only report/finish payloads, never raw multiprocessing objects.
    Unknown identities or kinds are protocol failures rather than ignored work.
    """

    def __init__(self, queue, worker_ids: set[str]):
        self._queue = queue
        self._worker_ids = frozenset(worker_ids)
        self.ready: set[str] = set()
        self.finished: set[str] = set()
        self.errors: list[Any] = []
        self._reports: dict[str, list[Any]] = defaultdict(list)

    def drain(self) -> None:
        while True:
            try:
                message = self._queue.get_nowait()
            except Empty:
                return
            if message.worker_id not in self._worker_ids:
                raise RuntimeError(f"message from unknown worker {message.worker_id!r}")
            if message.kind == "ready":
                if message.worker_id in self.ready:
                    raise RuntimeError(f"duplicate readiness from worker {message.worker_id!r}")
                self.ready.add(message.worker_id)
            elif message.kind == "finish":
                if message.worker_id not in self.ready:
                    raise RuntimeError(f"worker {message.worker_id!r} finished before readiness")
                if message.worker_id in self.finished:
                    raise RuntimeError(f"duplicate completion from worker {message.worker_id!r}")
                self.finished.add(message.worker_id)
                self._reports[message.worker_id].append(message.payload)
            elif message.kind == "report":
                self._reports[message.worker_id].append(message.payload)
            elif message.kind == "error":
                self.errors.append(message)
            else:
                raise RuntimeError(f"unknown worker message kind {message.kind!r}")

    def take_reports(self) -> dict[str, tuple[Any, ...]]:
        """Transfer pending batches once; retained algorithm state is explicit."""
        reports = {worker_id: tuple(items) for worker_id, items in self._reports.items()}
        self._reports.clear()
        return reports
