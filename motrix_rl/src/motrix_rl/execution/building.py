# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Parent-owned construction scope for process-backed algorithm plans."""

from __future__ import annotations

import multiprocessing as mp
from contextlib import ExitStack
from typing import Any


class BuildRuntime:
    """Allocate algorithm communication resources with one spawn context.

    Algorithms can allocate queues and synchronization primitives, and register
    cleanup immediately after acquiring other resources. They receive neither
    a Process factory nor a supervisor. The execution runtime owns the private
    context used later to materialize worker descriptions. All peers must be
    reaped before closing this scope.
    """

    def __init__(self, resources: ExitStack) -> None:
        self._context = mp.get_context("spawn")
        self._resources = resources

    def queue(self, maxsize: int = 0):
        """Own a queue without joining a feeder whose receiver may have died."""
        queue = self._context.Queue(maxsize=maxsize)
        self._resources.callback(queue.close)
        self._resources.callback(queue.cancel_join_thread)
        return queue

    def event(self):
        return self._context.Event()

    def lock(self):
        return self._context.Lock()

    def own(self, resource: Any, close=None):
        """Register a resource's explicit release, returning the same resource.

        Use an explicit callback when the resource has no ``close`` method;
        this does not assume environment or simulator shutdown interfaces.
        """
        self._resources.callback(resource.close if close is None else close)
        return resource

    def enter_context(self, context):
        return self._resources.enter_context(context)
