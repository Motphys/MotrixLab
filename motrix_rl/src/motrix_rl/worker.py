# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Small process-backed worker abstraction.

``Worker`` contains only the task executed in the child process. ``WorkerHandle``
contains the parent-side process lifecycle and placement state returned by
:meth:`Worker.start`.
"""

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.context import BaseContext

from motrix_rl.numa import apply_binding, spawn_placement


@dataclass(frozen=True)
class WorkerPlacement:
    """CPU and NUMA placement requested for a worker process."""

    numa_node: int | None = None
    cpus: tuple[int, ...] = ()


class WorkerHandle:
    """Parent-side lifecycle handle for a started worker."""

    def __init__(self, process: mp.Process, *, name: str, placement: WorkerPlacement) -> None:
        self._process = process
        self.name = name
        self.placement = placement

    def join(self, timeout: float | None = None) -> None:
        """Wait for the child process to exit."""
        self._process.join(timeout)

    def terminate(self) -> None:
        """Request graceful operating-system termination."""
        self._process.terminate()

    def kill(self) -> None:
        """Forcefully kill the child process."""
        self._process.kill()

    @property
    def process(self) -> mp.Process:
        """Return the underlying parent-side process object."""
        return self._process

    @property
    def pid(self) -> int | None:
        """Return the child PID."""
        return self._process.pid

    @property
    def exitcode(self) -> int | None:
        """Return the child exit code, or ``None`` while it is running."""
        return self._process.exitcode

    @property
    def is_alive(self) -> bool:
        """Return whether the child process is currently alive."""
        return self._process.is_alive()


class Worker(ABC):
    """Describe one spawn-compatible child task.

    Subclasses store only configuration and multiprocessing-safe IPC objects.
    Runtime-only objects such as environments, threads, writers, and simulator
    handles belong in :meth:`run`.
    """

    @abstractmethod
    def run(self) -> None:
        """Run the worker body inside the child process."""
        raise NotImplementedError

    def start(
        self,
        ctx: BaseContext,
        *,
        name: str | None = None,
        placement: WorkerPlacement | None = None,
    ) -> WorkerHandle:
        """Start this task and return its parent-side lifecycle handle."""
        worker_name = name or type(self).__name__
        worker_placement = placement or WorkerPlacement()
        process = ctx.Process(
            target=_worker_bootstrap,
            args=(self, worker_placement, worker_name),
            name=worker_name,
        )
        with spawn_placement(worker_placement.numa_node, worker_name):
            process.start()
        return WorkerHandle(process, name=worker_name, placement=worker_placement)


def _worker_bootstrap(worker: Worker, placement: WorkerPlacement, worker_name: str) -> None:
    """Apply child placement and invoke the worker task."""
    apply_binding(worker_name, placement.numa_node, list(placement.cpus))
    worker.run()
