# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Small process-backed worker abstraction.

The :class:`Worker` base class keeps process lifecycle management separate from
worker business logic. Subclasses only implement :meth:`Worker.run`; the base
class applies the requested NUMA and CPU placement before entering that method.

Worker instances are serialized by ``multiprocessing`` when a ``spawn`` context
starts a child process. Subclasses should therefore keep only configuration and
multiprocessing-safe IPC handles in instance attributes. Runtime objects such
as environments, threads, locks, writers, and simulator handles belong in
:meth:`Worker.run`.
"""

from __future__ import annotations

import multiprocessing as mp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any

from motrix_rl.numa import apply_binding, spawn_placement


@dataclass(frozen=True)
class WorkerPlacement:
    """CPU and NUMA placement requested for a worker process.

    Args:
        numa_node: NUMA node used for memory placement, or ``None`` to leave
            memory placement unchanged.
        cpus: CPU IDs assigned to the worker. An empty tuple leaves CPU
            affinity unchanged.
    """

    numa_node: int | None = None
    cpus: tuple[int, ...] = ()


class Worker(ABC):
    """Manage one spawn-compatible child process.

    A worker subclass implements :meth:`run` and stores only pickleable
    configuration or multiprocessing-safe IPC objects. The parent-side
    ``Process`` object is excluded from spawn serialization automatically.

    Args:
        name: Child process name. Defaults to the subclass name.
        placement: Requested NUMA node and CPU affinity.

    Example:
        ``Worker`` is intended to be subclassed with a module-level class::

            class ExampleWorker(Worker):
                def run(self) -> None:
                    print(self.name)


            worker = ExampleWorker(name="example")
            worker.start(mp.get_context("spawn"))
            worker.join()
    """

    def __init__(self, *, name: str | None = None, placement: WorkerPlacement | None = None) -> None:
        self.name = name or type(self).__name__
        self.placement = placement or WorkerPlacement()
        self._process: mp.Process | None = None
        self._started = False

    @abstractmethod
    def run(self) -> None:
        """Run the worker body inside the child process.

        Runtime-only objects should be created in this method rather than in
        ``__init__`` so they are not serialized from the parent process.
        """
        raise NotImplementedError

    def start(self, ctx: BaseContext) -> None:
        """Create and start the worker process.

        The parent is temporarily placed on the worker's NUMA node while
        ``Process.start`` performs the spawn. The child then applies its exact
        CPU affinity and memory policy in :func:`_worker_bootstrap`.

        Args:
            ctx: Multiprocessing context used to create the process.

        Raises:
            RuntimeError: If this worker has already been started.
        """
        if self._started:
            raise RuntimeError(f"worker {self.name!r} has already started")
        process = ctx.Process(
            target=_worker_bootstrap,
            args=(self,),
            name=self.name,
        )
        with spawn_placement(self.placement.numa_node, self.name):
            process.start()
        self._process = process
        self._started = True

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker process to exit.

        Args:
            timeout: Maximum number of seconds to wait, or ``None`` to wait
                indefinitely.
        """
        self._require_process().join(timeout)

    def terminate(self) -> None:
        """Request graceful operating-system termination of the worker."""
        self._require_process().terminate()

    def kill(self) -> None:
        """Forcefully kill the worker process."""
        self._require_process().kill()

    @property
    def process(self) -> mp.Process:
        """Return the underlying parent-side process object."""
        return self._require_process()

    @property
    def pid(self) -> int | None:
        """Return the child PID, or ``None`` before ``start``."""
        return None if self._process is None else self._process.pid

    @property
    def exitcode(self) -> int | None:
        """Return the child exit code, or ``None`` while it is running."""
        return None if self._process is None else self._process.exitcode

    @property
    def is_alive(self) -> bool:
        """Return whether the child process is currently alive."""
        return self._process is not None and self._process.is_alive()

    def __getstate__(self) -> dict[str, Any]:
        """Exclude parent-only lifecycle state during child serialization."""
        state: dict[str, Any] = {}
        if hasattr(self, "__dict__"):
            state.update(self.__dict__)
        for cls in type(self).__mro__:
            for slot in getattr(cls, "__slots__", ()):
                if slot in ("__dict__", "__weakref__"):
                    continue
                if hasattr(self, slot):
                    state[slot] = getattr(self, slot)
        state["_process"] = None
        state["_started"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore both dictionary and slotted subclass state after spawn."""
        for key, value in state.items():
            setattr(self, key, value)

    def _require_process(self) -> mp.Process:
        if self._process is None:
            raise RuntimeError(f"worker {self.name!r} has not been started")
        return self._process


def _worker_bootstrap(worker: Worker) -> None:
    """Apply child placement and invoke the subclass worker body."""
    placement = worker.placement
    apply_binding(worker.name, placement.numa_node, list(placement.cpus))
    worker.run()
