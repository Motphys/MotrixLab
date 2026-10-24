# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Small, algorithm-neutral worker protocol for spawned execution workers."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from motrix_rl.execution.contracts import ExecutionCancelled
from motrix_rl.execution.worker_runtime import (
    configure_process_logging,
    install_graceful_sigint,
    pin_worker_cpus,
    set_seed,
)
from motrix_rl.numa import apply_binding, resolve_cpu_set


@dataclass(frozen=True)
class WorkerMessage:
    """Low-frequency status message sent from a worker to its parent."""

    kind: str
    worker_id: Any
    payload: Any = None


@dataclass(frozen=True)
class WorkerSettings:
    """Optional process setup performed by :func:`worker_entry`."""

    seed: int | None = None
    numa_node: int | None = None
    cpus: tuple[int, ...] | None = None
    cpu_spec: str | None = None
    log_path: str | Path | None = None


class WorkerRuntime:
    """Shared lifecycle state and status transport for one worker."""

    def __init__(self, worker_id: Any, event_queue, start_event, stop_event) -> None:
        self.worker_id = worker_id
        self.event_queue = event_queue
        self.start_event = start_event
        self.stop_event = stop_event
        self.resources = ExitStack()
        self._closed = False
        self._ready = False
        self._finished = False

    def ready(self) -> None:
        """Report readiness, then wait for parent release or cooperative stop."""
        self._ready = True
        self.report(None, kind="ready")
        while not self.start_event.is_set():
            if self.should_stop():
                raise ExecutionCancelled("worker startup cancelled")
            self.start_event.wait(0.05)
        if self.should_stop():
            raise ExecutionCancelled("worker startup cancelled")

    @property
    def ready_reported(self) -> bool:
        return self._ready

    @property
    def finished(self) -> bool:
        return self._finished

    def should_stop(self) -> bool:
        return self.stop_event.is_set()

    def request_stop(self) -> None:
        self.stop_event.set()

    def report(self, payload: Any = None, *, kind: str = "report") -> None:
        self.event_queue.put(WorkerMessage(kind, self.worker_id, payload))

    def finish(self, payload: Any = None) -> None:
        self._finished = True
        self.report(payload, kind="finish")

    def callback(self, callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Register a private resource cleanup callback and return its result."""
        return self.resources.callback(callback, *args, **kwargs)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.resources.close()


def worker_entry(runtime: WorkerRuntime, target: Callable[..., Any], args=(), kwargs=None, settings=None) -> None:
    """Configure a child and invoke ``target`` without inventing completion."""
    if kwargs is None:
        kwargs = {}
    if settings is None:
        settings = WorkerSettings()
    primary: BaseException | None = None
    phase = "setup"
    try:
        if settings.log_path is not None:
            configure_process_logging(Path(settings.log_path))
        cpus = set(settings.cpus or ())
        if settings.numa_node is not None or cpus:
            apply_binding("worker", settings.numa_node, sorted(cpus))
        if settings.cpu_spec:
            cpus = resolve_cpu_set(settings.cpu_spec, "cpu_spec")
            if cpus:
                pin_worker_cpus(cpus)
        set_seed(settings.seed)
        install_graceful_sigint(runtime.request_stop)
        phase = "target"
        target(runtime, *args, **kwargs)
    except ExecutionCancelled as exc:
        if runtime.should_stop():
            return
        primary = exc
        runtime.report(
            {"traceback": traceback.format_exc(), "phase": phase, "message": str(exc)},
            kind="error",
        )
        runtime.request_stop()
        raise
    except BaseException as exc:
        primary = exc
        runtime.report(
            {
                "traceback": traceback.format_exc(),
                "phase": phase,
                "message": str(exc),
            },
            kind="error",
        )
        runtime.request_stop()
        raise
    finally:
        try:
            runtime.close()
        except BaseException as cleanup_error:
            if primary is None:
                runtime.report(
                    {
                        "traceback": traceback.format_exc(),
                        "phase": "cleanup",
                        "message": str(cleanup_error),
                    },
                    kind="error",
                )
                runtime.request_stop()
                raise
            logging.getLogger(__name__).exception("worker cleanup failed after target error")
