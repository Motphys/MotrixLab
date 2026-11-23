# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Small lifecycle helpers for benchmark worker resources."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class WorkerExecution(Generic[T]):
    """An acquired worker resource with idempotent cleanup.

    Benchmark workers commonly own an environment, transport, or wrapper whose
    cleanup must happen both on successful measurement and when a step raises.
    This handle centralizes that lifecycle while keeping construction lazy and
    dependency-free.
    """

    resource: T
    _cleanup: Callable[[T], object]
    _closed: bool = False

    def close(self) -> None:
        """Release the resource once."""
        if not self._closed:
            self._closed = True
            self._cleanup(self.resource)

    def __enter__(self) -> T:
        return self.resource

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        primary_error = exc_value
        try:
            self.close()
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            note = f"worker cleanup failed: {cleanup_error!r}"
            add_note = getattr(primary_error, "add_note", None)
            if add_note is not None:
                add_note(note)
            else:
                notes = getattr(primary_error, "__notes__", None)
                if notes is None:
                    notes = []
                    setattr(primary_error, "__notes__", notes)
                notes.append(note)
                primary_error.__context__ = cleanup_error
        return False


@contextmanager
def worker_execution(
    factory: Callable[[], T],
    cleanup: Callable[[T], object] | None = None,
) -> Iterator[T]:
    """Acquire a worker resource and guarantee cleanup on every exit path.

    ``cleanup`` defaults to the resource's ``close`` method.  The factory is
    called only once, and cleanup is skipped when acquisition itself fails.
    """
    resource = factory()
    close = cleanup or _default_cleanup
    execution = WorkerExecution(resource, close)
    with execution as acquired:
        yield acquired


def _default_cleanup(resource: T) -> object:
    close = getattr(resource, "close", None)
    if close is None:
        raise TypeError("worker resource has no close() method; provide cleanup")
    return close()


__all__ = ["WorkerExecution", "worker_execution"]
