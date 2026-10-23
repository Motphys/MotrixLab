# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Parent-side multiprocessing lifecycle, independent of worker protocols.

The caller owns readiness, transport, completion criteria and error reporting.
Keep this context open from the first start through monitoring: any exception,
including cancellation or a partial start, stops and reaps the started children.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from types import TracebackType

_LOG = logging.getLogger(__name__)


class ProcessExitError(RuntimeError):
    """A worker exited before the caller's completion condition allowed it."""

    def __init__(self, process: BaseProcess) -> None:
        self.process = process
        super().__init__(f"worker '{process.name}' exited unexpectedly with exit code {process.exitcode}")


class ProcessSupervisor:
    """Own started workers and perform bounded cooperative/terminate/kill cleanup.

    ``request_stop`` and optional ``unblock`` are parent-owned protocol hooks;
    neither the supervisor nor workers need to know an algorithm or transport.
    Clean exits are unexpected unless explicitly allowed by ``check_exits``.
    Shutdown errors are logged in isolation, never masking the original error or
    preventing cleanup attempts for the other children. The context is single-use.
    """

    def __init__(
        self,
        request_stop: Callable[[], None],
        *,
        unblock: Callable[[], None] | None = None,
        grace_s: float = 8.0,
        terminate_s: float = 3.0,
        kill_s: float = 3.0,
    ) -> None:
        if any(not math.isfinite(t) or t < 0 for t in (grace_s, terminate_s, kill_s)):
            raise ValueError("shutdown timeouts must be finite and nonnegative")
        self._request_stop = request_stop
        self._unblock = unblock
        self._timeouts = (grace_s, terminate_s, kill_s)
        self._processes: list[BaseProcess] = []
        self._closed = False
        self._cleanup_errors: list[BaseException] = []

    def __enter__(self) -> ProcessSupervisor:
        return self

    def start(self, process: BaseProcess) -> None:
        """Register before starting, including a start that raises after spawning."""
        if self._closed:
            raise RuntimeError("process supervisor is closed")
        self._processes.append(process)
        process.start()

    def check_exits(self, *, allow_clean_exit: Callable[[BaseProcess], bool] | None = None) -> list[BaseProcess]:
        """Raise on nonzero or unapproved zero exits; return allowed exits.

        Call during readiness as well as monitoring. In particular, a worker
        exiting zero without reporting ready must not leave a barrier waiting.
        """
        exited = []
        for process in self._processes:
            if process.exitcode is None:
                continue
            if process.exitcode != 0 or allow_clean_exit is None or not allow_clean_exit(process):
                raise ProcessExitError(process)
            exited.append(process)
        return exited

    def close(self) -> None:
        """Stop, unblock and reap each child, escalating with finite joins."""
        if self._closed:
            return
        self._closed = True
        for callback in (self._request_stop, self._unblock):
            if callback is not None:
                self._attempt(callback)
        for process in self._processes:
            self._attempt(lambda p=process: self._shutdown(p))
        if self._cleanup_errors:
            raise RuntimeError("process supervisor cleanup failed") from self._cleanup_errors[0]

    def _shutdown(self, process: BaseProcess) -> None:
        # A failed spawn may never have acquired a pid; joining it raises.
        if process.pid is None:
            return
        self._attempt(lambda: process.join(timeout=self._timeouts[0]))
        for action, timeout in ((process.terminate, self._timeouts[1]), (process.kill, self._timeouts[2])):
            if not process.is_alive():
                break
            self._attempt(action)
            self._attempt(lambda t=timeout: process.join(timeout=t))
        if process.is_alive():
            raise RuntimeError(f"worker {process.name} is still alive after shutdown escalation")

    def _attempt(self, action: Callable[[], None]) -> None:
        try:
            action()
        except BaseException as exc:  # cleanup must also survive a second Ctrl+C
            self._cleanup_errors.append(exc)
            _LOG.warning("process supervisor cleanup failed", exc_info=True)

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if exc is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException:
                _LOG.warning("cleanup failed while handling worker error", exc_info=True)
