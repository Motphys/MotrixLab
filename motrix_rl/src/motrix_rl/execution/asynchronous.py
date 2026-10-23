# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""A bounded, algorithm-neutral collector/learner reference pipeline."""

from __future__ import annotations

import math
from collections.abc import Callable
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import Any, Protocol

from motrix_rl.execution.contracts import (
    Cancellation,
    ExecutionCancelled,
    Parameters,
    Progress,
    Sample,
)


class Collector(Protocol):
    def collect(self, parameters: Parameters, cancellation: Cancellation) -> Any: ...

    def close(self) -> None: ...


class Learner(Protocol):
    def learn(self, sample: Sample, cancellation: Cancellation) -> Parameters | None: ...

    def close(self) -> None: ...


class AsyncExecution:
    """Collect on one daemon thread; learn, publish and observe on the caller thread.

    The queue bounds waiting samples, with at most one additional collection in
    flight. Exactly ``max_iterations`` samples are collected and learned on normal
    completion, including samples rejected by a learner returning None. Algorithms
    own acceptance, optimizer scheduling, and strictly increasing parameter versions.
    Each collection captures the latest published Parameters at its boundary; queued
    samples can therefore be stale. Payloads transfer ownership, and snapshots must
    be immutable/detached: no copying or algorithm-specific synchronization is done.

    Cancellation discards queued/in-flight work and raises ExecutionCancelled. Queue
    waits poll cancellation; callbacks must cooperate with the supplied token. Python
    cannot kill threads: after shutdown_timeout a still-running collector (including
    its close callback) produces RuntimeError, chained from any original failure.
    Its daemon thread may remain alive until the callback returns. Collector.close
    runs on that worker, never concurrently with collect (or on the caller if thread
    startup fails); Learner.close runs on the caller after the bounded join.
    Learner/observer/learner-close calls themselves
    cannot be preempted or time-limited. Resources are owned for a valid run and are
    closed even for zero iterations or a pre-cancelled token. Worker SystemExit and
    KeyboardInterrupt become RuntimeError with the original exception as the cause:
    a collector exiting is a failure, not caller cancellation or successful training.
    Other callback failures, including caller-thread SystemExit/KeyboardInterrupt, propagate
    unchanged; an earlier failure takes precedence over cleanup failures. Successful
    runs do not cancel the caller's token.
    """

    _poll_timeout = 0.02

    def __init__(self, *, queue_capacity: int = 1, shutdown_timeout: float = 5.0) -> None:
        if isinstance(queue_capacity, bool) or not isinstance(queue_capacity, int) or queue_capacity <= 0:
            raise ValueError("queue_capacity must be a positive integer")
        if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be positive and finite")
        self.queue_capacity = queue_capacity
        self.shutdown_timeout = shutdown_timeout

    def run(
        self,
        collector: Collector,
        learner: Learner,
        *,
        initial_parameters: Parameters,
        max_iterations: int,
        cancellation: Cancellation | None = None,
        on_step: Callable[[Progress], None] | None = None,
    ) -> Progress:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
            raise ValueError("max_iterations must be a nonnegative integer")
        cancellation = cancellation if cancellation is not None else Cancellation()
        queue: Queue[Sample] = Queue(maxsize=self.queue_capacity)
        stop = Event()
        lock = Lock()
        parameters = initial_parameters
        worker_errors: list[BaseException] = []
        progress = Progress(policy_version=parameters.version)

        def record_error(error: BaseException) -> None:
            with lock:
                worker_errors.append(error)

        def raise_worker_error() -> None:
            with lock:
                error = worker_errors[0] if worker_errors else None
            if isinstance(error, (SystemExit, KeyboardInterrupt)):
                # Do not reinterpret a worker exit as caller-thread success/cancel.
                raise RuntimeError("Collector worker exited unexpectedly") from error
            if error is not None:
                raise error

        def collect() -> None:
            try:
                for sequence in range(max_iterations):
                    if stop.is_set():
                        break
                    cancellation.raise_if_cancelled()
                    with lock:
                        snapshot = parameters
                    value = collector.collect(snapshot, cancellation)
                    sample = Sample(value, snapshot.version, sequence)
                    while not stop.is_set():
                        cancellation.raise_if_cancelled()
                        try:
                            queue.put(sample, timeout=self._poll_timeout)
                            break
                        except Full:
                            pass
            except BaseException as error:
                if not isinstance(error, ExecutionCancelled) or not (stop.is_set() or cancellation.cancelled):
                    record_error(error)
            finally:
                try:
                    collector.close()
                except BaseException as error:
                    record_error(error)

        worker = Thread(target=collect, name="motrix-collector", daemon=True)
        failure: BaseException | None = None
        started = False
        try:
            worker.start()
            started = True
            cancellation.raise_if_cancelled()
            for sequence in range(max_iterations):
                while True:
                    raise_worker_error()
                    cancellation.raise_if_cancelled()
                    try:
                        sample = queue.get(timeout=self._poll_timeout)
                        break
                    except Empty:
                        pass
                raise_worker_error()
                cancellation.raise_if_cancelled()
                published = learner.learn(sample, cancellation)
                raise_worker_error()
                cancellation.raise_if_cancelled()
                if published is not None:
                    with lock:
                        if published.version <= parameters.version:
                            raise ValueError("Published parameter versions must increase")
                        parameters = published
                progress = Progress(
                    iterations=sequence + 1,
                    samples=sequence + 1,
                    updates=progress.updates + (published is not None),
                    policy_version=parameters.version,
                )
                if on_step is not None:
                    on_step(progress)
                cancellation.raise_if_cancelled()
        except BaseException as error:
            failure = error
        finally:
            stop.set()
            if failure is not None:
                cancellation.cancel()
            if started:
                worker.join(timeout=self.shutdown_timeout)
            else:
                try:
                    collector.close()
                except BaseException as error:
                    if failure is None:
                        failure = error
            try:
                learner.close()
            except BaseException as error:
                if failure is None:
                    failure = error

        if worker.is_alive():
            raise RuntimeError(
                "Collector did not stop within shutdown_timeout; Python threads cannot be killed"
            ) from failure
        if failure is not None:
            raise failure
        raise_worker_error()
        cancellation.raise_if_cancelled()
        return progress
