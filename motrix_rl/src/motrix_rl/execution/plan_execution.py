# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Process materialization and protocol execution for declarative training plans."""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path

from motrix_rl.execution.contracts import ExecutionCancelled
from motrix_rl.execution.messages import MessageInbox
from motrix_rl.execution.processes import ProcessExitError, ProcessSupervisor
from motrix_rl.execution.training import TrainingEvent, TrainingProgress, WorkerReports, WorkerStatus
from motrix_rl.execution.worker_runtime import inherit_log_stdio
from motrix_rl.execution.workers import WorkerRuntime, worker_entry
from motrix_rl.numa import spawn_placement


class PlanExecution:
    """Own raw processes; expose only semantic messages and immutable statuses.

    ``observe`` consumes report batches, while ``decide`` interprets a snapshot
    of runtime states. Neither callback receives OS handles. This internal
    execution mechanism is used by the trainer template, not an algorithm hook.
    """

    def __init__(
        self, build, plan, *, run_dir: Path, startup_timeout=600.0, poll_interval=0.25, completion_timeout=60.0
    ):
        if any(
            not math.isfinite(value) or value <= 0 for value in (startup_timeout, poll_interval, completion_timeout)
        ):
            raise ValueError("execution timeouts must be positive and finite")
        self._build = build
        self._plan = plan
        self._run_dir = Path(run_dir)
        self._startup_timeout = startup_timeout
        self._poll_interval = poll_interval
        self._completion_timeout = completion_timeout
        self._queue = build.queue()
        self._start = build.event()
        self._stop = build.event()
        ids = [worker.id for worker in plan.workers]
        if len(set(ids)) != len(ids):
            raise ValueError("worker IDs must be unique")
        if any(
            not worker_id
            or any(not (char.isalnum() or char in "-_ .") for char in worker_id)
            or worker_id in {".", ".."}
            or worker_id.startswith(".")
            for worker_id in ids
        ):
            raise ValueError("worker IDs must be safe filename components")
        self._inbox = MessageInbox(self._queue, set(ids))
        self._processes = {}
        self._released = False
        self._completed_workers: frozenset[str] = frozenset()
        self._error_paths = {}
        self._crashed: set[str] = set()
        self._started_event = False

    def _snapshot(self):
        workers = {}
        for worker in self._plan.workers:
            process = self._processes.get(worker.id)
            code = process.exitcode if process is not None else None
            ready = worker.id in self._inbox.ready
            finished = worker.id in self._inbox.finished
            state = "starting"
            if ready:
                state = "running" if self._released else "ready"
            if finished:
                state = "finished"
            if code is not None:
                state = "exited" if code == 0 else "failed"
            workers[worker.id] = WorkerStatus(state, ready, finished, code is not None, code)
        return TrainingProgress(workers=workers, cancelled=self._stop.is_set())

    def _persist_errors(self):
        if self._inbox.errors:
            error_dir = self._run_dir / "async_errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            for error in self._inbox.errors:
                path = error_dir / f"{error.worker_id}_error.log"
                if error.worker_id not in self._error_paths:
                    path.write_text(error.payload["traceback"])
                    self._error_paths[error.worker_id] = path
                    spec = next((item for item in self._plan.workers if item.id == error.worker_id), None)
                    if spec is not None:
                        role_path = error_dir / f"{spec.role}_error.log"
                        if role_path != path:
                            role_path.write_text(error.payload["traceback"])

    def _check_errors(self):
        if self._inbox.errors:
            message = self._inbox.errors[0]
            raise RuntimeError(
                f"worker {message.worker_id} failed during {message.payload['phase']} "
                f"(traceback: {self._error_paths[message.worker_id]})"
            )

    def _observe(self, observe, *, force=False):
        self._inbox.drain()
        self._persist_errors()
        events = []
        for error in self._inbox.errors:
            if error.worker_id not in self._crashed:
                self._crashed.add(error.worker_id)
                process = self._processes[error.worker_id]
                events.append(
                    TrainingEvent(
                        "worker_crashed",
                        step=0,
                        data={
                            "worker": error.worker_id,
                            "exitcode": process.exitcode,
                            "error_log": f"async_errors/{error.worker_id}_error.log",
                        },
                    )
                )
        try:
            observe(WorkerReports(self._inbox.take_reports(), now=time.time(), force=force, events=tuple(events)))
        finally:
            self._check_errors()

    def run(self, observe, decide, startup=None):
        """Execute one plan and final-drain reports before build resources close."""
        primary = None
        supervisor = ProcessSupervisor(self._stop.set, unblock=self._start.set)
        try:
            log_dir = self._run_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            for spec in self._plan.workers:
                settings = replace(spec.settings, log_path=log_dir / f"{spec.id}.log")
                runtime = WorkerRuntime(spec.id, self._queue, self._start, self._stop)
                process = self._build._context.Process(
                    name=spec.id,
                    target=worker_entry,
                    args=(runtime, spec.target, spec.args, dict(spec.kwargs), settings),
                )
                self._processes[spec.id] = process
                with (
                    spawn_placement(settings.numa_node, spec.id),
                    inherit_log_stdio(settings.log_path),
                ):
                    supervisor.start(process)
            deadline = time.monotonic() + self._startup_timeout
            expected = set(self._processes)
            while True:
                self._observe(observe)
                if startup is not None:
                    startup(self._plan.workers, frozenset(self._inbox.ready), False)
                if any(process.exitcode is not None for process in self._processes.values()):
                    for process in self._processes.values():
                        if process.exitcode is not None:
                            process.join(timeout=0)
                    self._observe(observe, force=True)
                    supervisor.check_exits()
                if expected <= self._inbox.ready:
                    break
                if self._stop.is_set():
                    raise ExecutionCancelled("training cancelled during startup")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"worker startup timed out; missing: {sorted(expected - self._inbox.ready)}")
                time.sleep(self._poll_interval)
            self._released = True
            self._start.set()
            observe(
                WorkerReports(
                    now=time.time(),
                    force=True,
                    events=(
                        TrainingEvent(
                            "run_started",
                            step=0,
                            data={"workers": len(self._plan.workers)},
                        ),
                    ),
                )
            )
            if startup is not None:
                startup(self._plan.workers, frozenset(self._inbox.ready), True)
            completion_deadline = None
            checked_exits = set()
            while True:
                self._observe(observe)
                # A child may exit between draining the queue and sampling its
                # status. Joining its already-exited process flushes its feeder;
                # consume those final messages before classifying the exit.
                for worker_id, process in self._processes.items():
                    if process.exitcode is not None and worker_id not in checked_exits:
                        process.join(timeout=0)
                        checked_exits.add(worker_id)
                        self._observe(observe)
                snapshot = self._snapshot()
                decision = decide(snapshot)
                if self._inbox.finished and completion_deadline is None:
                    completion_deadline = time.monotonic() + self._completion_timeout
                # finish is an explicit declaration AND must be allowed by the
                # algorithm budget. A clean OS exit alone cannot prove success.
                # Classify only exits in the same reconciled snapshot. A peer
                # exiting during algorithm evaluation is consumed next round,
                # rather than checked against a stale completion decision.
                for worker_id, status in snapshot.workers.items():
                    if status.exited and (status.exitcode != 0 or not status.finished):
                        raise ProcessExitError(self._processes[worker_id])
                if decision.complete:
                    self._completed_workers = decision.completed_workers
                    break
                if self._stop.is_set():
                    self._observe(observe, force=True)
                    if self._inbox.errors:
                        self._check_errors()
                    raise ExecutionCancelled("training cancelled")
                if decision.complete:
                    self._completed_workers = decision.completed_workers
                    break
                if completion_deadline is not None and time.monotonic() >= completion_deadline:
                    raise TimeoutError("workers did not complete within the peer completion deadline")
                time.sleep(self._poll_interval)
        except BaseException as error:
            primary = error
            raise
        finally:
            cleanup_error = None
            try:
                supervisor.close()
            except BaseException as error:
                cleanup_error = error
            try:
                self._observe(observe, force=True)
                if primary is None:
                    supervisor.check_exits(
                        allow_clean_exit=lambda process: (
                            process.name not in self._completed_workers or process.name in self._inbox.finished
                        )
                    )
            except BaseException as final_error:
                # A worker error sent just before its stop event may arrive
                # only after reaping. It outranks a provisional cancellation,
                # but never replaces an actual parent/algorithm exception.
                if primary is None or (
                    isinstance(primary, (ExecutionCancelled, ProcessExitError)) and self._inbox.errors
                ):
                    raise final_error
            if primary is None and cleanup_error is not None:
                raise cleanup_error
