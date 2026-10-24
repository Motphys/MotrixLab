# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deterministic edge contracts for public plan supervision.

The fake processes expose messages exactly after a parent drain, reproducing
queue-feeder races without real subprocess timing or wall-clock sleeps.
"""

from contextlib import nullcontext
from queue import Queue
from threading import Event
from types import SimpleNamespace

import pytest

from motrix_rl.execution import plan_execution
from motrix_rl.execution.contracts import ExecutionCancelled
from motrix_rl.execution.plan_execution import PlanExecution
from motrix_rl.execution.training import TrainingDecision, TrainingPlan, WorkerSpec
from motrix_rl.execution.workers import WorkerMessage


def _target(runtime):
    """Pickle-friendly declaration; fake processes never execute the target."""


class _Process:
    def __init__(self, *, name, target, args):
        self.name = name
        self.runtime = args[0]
        self.exitcode = None
        self.pid = None
        self.joined = False

    def start(self):
        self.pid = 1
        self.runtime.report(kind="ready")

    def join(self, timeout=None):
        self.joined = True
        self.exitcode = 0 if self.exitcode is None else self.exitcode

    def is_alive(self):
        return self.exitcode is None

    def terminate(self):
        self.exitcode = -15

    def kill(self):
        self.exitcode = -9


@pytest.fixture
def rig(monkeypatch, tmp_path):
    processes = {}

    def process(**kwargs):
        child = _Process(**kwargs)
        processes[child.name] = child
        return child

    clock = SimpleNamespace(now=0.0)

    def sleep(duration):
        clock.now += duration
        if clock.now > 2:
            raise AssertionError("execution failed to enforce its completion deadline")

    monkeypatch.setattr(
        plan_execution,
        "time",
        SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now, sleep=sleep),
    )
    monkeypatch.setattr(plan_execution, "spawn_placement", lambda *args: nullcontext())
    monkeypatch.setattr(plan_execution, "inherit_log_stdio", lambda *args: nullcontext())
    build = SimpleNamespace(queue=Queue, event=Event, _context=SimpleNamespace(Process=process))
    plan = TrainingPlan((WorkerSpec("collector-0", "collector", _target),), None)
    return SimpleNamespace(build=build, plan=plan, processes=processes, run_dir=tmp_path, clock=clock)


@pytest.mark.parametrize("worker_id", ["", ".", "..", "../escape", "/tmp/escape", "nested/worker", r"nested\worker"])
def test_worker_ids_cannot_escape_runtime_output_directories(rig, worker_id):
    with pytest.raises(ValueError, match="worker ID|filename|path"):
        plan = TrainingPlan((WorkerSpec(worker_id, "collector", _target),), None)
        PlanExecution(rig.build, plan, run_dir=rig.run_dir)
    assert not rig.processes


def test_finish_delivered_between_drain_and_exit_check_is_accepted(rig):
    delivered = False
    seen = []

    def observe(reports):
        nonlocal delivered
        seen.extend(reports.reports.get("collector-0", ()))
        child = rig.processes["collector-0"]
        if child.runtime.start_event.is_set() and not delivered:
            delivered = True
            # The preceding get_nowait observed Empty. The feeder now delivers
            # finish and the OS reports zero exit before the next parent check.
            child.runtime.finish({"completed": 1})
            child.exitcode = 0

    def decide(progress):
        finished = progress.workers["collector-0"].finished
        return TrainingDecision(finished, frozenset({"collector-0"}) if finished else frozenset())

    PlanExecution(rig.build, rig.plan, run_dir=rig.run_dir, poll_interval=0.01).run(observe, decide)
    assert {"completed": 1} in seen
    assert rig.processes["collector-0"].joined


def test_error_feeder_lag_cannot_turn_worker_failure_into_cancellation(rig):
    delivered = False

    def observe(reports):
        nonlocal delivered
        child = rig.processes["collector-0"]
        if child.runtime.start_event.is_set() and not delivered:
            delivered = True
            child.runtime.request_stop()
            child.runtime.event_queue.put(
                WorkerMessage(
                    "error",
                    child.name,
                    {"phase": "target", "message": "broken update", "traceback": "ORIGINAL WORKER TRACEBACK"},
                )
            )

    with pytest.raises(RuntimeError, match="collector-0.*failed") as caught:
        PlanExecution(rig.build, rig.plan, run_dir=rig.run_dir, poll_interval=0.01).run(
            observe, lambda progress: TrainingDecision(False)
        )
    assert not isinstance(caught.value, ExecutionCancelled)
    assert (rig.run_dir / "async_errors" / "collector-0_error.log").read_text() == "ORIGINAL WORKER TRACEBACK"
    assert rig.processes["collector-0"].joined


def test_accepted_peer_finish_has_bounded_wait_for_remaining_completion(rig):
    rig.plan = TrainingPlan(
        (
            WorkerSpec("collector-0", "collector", _target),
            WorkerSpec("learner", "learner", _target),
        ),
        None,
    )
    delivered = False

    def observe(reports):
        nonlocal delivered
        child = rig.processes["collector-0"]
        if child.runtime.start_event.is_set() and not delivered:
            delivered = True
            child.runtime.finish()

    def decide(progress):
        finished = progress.workers["collector-0"].finished
        if finished:
            rig.processes["collector-0"].exitcode = 0
        # A finished collector is legitimate, but the algorithm's learner
        # boundary has not arrived and must not be waited on indefinitely.
        return TrainingDecision(False, frozenset({"collector-0"}) if finished else frozenset())

    with pytest.raises(TimeoutError):
        PlanExecution(
            rig.build,
            rig.plan,
            run_dir=rig.run_dir,
            poll_interval=0.005,
            completion_timeout=0.01,
        ).run(observe, decide)
    assert rig.clock.now < 2
    assert all(child.joined for child in rig.processes.values())
