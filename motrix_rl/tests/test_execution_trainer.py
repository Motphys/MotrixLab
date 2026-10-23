# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle contracts for the declarative AsyncTrainerBase API."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from motrix_rl.execution import plan_execution
from motrix_rl.execution.processes import ProcessExitError
from motrix_rl.execution.trainer import AsyncTrainerBase
from motrix_rl.execution.training import TrainingDecision, TrainingPlan, WorkerSpec
from motrix_rl.execution.workers import WorkerSettings


class FakeProcess:
    behaviors = {}

    def __init__(self, name="worker", target=None, args=(), **_kwargs):
        self.name = name
        self.target = target
        self.args = args
        self.pid = None
        self.exitcode = None
        self.started = False
        self.start_error = self.behaviors.get(name)

    def start(self):
        if isinstance(self.start_error, BaseException):
            raise self.start_error
        self.started = True
        self.pid = 1
        behavior = self.start_error or "ready"
        runtime = self.args[0]
        if behavior == "ready":
            runtime.ready = True
            runtime.report(kind="ready")
        elif behavior == "exit-zero":
            self.exitcode = 0
        elif behavior == "late-failure":
            runtime.ready = True
            runtime.report(kind="ready")
        elif behavior == "never-ready":
            return

    def is_alive(self):
        return self.pid is not None and self.exitcode is None

    def join(self, timeout):
        if self.start_error == "late-failure":
            self.exitcode = 7

    def terminate(self):
        self.exitcode = -15

    def kill(self):
        self.exitcode = -9


class FakeContext:
    Process = FakeProcess


class FakeBuild:
    def __init__(self, resources):
        self._context = FakeContext()
        self._queue = queue.Queue()
        self._resources = resources

    def queue(self):
        return self._queue

    def event(self):
        return threading.Event()

    def own(self, resource, close=None):
        self._resources.callback(close or resource.close)
        return resource


class Presentation:
    def __init__(self, *_args, **_kwargs):
        self.frames = []
        self.closed = False

    def publish(self, metrics):
        self.frames.append(metrics)

    def close(self):
        self.closed = True


class ToyTrainer(AsyncTrainerBase):
    def __init__(self, tmp_path, *, workers=("worker",), build_failure=None):
        super().__init__(context=SimpleNamespace(run_dir=Path(tmp_path), env_name="toy", metrics=None))
        self.worker_ids = workers
        self.build_failure = build_failure
        self.processes = {}
        self.calls = []

    def build_training(self, runtime):
        runtime._context = FakeContext()
        if self.build_failure is not None:
            runtime._resources.callback(self.build_failure[1])
            raise self.build_failure[0]
        specs = tuple(WorkerSpec(worker_id, "toy", _noop, settings=WorkerSettings()) for worker_id in self.worker_ids)
        return TrainingPlan(specs, self)

    def evaluate_progress(self, state, progress):
        self.calls.append(progress)
        if progress.cancelled:
            raise KeyboardInterrupt()
        if all(status.finished for status in progress.workers.values()):
            for worker_id in progress.workers:
                _processes[worker_id].exitcode = 0
            return TrainingDecision(True, frozenset(progress.workers))
        if all(status.ready for status in progress.workers.values()):
            for worker_id, status in progress.workers.items():
                if status.state == "running":
                    runtime = _processes[worker_id].args[0]
                    runtime.finish({"value": 42})
        return TrainingDecision(False)

    def play(self, policy):
        raise NotImplementedError


_processes = {}


def _noop(_runtime):
    return None


def _install(monkeypatch):
    monkeypatch.setattr("motrix_rl.execution.trainer.BuildRuntime", FakeBuild)
    monkeypatch.setattr("motrix_rl.execution.trainer.TrainingPresentation", Presentation)
    monkeypatch.setattr(plan_execution, "time", plan_execution.time)


def _run(monkeypatch, tmp_path, trainer):
    _install(monkeypatch)
    original = FakeProcess.__init__

    def capture(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _processes[self.name] = self

    monkeypatch.setattr(FakeProcess, "__init__", capture)
    trainer.train()


def test_base_runs_readiness_poll_and_completion(monkeypatch, tmp_path):
    FakeProcess.behaviors = {}
    trainer = ToyTrainer(tmp_path)
    _run(monkeypatch, tmp_path, trainer)
    assert any(progress.workers["worker"].state == "running" for progress in trainer.calls)
    assert trainer.calls[-1].workers["worker"].finished


@pytest.mark.parametrize("field", ["startup_timeout", "poll_interval"])
def test_plan_rejects_nonpositive_timeout(field, tmp_path):
    from motrix_rl.execution.plan_execution import PlanExecution

    values = {"run_dir": tmp_path, "startup_timeout": 1, "poll_interval": 1}
    values[field] = 0
    with pytest.raises(ValueError, match="timeouts"):
        PlanExecution(FakeBuild(__import__("contextlib").ExitStack()), TrainingPlan((), None), **values)


def test_partial_spawn_failure_reaps_started_peer_and_preserves_error(monkeypatch, tmp_path):
    failure = OSError("spawn failed")
    FakeProcess.behaviors = {"second": failure}
    trainer = ToyTrainer(tmp_path, workers=("first", "second"))
    with pytest.raises(OSError) as caught:
        _run(monkeypatch, tmp_path, trainer)
    assert caught.value is failure
    assert _processes["first"].exitcode is not None
    assert _processes["second"].pid is None


def test_startup_timeout_reaps_worker(monkeypatch, tmp_path):
    FakeProcess.behaviors = {"worker": "never-ready"}
    monkeypatch.setenv("MOTRIX_BOOT_TIMEOUT_S", "0.001")
    trainer = ToyTrainer(tmp_path)
    with pytest.raises(TimeoutError, match="startup timed out"):
        _run(monkeypatch, tmp_path, trainer)
    assert _processes["worker"].exitcode is not None


def test_prepare_failure_preserves_original_when_resource_cleanup_fails(monkeypatch, tmp_path):
    failure = ValueError("prepare failed")
    cleanup = RuntimeError("resource close failed")
    trainer = ToyTrainer(tmp_path, build_failure=(failure, lambda: (_ for _ in ()).throw(cleanup)))
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, trainer)
    assert caught.value is failure


def test_clean_exit_before_ready_is_failure(monkeypatch, tmp_path):
    FakeProcess.behaviors = {"worker": "exit-zero"}
    trainer = ToyTrainer(tmp_path)
    with pytest.raises(ProcessExitError):
        _run(monkeypatch, tmp_path, trainer)


def test_failure_reported_only_during_final_join_is_not_success(monkeypatch, tmp_path):
    FakeProcess.behaviors = {"worker": "late-failure"}
    trainer = ToyTrainer(tmp_path)
    with pytest.raises(ProcessExitError):
        _run(monkeypatch, tmp_path, trainer)
