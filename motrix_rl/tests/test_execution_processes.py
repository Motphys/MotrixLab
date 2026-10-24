# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""CPU-only lifecycle tests; fake children keep failure injection deterministic."""

from __future__ import annotations

import multiprocessing as mp

import pytest

from motrix_rl.execution.processes import ProcessExitError, ProcessSupervisor


class Child:
    def __init__(self, name="child", *, start_error=None, exitcode=None, stubborn=False):
        self.name = name
        self.pid = None
        self.exitcode = exitcode
        self.start_error = start_error
        self.stubborn = stubborn
        self.terminated = False
        self.killed = False
        self.joins = []

    def start(self):
        if self.start_error:
            raise self.start_error
        self.pid = 1

    def is_alive(self):
        return self.pid is not None and self.exitcode is None

    def join(self, timeout):
        self.joins.append(timeout)

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self.exitcode = -15

    def kill(self):
        self.killed = True
        self.exitcode = -9


def test_partial_start_reaps_started_children_and_propagates_error():
    stopped = []
    first, second = Child(), Child(start_error=OSError("spawn failed"))
    with pytest.raises(OSError, match="spawn failed"):
        with ProcessSupervisor(lambda: stopped.append(True), grace_s=0) as supervisor:
            supervisor.start(first)
            supervisor.start(second)
    assert stopped == [True]
    assert not first.is_alive()
    assert second.joins == []


def test_cancellation_unblocks_and_escalates_without_suppressing():
    callbacks = []
    child = Child(stubborn=True)
    with pytest.raises(KeyboardInterrupt):
        with ProcessSupervisor(
            lambda: callbacks.append("stop"),
            unblock=lambda: callbacks.append("release"),
            grace_s=0.1,
            terminate_s=0.2,
            kill_s=0.3,
        ) as supervisor:
            supervisor.start(child)
            raise KeyboardInterrupt
    assert callbacks == ["stop", "release"]
    assert child.terminated and child.killed
    assert child.joins == [0.1, 0.2, 0.3]
    supervisor.close()
    assert callbacks == ["stop", "release"]


@pytest.mark.parametrize("exitcode", [0, 7, -9])
def test_exit_before_readiness_is_error_including_zero(exitcode):
    child = Child(exitcode=exitcode)
    with ProcessSupervisor(lambda: None) as supervisor:
        supervisor.start(child)
        with pytest.raises(ProcessExitError, match=f"exit code {exitcode}") as error:
            supervisor.check_exits()
        assert error.value.process is child


def test_only_explicitly_expected_clean_exits_are_allowed():
    child = Child(exitcode=0)
    with ProcessSupervisor(lambda: None) as supervisor:
        supervisor.start(child)
        assert supervisor.check_exits(allow_clean_exit=lambda p: p is child) == [child]
        child.exitcode = 2
        with pytest.raises(ProcessExitError):
            supervisor.check_exits(allow_clean_exit=lambda p: True)


def test_cleanup_failures_do_not_mask_exception_or_skip_other_children():
    def fail():
        raise RuntimeError("cleanup failed")

    bad, good = Child(), Child()
    with pytest.raises(ValueError, match="original"):
        with ProcessSupervisor(fail, unblock=fail, grace_s=0) as supervisor:
            supervisor.start(bad)
            supervisor.start(good)
            bad.join = lambda timeout: fail()
            bad.terminate = fail
            raise ValueError("original")
    assert bad.killed
    assert not good.is_alive()


def test_start_failure_after_pid_assignment_is_still_reaped():
    child = Child()

    def partial_start():
        child.pid = 1
        raise OSError("after spawn")

    child.start = partial_start
    with pytest.raises(OSError):
        with ProcessSupervisor(lambda: None, grace_s=0) as supervisor:
            supervisor.start(child)
    assert not child.is_alive()


@pytest.mark.parametrize("timeout", [-1.0, float("inf"), float("nan")])
def test_shutdown_timeout_must_be_finite_and_nonnegative(timeout):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ProcessSupervisor(lambda: None, grace_s=timeout)


def test_cleanup_failure_is_reported_without_an_original_error():
    child = Child(stubborn=True)
    child.kill = lambda: None
    with pytest.raises(RuntimeError, match="cleanup failed"):
        with ProcessSupervisor(lambda: None, grace_s=0, terminate_s=0, kill_s=0) as supervisor:
            supervisor.start(child)
    assert child.is_alive()


def test_stop_callback_failure_is_reported_after_other_children_are_reaped():
    def fail():
        raise OSError("stop failed")

    child = Child()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        with ProcessSupervisor(fail, grace_s=0) as supervisor:
            supervisor.start(child)
    assert not child.is_alive()


def _wait_for_stop(stop):
    stop.wait()


def _exit_before_ready(code):
    raise SystemExit(code)


@pytest.mark.integration
@pytest.mark.parametrize("code", [0, 7])
def test_spawned_worker_exit_before_readiness_is_failure(code):
    ctx = mp.get_context("spawn")
    child = ctx.Process(target=_exit_before_ready, args=(code,))
    with ProcessSupervisor(lambda: None, grace_s=1) as supervisor:
        supervisor.start(child)
        child.join(timeout=10)
        assert not child.is_alive()
        with pytest.raises(ProcessExitError, match=f"exit code {code}"):
            supervisor.check_exits()
    assert child.exitcode == code


@pytest.mark.integration
def test_spawned_cpu_worker_is_reaped_on_cancellation():
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    child = ctx.Process(target=_wait_for_stop, args=(stop,))
    with pytest.raises(KeyboardInterrupt):
        with ProcessSupervisor(stop.set, grace_s=5) as supervisor:
            supervisor.start(child)
            raise KeyboardInterrupt
    assert child.exitcode == 0
    assert not child.is_alive()
