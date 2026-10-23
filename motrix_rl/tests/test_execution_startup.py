# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Startup gate and opaque resource delivery contracts without an algorithm."""

from queue import Queue
from threading import Event, Thread

import pytest

from motrix_rl.execution.contracts import ExecutionCancelled
from motrix_rl.execution.startup import StartupBarrier, await_startup_resource


class Context:
    Queue = staticmethod(Queue)
    Event = staticmethod(Event)


def test_startup_gate_reports_and_releases_worker():
    gate = StartupBarrier(Context())
    gate.report_ready("sampler", 3)
    assert gate.drain_ready() == {("sampler", 3)}
    assert gate.drain_ready() == set()
    entered, finished = Event(), Event()

    def worker():
        entered.set()
        gate.wait_for_start(lambda: False)
        finished.set()

    thread = Thread(target=worker, daemon=True)
    thread.start()
    assert entered.wait(1)
    assert not finished.is_set()
    gate.release()
    thread.join(timeout=1)
    assert finished.is_set()


def test_startup_gate_cancel_does_not_allow_algorithm_work():
    gate = StartupBarrier(Context())
    with pytest.raises(ExecutionCancelled):
        gate.wait_for_start(lambda: True)
    gate.release()
    with pytest.raises(ExecutionCancelled):
        gate.wait_for_start(lambda: True)


def test_disabled_gate_does_not_report_or_block():
    gate = StartupBarrier(Context(), enabled=False)
    gate.report_ready("sampler", 1)
    assert gate.drain_ready() == set()
    gate.wait_for_start(lambda: True)


def test_startup_resource_delivery_preserves_opaque_values_and_none():
    queue = Queue()
    for value in (None, {"slots": [1, 2]}):
        queue.put(value)
        assert await_startup_resource(queue, should_stop=lambda: False, timeout=1, description="slots") is value


def test_startup_resource_wait_checks_cancel_while_blocked():
    queue = Queue()
    waiting, stop, finished = Event(), Event(), Event()
    errors = []

    def should_stop():
        waiting.set()
        return stop.is_set()

    def receive():
        try:
            await_startup_resource(queue, should_stop=should_stop, timeout=2, description="slots")
        except ExecutionCancelled as error:
            errors.append(error)
        finally:
            finished.set()

    thread = Thread(target=receive, daemon=True)
    thread.start()
    assert waiting.wait(1)
    stop.set()
    thread.join(timeout=1)
    assert finished.is_set() and len(errors) == 1


def test_startup_resource_wait_observes_cancel_and_timeout():
    queue = Queue()
    with pytest.raises(ExecutionCancelled, match="slots"):
        await_startup_resource(queue, should_stop=lambda: True, timeout=1, description="slots")
    with pytest.raises(TimeoutError, match="slots"):
        await_startup_resource(queue, should_stop=lambda: False, timeout=0.001, description="slots")


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_startup_resource_rejects_unbounded_timeout(timeout):
    with pytest.raises(ValueError):
        await_startup_resource(Queue(), should_stop=lambda: False, timeout=timeout, description="slots")
