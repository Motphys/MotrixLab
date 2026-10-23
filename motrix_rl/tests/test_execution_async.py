# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contracts of the algorithm-neutral asynchronous executor."""

from threading import Event, current_thread

import pytest

from motrix_rl.execution.asynchronous import AsyncExecution
from motrix_rl.execution.contracts import Cancellation, ExecutionCancelled, Parameters, Progress


class Collector:
    def __init__(self, callback=None):
        self.callback = callback
        self.parameters = []
        self.closed = Event()
        self.thread = None
        self.close_thread = None

    def collect(self, parameters, cancellation):
        self.thread = current_thread()
        self.parameters.append(parameters)
        if self.callback is not None:
            return self.callback(parameters, cancellation)
        return len(self.parameters) - 1

    def close(self):
        self.close_thread = current_thread()
        self.closed.set()


class Learner:
    def __init__(self, callback=None):
        self.callback = callback
        self.samples = []
        self.closed = False
        self.threads = []

    def learn(self, sample, cancellation):
        self.threads.append(current_thread())
        self.samples.append(sample)
        if self.callback is not None:
            return self.callback(sample, cancellation)
        return None

    def close(self):
        self.threads.append(current_thread())
        self.closed = True


def test_order_data_versions_rejections_and_main_thread_hooks():
    collector = Collector()
    learner = Learner(
        lambda sample, _: Parameters(sample.value, 10 + sample.sequence) if sample.sequence % 2 == 0 else None
    )
    observed = []
    token = Cancellation()

    def observe(progress):
        assert current_thread() is main
        assert len(learner.samples) == progress.iterations
        observed.append(progress)

    main = current_thread()
    result = AsyncExecution(queue_capacity=2).run(
        collector,
        learner,
        initial_parameters=Parameters("initial", 3),
        max_iterations=5,
        cancellation=token,
        on_step=observe,
    )
    assert result == Progress(iterations=5, samples=5, updates=3, policy_version=14)
    assert [s.sequence for s in learner.samples] == list(range(5))
    assert [s.value for s in learner.samples] == list(range(5))
    assert [s.policy_version for s in learner.samples] == [p.version for p in collector.parameters]
    assert observed == [
        Progress(1, 1, 1, 10),
        Progress(2, 2, 1, 10),
        Progress(3, 3, 2, 12),
        Progress(4, 4, 2, 12),
        Progress(5, 5, 3, 14),
    ]
    assert all(thread is main for thread in learner.threads)
    assert collector.thread is not main and collector.thread.daemon
    assert collector.close_thread is collector.thread
    assert not collector.thread.is_alive()
    assert collector.closed.is_set() and learner.closed
    assert not token.cancelled


def test_next_collection_uses_latest_snapshot_at_its_boundary():
    # The first learner waits until the second collect is in flight. That collect
    # waits for the first publish hook, so the third boundary must see the update.
    second_started = Event()
    published = Event()
    collector = None

    def collect(parameters, cancellation):
        sequence = len(collector.parameters) - 1
        if sequence == 1:
            second_started.set()
            assert published.wait(2)
        return sequence

    collector = Collector(collect)

    def learn(sample, cancellation):
        if sample.sequence == 0:
            assert second_started.wait(2)
            return Parameters("new", 8)
        return None

    learner = Learner(learn)
    AsyncExecution().run(
        collector,
        learner,
        initial_parameters=Parameters("old", 7),
        max_iterations=3,
        on_step=lambda _: published.set(),
    )
    assert [s.policy_version for s in learner.samples] == [7, 7, 8]
    assert collector.parameters[2] == Parameters("new", 8)


def test_zero_iterations_still_closes_resources():
    collector, learner = Collector(), Learner()
    assert AsyncExecution().run(
        collector,
        learner,
        initial_parameters=Parameters(None, 4),
        max_iterations=0,
    ) == Progress(policy_version=4)
    assert not collector.parameters and not learner.samples
    assert collector.closed.is_set() and learner.closed


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_invalid_iteration_limit_does_not_take_resource_ownership(limit):
    collector, learner = Collector(), Learner()
    with pytest.raises(ValueError, match="max_iterations"):
        AsyncExecution().run(collector, learner, initial_parameters=Parameters(None), max_iterations=limit)
    assert not collector.closed.is_set() and not learner.closed


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_invalid_queue_capacity(capacity):
    with pytest.raises(ValueError, match="queue_capacity"):
        AsyncExecution(queue_capacity=capacity)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_shutdown_timeout(timeout):
    with pytest.raises(ValueError, match="shutdown_timeout"):
        AsyncExecution(shutdown_timeout=timeout)


def test_pre_cancelled_token_closes_without_learning():
    token = Cancellation()
    token.cancel()
    collector, learner = Collector(), Learner()
    with pytest.raises(ExecutionCancelled):
        AsyncExecution().run(
            collector,
            learner,
            initial_parameters=Parameters(None),
            max_iterations=3,
            cancellation=token,
        )
    assert not collector.parameters and not learner.samples
    assert collector.closed.is_set() and learner.closed


def test_cancellation_during_collection_discards_inflight_sample():
    def collect(parameters, cancellation):
        cancellation.cancel()
        return "discard"

    collector, learner = Collector(collect), Learner()
    with pytest.raises(ExecutionCancelled):
        AsyncExecution().run(collector, learner, initial_parameters=Parameters(None), max_iterations=1)
    assert not learner.samples
    assert collector.closed.is_set() and learner.closed


def test_cancellation_during_learning_prevents_publication_and_hook():
    def learn(sample, cancellation):
        cancellation.cancel()
        return Parameters("must not publish", 1)

    collector, learner = Collector(), Learner(learn)
    hooks = []
    with pytest.raises(ExecutionCancelled):
        AsyncExecution().run(
            collector,
            learner,
            initial_parameters=Parameters(None),
            max_iterations=5,
            on_step=hooks.append,
        )
    assert len(learner.samples) == 1 and not hooks
    assert collector.closed.is_set() and learner.closed


@pytest.mark.parametrize("error", [RuntimeError("broken"), SystemExit(0), SystemExit(7), KeyboardInterrupt()])
@pytest.mark.parametrize("stage", ["collect", "learn", "hook", "collector_close", "learner_close"])
def test_callback_failures_keep_original_exception_or_worker_exit_cause(stage, error):
    def fail(*args):
        raise error

    collector = Collector(fail if stage == "collect" else None)
    learner = Learner(fail if stage == "learn" else None)
    if stage == "collector_close":

        def close_collector():
            collector.closed.set()
            raise error

        collector.close = close_collector
    if stage == "learner_close":

        def close_learner():
            learner.closed = True
            raise error

        learner.close = close_learner
    with pytest.raises(BaseException) as caught:
        AsyncExecution(shutdown_timeout=1).run(
            collector,
            learner,
            initial_parameters=Parameters(None),
            max_iterations=3,
            on_step=fail if stage == "hook" else None,
        )
    if stage in ("collect", "collector_close") and isinstance(error, (SystemExit, KeyboardInterrupt)):
        assert isinstance(caught.value, RuntimeError)
        assert caught.value.__cause__ is error
    else:
        assert caught.value is error
    assert collector.closed.is_set() and learner.closed


def test_invalid_publication_version_fails_and_closes():
    collector, learner = Collector(), Learner(lambda *_: Parameters(None, 2))
    with pytest.raises(ValueError, match="versions must increase"):
        AsyncExecution().run(collector, learner, initial_parameters=Parameters(None, 2), max_iterations=1)
    assert collector.closed.is_set() and learner.closed


def test_backpressure_bounds_collection_and_cancellation_unblocks_full_queue():
    third_started = Event()
    fourth_started = Event()
    collector = None

    def collect(parameters, cancellation):
        if len(collector.parameters) == 3:
            third_started.set()
        elif len(collector.parameters) == 4:
            fourth_started.set()
        return len(collector.parameters)

    collector = Collector(collect)

    def learn(sample, cancellation):
        # One consumed, one queued, one in-flight: no fourth collection can start.
        assert third_started.wait(2)
        assert not fourth_started.wait(0.05)
        cancellation.cancel()
        return None

    learner = Learner(learn)
    with pytest.raises(ExecutionCancelled):
        AsyncExecution(queue_capacity=1, shutdown_timeout=1).run(
            collector,
            learner,
            initial_parameters=Parameters(None),
            max_iterations=100,
        )
    assert len(collector.parameters) == 3
    assert collector.closed.is_set() and learner.closed
    assert not collector.thread.is_alive()


def test_uncooperative_collector_has_bounded_shutdown_and_deferred_close():
    blocked = Event()
    release = Event()
    token = Cancellation()
    collector = None

    def collect(parameters, cancellation):
        if len(collector.parameters) == 2:
            blocked.set()
            release.wait()
        return None

    collector = Collector(collect)

    def learn(sample, cancellation):
        assert blocked.wait(2)
        cancellation.cancel()
        return None

    learner = Learner(learn)
    try:
        with pytest.raises(RuntimeError, match="did not stop") as caught:
            AsyncExecution(shutdown_timeout=0.03).run(
                collector,
                learner,
                initial_parameters=Parameters(None),
                max_iterations=2,
                cancellation=token,
            )
        assert isinstance(caught.value.__cause__, ExecutionCancelled)
        assert learner.closed
        assert not collector.closed.is_set()  # close must not race collect
        assert collector.thread.is_alive() and collector.thread.daemon
    finally:
        release.set()
        assert collector.closed.wait(2)
        collector.thread.join(2)
    assert not collector.thread.is_alive()


def test_thread_start_failure_closes_both_resources_on_caller(monkeypatch):
    error = RuntimeError("cannot start thread")

    def fail_start(self):
        raise error

    monkeypatch.setattr("motrix_rl.execution.asynchronous.Thread.start", fail_start)
    collector, learner = Collector(), Learner()
    with pytest.raises(RuntimeError) as caught:
        AsyncExecution().run(collector, learner, initial_parameters=Parameters(None), max_iterations=1)
    assert caught.value is error
    assert collector.closed.is_set() and learner.closed
    assert collector.close_thread is current_thread()
    assert not collector.parameters and not learner.samples


def test_independent_collector_cancellation_is_propagated_without_hanging():
    error = ExecutionCancelled("collector aborted independently")

    def collect(*args):
        raise error

    collector, learner = Collector(collect), Learner()
    with pytest.raises(ExecutionCancelled) as caught:
        AsyncExecution().run(collector, learner, initial_parameters=Parameters(None), max_iterations=1)
    assert caught.value is error
    assert collector.closed.is_set() and learner.closed


def test_primary_failure_is_not_replaced_by_teardown_failure():
    original = RuntimeError("learn failed")
    collector, learner = Collector(), Learner()

    def learn(*args):
        raise original

    def close():
        raise SystemExit("cleanup failed")

    learner.learn, learner.close = learn, close
    with pytest.raises(RuntimeError) as caught:
        AsyncExecution().run(collector, learner, initial_parameters=Parameters(None), max_iterations=1)
    assert caught.value is original
    assert collector.closed.is_set()
