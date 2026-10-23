# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest

from motrix_rl.execution.contracts import Cancellation, ExecutionCancelled, Parameters, Progress
from motrix_rl.execution.synchronous import SyncExecution


def test_collect_learn_publish_and_checkpoint_boundary():
    events = []
    observed = []

    def collect(parameters, cancellation):
        events.append(("collect", parameters.value, parameters.version))
        return parameters.value + 2

    def learn(sample, cancellation):
        events.append(("learn", sample.value, sample.policy_version, sample.sequence))
        # Algorithms may reject data or warm up without publishing.
        if sample.sequence == 0:
            return None
        return Parameters(sample.value, sample.policy_version + 1)

    def checkpoint(progress):
        events.append(("checkpoint", progress.policy_version))
        observed.append(progress)

    result = SyncExecution().run(collect, learn, initial_parameters=Parameters(1), max_iterations=3, on_step=checkpoint)
    assert result == Progress(iterations=3, samples=3, updates=2, policy_version=2)
    assert events == [
        ("collect", 1, 0),
        ("learn", 3, 0, 0),
        ("checkpoint", 0),
        ("collect", 1, 0),
        ("learn", 3, 0, 1),
        ("checkpoint", 1),
        ("collect", 3, 1),
        ("learn", 5, 1, 2),
        ("checkpoint", 2),
    ]
    assert observed[-1] == result


@pytest.mark.parametrize("phase", ["collect", "learn", "observer"])
def test_callback_errors_propagate_without_next_phase(phase):
    events = []
    error = RuntimeError("algorithm failure")

    def callback(name, result):
        def run(*args):
            events.append(name)
            if name == phase:
                raise error
            return result

        return run

    with pytest.raises(RuntimeError) as exc:
        SyncExecution().run(
            callback("collect", 0),
            callback("learn", None),
            initial_parameters=Parameters(None),
            max_iterations=2,
            on_step=callback("observer", None),
        )
    assert exc.value is error
    assert events == ["collect", "learn", "observer"][: ["collect", "learn", "observer"].index(phase) + 1]


def test_cancel_during_collection_prevents_learning():
    cancellation = Cancellation()
    learned = []

    def collect(parameters, token):
        token.cancel()

    with pytest.raises(ExecutionCancelled):
        SyncExecution().run(
            collect,
            lambda *args: learned.append(args),
            initial_parameters=Parameters(None),
            max_iterations=10,
            cancellation=cancellation,
        )
    assert learned == []


@pytest.mark.parametrize("phase", ["learn", "observer"])
def test_cancellation_at_learning_boundary_prevents_next_iteration(phase):
    token = Cancellation()
    events = []

    def collect(*args):
        events.append("collect")
        return 1

    def learn(*args):
        events.append("learn")
        if phase == "learn":
            token.cancel()
        return Parameters(2, 1)

    def observe(progress):
        events.append("observer")
        assert progress.iterations == progress.updates == 1
        token.cancel()

    with pytest.raises(ExecutionCancelled):
        SyncExecution().run(
            collect,
            learn,
            initial_parameters=Parameters(0),
            max_iterations=2,
            cancellation=token,
            on_step=observe,
        )
    assert events == (["collect", "learn"] if phase == "learn" else ["collect", "learn", "observer"])


def test_zero_iterations_and_invalid_publication():
    def unexpected(*args):
        pytest.fail("callback should not run")

    result = SyncExecution().run(unexpected, unexpected, initial_parameters=Parameters(None, 7), max_iterations=0)
    assert result == Progress(policy_version=7)
    with pytest.raises(ValueError, match="nonnegative integer"):
        SyncExecution().run(unexpected, unexpected, initial_parameters=Parameters(None), max_iterations=-1)
    for invalid in (True, 1.5, "1"):
        with pytest.raises(ValueError, match="nonnegative integer"):
            SyncExecution().run(unexpected, unexpected, initial_parameters=Parameters(None), max_iterations=invalid)
    with pytest.raises(ValueError, match="versions must increase"):
        SyncExecution().run(
            lambda *args: 1,
            lambda *args: Parameters(None, 2),
            initial_parameters=Parameters(None, 2),
            max_iterations=1,
        )
