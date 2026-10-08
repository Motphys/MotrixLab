# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""PolicyController computes commands from measurements without backend side effects."""

import numpy as np
import pytest
from controller_helpers import ConstantPolicy, _StatefulPolicy, _StatefulPolicyProcessor, _TestPolicyProcessor
from fake_robot import FakeRobotInterface

from motrix_deploy.errors import ControlFailure
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.control import PolicyController
from motrix_env_core.input import PlanarVelocityCommand


def context(step=0, time_s=0.0):
    return ControlContext(step, time_s, PlanarVelocityCommand(np.zeros((1, 3), dtype=np.float32)), dt_s=0.02)


def test_policy_controller_uses_supplied_state_and_returns_command(manifest_factory):
    manifest = manifest_factory()
    robot = FakeRobotInterface(manifest.robot)
    task = _TestPolicyProcessor(manifest)
    model = ConstantPolicy(4, np.zeros(2, np.float32))
    controller = PolicyController(task, model, steps=1)
    assert controller.policy_io is task
    assert controller.policy is model
    robot.open()
    state = robot.read_state(0.1)
    events = list(robot.events)
    seen = []
    observe = task.build_observation

    def observation(measurement, inputs):
        seen.append((measurement, inputs))
        return observe(measurement, inputs)

    task.build_observation = observation
    controller.reset(state)
    decision = controller.step(state, context())
    assert decision.command is not None and not decision.complete
    assert seen[0][0] is state
    assert robot.events == events
    assert controller.completed_steps == 1
    assert decision.policy_tick
    assert decision.trace_bytes
    assert decision.latency_ns["inference"] >= 0
    final = controller.step(state, context(1, 0.02))
    assert final.complete and final.command is None
    assert controller.completed_steps == 1


def test_model_failure_propagates_without_backend_side_effects(manifest_factory):
    manifest = manifest_factory()
    robot = FakeRobotInterface(manifest.robot)
    model = ConstantPolicy(4, np.zeros(2, np.float32))
    controller = PolicyController(_TestPolicyProcessor(manifest), model)
    robot.open()
    state = robot.read_state(0.1)
    controller.reset(state)
    events = list(robot.events)

    def infer(observation):
        raise RuntimeError("controller failed")

    model.infer = infer
    with pytest.raises(ControlFailure, match="controller failed") as error:
        controller.step(state, context())
    assert error.value.reason == "policy_error"
    assert controller.completed_steps == 0
    assert robot.events == events


def test_policy_controller_owns_local_count_clock_and_history_reset(manifest_factory):
    manifest = manifest_factory()
    robot = FakeRobotInterface(manifest.robot)
    robot.open()
    state = robot.read_state(0.1)
    task = _StatefulPolicyProcessor(manifest)
    model = _StatefulPolicy()
    controller = PolicyController(task, model, steps=2)
    controller.reset(state)
    controller.step(state, context(42, 3.0))
    controller.step(state, context(43, 3.07))
    assert model.reset_count == 1
    assert len(task.reset_contexts) == 1
    assert [inputs.step for inputs in task.contexts] == [0, 1]
    assert [inputs.elapsed_time_s for inputs in task.contexts] == pytest.approx([0, 0.07])
    assert controller.step(state, context(44, 3.09)).complete


def test_requested_stop_completes_without_another_inference(manifest_factory):
    manifest = manifest_factory()
    robot = FakeRobotInterface(manifest.robot)
    robot.open()
    state = robot.read_state(0.1)
    model = ConstantPolicy(4, np.zeros(2, np.float32))
    controller = PolicyController(_TestPolicyProcessor(manifest), model)
    controller.reset(state)
    assert controller.step(state, context()).command is not None
    controller.request_stop()
    assert controller.step(state, context(1, 0.02)).complete
    assert len(model.observations) == 1
