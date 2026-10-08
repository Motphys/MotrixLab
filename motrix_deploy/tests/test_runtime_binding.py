# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Runtime binds robot session; session binds a backend-independent controller."""

from types import SimpleNamespace

import numpy as np
import pytest
from controller_helpers import ConstantPolicy, _TestPolicyProcessor
from fake_robot import FakeRobotInterface

import motrix_deploy.runtime.factory as runtime_factory
from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime.control import ControlSession, PolicyController
from motrix_deploy.runtime.factory import create_hardware_runtime
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_env_core.input import ConstantPlanarVelocityBinding


def make_controller(manifest, steps=2):
    return PolicyController(_TestPolicyProcessor(manifest), ConstantPolicy(4, np.zeros(2, np.float32)), steps=steps)


def bind(runtime, controller, period_s=0.02):
    session = ControlSession(
        controller=controller,
        robot=runtime.robot,
        command_binding=ConstantPlanarVelocityBinding((0, 0, 0)),
        period_s=period_s,
        state_timeout_s=0.1,
    )
    runtime.bind_session(session)
    return session


def test_runtime_rejects_conflicting_session_binding(manifest_factory):
    manifest = manifest_factory()
    runtime = HardwareRuntime(FakeRobotInterface(manifest.robot), realtime=False)
    first = bind(runtime, make_controller(manifest))
    with pytest.raises(ValidationError, match="bound"):
        bind(runtime, make_controller(manifest))
    assert runtime.session is first


def test_runtime_rejects_session_for_another_robot(manifest_factory):
    manifest = manifest_factory()
    runtime = HardwareRuntime(FakeRobotInterface(manifest.robot), realtime=False)
    foreign = ControlSession(
        controller=make_controller(manifest),
        robot=FakeRobotInterface(manifest.robot),
        command_binding=ConstantPlanarVelocityBinding((0, 0, 0)),
        period_s=0.02,
        state_timeout_s=0.1,
    )
    with pytest.raises(ValidationError, match="robot"):
        runtime.bind_session(foreign)
    assert runtime.session is None


def test_unbound_runtime_requires_session(manifest_factory):
    runtime = HardwareRuntime(FakeRobotInterface(manifest_factory().robot), realtime=False)
    with pytest.raises(ValidationError, match="session"):
        runtime.run()


def test_session_supplies_its_execution_period_to_controller(manifest_factory):
    manifest = manifest_factory()
    controller = make_controller(manifest)
    contexts = []
    step = controller.step

    def observe(state, context):
        contexts.append(context)
        return step(state, context)

    controller.step = observe
    runtime = HardwareRuntime(FakeRobotInterface(manifest.robot), realtime=False)
    session = bind(runtime, controller, period_s=0.04)
    assert runtime.run().success
    assert session.period_s == 0.04
    assert [context.dt_s for context in contexts] == [0.04, 0.04, 0.04]
    assert [context.elapsed_time_s for context in contexts] == pytest.approx([0, 0.04, 0.08])


@pytest.mark.parametrize("period,timeout", [(0.0, 0.1), (0.02, 0.0)])
def test_session_validates_public_timing_boundary(manifest_factory, period, timeout):
    manifest = manifest_factory()
    with pytest.raises(ValidationError):
        ControlSession(
            controller=make_controller(manifest),
            robot=FakeRobotInterface(manifest.robot),
            command_binding=ConstantPlanarVelocityBinding((0, 0, 0)),
            period_s=period,
            state_timeout_s=timeout,
        )


def test_factory_produces_host_for_explicit_session_binding(monkeypatch, manifest_factory):
    manifest = manifest_factory()
    robot = FakeRobotInterface(manifest.robot)
    calls = []

    def factory(config, context):
        calls.append((config, context))
        return HardwareRuntime(robot, realtime=False)

    entry = SimpleNamespace(name="fake", value="fake:factory", load=lambda: factory)
    monkeypatch.setattr(runtime_factory, "_runtime_entry_points", lambda: (entry,))
    context = runtime_factory.RuntimeCreateContext(robot=manifest.robot, control=manifest.control)
    runtime = create_hardware_runtime("fake", {}, context)
    assert calls == [({}, context)]
    assert runtime.robot is robot and not runtime.opened
    controller = make_controller(manifest, steps=2)
    session = bind(runtime, controller)
    assert runtime.session is session and session.controller is controller
    assert robot.events == []
    with runtime:
        assert runtime.opened and runtime.session is session
        result = runtime.run()
        assert result.success and result.completed_steps == 2
        assert len(robot.commands) == 2
        assert runtime.session is session
    assert robot.events[-2:] == ["stop", "close"]
    assert not runtime.opened
