# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Runtime/control ownership binding contracts."""

from types import SimpleNamespace

import pytest
from fake_robot import FakeRobotInterface
from test_runtime import _session

import motrix_deploy.runtime.factory as runtime_factory
from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime.factory import create_hardware_runtime
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_deploy.runtime.scheduler import FixedStepScheduler


def test_hardware_runtime_runs_bound_control_without_duplicate_session(manifest_factory):
    manifest = manifest_factory()
    robot = FakeRobotInterface(manifest.robot)
    control = _session(manifest, robot)
    runtime = HardwareRuntime(robot, scheduler=FixedStepScheduler(manifest.control.period_s))

    runtime.bind_control_session(control)
    assert runtime.control is control
    result = runtime.run(steps=2)

    assert result.success
    assert runtime.control is control
    assert robot.events[-2:] == ["stop", "close"]


def test_binding_rejects_conflicting_robot_and_duplicate_control(manifest_factory):
    manifest = manifest_factory()
    robot = FakeRobotInterface(manifest.robot)
    other = FakeRobotInterface(manifest.robot)
    runtime = HardwareRuntime(robot)
    first = _session(manifest, robot)
    second = _session(manifest, robot)
    foreign = _session(manifest, other)

    runtime.bind_control_session(first)
    runtime.bind_control_session(first)
    with pytest.raises(ValidationError, match="already bound"):
        runtime.bind_control_session(second)
    with pytest.raises(ValidationError, match="runtime.control.robot"):
        runtime.bind_control_session(foreign)
    with pytest.raises(ValidationError, match="already bound"):
        runtime.run(second, steps=1)


def test_unbound_hardware_runtime_requires_control(manifest_factory):
    manifest = manifest_factory()
    runtime = HardwareRuntime(FakeRobotInterface(manifest.robot))

    with pytest.raises(ValidationError, match="runtime.control"):
        runtime.run(steps=1)


def test_hardware_factory_prepares_robot_for_explicit_control_binding(monkeypatch, manifest_factory):
    manifest = manifest_factory()
    robot = FakeRobotInterface(manifest.robot)
    calls = []

    def factory(config, context):
        calls.append((config, context))
        return HardwareRuntime(robot, realtime=False)

    entry = SimpleNamespace(name="fake", value="fake:factory", load=lambda: factory)
    monkeypatch.setattr(runtime_factory, "_runtime_entry_points", lambda: (entry,))

    context = runtime_factory.RuntimeCreateContext(robot=manifest.robot, control=manifest.control)
    config = {}
    runtime = create_hardware_runtime("fake", config, context)
    assert calls == [(config, context)]
    assert runtime.robot is robot
    assert not runtime.opened
    assert robot.events == []

    control = _session(manifest, runtime.robot)
    runtime.bind_control_session(control)
    assert runtime.control is control
    assert control.robot is runtime.robot
    assert robot.events == []

    with runtime:
        assert runtime.opened
        result = runtime.run(steps=1)
    assert result.success
    assert robot.events[-2:] == ["stop", "close"]
    assert not runtime.opened
