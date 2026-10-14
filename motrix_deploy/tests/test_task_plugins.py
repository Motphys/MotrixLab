# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deployment task plugin discovery and complete controller loading boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from controller_helpers import ConstantPolicy
from fake_robot import FakeRobotInterface
from task_specs import TestDeployTask as FixtureDeployTask

from motrix_deploy import task as task_module
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_deploy.task import TASK_ENTRY_POINT_GROUP, available_tasks, create_task


def _install(monkeypatch, entries):
    def entry_points(*, group):
        assert group == TASK_ENTRY_POINT_GROUP
        return entries

    monkeypatch.setattr(task_module.metadata, "entry_points", entry_points)


def policy():
    return ConstantPolicy(4, np.zeros(2, np.float32))


def test_listing_is_sorted_and_lazy(monkeypatch):
    entries = [SimpleNamespace(name=name, load=Mock()) for name in ("other/v1", "test/v1")]
    _install(monkeypatch, entries)
    assert available_tasks() == ("other/v1", "test/v1")
    for entry in entries:
        entry.load.assert_not_called()


def test_selected_class_receives_policy_and_budget_and_executes_as_session_controller(monkeypatch, manifest_factory):
    manifest = manifest_factory()
    selected = SimpleNamespace(name=manifest.task.name, load=Mock(return_value=FixtureDeployTask))
    other = SimpleNamespace(name="other/v1", load=Mock())
    _install(monkeypatch, [other, selected])
    model = policy()
    task = create_task(manifest.task, manifest.robot, model, steps=2)
    assert isinstance(task, FixtureDeployTask)
    assert task.spec is manifest.task
    assert task.robot is manifest.robot
    assert task.policy is model
    assert task.steps == 2
    selected.load.assert_called_once_with()
    other.load.assert_not_called()
    backend = FakeRobotInterface(manifest.robot)
    session = ControlSession(robot=backend, controller=task, period_s=0.02, state_timeout_s=0.1)
    runtime = HardwareRuntime(backend, realtime=False)
    runtime.bind_session(session)
    result = runtime.run()
    assert result.success
    assert session.controller is task
    assert len(task.states) == 2
    assert backend.events[-2:] == ["stop", "close"]


def test_missing_plugin_reports_available_names(monkeypatch, manifest_factory):
    manifest = manifest_factory()
    _install(monkeypatch, [SimpleNamespace(name="other/v1", load=Mock())])
    with pytest.raises(ValueError, match="test/v1.*available tasks: other/v1"):
        create_task(manifest.task, manifest.robot, policy())


def test_duplicate_plugins_fail_before_loading(monkeypatch, manifest_factory):
    manifest = manifest_factory()
    entries = [SimpleNamespace(name=manifest.task.name, load=Mock()) for _ in range(2)]
    _install(monkeypatch, entries)
    with pytest.raises(ValueError, match="Multiple deployment task plugins"):
        create_task(manifest.task, manifest.robot, policy())
    for entry in entries:
        entry.load.assert_not_called()


@pytest.mark.parametrize("factory", [object(), dict, lambda spec, robot: object()])
def test_invalid_plugin_class_contract(monkeypatch, manifest_factory, factory):
    manifest = manifest_factory()
    _install(monkeypatch, [SimpleNamespace(name=manifest.task.name, load=Mock(return_value=factory))])
    with pytest.raises(TypeError, match="Deployment task plugin test/v1"):
        create_task(manifest.task, manifest.robot, policy())


def test_plugin_task_exposes_root_reset_step_and_stop(manifest_factory):
    manifest = manifest_factory()
    task = create_task(manifest.task, manifest.robot, policy())
    backend = FakeRobotInterface(manifest.robot)
    backend.open()
    try:
        state = backend.read_state(0.1)
        task.reset(state)
        assert task.step(state, ControlContext(0, 0.0, None, dt_s=0.02)).complete
        task.request_stop()
        assert task.stop_requested
        task.reset(state)
        assert task.states == [state]
        assert not task.stop_requested
    finally:
        backend.close()
