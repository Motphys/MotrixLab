# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deployment task plugin discovery and loading boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from task_specs import TestDeployTask as FixtureDeployTask

from motrix_deploy import task as task_module
from motrix_deploy.task import TASK_ENTRY_POINT_GROUP, available_tasks, create_task


def _install(monkeypatch, entries):
    def entry_points(*, group):
        assert group == TASK_ENTRY_POINT_GROUP
        return entries

    monkeypatch.setattr(task_module.metadata, "entry_points", entry_points)


def test_listing_is_sorted_and_lazy(monkeypatch):
    entries = [SimpleNamespace(name=name, load=Mock()) for name in ("other/v1", "test/v1")]
    _install(monkeypatch, entries)
    assert available_tasks() == ("other/v1", "test/v1")
    for entry in entries:
        entry.load.assert_not_called()


def test_selected_class_receives_artifact_contract(monkeypatch, manifest_factory):
    manifest = manifest_factory()
    selected = SimpleNamespace(name=manifest.task.name, load=Mock(return_value=FixtureDeployTask))
    other = SimpleNamespace(name="other/v1", load=Mock())
    _install(monkeypatch, [other, selected])
    task = create_task(manifest.task, manifest.robot)
    assert isinstance(task, FixtureDeployTask)
    assert task.spec is manifest.task
    assert task.robot is manifest.robot
    selected.load.assert_called_once_with()
    other.load.assert_not_called()


def test_missing_plugin_reports_available_names(monkeypatch, manifest_factory):
    manifest = manifest_factory()
    _install(monkeypatch, [SimpleNamespace(name="other/v1", load=Mock())])
    with pytest.raises(ValueError, match="test/v1.*available tasks: other/v1"):
        create_task(manifest.task, manifest.robot)


def test_duplicate_plugins_fail_before_loading(monkeypatch, manifest_factory):
    manifest = manifest_factory()
    entries = [SimpleNamespace(name=manifest.task.name, load=Mock()) for _ in range(2)]
    _install(monkeypatch, entries)
    with pytest.raises(ValueError, match="Multiple deployment task plugins"):
        create_task(manifest.task, manifest.robot)
    for entry in entries:
        entry.load.assert_not_called()


@pytest.mark.parametrize("factory", [object(), dict, lambda spec, robot: object()])
def test_invalid_plugin_class_contract(monkeypatch, manifest_factory, factory):
    manifest = manifest_factory()
    _install(monkeypatch, [SimpleNamespace(name=manifest.task.name, load=Mock(return_value=factory))])
    with pytest.raises(TypeError, match="Deployment task plugin test/v1"):
        create_task(manifest.task, manifest.robot)
