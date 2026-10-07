# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Robot-independent deployment environment discovery contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from motrix_deploy import env as env_module
from motrix_deploy.env import (
    DEPLOY_ENV_ENTRY_POINT_GROUP,
    assemble_deploy_scene,
    available_deploy_envs,
    create_deploy_env,
)
from motrix_env_core.config.scene import MjcfFileCfg, RobotCfg, SceneCfg


def _install(monkeypatch, entries):
    def entry_points(*, group):
        assert group == DEPLOY_ENV_ENTRY_POINT_GROUP
        return entries

    monkeypatch.setattr(env_module.metadata, "entry_points", entry_points)


def test_world_discovery_is_lazy(monkeypatch):
    entries = [SimpleNamespace(name=name, load=Mock()) for name in ("rough", "flat")]
    _install(monkeypatch, entries)
    assert available_deploy_envs() == ("flat", "rough")
    for entry in entries:
        entry.load.assert_not_called()


def test_factory_creates_instance_local_worlds(monkeypatch):
    factory = Mock(side_effect=SceneCfg)
    entry = SimpleNamespace(name="flat", load=Mock(return_value=factory))
    other = SimpleNamespace(name="rough", load=Mock())
    _install(monkeypatch, [entry, other])
    first = create_deploy_env("flat")
    second = create_deploy_env("flat")
    assert first is not second
    assert first.objs is not second.objs
    assert first.objs.robot is None
    other.load.assert_not_called()


def test_assembly_uses_independently_registered_robot(monkeypatch):
    _install(monkeypatch, [SimpleNamespace(name="world", load=Mock(return_value=SceneCfg))])
    robot = RobotCfg(model=MjcfFileCfg(file="synthetic.xml"), base_link_name="base")
    factory = Mock(return_value=robot)
    monkeypatch.setattr(env_module.registry, "make_robot_config", factory)
    scene = assemble_deploy_scene("world", "custom-robot")
    assert scene.objs.robot is robot
    factory.assert_called_once_with("custom-robot")


def test_unknown_environment_reports_choices(monkeypatch):
    _install(monkeypatch, [SimpleNamespace(name="flat", load=Mock())])
    with pytest.raises(ValueError, match="Unknown deployment environment.*available environments: flat"):
        create_deploy_env("unknown")


def test_duplicate_environment_rejected_before_loading(monkeypatch):
    entries = [SimpleNamespace(name="flat", load=Mock()) for _ in range(2)]
    _install(monkeypatch, entries)
    with pytest.raises(ValueError, match="Multiple deployment environment plugins"):
        create_deploy_env("flat")
    for entry in entries:
        entry.load.assert_not_called()


@pytest.mark.parametrize("scene", [object(), SceneCfg()])
def test_environment_factory_contract(monkeypatch, scene):
    if isinstance(scene, SceneCfg):
        scene.objs.robot = RobotCfg(model=MjcfFileCfg(file="synthetic.xml"), base_link_name="base")
    _install(monkeypatch, [SimpleNamespace(name="bad", load=Mock(return_value=lambda: scene))])
    with pytest.raises((TypeError, ValueError), match="Deployment environment 'bad'"):
        create_deploy_env("bad")
