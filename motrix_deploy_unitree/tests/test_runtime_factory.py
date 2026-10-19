# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Concrete hardware/transport dispatch without constructing SDK resources."""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from motrix_deploy_unitree.config.control import UnitreeControlConfig, load_control_config
from motrix_deploy_unitree.config.runtime import UnitreeRuntimeConfig, UnitreeTransportConfig
from motrix_deploy_unitree.plugin import create_runtime


@pytest.mark.parametrize(
    ("robot", "recipe", "implementation", "adapter"),
    [
        ("go2", "go2", "python_sdk", "UnitreeGo2Robot"),
        ("g1-29dof", "g1", "python_sdk", "UnitreeG1_29DOFRobot"),
        ("g1-29dof", "g1", "native_cpp", "UnitreeG1NativeRobot"),
    ],
)
def test_factory_dispatches_hardware_and_transport(monkeypatch, robot, recipe, implementation, adapter):
    config = UnitreeRuntimeConfig(
        robot=robot,
        control=load_control_config(recipe),
        transport=UnitreeTransportConfig(network_interface="eth0", implementation=implementation),
    )
    context = SimpleNamespace(robot=object(), control=SimpleNamespace(period_s=0.02, state_timeout_s=0.1))
    captured = {}
    sentinel = object()

    def construct(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return sentinel

    for name in ("UnitreeGo2Robot", "UnitreeG1_29DOFRobot", "UnitreeG1NativeRobot"):

        def forbidden(*args, **kwargs):
            pytest.fail("factory selected an unexpected adapter")

        monkeypatch.setattr(f"motrix_deploy_unitree.plugin.{name}", forbidden)
    monkeypatch.setattr(f"motrix_deploy_unitree.plugin.{adapter}", construct)
    runtime_options = {}

    def create_host(robot, **kwargs):
        runtime_options.update(kwargs)
        return robot

    monkeypatch.setattr("motrix_deploy_unitree.plugin.HardwareRuntime", create_host)
    assert create_runtime(config, context) is sentinel
    assert runtime_options["realtime"] is True
    assert captured["kwargs"]["spec"] is context.robot
    assert captured["kwargs"]["transport"] is config.transport
    assert captured["kwargs"]["control"] is config.control
    if robot != "go2":
        assert captured["kwargs"]["operator"] is config.operator


def test_factory_rejects_unimplemented_go2_native_transport(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported transport must not construct a robot")

    monkeypatch.setattr("motrix_deploy_unitree.plugin.UnitreeGo2Robot", forbidden)
    with pytest.raises(ValueError, match="python_sdk.*Go2"):
        config = UnitreeRuntimeConfig(
            robot="go2",
            control=UnitreeControlConfig(),
            transport=UnitreeTransportConfig(network_interface="eth0", implementation="native_cpp"),
        )
        create_runtime(config, SimpleNamespace())


@pytest.mark.parametrize("recipe", ["go2", "g1"])
def test_control_yaml_overrides_produce_shared_typed_config(recipe):
    defaults = load_control_config(recipe)
    overrides = OmegaConf.create("command_timeout_s: 0.25\ndamping_kd: 2.5\n")
    control = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(defaults), overrides))
    assert type(control) is UnitreeControlConfig
    assert control.command_timeout_s == pytest.approx(0.25)
    assert control.damping_kd == pytest.approx(2.5)
    assert control.command_publish_period_s == defaults.command_publish_period_s


@pytest.mark.parametrize("recipe", ["go2", "g1"])
def test_control_yaml_rejects_unknown_fields(recipe):
    with pytest.raises(ConfigKeyError, match="unknown_setting"):
        OmegaConf.merge(OmegaConf.structured(load_control_config(recipe)), {"unknown_setting": 1})
