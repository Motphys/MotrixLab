# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deployment runtime plugin discovery contract tests."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from fake_robot import FakeRobotInterface

import motrix_deploy.runtime.factory as runtime_factory
from motrix_deploy.artifact import ControlSpec
from motrix_deploy.contracts import RobotSpec
from motrix_deploy.runtime.config import SimulationRuntimeConfig
from motrix_deploy.runtime.factory import (
    RuntimeCreateContext,
    create_hardware_runtime,
    create_simulation_runtime,
    registered_runtimes,
)
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_env_core.config.scene import SceneCfg


@dataclass(frozen=True)
class _EntryPoint:
    name: str
    value: str
    target: Any

    def load(self) -> Any:
        return self.target


def test_runtime_plugin_is_discovered_and_constructed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    spec = RobotSpec(
        base_link_name="base",
        joint_names=("joint",),
        default_joint_position=np.zeros(1, dtype=np.float32),
        position_lower=np.full(1, -1.0, dtype=np.float32),
        position_upper=np.full(1, 1.0, dtype=np.float32),
        torque_limit=np.ones(1, dtype=np.float32),
    )

    def factory(config, context):
        calls.append((config, context))
        robot = FakeRobotInterface(context.robot, response=config["response"])
        return HardwareRuntime(robot=robot, realtime=True)

    entry_point = _EntryPoint("fake", "test:factory", factory)
    monkeypatch.setattr(runtime_factory, "_runtime_entry_points", lambda: (entry_point,))
    context = RuntimeCreateContext(
        robot=spec,
        control=ControlSpec(period_s=0.02, state_timeout_s=0.1),
    )

    runtime = create_hardware_runtime("fake", {"name": "fake", "response": 0.5}, context)

    assert isinstance(runtime, HardwareRuntime)
    assert isinstance(runtime.robot, FakeRobotInterface)
    assert runtime.robot.spec is context.robot
    assert registered_runtimes() == ("fake",)
    assert calls == [({"name": "fake", "response": 0.5}, context)]
    with pytest.raises(TypeError, match="does not provide a SimulationRuntime"):
        monkeypatch.setattr(
            runtime_factory,
            "_runtime_entry_points",
            lambda: (_EntryPoint("fake", "test:factory", lambda config: runtime),),
        )
        create_simulation_runtime("fake", SimulationRuntimeConfig(scene=SceneCfg()))


def test_runtime_plugin_discovery_rejects_missing_duplicate_and_invalid_factories(
    monkeypatch: pytest.MonkeyPatch,
    manifest_factory,
) -> None:
    spec = manifest_factory().robot
    context = RuntimeCreateContext(
        robot=spec,
        control=ControlSpec(period_s=0.02, state_timeout_s=0.1),
    )
    monkeypatch.setattr(runtime_factory, "_runtime_entry_points", lambda: ())
    with pytest.raises(ValueError, match="installed runtimes: none"):
        create_hardware_runtime("missing", {}, context)

    duplicates = (
        _EntryPoint("duplicate", "first:factory", lambda config, context: object()),
        _EntryPoint("duplicate", "second:factory", lambda config, context: object()),
    )
    monkeypatch.setattr(runtime_factory, "_runtime_entry_points", lambda: duplicates)
    with pytest.raises(ValueError, match="Multiple deployment runtime plugins"):
        create_hardware_runtime("duplicate", {}, context)

    invalid = _EntryPoint("invalid", "invalid:factory", lambda config, context: object())
    monkeypatch.setattr(runtime_factory, "_runtime_entry_points", lambda: (invalid,))
    with pytest.raises(TypeError, match="expected DeploymentRuntime"):
        create_hardware_runtime("invalid", {}, context)
