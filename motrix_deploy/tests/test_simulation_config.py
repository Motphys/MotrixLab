# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Simulation scene, timing and sensor recipe decoding contracts."""

from copy import deepcopy

import pytest

from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime.config import SensorBindings, SimulationRuntimeConfig
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.config.sim import SimCfg


def test_from_mapping_decodes_recipe_without_mutating_input():
    scene = SceneCfg()
    nested = {
        "physics": {"dt": 0.005, "solver_iterations": 25, "solver_tolerance": 0.0001, "gravity": (0.0, 0.0, -9.81)},
        "sensor_bindings": {
            "base_angular_velocity": "gyro",
            "base_linear_acceleration": "accelerometer",
            "base_linear_velocity": "velocimeter",
        },
    }
    original = deepcopy(nested)
    values = {"name": "generic-backend", "scene": scene, **nested}
    config = SimulationRuntimeConfig.from_mapping(values, render=True, realtime=False)
    assert config.scene is scene
    assert isinstance(config.physics, SimCfg)
    for name, value in nested["physics"].items():
        assert getattr(config.physics, name) == value
    assert config.sensor_bindings == SensorBindings(**nested["sensor_bindings"])
    assert config.render is True
    assert config.realtime is False
    assert {name: values[name] for name in nested} == original


def test_from_mapping_supplies_independent_optional_defaults():
    values = {"scene": SceneCfg(), "physics": {"dt": 0.005, "solver_iterations": 25}}
    first = SimulationRuntimeConfig.from_mapping(values)
    second = SimulationRuntimeConfig.from_mapping(values)
    assert first.sensor_bindings == SensorBindings()
    assert first.render is False
    assert first.realtime is None
    assert first.sensor_bindings is not second.sensor_bindings
    assert first.physics is not second.physics
    assert set(values) == {"scene", "physics"}


def test_from_mapping_rejects_unknown_top_level_fields():
    unknown_name = "unrecognized_option"
    values = {"scene": SceneCfg(), "physics": {"dt": 0.005, "solver_iterations": 25}, unknown_name: True}
    with pytest.raises(ValidationError) as error:
        SimulationRuntimeConfig.from_mapping(values)
    assert error.value.path == "runtime"
    assert unknown_name in str(error.value)


@pytest.mark.parametrize("section", ["physics", "sensor_bindings"])
def test_from_mapping_rejects_unknown_nested_fields(section):
    unknown_name = "unrecognized_option"
    values = {
        "scene": SceneCfg(),
        "physics": {"dt": 0.005, "solver_iterations": 25},
        "sensor_bindings": {"base_angular_velocity": "gyro"},
    }
    values[section][unknown_name] = True
    with pytest.raises(TypeError, match=unknown_name):
        SimulationRuntimeConfig.from_mapping(values)
