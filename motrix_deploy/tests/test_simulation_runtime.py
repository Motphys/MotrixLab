# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Public simulation runtime configuration and plugin selection contracts."""

from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace

import numpy as np
import pytest
from fake_robot import FakeRobotInterface

import motrix_deploy.runtime.factory as runtime_factory
from motrix_deploy.runtime.base import SimulationRuntime
from motrix_deploy.runtime.config import SensorBindings, SimulationRuntimeConfig
from motrix_deploy.runtime.factory import create_simulation_runtime
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.config.sim import SimCfg


def test_simulation_factory_uses_installed_plugin_and_public_config(monkeypatch, manifest_factory):
    spec = manifest_factory().robot

    class Runtime(SimulationRuntime):
        def __init__(self):
            super().__init__()
            self._robot = FakeRobotInterface(spec)

        @property
        def robot(self):
            return self._robot

        def run(self, control, *, steps=None):
            raise NotImplementedError

    calls = []

    def factory(config):
        calls.append(config)
        assert config.scene.objs.robot is robot_cfg
        return Runtime()

    entry = SimpleNamespace(name="test", value="test:factory", load=lambda: factory)
    monkeypatch.setattr(runtime_factory, "_runtime_entry_points", lambda: (entry,))
    robot_cfg = object()
    config = SimulationRuntimeConfig(
        scene=SceneCfg(objs=SimpleNamespace(robot=robot_cfg)),
    )

    runtime = create_simulation_runtime("test", config)

    assert isinstance(runtime, SimulationRuntime)
    assert calls == [config]
    assert isinstance(runtime.robot, FakeRobotInterface)
    assert runtime.robot.spec is spec
    with runtime:
        assert runtime.opened
    assert not runtime.opened


def test_simulation_config_uses_independent_typed_sensor_defaults():
    first = SimulationRuntimeConfig(scene=SceneCfg())
    second = SimulationRuntimeConfig(scene=SceneCfg())

    assert isinstance(first.sensor_bindings, SensorBindings)
    assert first.sensor_bindings is not second.sensor_bindings
    assert asdict(first.sensor_bindings) == {
        "base_angular_velocity": None,
        "base_linear_acceleration": None,
        "base_linear_velocity": None,
    }
    with pytest.raises(FrozenInstanceError):
        first.sensor_bindings.base_angular_velocity = "gyro"


@pytest.mark.parametrize(
    "field_name",
    ["base_angular_velocity", "base_linear_acceleration", "base_linear_velocity"],
)
@pytest.mark.parametrize("invalid_name", ["", " ", "\t\n", 1, True, b"sensor", ["sensor"]])
def test_sensor_bindings_reject_invalid_sensor_names(field_name, invalid_name):
    expected_error = ValueError if isinstance(invalid_name, str) else TypeError
    with pytest.raises(expected_error, match=field_name):
        SensorBindings(**{field_name: invalid_name})


def test_sensor_bindings_reject_unknown_semantics():
    with pytest.raises(TypeError, match="base_orientation"):
        SensorBindings(**{"base_orientation": "orientation"})


def test_sensor_bindings_validate_plain_dict_without_changing_sensor_names():
    # YAML parsing produces a plain dictionary; validation belongs at that boundary.
    bindings = SensorBindings(
        **{
            "base_angular_velocity": " gyro-local ",
            "base_linear_acceleration": "robot/imu.accel",
            "base_linear_velocity": "robot-velocity",
        }
    )
    config = SimulationRuntimeConfig(scene=SceneCfg(), sensor_bindings=bindings)

    assert config.sensor_bindings is bindings
    assert bindings.base_angular_velocity == " gyro-local "
    assert bindings.base_linear_acceleration == "robot/imu.accel"
    assert bindings.base_linear_velocity == "robot-velocity"


@pytest.mark.parametrize(
    "values",
    [{}, {"base_linear_velocity": None}, {"base_linear_velocity": "velocity"}],
)
def test_sensor_bindings_allow_partial_and_none_bindings(values):
    bindings = SensorBindings(**values)

    assert bindings.base_angular_velocity is None
    assert bindings.base_linear_acceleration is None
    assert bindings.base_linear_velocity == values.get("base_linear_velocity")


def test_sensor_bindings_are_available_on_public_runtime_facade():
    from motrix_deploy.runtime import SensorBindings as PublicSensorBindings

    assert PublicSensorBindings is SensorBindings


def test_simulation_config_uses_independent_canonical_physics_defaults():
    first = SimulationRuntimeConfig(scene=SceneCfg())
    second = SimulationRuntimeConfig(scene=SceneCfg())

    assert isinstance(first.physics, SimCfg)
    # Deployment's timestep and solver defaults are a preserved runtime contract.
    assert first.physics.dt == 0.002
    assert first.physics.solver_iterations == 100
    assert first.physics is not second.physics
    first.physics.dt = 0.01
    assert second.physics.dt == 0.002


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf, -np.inf, True, False])
def test_simulation_config_rejects_invalid_timestep_at_boundary(dt):
    physics = SimCfg(dt=dt)
    with pytest.raises(ValueError, match=r"sim\.dt must be positive and finite"):
        SimulationRuntimeConfig(scene=SceneCfg(), physics=physics)


@pytest.mark.parametrize("iterations", [0, -1, 1.5, 1.0, True, False, "100"])
def test_simulation_config_rejects_invalid_solver_iterations_at_boundary(iterations):
    with pytest.raises(ValueError, match=r"sim\.solver_iterations must be a positive integer"):
        SimulationRuntimeConfig(scene=SceneCfg(), physics=SimCfg(solver_iterations=iterations))


@pytest.mark.parametrize("tolerance", [0.0, -0.1, np.nan, np.inf])
def test_simulation_config_rejects_invalid_solver_tolerance_at_boundary(tolerance):
    with pytest.raises(ValueError, match=r"sim\.solver_tolerance must be positive and finite"):
        SimulationRuntimeConfig(scene=SceneCfg(), physics=SimCfg(solver_tolerance=tolerance))


def test_simulation_config_preserves_optional_backend_physics_settings():
    physics = SimCfg(solver_iterations=None, solver_tolerance=None, gravity=None)
    config = SimulationRuntimeConfig(scene=SceneCfg(), physics=physics)

    assert config.physics is physics
    assert config.physics.solver_iterations is None
    assert config.physics.solver_tolerance is None
    assert config.physics.gravity is None
