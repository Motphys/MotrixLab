# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo Go2 adapter contract tests."""

import logging
import subprocess
import sys
import textwrap
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from motrix_robots.unitree import UnitreeGo2Robot
from omegaconf import OmegaConf
from scene_helpers import AdapterSceneObjsCfg, build_scene

from motrix_deploy.contracts import JointServoCommand, RobotSpec
from motrix_deploy.errors import ValidationError
from motrix_deploy.robot.interface import KeyboardDeviceProvider
from motrix_deploy.runtime.config import SensorBindings, SimulationRuntimeConfig
from motrix_deploy_mujoco import (
    MujocoRuntime,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from motrix_env_core.config.scene import FlatTerrainCfg, SceneCfg, SystemCameraCfg
from motrix_env_core.config.sim import SimCfg
from motrix_env_mujoco.compiler import MuJoCoSceneCompiler

ROOT = Path(__file__).parents[2]
CONFIG_PATH = ROOT / "motrix_deploy_tasks/src/motrix_deploy_tasks/config/task/go2-walk-flat/sim.yaml"
FLAT_CONFIG_PATH = CONFIG_PATH
JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
CONTROL_PERIOD_S = 0.02


def _mapping(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    values = OmegaConf.to_container(OmegaConf.load(config_path).runtime)
    values.pop("backend")
    values.pop("deploy_env_id")
    values.pop("robot_id")
    values["scene"] = build_scene()
    values["scene"].objs.robot.translation = tuple(values.pop("robot_translation"))
    values["scene"].objs.robot.rotation = tuple(values.pop("robot_rotation"))
    return values


def _config(config_path: Path | None = None) -> SimulationRuntimeConfig:
    if config_path is not None:
        return SimulationRuntimeConfig.from_mapping(_mapping(config_path))
    return SimulationRuntimeConfig(
        scene=build_scene(),
        physics=SimCfg(dt=0.002, solver_iterations=100),
        sensor_bindings=SensorBindings(
            base_angular_velocity="gyro",
            base_linear_acceleration="accelerometer",
            base_linear_velocity="global_linvel",
        ),
    )


def create_runtime(config, *, control_period_s, render=False, **kwargs):
    runtime = MujocoRuntime(replace(config, render=render), control_period_s=control_period_s, **kwargs)
    return runtime


def _backend(
    *,
    render: bool = False,
    control_period_s: float = CONTROL_PERIOD_S,
    viewer_factory: Callable[[Any, SystemCameraCfg], Any] | None = None,
) -> MujocoRuntime:
    kwargs = {} if viewer_factory is None else {"viewer_factory": viewer_factory}
    return create_runtime(
        _config(),
        control_period_s=control_period_s,
        render=render,
        **kwargs,
    )


def _spec(joint_names: tuple[str, ...] = JOINT_NAMES) -> RobotSpec:
    return RobotSpec(
        base_link_name="base",
        joint_names=joint_names,
        default_joint_position=np.array(
            [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
            dtype=np.float32,
        ),
        position_lower=np.tile(np.array([-0.9472, -1.4, -2.6227], dtype=np.float32), 4),
        position_upper=np.tile(np.array([0.9472, 2.5, -0.84776], dtype=np.float32), 4),
        torque_limit=np.full(12, 24.0, dtype=np.float32),
    )


def _command(position: np.ndarray, *, kp: float = 35.0, kd: float = 0.5) -> JointServoCommand:
    zeros = np.zeros(12, dtype=np.float32)
    return JointServoCommand(
        joint_position=position,
        joint_velocity=zeros,
        feedforward_torque=zeros,
        kp=np.full(12, kp, dtype=np.float32),
        kd=np.full(12, kd, dtype=np.float32),
    )


def test_backend_control_period_requires_integral_physics_substeps() -> None:
    with pytest.raises(ValidationError, match="integer multiple"):
        _backend(control_period_s=0.015)


def test_quaternion_order_conversion_is_explicit_and_round_trips() -> None:
    wxyz = np.array([0.5, 0.5, -0.5, 0.5], dtype=np.float64)

    xyzw = wxyz_to_xyzw(wxyz)

    np.testing.assert_array_equal(xyzw, [0.5, -0.5, 0.5, 0.5])
    np.testing.assert_array_equal(xyzw_to_wxyz(xyzw), wxyz)


def test_from_mapping_accepts_ready_scene_without_reconstructing_it() -> None:
    values = _mapping()
    robot = values["scene"].objs.robot
    robot.translation = (1.0, -0.5, 2.0)
    robot.rotation = (0.0, 0.0, 1.0, 0.0)
    config = SimulationRuntimeConfig.from_mapping(values)
    assert config.scene is values["scene"]
    assert config.scene.objs.robot is robot
    assert robot.translation == (1.0, -0.5, 2.0)
    assert robot.rotation == (0.0, 0.0, 1.0, 0.0)
    assert config.physics.dt == values["physics"]["dt"]


def test_mapping_parses_all_canonical_physics_fields() -> None:
    values = _mapping()
    physics = {"dt": 0.004, "solver_iterations": 19, "solver_tolerance": 4e-7, "gravity": (0.2, -0.3, -5.0)}
    values["physics"] = physics
    config = SimulationRuntimeConfig.from_mapping(values)
    assert isinstance(config.physics, SimCfg)
    for name, value in physics.items():
        assert getattr(config.physics, name) == value


@pytest.mark.parametrize("field", ["dt", "solver_iterations"])
def test_mapping_preserves_required_physics_fields(field) -> None:
    values = _mapping()
    del values["physics"][field]
    with pytest.raises(ValidationError, match=field):
        SimulationRuntimeConfig.from_mapping(values)


def test_mapping_rejects_unknown_physics_fields() -> None:
    values = _mapping()
    values["physics"]["unexpected_physics_setting"] = 1
    with pytest.raises(TypeError, match="unexpected_physics_setting"):
        SimulationRuntimeConfig.from_mapping(values)


@pytest.mark.parametrize(
    "field,value", [("dt", 0), ("solver_iterations", -1), ("solver_tolerance", 0), ("gravity", (1, 2))]
)
def test_backend_config_validates_physics_at_public_boundary(field, value) -> None:
    with pytest.raises(ValueError, match=f"sim.{field}"):
        SimulationRuntimeConfig(scene=build_scene(), physics=SimCfg(**{field: value}))


@pytest.mark.parametrize(
    "scene",
    [None, "flat", [], {}, {"_target_": "motrix_env_core.config.scene.SceneCfg"}],
)
def test_from_mapping_rejects_unassembled_scene_at_backend_boundary(scene: object) -> None:
    values = _mapping()
    values["scene"] = scene
    with pytest.raises(ValidationError, match="runtime.scene.*complete SceneCfg"):
        SimulationRuntimeConfig.from_mapping(values)


@pytest.mark.parametrize("sensor_name", [7, "", "   "])
def test_state_sensor_mapping_validates_model_local_names(sensor_name: object) -> None:
    values = _mapping()
    values["sensor_bindings"]["base_angular_velocity"] = sensor_name

    with pytest.raises((TypeError, ValueError), match="base_angular_velocity"):
        SimulationRuntimeConfig.from_mapping(values)


def test_state_sensor_mapping_rejects_unknown_roles() -> None:
    values = _mapping()
    values["sensor_bindings"]["unexpected_role"] = "gyro"
    with pytest.raises(TypeError, match="unexpected_role"):
        SimulationRuntimeConfig.from_mapping(values)


@pytest.mark.parametrize("role", ["base_angular_velocity", "base_linear_acceleration", "base_linear_velocity"])
def test_missing_state_sensor_fails_at_model_binding(role: str) -> None:
    config = _config()
    config = replace(config, sensor_bindings=replace(config.sensor_bindings, **{role: None}))
    backend = create_runtime(config, control_period_s=CONTROL_PERIOD_S)
    with pytest.raises(ValidationError, match=f"runtime.sensor_bindings.{role}"):
        backend.open()
    backend.close()


def test_initial_state_command_step_contract_is_headless() -> None:
    spec = _spec()
    backend = _backend()
    viewer_modules_before = {name for name in sys.modules if name.startswith("mujoco.viewer")}
    backend.open()

    initial = backend.robot.read_state(timeout_s=0.1)
    assert backend.data is not None
    assert backend.data.ncon == 0
    target = spec.default_joint_position + np.float32(0.01)
    backend.robot.write_command(_command(target))
    backend.advance_control_period()
    stepped = backend.robot.read_state(timeout_s=0.1)
    backend.robot.stop()
    backend.close()
    backend.robot.stop()
    backend.close()

    np.testing.assert_allclose(initial.joint_position, spec.default_joint_position, atol=1e-6)
    np.testing.assert_array_equal(initial.base_orientation_xyzw, [0.0, 0.0, 0.0, 1.0])
    assert stepped.sample_time_ns == 20_000_000
    assert np.isfinite(stepped.joint_position).all()
    # The robot's physical command-rate limit is independent of controller cadence.
    assert backend.robot.capabilities.max_command_rate_hz == pytest.approx(1.0 / backend.sim.dt)
    assert {name for name in sys.modules if name.startswith("mujoco.viewer")} == viewer_modules_before


def test_glfw_viewer_syncs_without_owning_simulation_step() -> None:
    class FakeViewer:
        def __init__(self) -> None:
            self.running = False
            self.open_count = 0
            self.sync_count = 0
            self.close_count = 0
            self.keyboard_device = object()

        def open(self, model: Any, data: Any) -> None:
            del model, data
            self.open_count += 1
            self.running = True

        def is_running(self) -> bool:
            return self.running

        def sync(self) -> None:
            self.sync_count += 1

        def close(self) -> None:
            self.close_count += 1
            self.running = False

    viewer = FakeViewer()
    camera_configs: list[SystemCameraCfg] = []

    def viewer_factory(mujoco: Any, camera_config: SystemCameraCfg) -> FakeViewer:
        del mujoco
        camera_configs.append(camera_config)
        return viewer

    backend = _backend(render=True, viewer_factory=viewer_factory)
    spec = _spec()

    assert backend.get_keyboard_device() is viewer.keyboard_device
    assert camera_configs == [backend.scene.system_camera]
    backend.open()
    initial = backend.robot.read_state(timeout_s=0.1)
    backend.robot.write_command(_command(spec.default_joint_position))
    backend.advance_control_period()
    stepped = backend.robot.read_state(timeout_s=0.1)

    assert backend.robot.capabilities.supports_rendering is True
    assert viewer.open_count == 1
    assert initial.sample_time_ns == 0
    assert stepped.sample_time_ns == 20_000_000
    assert viewer.sync_count == 2

    viewer.running = False
    with pytest.raises(KeyboardInterrupt, match="viewer was closed"):
        backend.robot.read_state(timeout_s=0.1)
    backend.close()
    assert viewer.close_count == 1

    backend.open()
    assert backend.get_keyboard_device() is viewer.keyboard_device
    assert viewer.open_count == 2
    backend.close()
    assert viewer.close_count == 2


def test_headless_backend_does_not_provide_keyboard_input() -> None:
    backend = _backend(render=False)

    assert isinstance(backend, KeyboardDeviceProvider)
    with pytest.raises(RuntimeError, match="viewer=true"):
        backend.get_keyboard_device()


def test_swapped_artifact_joint_contract_fails_compatibility_comparison() -> None:
    backend = _backend()
    actual = backend.robot.spec
    swapped = list(actual.joint_names)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    expected = replace(actual, joint_names=tuple(swapped))

    with pytest.raises(ValidationError, match="joint_names"):
        expected.validate_compatible(actual)


def test_artifact_base_link_must_match_selected_robot() -> None:
    backend = _backend()
    actual = backend.robot.spec
    expected = replace(actual, base_link_name="wrong_base")
    with pytest.raises(ValidationError, match="base_link_name"):
        expected.validate_compatible(actual)


def test_missing_artifact_joint_fails_compatibility_comparison() -> None:
    backend = _backend()
    actual = backend.robot.spec
    joint_names = list(actual.joint_names)
    joint_names[0] = "missing_joint"
    expected = replace(actual, joint_names=tuple(joint_names))

    with pytest.raises(ValidationError, match="joint_names"):
        expected.validate_compatible(actual)


@pytest.mark.parametrize(
    ("role", "sensor_name", "message"),
    [
        ("base_angular_velocity", "missing_gyro", "missing_gyro"),
        ("base_angular_velocity", "global_angvel", "mjSENS_GYRO"),
        ("base_linear_acceleration", "gyro", "mjSENS_ACCELEROMETER"),
        ("base_linear_velocity", "local_linvel", "mjSENS_FRAMELINVEL"),
        ("base_linear_velocity", "FR_global_linvel", "on the base body"),
    ],
)
def test_invalid_state_sensor_mapping_fails_during_open(role: str, sensor_name: str, message: str) -> None:
    config = _config()
    config = replace(config, sensor_bindings=replace(config.sensor_bindings, **{role: sensor_name}))
    backend = create_runtime(config, control_period_s=CONTROL_PERIOD_S)

    with pytest.raises(ValidationError, match=message):
        backend.open()
    backend.close()


@pytest.mark.parametrize("role", ["base_angular_velocity", "base_linear_acceleration"])
def test_imu_sensor_must_use_base_body_frame(role: str, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend()
    build_model = backend.build_model

    def rotated_imu_model() -> Any:
        model = build_model()
        sensor_id = backend.mj.mj_name2id(
            model,
            backend.mj.mjtObj.mjOBJ_SENSOR,
            backend.robot._cfg.resolve_name(getattr(backend.config.sensor_bindings, role)),
        )
        site_id = model.sensor_objid[sensor_id]
        model.site_quat[site_id] = [0.7071067812, 0.0, 0.0, 0.7071067812]
        return model

    monkeypatch.setattr(backend, "build_model", rotated_imu_model)
    with pytest.raises(ValidationError, match="aligned with its body frame"):
        backend.open()
    backend.close()


def test_sensor_bindings_resolve_robot_names_and_are_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config()
    config.scene.objs.robot.prefix = "robot_"
    config.scene.objs.robot.suffix = "_instance"
    backend = create_runtime(config, control_period_s=CONTROL_PERIOD_S)
    assert backend.robot.spec.base_link_name == "base"
    assert backend.robot.spec.joint_names == JOINT_NAMES
    backend.open()

    for index, name in enumerate(backend.robot.spec.joint_names):
        joint = backend.model.joint(backend.robot._cfg.resolve_name(name))
        assert backend.robot._joint_qpos_indices[index] == joint.qposadr[0]
        assert backend.robot._joint_qvel_indices[index] == joint.dofadr[0]
        actuator = backend.robot._actuator_indices[index]
        assert backend.model.actuator_trnid[actuator, 0] == joint.id
    assert backend.robot._base_body_id == backend.model.body(backend.robot._cfg.resolve_name("base")).id

    def unexpected_lookup(*args: Any) -> int:
        pytest.fail("State reads must use bindings cached during open, not name lookups")

    monkeypatch.setattr(backend.mj, "mj_name2id", unexpected_lookup)
    assert backend.data is not None
    backend.data.qvel[:6] = [0.3, -0.2, 0.1, 0.4, -0.5, 0.6]
    backend.mj.mj_forward(backend.model, backend.data)
    state = backend.robot.read_state(0.1)
    for role in ("base_angular_velocity", "base_linear_acceleration", "base_linear_velocity"):
        np.testing.assert_allclose(getattr(state, role), backend.data.sensordata[backend.robot._sensor_slices[role]])
    np.testing.assert_allclose(state.base_position, backend.model.qpos0[:3])
    np.testing.assert_allclose(state.base_angular_velocity, [0.4, -0.5, 0.6], atol=1e-6)
    backend.close()


def test_actuator_force_range_mismatch_fails_during_open(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend()
    prepare = backend.robot.prepare

    def invalid_actuator_model(source_model: Any, model_spec: Any) -> None:
        source_model.actuator_forcerange[:, 0] = -23.0
        prepare(source_model, model_spec)

    monkeypatch.setattr(backend.robot, "prepare", invalid_actuator_model)
    with pytest.raises(ValidationError, match="force_lower"):
        backend.open()
    backend.close()


def test_open_converts_actuators_and_uses_scene_cfg_ground(caplog: pytest.LogCaptureFixture) -> None:
    spec = _spec()
    backend = _backend()
    with caplog.at_level(logging.INFO, logger="motrix_deploy_mujoco"):
        backend.open()

    model = backend.model
    assert model is not None
    assert model.opt.timestep == pytest.approx(0.002)
    assert model.opt.iterations == 100
    floor_id = backend.mj.mj_name2id(model, backend.mj.mjtObj.mjOBJ_GEOM, "floor")
    assert floor_id >= 0
    np.testing.assert_allclose(model.geom_friction[floor_id], [0.6, 0.005, 0.0001])
    np.testing.assert_array_equal(
        model.actuator_biastype[backend.robot._actuator_indices],
        backend.mj.mjtBias.mjBIAS_NONE,
    )
    np.testing.assert_allclose(model.actuator_gainprm[backend.robot._actuator_indices, 0], 1.0)
    np.testing.assert_allclose(
        model.actuator_ctrlrange[backend.robot._actuator_indices],
        np.column_stack((-spec.torque_limit, spec.torque_limit)),
    )
    assert "Converting 12 actuators from position servos to torque motors" in caplog.text
    assert "Built MuJoCo deployment model from SceneCfg" in caplog.text
    backend.close()


def test_direct_scene_cfg_headless_uses_supplied_robot_floor_and_placement() -> None:
    robot = UnitreeGo2Robot(translation=(1.0, -0.5, 0.155), rotation=(0.0, 0.0, 1.0, 0.0))
    scene = SceneCfg(objs=AdapterSceneObjsCfg(robot=robot, floor=FlatTerrainCfg(height=0.2)))
    config = replace(_config(), scene=scene)
    # Direct scenes have no training ctrl_dt: only integer physics substeps are required.
    backend = create_runtime(config, control_period_s=0.01)
    assert backend.robot._cfg is not robot
    backend.open()
    try:
        state = backend.robot.read_state(0.1)
        assert backend.robot.capabilities.privileged_state_fields == frozenset(
            {"base_position", "base_linear_velocity"}
        )
        np.testing.assert_allclose(state.base_position, backend.model.qpos0[:3])
        np.testing.assert_allclose(state.base_orientation_xyzw, robot.rotation)
        floor_id = backend.mj.mj_name2id(backend.model, backend.mj.mjtObj.mjOBJ_GEOM, "floor")
        assert backend.model.geom_pos[floor_id, 2] == pytest.approx(0.2)
        backend.robot.write_command(_command(_spec().default_joint_position))
        backend.advance_control_period()
        assert backend.robot.read_state(0.1).sample_time_ns == 10_000_000
    finally:
        backend.close()


@pytest.mark.parametrize("robot_type", [UnitreeGo2Robot])
def test_robot_description_switch_needs_no_backend_specific_wiring(robot_type) -> None:
    robot = robot_type()
    scene = SceneCfg(objs=AdapterSceneObjsCfg(robot=robot, floor=FlatTerrainCfg()))
    source = MuJoCoSceneCompiler().create_spec(scene, SimCfg()).compile()
    joints = tuple(robot.key_pose.joint_names)
    actuators = np.asarray([np.flatnonzero(source.actuator_trnid[:, 0] == source.joint(name).id)[0] for name in joints])
    spec = RobotSpec(
        base_link_name=robot.base_link_name,
        joint_names=joints,
        default_joint_position=np.asarray(robot.key_pose.poses[robot.init_key_pose], dtype=np.float32),
        position_lower=source.actuator_ctrlrange[actuators, 0].astype(np.float32),
        position_upper=source.actuator_ctrlrange[actuators, 1].astype(np.float32),
        torque_limit=source.actuator_forcerange[actuators, 1].astype(np.float32),
    )
    config = replace(_config(), scene=scene)
    backend = create_runtime(config, control_period_s=CONTROL_PERIOD_S)
    spec.validate_compatible(backend.robot.spec)
    backend.open()
    try:
        state = backend.robot.read_state(0.1)
        np.testing.assert_allclose(state.joint_position, spec.default_joint_position)
        zeros = np.zeros(spec.joint_count, dtype=np.float32)
        backend.robot.write_command(JointServoCommand(spec.default_joint_position, zeros, zeros, zeros, zeros))
        backend.advance_control_period()
        assert backend.robot.read_state(0.1).sample_time_ns > state.sample_time_ns
        assert backend.robot.health().healthy
    finally:
        backend.close()


def test_fallen_pose_does_not_change_resource_health() -> None:
    config = _config(FLAT_CONFIG_PATH)
    config.scene.objs.robot.translation = (0.0, 0.0, 0.545)
    config.scene.objs.robot.rotation = (1.0, 0.0, 0.0, 0.0)
    backend = create_runtime(config, control_period_s=CONTROL_PERIOD_S)
    backend.open()
    try:
        backend.robot.write_command(_command(_spec().default_joint_position, kp=0.0, kd=0.0))
        backend.advance_control_period()
        assert backend.robot.health().healthy
    finally:
        backend.close()


def test_direct_scene_recipe_runs_with_training_imports_blocked() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        class BlockTraining(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'motrix_envs' or fullname.startswith('motrix_envs.'):
                    raise AssertionError('Direct deployment must not import training environments')
        sys.meta_path.insert(0, BlockTraining())
        from omegaconf import OmegaConf
        from motrix_deploy_mujoco import MujocoRuntime
        from motrix_deploy.runtime.config import SimulationRuntimeConfig
        import numpy as np
        from motrix_env_core.config import configclass
        from motrix_env_core.config.scene import SceneCfg, SceneObjsCfg, FlatTerrainCfg
        @configclass
        class FlatWorldObjsCfg(SceneObjsCfg):
            floor: FlatTerrainCfg = FlatTerrainCfg()
        from motrix_robots.unitree import UnitreeGo2Robot
        values = OmegaConf.to_container(OmegaConf.load(sys.argv[1]).runtime)
        values.pop('backend')
        values.pop('deploy_env_id')
        values.pop('robot_id')
        robot = UnitreeGo2Robot(
            translation=tuple(values.pop('robot_translation')),
            rotation=tuple(values.pop('robot_rotation')),
        )
        values['scene'] = SceneCfg(objs=FlatWorldObjsCfg(robot=robot, floor=FlatTerrainCfg()))
        config = SimulationRuntimeConfig.from_mapping(values)
        backend = MujocoRuntime(config, control_period_s=0.02)
        backend.open()
        assert np.isfinite(backend.robot.read_state(0.1).joint_position).all()
        backend.close()
        assert not any(name == 'motrix_envs' or name.startswith('motrix_envs.') for name in sys.modules)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(FLAT_CONFIG_PATH)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hybrid_pd_command_is_recomputed_as_torque_each_physics_substep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec()
    backend = _backend()
    backend.open()
    initial = backend.robot.read_state(0.1)
    desired_position = initial.joint_position + np.linspace(0.01, 0.02, 12, dtype=np.float32)
    desired_velocity = np.linspace(-0.3, 0.3, 12, dtype=np.float32)
    feedforward_torque = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    kp = np.linspace(10.0, 21.0, 12, dtype=np.float32)
    kd = np.linspace(0.1, 1.2, 12, dtype=np.float32)
    command = JointServoCommand(
        joint_position=desired_position,
        joint_velocity=desired_velocity,
        feedforward_torque=feedforward_torque,
        kp=kp,
        kd=kd,
    )
    snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    mj_step = backend.mj.mj_step

    def record_step(model: object, data: object) -> None:
        snapshots.append(
            (
                data.qpos[backend.robot._joint_qpos_indices].copy(),
                data.qvel[backend.robot._joint_qvel_indices].copy(),
                data.ctrl[backend.robot._actuator_indices].copy(),
            )
        )
        mj_step(model, data)

    monkeypatch.setattr(backend.mj, "mj_step", record_step)

    backend.robot.write_command(command)
    backend.advance_control_period()

    assert backend.data is not None
    assert len(snapshots) == 10
    for position, velocity, applied_torque in snapshots:
        expected_torque = np.clip(
            kp * (desired_position - position) + kd * (desired_velocity - velocity) + feedforward_torque,
            -spec.torque_limit,
            spec.torque_limit,
        )
        np.testing.assert_allclose(applied_torque, expected_torque, atol=1e-6)
    assert not np.array_equal(snapshots[0][2], snapshots[1][2])
    assert backend.robot.read_state(0.1).sample_time_ns == 20_000_000
    backend.robot.stop()
    np.testing.assert_array_equal(backend.data.ctrl[backend.robot._actuator_indices], 0.0)
    backend.close()
