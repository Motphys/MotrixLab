# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Complete scene ownership, absolute placement and runtime lifecycle contracts."""

from dataclasses import replace

import numpy as np
import pytest
from motrix_robots.unitree import UnitreeGo2Robot
from scene_helpers import AdapterSceneObjsCfg, build_scene
from test_mujoco_deploy_backend import _command, _spec, create_runtime

from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime.config import SensorBindings, SimulationRuntimeConfig
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, SceneCfg, SceneObjsCfg, SystemCameraCfg
from motrix_env_core.config.sim import SimCfg
from motrix_env_mujoco.compiler import MuJoCoSceneCompiler


@configclass
class CustomSceneObjsCfg(SceneObjsCfg):
    platform: FlatTerrainCfg = FlatTerrainCfg(height=0.2)


def _config(scene=None):
    return SimulationRuntimeConfig(
        scene=build_scene() if scene is None else scene,
        physics=SimCfg(dt=0.002, solver_iterations=20),
        sensor_bindings=SensorBindings(
            base_angular_velocity="gyro",
            base_linear_acceleration="accelerometer",
            base_linear_velocity="global_linvel",
        ),
    )


def test_complete_scene_copies_all_fields_without_reconstructing_derived_contacts() -> None:
    @configclass
    class DerivedScene(SceneCfg):
        contact_name: str | None = None

        def __post_init__(self):
            self.contact_name = self.objs.robot.resolved_base_link_name

    robot = UnitreeGo2Robot(translation=(4, 5, 6), rotation=(0, 0, 1, 0), prefix="deployment_")
    source = DerivedScene(objs=CustomSceneObjsCfg(robot=robot), system_camera=SystemCameraCfg(distance=4.0))
    backend = create_runtime(_config(source), control_period_s=0.02)
    assert backend.robot.spec.base_link_name == robot.base_link_name
    assert backend.robot.spec.joint_names == tuple(robot.key_pose.joint_names)
    scene = backend.scene
    assert type(scene.objs) is CustomSceneObjsCfg
    assert scene.objs.platform is not source.objs.platform
    assert scene.assets is not source.assets and scene.sensors is not source.sensors
    assert scene.system_camera is not source.system_camera
    assert scene.system_camera.distance == source.system_camera.distance
    assert scene.contact_name == source.contact_name == "deployment_base"
    assert robot.translation == (4, 5, 6) and robot.rotation == (0, 0, 1, 0)
    assert scene.objs.robot.translation == robot.translation and scene.objs.robot.rotation == robot.rotation
    scene.objs.platform.height = 9
    assert source.objs.platform.height == 0.2


def test_complete_scene_reuses_one_copy_without_inserting_another_robot() -> None:
    robot = UnitreeGo2Robot(translation=(4, 5, 6), rotation=(0, 0, 1, 0))
    scene = SceneCfg(objs=AdapterSceneObjsCfg(robot=robot, floor=FlatTerrainCfg()))
    backend = create_runtime(_config(scene), control_period_s=0.02)
    assert backend.scene is not scene
    assert len(list(backend.scene.iter_objs())) == 2
    backend.open()
    try:
        assert backend.model.njnt == 13
        np.testing.assert_allclose(backend.robot.read_state(0.1).base_orientation_xyzw, robot.rotation, atol=1e-6)
    finally:
        backend.close()
    assert robot.translation == (4, 5, 6)


def test_file_scene_and_custom_objects_preserved_placement_has_no_double_transform(tmp_path) -> None:
    path = tmp_path / "world.xml"
    path.write_text(
        '<mujoco><worldbody><geom name="obstacle" type="box" size=".2 .3 .4" pos="5 6 7"/></worldbody></mujoco>'
    )
    robot = UnitreeGo2Robot(translation=(5, 6, 7), rotation=(0.6, 0, 0, 0.8))
    scene = SceneCfg(file=path, objs=CustomSceneObjsCfg(robot=robot), system_camera=SystemCameraCfg(distance=3))
    config = _config(scene)
    backend = create_runtime(config, control_period_s=0.02)
    backend.open()
    try:
        state = backend.robot.read_state(0.1)
        np.testing.assert_allclose(state.base_position, backend.model.qpos0[:3])
        np.testing.assert_allclose(state.base_orientation_xyzw, robot.rotation, atol=1e-6)
        np.testing.assert_allclose(backend.model.geom("obstacle").pos, [5, 6, 7])
        assert backend.model.geom("platform").pos[2] == pytest.approx(0.2)
        assert state.sample_time_ns == 0
        assert backend.robot.read_state(0.1).receive_time_ns >= state.receive_time_ns
        backend.robot.write_command(_command(_spec().default_joint_position))
        backend.advance_control_period()
        assert backend.robot.read_state(0.1).sample_time_ns == 20_000_000
    finally:
        backend.close()
    assert scene.file == path and scene.objs.robot is robot
    assert robot.translation == (5, 6, 7) and robot.rotation == (0.6, 0, 0, 0.8)


@pytest.mark.parametrize("rotation", [None, (0.6, 0.0, 0.0, 0.8)])
def test_robot_attach_transform_and_init_key_pose_are_reset_once(rotation) -> None:
    import mujoco

    from motrix_deploy.runtime.config import SimulationRuntimeConfig
    from motrix_deploy.runtime.factory import create_simulation_runtime

    robot = UnitreeGo2Robot()
    scene = SceneCfg(objs=AdapterSceneObjsCfg(robot=robot))
    intrinsic = MuJoCoSceneCompiler().create_spec(scene, SimCfg()).compile().qpos0[:7].copy()
    translation = (1.2, -0.7, 2.1)
    robot.translation = translation
    robot.rotation = rotation
    robot.key_pose.poses["custom"] = tuple(np.asarray(robot.key_pose.poses["default"]) + 0.01)
    robot.init_key_pose = "custom"
    attach_quat = np.asarray((0, 0, 0, 1) if rotation is None else rotation)[[3, 0, 1, 2]]
    rotated_position = np.empty(3)
    expected_quat = np.empty(4)
    mujoco.mju_rotVecQuat(rotated_position, intrinsic[:3], attach_quat)
    mujoco.mju_mulQuat(expected_quat, attach_quat, intrinsic[3:])
    expected_position = np.asarray(translation) + rotated_position
    runtime = create_simulation_runtime(
        "mujoco", SimulationRuntimeConfig(scene=scene, sensor_bindings=_config().sensor_bindings)
    )
    with runtime:
        for _ in range(2):
            state = runtime.robot.read_state(0.1)
            np.testing.assert_allclose(state.base_position, expected_position, atol=1e-6)
            np.testing.assert_allclose(state.base_orientation_xyzw, expected_quat[[1, 2, 3, 0]], atol=1e-6)
            np.testing.assert_allclose(state.joint_position, robot.key_pose.poses["custom"], atol=1e-6)
            runtime.data.qpos[:] = 0.0
            runtime.data.qvel[:] = 1.0
            runtime.reset(runtime.robot._joint_qpos_indices, runtime.robot.spec.default_joint_position)
            np.testing.assert_array_equal(runtime.data.qvel, 0.0)
    assert scene.objs.robot.translation == translation and scene.objs.robot.rotation == rotation


def test_environment_scene_is_resolved_by_application_before_backend() -> None:
    import motrix_envs  # noqa: F401 registers configurations for this application test
    from motrix_env_core import registry

    source = registry.make_env_config("go2-walk-rough", mode="play").scene
    backend = create_runtime(_config(source), control_period_s=0.02)
    backend.open()
    try:
        assert backend.model.njnt == 13
        assert backend.scene is not source
        np.testing.assert_allclose(backend.robot.read_state(0.1).base_position, backend.model.qpos0[:3])
    finally:
        backend.close()


@pytest.mark.parametrize("failure", ["prepare", "bind", "viewer"])
def test_failed_open_releases_session_and_viewer_idempotently(failure, monkeypatch) -> None:
    class Viewer:
        running = False
        closes = 0

        def open(self, model, data):
            self.running = True
            if failure == "viewer":
                raise ValueError("viewer failed")

        def close(self):
            self.closes += 1
            self.running = False

        def is_running(self):
            return self.running

        def sync(self):
            pass

    viewer = Viewer()
    config = _config()
    if failure == "bind":
        config = replace(config, sensor_bindings=SensorBindings())
    backend = create_runtime(config, control_period_s=0.02, render=True, viewer_factory=lambda *args: viewer)
    if failure == "prepare":
        prepare = backend.robot.prepare

        def invalid_actuator_model(source_model, model_spec):
            source_model.actuator_forcerange[:, 0] = -23.0
            prepare(source_model, model_spec)

        monkeypatch.setattr(backend.robot, "prepare", invalid_actuator_model)
    with pytest.raises((ValidationError, ValueError)):
        backend.open()
    assert backend.model is None and backend.data is None
    assert not backend.robot.health().healthy
    assert not viewer.running and viewer.closes == 1
    backend.close()
    backend.close()
    assert viewer.closes == 1
    with pytest.raises(RuntimeError, match="not open"):
        backend.robot.read_state(0.1)


def test_robot_io_lifecycle_preserves_world_and_rebinds_on_runtime_reopen(control_scene_config) -> None:
    from motrix_deploy.contracts import JointTorqueCommand
    from motrix_deploy.robot.interface import RobotInterface
    from motrix_deploy_mujoco.runtime import MujocoRuntime

    runtime = MujocoRuntime(control_scene_config, control_period_s=0.02)
    robot = runtime.robot
    assert isinstance(robot, RobotInterface)
    spec = robot.spec
    with pytest.raises(RuntimeError, match="not open"):
        robot.open()

    with runtime:
        robot.open()
        model, data = runtime.model, runtime.data
        robot.write_command(JointTorqueCommand(np.ones(spec.joint_count, dtype=np.float32)))
        runtime.advance_control_period()
        time_before_close = data.time
        robot.close()
        robot.close()
        robot.stop()
        assert runtime.opened
        assert runtime.model is model and runtime.data is data
        assert data.time == time_before_close
        np.testing.assert_array_equal(data.ctrl, 0.0)
        assert not robot.health().healthy
        with pytest.raises(RuntimeError, match="not open"):
            robot.read_state(0.1)
        with pytest.raises(RuntimeError, match="not open"):
            robot.open()

    assert runtime.model is None and runtime.data is None
    with runtime:
        assert runtime.robot is robot and robot.spec is spec
        robot.open()
        assert robot.health().healthy
        state = robot.read_state(0.1)
        assert state.sample_time_ns == 0
        np.testing.assert_array_equal(state.joint_position, spec.default_joint_position)
    assert not robot.health().healthy
    with pytest.raises(RuntimeError, match="not open"):
        robot.read_state(0.1)


def test_runtime_requires_primary_scene_robot() -> None:
    with pytest.raises(ValidationError, match="backend.scene.objs.robot"):
        create_runtime(_config(SceneCfg()), control_period_s=0.02)


def test_physics_requires_integral_substeps() -> None:
    with pytest.raises(ValidationError, match="integer multiple"):
        create_runtime(_config(), control_period_s=0.003)


@pytest.mark.parametrize("gravity,tolerance", [((0.3, -0.2, -4.1), 2e-7), ((-0.1, 0.4, -6.3), 3e-6)])
def test_public_factory_forwards_shared_physics_once_to_compiler(monkeypatch, gravity, tolerance) -> None:
    from motrix_deploy.runtime.config import SimulationRuntimeConfig
    from motrix_deploy.runtime.factory import create_simulation_runtime

    physics = SimCfg(dt=0.003, solver_iterations=17, solver_tolerance=tolerance, gravity=gravity)
    config = _config()
    calls = []
    original = MuJoCoSceneCompiler.create_spec

    def create_spec(compiler, scene, sim):
        calls.append(sim)
        return original(compiler, scene, sim)

    monkeypatch.setattr(MuJoCoSceneCompiler, "create_spec", create_spec)
    runtime = create_simulation_runtime(
        "mujoco", SimulationRuntimeConfig(scene=config.scene, physics=physics, sensor_bindings=config.sensor_bindings)
    )
    assert runtime.config.physics is physics
    assert runtime.config.sensor_bindings is config.sensor_bindings
    with runtime:
        assert calls == [physics]
        assert calls[0] is physics
        assert runtime.model.opt.timestep == physics.dt
        assert runtime.model.opt.iterations == physics.solver_iterations
        assert runtime.model.opt.tolerance == physics.solver_tolerance
        np.testing.assert_allclose(runtime.model.opt.gravity, physics.gravity)


@pytest.mark.parametrize("render", [False, True])
@pytest.mark.parametrize("realtime", [None, False, True])
def test_direct_runtime_and_plugin_use_config_render_and_realtime(control_scene_config, render, realtime) -> None:
    from motrix_deploy_mujoco.plugin import create_runtime as create_plugin_runtime
    from motrix_deploy_mujoco.runtime import MujocoRuntime

    config = replace(control_scene_config, render=render, realtime=realtime)
    direct = MujocoRuntime(config)
    plugin = create_plugin_runtime(config)
    for runtime in (direct, plugin):
        assert runtime.config is config
        assert (runtime.viewer is not None) == render
        assert runtime.realtime is realtime
        assert runtime.model is None and runtime.data is None


def test_artifact_recipe_normalizes_to_simulation_config(control_scene_config) -> None:
    from motrix_deploy_mujoco.plugin import create_runtime as create_plugin_runtime

    config = control_scene_config
    runtime_config = SimulationRuntimeConfig.from_mapping(
        {
            "name": "mujoco",
            "scene": config.scene,
            "physics": {"dt": config.physics.dt, "solver_iterations": config.physics.solver_iterations},
            "sensor_bindings": {
                "base_angular_velocity": "gyro",
                "base_linear_acceleration": "accelerometer",
                "base_linear_velocity": "global_linvel",
            },
        },
        render=True,
        realtime=False,
    )
    runtime = create_plugin_runtime(runtime_config)
    assert isinstance(runtime.config, SimulationRuntimeConfig)
    assert runtime.config.scene is config.scene
    assert runtime.config.render is True and runtime.config.realtime is False
    assert runtime.viewer is not None and runtime.realtime is False
    assert runtime.robot.spec.joint_names == tuple(config.scene.objs.robot.key_pose.joint_names)


def test_public_simulation_factory_returns_model_derived_robot_spec() -> None:
    from motrix_deploy.runtime.base import SimulationRuntime
    from motrix_deploy.runtime.config import SimulationRuntimeConfig
    from motrix_deploy.runtime.factory import create_simulation_runtime as create_selected_runtime

    config = _config()
    runtime = create_selected_runtime(
        "mujoco", SimulationRuntimeConfig(scene=config.scene, sensor_bindings=config.sensor_bindings)
    )
    assert isinstance(runtime, SimulationRuntime)
    assert runtime.config.scene is config.scene
    spec = runtime.robot.spec
    assert runtime.model is None and runtime.data is None
    assert spec.joint_names == tuple(config.scene.objs.robot.key_pose.joint_names)
    port = runtime.robot
    assert runtime.robot is port and port.spec is spec
    with runtime:
        port.open()
        np.testing.assert_array_equal(port.read_state(0.1).joint_position, spec.default_joint_position)
    assert runtime.model is None and runtime.data is None


@pytest.mark.parametrize("field", ["default_joint_position", "position_lower", "position_upper", "torque_limit"])
def test_artifact_robot_vectors_are_compared_against_derived_spec(field) -> None:
    runtime = create_runtime(_config(), control_period_s=0.02)
    actual = runtime.robot.spec
    expected = replace(actual, **{field: getattr(actual, field) + np.float32(0.01)})
    with pytest.raises(ValidationError, match=field):
        expected.validate_compatible(actual)
    assert runtime.model is None and runtime.data is None


@pytest.mark.parametrize("field", ["default_joint_position", "position_lower", "position_upper", "torque_limit"])
def test_artifact_robot_vector_comparison_uses_absolute_tolerance(field) -> None:
    runtime = create_runtime(_config(), control_period_s=0.02)
    actual = runtime.robot.spec
    expected = replace(actual, **{field: getattr(actual, field) + np.float32(5e-7)})
    expected.validate_compatible(actual)
    # Values such as torque limits must not get a scale-dependent relative tolerance.
    expected = replace(actual, **{field: getattr(actual, field) + np.float32(5e-6)})
    with pytest.raises(ValidationError, match=field):
        expected.validate_compatible(actual)
