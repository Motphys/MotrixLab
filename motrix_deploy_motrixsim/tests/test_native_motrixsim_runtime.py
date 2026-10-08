# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Real installed SDK coverage of native deployment contracts."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import motrixsim as mtx
import numpy as np
import pytest
from motrix_deploy_motrixsim.runtime import MotrixSimRuntime
from motrix_deploy_motrixsim.viewer import MotrixSimKeyboard, MotrixSimViewer
from motrix_robots.unitree import UnitreeGo2Robot

from motrix_deploy.contracts import JointControlMode, JointServoCommand, JointTorqueCommand
from motrix_deploy.errors import ValidationError
from motrix_deploy.policy import NoOpPolicyRuntime
from motrix_deploy.runtime.config import SensorBindings, SimulationRuntimeConfig
from motrix_deploy.runtime.control import ControlSession
from motrix_deploy.runtime.factory import create_simulation_runtime
from motrix_deploy.task import DeployTask
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, SceneCfg, SceneObjsCfg
from motrix_env_core.config.scene.actuator import MotorActuatorCfg
from motrix_env_core.config.sim import SimCfg
from motrix_env_core.input import ConstantPlanarVelocityBinding


@configclass
class NativeSceneObjs(SceneObjsCfg):
    floor: FlatTerrainCfg | None = None


def config(**kwargs):
    robot = UnitreeGo2Robot(translation=(0.2, -0.1, 0.4), rotation=(0.0, 0.0, np.sin(0.2), np.cos(0.2)))
    return SimulationRuntimeConfig(
        scene=SceneCfg(objs=NativeSceneObjs(robot=robot, floor=FlatTerrainCfg())),
        physics=SimCfg(dt=0.002, solver_iterations=30),
        sensor_bindings=SensorBindings(
            base_angular_velocity="gyro",
            base_linear_acceleration="accelerometer",
            base_linear_velocity="global_linvel",
        ),
        **kwargs,
    )


class HoldTask(DeployTask):
    def __init__(self, spec):
        self.spec = spec

    def reset(self, state, context):
        pass

    def validate_command(self, command):
        pass

    def build_observation(self, state, context):
        return np.zeros(self.spec.joint_count, dtype=np.float32)

    def process_action(self, action):
        z = np.zeros(self.spec.joint_count, dtype=np.float32)
        return JointServoCommand(self.spec.default_joint_position, z, z, z + 30, z + 1)


def session(runtime, period=0.006):
    return ControlSession(
        robot=runtime.robot,
        task=HoldTask(runtime.robot.spec),
        policy=NoOpPolicyRuntime(runtime.robot.spec.joint_count),
        command_binding=ConstantPlanarVelocityBinding((0, 0, 0)),
        period_s=period,
        state_timeout_s=0.1,
    )


@pytest.mark.parametrize("gravity,tolerance", [((0.3, -0.2, -4.1), 2e-7), ((-0.1, 0.4, -6.3), 3e-6)])
def test_native_factory_forwards_shared_physics_once_to_compiler(monkeypatch, gravity, tolerance):
    from motrix_env_motrixsim.compiler import MotrixSimSceneCompiler

    cfg = config()
    physics = SimCfg(dt=0.003, solver_iterations=17, solver_tolerance=tolerance, gravity=gravity)
    cfg = replace(cfg, physics=physics)
    calls = []
    original = MotrixSimSceneCompiler.configure_world

    def configure_world(compiler, world, sim):
        calls.append(sim)
        return original(compiler, world, sim)

    monkeypatch.setattr(MotrixSimSceneCompiler, "configure_world", configure_world)
    runtime = create_simulation_runtime("motrixsim", cfg)
    try:
        assert runtime.config.physics is physics
        assert calls == [physics]
        assert calls[0] is physics
        options = runtime.world.simulate_option
        assert options.timestep == pytest.approx(physics.dt)
        assert options.constraint_solver_iterations == physics.solver_iterations
        assert options.constraint_solver_tolerance == pytest.approx(physics.solver_tolerance)
        np.testing.assert_allclose(options.gravity, physics.gravity)
    finally:
        runtime.close()


def test_native_factory_session_cadence_and_owner(monkeypatch):
    runtime = create_simulation_runtime("motrixsim", config())
    spec = runtime.robot.spec
    assert spec.joint_names == tuple(runtime.scene.objs.robot.key_pose.joint_names)
    assert runtime.robot.capabilities.control_modes == (JointControlMode.SERVO, JointControlMode.TORQUE)
    assert runtime.robot.capabilities.max_command_rate_hz == 1 / runtime.config.physics.dt
    with pytest.raises(RuntimeError):
        runtime.robot.read_state(0.1)
    control = session(runtime)
    runtime.bind_control_session(control)
    calls = []
    original = runtime.robot.apply_control

    def count_apply():
        calls.append(runtime.simulation_time_s)
        original()

    monkeypatch.setattr(runtime.robot, "apply_control", count_apply)
    with runtime:
        result = runtime.run(steps=5)
        assert result.success
        assert result.completed_steps == 5
        assert result.simulation_time_s == pytest.approx(5 * control.period_s)
        assert len(calls) == 5 * 3
        assert runtime.opened
        runtime.robot.close()
        assert runtime.opened and runtime.model is not None
        runtime.advance_control_period()
    assert runtime.model is None and runtime.data is None


def test_native_servo_feedforward_snapshots_and_clipping():
    runtime = MotrixSimRuntime(config())
    with runtime:
        state = runtime.robot.read_state(0.1)
        old_position = state.joint_position.copy()
        count = runtime.robot.spec.joint_count
        zeros = np.zeros(count, dtype=np.float32)
        target = state.joint_position.copy() + 0.01
        feedforward = zeros + 0.5
        command = JointServoCommand(target, zeros, feedforward, zeros + 10, zeros + 1)
        runtime.robot.write_command(command)
        command.joint_position[:] += 0.1
        runtime.robot.apply_control()
        expected = np.full(count, 0.6)
        actual = np.asarray(runtime.model.get_actuator_ctrls(runtime.data)).reshape(-1)
        np.testing.assert_allclose(actual, expected, atol=1e-5)
        runtime._step()
        runtime.robot.apply_control()
        current = runtime.robot.read_state(0.1)
        expected = 10 * (target - 0.1 - current.joint_position) - current.joint_velocity + 0.5
        expected = np.clip(expected, -runtime.robot.spec.torque_limit, runtime.robot.spec.torque_limit)
        np.testing.assert_allclose(
            np.asarray(runtime.model.get_actuator_ctrls(runtime.data)).reshape(-1), expected, atol=1e-5
        )
        np.testing.assert_array_equal(state.joint_position, old_position)
        runtime.robot.write_command(
            JointServoCommand(runtime.robot.spec.position_upper, zeros, zeros, zeros + 1e6, zeros)
        )
        runtime.robot.apply_control()
        np.testing.assert_allclose(
            np.asarray(runtime.model.get_actuator_ctrls(runtime.data)).reshape(-1), runtime.robot.spec.torque_limit
        )
        runtime.robot.stop()
        np.testing.assert_array_equal(runtime.model.get_actuator_ctrls(runtime.data), 0)


def test_native_servo_substeps_only_query_joint_state(monkeypatch):
    runtime = MotrixSimRuntime(config())
    with runtime:
        state = runtime.robot.read_state(0.1)
        zeros = np.zeros(runtime.robot.spec.joint_count, dtype=np.float32)
        runtime.robot.write_command(JointServoCommand(state.joint_position + 0.01, zeros, zeros, zeros + 10, zeros + 1))

        def unexpected_full_state_query(data):
            pytest.fail("Servo substeps must not query base state or sensors")

        monkeypatch.setattr(runtime.robot, "_read", SimpleNamespace(execute=unexpected_full_state_query))
        runtime._step()
        assert runtime.simulation_time_s == pytest.approx(runtime.config.physics.dt)


def test_native_repeated_reset_preserves_compiled_placement_and_sensor_state():
    runtime = MotrixSimRuntime(config())
    with runtime:
        initial = runtime.robot.read_state(0.1)
        assert initial.base_position[0] == pytest.approx(0.2)
        assert initial.base_position[1] == pytest.approx(-0.1)
        assert initial.base_position[2] > 0.4  # placement composes with imported root, never replaces it
        for _ in range(2):
            runtime.robot.write_command(JointTorqueCommand(np.ones(runtime.robot.spec.joint_count, dtype=np.float32)))
            for _ in range(5):
                runtime._step()
            assert runtime.robot.read_state(0.1).sample_time_ns == 10_000_000
            runtime.reset()
            reset = runtime.robot.read_state(0.1)
            assert reset.sample_time_ns == 0
            for field in (
                "base_position",
                "base_orientation_xyzw",
                "joint_position",
                "joint_velocity",
                "base_angular_velocity",
                "base_linear_acceleration",
                "base_linear_velocity",
            ):
                np.testing.assert_allclose(getattr(reset, field), getattr(initial, field), atol=1e-6)
        runtime.robot.write_command(JointTorqueCommand(np.zeros(runtime.robot.spec.joint_count, dtype=np.float32)))
    with runtime:
        np.testing.assert_allclose(runtime.robot.read_state(0.1).base_position, initial.base_position)


def test_native_sensor_bindings_are_actual_live_native_values():
    runtime = MotrixSimRuntime(config())
    with runtime:
        q = runtime.data.dof_pos.copy()
        v = runtime.data.dof_vel.copy()
        base = runtime.model.floating_bases[0]
        v[0, base.dof_vel_indices] = [0.7, 0.4, 0.2, 0.1, 0.2, 0.3]
        runtime.data.reset(runtime.model, dof_pos=q, dof_vel=v, forward_kinematic=True)
        runtime._step()
        state = runtime.robot.read_state(0.1)
        for role in ("base_angular_velocity", "base_linear_acceleration", "base_linear_velocity"):
            name = getattr(runtime.config.sensor_bindings, role)
            expected = runtime.model.get_sensor_value(name, runtime.data).reshape(3)
            np.testing.assert_allclose(getattr(state, role), expected, atol=1e-6)
        assert np.linalg.norm(state.base_linear_velocity) > 0.1
        assert np.linalg.norm(state.base_angular_velocity) > 0.1
        assert np.isfinite(state.base_linear_acceleration).all()


def test_native_instance_names_preserve_artifact_local_robot_contract():
    cfg = config()
    cfg.scene.objs.robot.prefix = "deployed_"
    cfg.scene.objs.robot.suffix = "_robot"
    with MotrixSimRuntime(cfg) as runtime:
        assert runtime.robot.spec.joint_names == tuple(cfg.scene.objs.robot.key_pose.joint_names)
        assert runtime.robot.spec.base_link_name == cfg.scene.objs.robot.base_link_name
        np.testing.assert_allclose(
            runtime.robot.read_state(0.1).joint_position, runtime.robot.spec.default_joint_position, atol=1e-6
        )
        runtime.bind_control_session(session(runtime))
        assert runtime.run(steps=2).success


def test_native_motor_source_supported():
    cfg = config()
    robot = cfg.scene.objs.robot
    names = robot.key_pose.joint_names
    source = MotrixSimRuntime(cfg)
    limits = source.robot.spec.torque_limit
    robot.model.actuators = {
        name: MotorActuatorCfg(
            joint_name=name, ctrl_range=(-float(limit), float(limit)), force_range=(-float(limit), float(limit))
        )
        for name, limit in zip(names, limits)
    }
    runtime = MotrixSimRuntime(cfg)
    assert all(isinstance(actuator, mtx.MotorActuator) for actuator in runtime.model.actuators)
    with runtime:
        runtime.robot.write_command(JointTorqueCommand(np.ones(len(names), dtype=np.float32)))
        runtime._step()
        assert runtime.robot.read_state(0.1).sample_time_ns == 2_000_000


@pytest.mark.parametrize("period", [0.001, 0.003])
def test_native_session_period_requires_whole_physics_steps(period):
    runtime = MotrixSimRuntime(config())
    with pytest.raises(ValidationError, match="integer multiple"):
        runtime.bind_control_session(session(runtime, period))
    runtime.close()


@pytest.mark.parametrize(
    "role,name",
    [
        ("base_angular_velocity", "missing"),
        ("base_angular_velocity", "global_angvel"),
        ("base_linear_acceleration", "gyro"),
        ("base_linear_velocity", "local_linvel"),
    ],
)
def test_native_explicit_sensor_semantics(role, name):
    cfg = config()
    cfg = replace(cfg, sensor_bindings=replace(cfg.sensor_bindings, **{role: name}))
    with pytest.raises(ValidationError, match="sensor_bindings"):
        with MotrixSimRuntime(cfg):
            pass


@pytest.mark.parametrize("role", ["base_angular_velocity", "base_linear_acceleration", "base_linear_velocity"])
def test_native_missing_sensor_fails_at_model_binding(role):
    cfg = config()
    cfg = replace(cfg, sensor_bindings=replace(cfg.sensor_bindings, **{role: None}))
    with pytest.raises(ValidationError, match=f"backend.sensor_bindings.{role}"):
        with MotrixSimRuntime(cfg):
            pass


@pytest.mark.parametrize(
    "bindings,message",
    [
        ({"base_angular_velocity": "   "}, "base_angular_velocity"),
        ({"base_angular_velocity": 7}, "base_angular_velocity"),
        ({"unexpected_role": "gyro"}, "unexpected_role"),
    ],
)
def test_native_artifact_mapping_validates_sensor_fields(bindings, message):
    cfg = config()
    with pytest.raises((TypeError, ValueError), match=message):
        SimulationRuntimeConfig.from_mapping(
            {"scene": cfg.scene, "physics": {"dt": 0.002, "solver_iterations": 30}, "sensor_bindings": bindings}
        )


def test_native_artifact_mapping_same_contract():
    cfg = config()
    runtime = create_simulation_runtime(
        "motrixsim",
        SimulationRuntimeConfig.from_mapping(
            {
                "scene": cfg.scene,
                "physics": {
                    "dt": 0.002,
                    "solver_iterations": 30,
                    "solver_tolerance": 4e-7,
                    "gravity": (0.2, -0.3, -5.0),
                },
                "sensor_bindings": asdict(cfg.sensor_bindings),
            }
        ),
    )
    assert isinstance(runtime.config.physics, SimCfg)
    assert runtime.world.simulate_option.constraint_solver_tolerance == pytest.approx(4e-7)
    np.testing.assert_allclose(runtime.world.simulate_option.gravity, (0.2, -0.3, -5.0))
    runtime.bind_control_session(session(runtime))
    with runtime:
        assert runtime.run(steps=2).success


@pytest.mark.parametrize("physics", [{}, {"dt": 0.004}, {"solver_iterations": None}])
def test_native_artifact_mapping_requires_complete_physics_recipe(physics):
    cfg = config()
    with pytest.raises(ValidationError, match="dt and solver_iterations"):
        SimulationRuntimeConfig.from_mapping({"scene": cfg.scene, "physics": physics})


def test_native_artifact_mapping_rejects_unknown_physics_field():
    cfg = config()
    with pytest.raises(TypeError, match="unexpected_physics_setting"):
        SimulationRuntimeConfig.from_mapping(
            {"scene": cfg.scene, "physics": {"dt": 0.002, "solver_iterations": 30, "unexpected_physics_setting": 1}}
        )


def test_sdk_viewer_lifecycle_and_keyboard(monkeypatch):
    calls = []

    class Render:
        is_closed = False
        input = SimpleNamespace(is_key_pressed=lambda key: False, is_key_just_pressed=lambda key: False)
        system_camera = SimpleNamespace(set_view=lambda *args: calls.append("camera"))

        def __init__(self, **kwargs):
            assert kwargs == {"headless": False}

        def launch(self, model, batch):
            assert batch == 1
            calls.append("launch")

        def sync(self, data):
            calls.append("sync")

        def close(self):
            self.is_closed = True
            calls.append("close")

    monkeypatch.setattr("motrix_deploy_motrixsim.viewer.RenderApp", Render)
    runtime = MotrixSimRuntime(config(render=True), viewer_factory=MotrixSimViewer)
    # CLI and application assembly obtain input before the world/viewer opens.
    device = runtime.get_keyboard_device()
    assert device is runtime.get_keyboard_device()
    assert calls == [] and not runtime.viewer.is_running()
    with pytest.raises(RuntimeError, match="Open the MotrixSim viewer"):
        device.poll()
    control = session(runtime)
    runtime.bind_control_session(control)
    with runtime:
        assert runtime.viewer.is_running()
        assert device is runtime.get_keyboard_device()
        device.poll()
        assert not device.is_pressing("w")
    assert calls == ["launch", "camera", "sync", "close"]
    assert device is runtime.get_keyboard_device()
    assert not runtime.viewer.is_running()
    with pytest.raises(RuntimeError, match="Open the MotrixSim viewer"):
        device.is_pressing("w")
    runtime.close()
    assert calls == ["launch", "camera", "sync", "close"]
    held = {"w": True}
    keyboard = MotrixSimKeyboard(
        SimpleNamespace(is_key_pressed=lambda key: held[key], is_key_just_pressed=lambda key: held[key])
    )
    keyboard.poll()
    assert keyboard.is_key_down("w") and keyboard.is_pressing("w")
    held["w"] = False
    keyboard.poll()
    assert keyboard.is_key_up("w") and not keyboard.is_pressing("w")
