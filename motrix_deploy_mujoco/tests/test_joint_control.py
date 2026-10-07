# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Servo and torque command boundaries and physics-owned application cadence."""

import numpy as np
import pytest

from motrix_deploy.contracts import JointControlMode, JointServoCommand, JointTorqueCommand
from motrix_deploy.errors import ValidationError
from motrix_deploy_mujoco.runtime import MujocoRuntime
from motrix_env_core.config.scene.actuator import MotorActuatorCfg
from motrix_env_mujoco.compiler import MuJoCoSceneCompiler


def _servo(spec, **overrides):
    zeros = np.zeros(spec.joint_count, dtype=np.float32)
    fields = dict(
        joint_position=spec.default_joint_position, joint_velocity=zeros, feedforward_torque=zeros, kp=zeros, kd=zeros
    )
    fields.update(overrides)
    return JointServoCommand(**fields)


def test_auto_spec_inspection_reuses_original_source_spec_on_first_open(control_scene_config, monkeypatch):
    original = MuJoCoSceneCompiler.create_spec
    specs = []

    def record_spec(compiler, scene, sim):
        spec = original(compiler, scene, sim)
        specs.append(spec)
        return spec

    monkeypatch.setattr(MuJoCoSceneCompiler, "create_spec", record_spec)
    runtime = MujocoRuntime(control_scene_config, 0.02)
    assert len(specs) == 1 and runtime._source_spec is specs[0]
    assert runtime.model is runtime.data is None
    with runtime:
        assert len(specs) == 1
    with runtime:
        assert len(specs) == 2  # Reopen starts from source, not already transformed motors.


def test_torque_is_snapshotted_and_applied_directly_every_physics_substep(control_scene_config, monkeypatch):
    with MujocoRuntime(control_scene_config, 0.02) as runtime:
        assert runtime.robot.capabilities.control_modes == (JointControlMode.SERVO, JointControlMode.TORQUE)
        torque = np.linspace(-1, 1, runtime.robot.spec.joint_count, dtype=np.float32)
        expected = torque.copy()
        snapshots = []
        step = runtime.mj.mj_step

        def record_step(model, data):
            snapshots.append(data.ctrl[runtime.robot._actuator_indices].copy())
            step(model, data)

        monkeypatch.setattr(runtime.mj, "mj_step", record_step)
        runtime.robot.write_command(JointTorqueCommand(torque))
        torque[:] = 0
        assert runtime.data.time == 0
        runtime.advance_control_period()
        assert len(snapshots) == runtime.physics_substeps
        for applied in snapshots:
            np.testing.assert_array_equal(applied, expected)
        assert runtime.data.time == pytest.approx(runtime.control_period_s)
        runtime.robot.stop()
        np.testing.assert_array_equal(runtime.data.ctrl, 0)


@pytest.mark.parametrize("kind", ["torque", "feedforward", "position", "shape"])
def test_command_limits_are_validated_before_application(control_scene_config, kind):
    with MujocoRuntime(control_scene_config, 0.02) as runtime:
        spec = runtime.robot.spec
        if kind == "torque":
            command = JointTorqueCommand(spec.torque_limit + np.float32(1))
            field = "torque"
        elif kind == "feedforward":
            command = _servo(spec, feedforward_torque=spec.torque_limit + np.float32(1))
            field = "feedforward_torque"
        elif kind == "position":
            command = _servo(spec, joint_position=spec.position_upper + np.float32(1))
            field = "joint_position"
        else:
            command = JointTorqueCommand(np.zeros(spec.joint_count - 1, dtype=np.float32))
            field = "torque"
        with pytest.raises(ValidationError, match=f"command.{field}"):
            runtime.robot.write_command(command)
        np.testing.assert_array_equal(runtime.data.ctrl, 0)


def test_servo_total_pd_torque_is_clipped_and_mode_can_change(control_scene_config):
    with MujocoRuntime(control_scene_config, 0.02) as runtime:
        spec = runtime.robot.spec
        runtime.robot.write_command(
            _servo(spec, joint_position=spec.position_upper, kp=np.full(spec.joint_count, 1e6, dtype=np.float32))
        )
        runtime.robot.apply_control()
        np.testing.assert_allclose(runtime.data.ctrl[runtime.robot._actuator_indices], spec.torque_limit)
        torque = np.zeros(spec.joint_count, dtype=np.float32)
        runtime.robot.write_command(JointTorqueCommand(torque))
        runtime.robot.apply_control()
        np.testing.assert_array_equal(runtime.data.ctrl, 0)


def test_source_motor_model_derives_joint_limits_and_accepts_torque(control_scene_config):
    robot = control_scene_config.scene.objs.robot
    # Test-owned effort bounds differ from joint limits to exercise motor spec derivation.
    robot.actuators = {
        f"motor_{name}": MotorActuatorCfg(joint_name=name, ctrl_range=(-5.0, 5.0), force_range=(-5.0, 5.0))
        for name in robot.key_pose.joint_names
    }
    with MujocoRuntime(control_scene_config, 0.02) as runtime:
        ids = [runtime.model.joint(name).id for name in runtime.robot.spec.joint_names]
        np.testing.assert_allclose(runtime.robot.spec.position_lower, runtime.model.jnt_range[ids, 0])
        np.testing.assert_allclose(runtime.robot.spec.position_upper, runtime.model.jnt_range[ids, 1])
        np.testing.assert_array_equal(runtime.robot.spec.torque_limit, 5.0)
        runtime.robot.write_command(JointTorqueCommand(np.ones(runtime.robot.spec.joint_count, dtype=np.float32)))
        runtime.advance_control_period()
        np.testing.assert_array_equal(runtime.data.ctrl, 1)


@pytest.mark.parametrize("field", ["gear", "ctrllimited", "forcelimited", "dyntype"])
def test_noncanonical_source_actuation_is_rejected(control_scene_config, monkeypatch, field):
    original = MuJoCoSceneCompiler.create_spec

    def invalid_spec(compiler, scene, sim):
        spec = original(compiler, scene, sim)
        actuator = next(iter(spec.actuators))
        if field == "gear":
            actuator.gear = [2, 0, 0, 0, 0, 0]
        elif field in {"ctrllimited", "forcelimited"}:
            setattr(actuator, field, 0)
        else:
            import mujoco

            actuator.dyntype = mujoco.mjtDyn.mjDYN_FILTER
            actuator.dynprm[0] = 0.01
        return spec

    monkeypatch.setattr(MuJoCoSceneCompiler, "create_spec", invalid_spec)
    runtime = MujocoRuntime(control_scene_config, 0.02)
    with pytest.raises(ValidationError, match=f"backend.actuators.{field}"):
        runtime.open()
    assert runtime.model is runtime.data is None
