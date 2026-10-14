# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Native SDK motion-state and software-PD behavior contracts."""

from dataclasses import replace
from types import SimpleNamespace

import motrixsim as mtx
import numpy as np
from motrix_deploy_motrixsim.runtime import MotrixSimRuntime
from test_native_motrixsim_runtime import config

from motrix_deploy.contracts import JointControlMode, JointServoCommand, JointTorqueCommand


def set_simulation_state(
    runtime,
    *,
    root_position,
    root_orientation_xyzw,
    root_linear_velocity,
    root_angular_velocity,
    joint_position,
    joint_velocity,
):
    """Arrange native SDK state for sensor and control behavior tests only."""
    base = (runtime.scene.objs.robot.resolved_base_link_name,)
    joints = tuple(runtime.scene.objs.robot.resolve_name(name) for name in runtime.robot.spec.joint_names)
    program = runtime.model.compile_write(
        {
            "root_position": mtx.write.BodyPosition(base),
            "root_orientation_xyzw": mtx.write.BodyRotation(base),
            "root_linear_velocity": mtx.write.BodyLinearVelocity(base),
            "root_angular_velocity": mtx.write.BodyAngularVelocity(base),
            "joint_position": mtx.write.BodyJointPosition(joints),
            "joint_velocity": mtx.write.BodyJointVelocity(joints),
        },
        reset=True,
        forward_kinematic=True,
    ).allocate(runtime.data)
    program["root_position"][:] = root_position
    program["root_orientation_xyzw"][:] = root_orientation_xyzw
    program["root_linear_velocity"][:] = root_linear_velocity
    program["root_angular_velocity"][:] = root_angular_velocity
    program["joint_position"][:] = joint_position
    program["joint_velocity"][:] = joint_velocity
    program.execute(runtime.data)


def motion(spec):
    return SimpleNamespace(
        root_position=np.asarray([1.2, -0.7, 2.0], dtype=np.float32),
        root_orientation_xyzw=np.asarray([0.0, 0.0, np.sin(0.4), np.cos(0.4)], dtype=np.float32),
        root_linear_velocity=np.asarray([0.4, -0.2, 0.3], dtype=np.float32),
        root_angular_velocity=np.asarray([0.1, 0.3, -0.2], dtype=np.float32),
        joint_position=spec.default_joint_position + np.float32(0.02),
        joint_velocity=np.linspace(-0.2, 0.2, spec.joint_count, dtype=np.float32),
    )


def test_world_motion_sensor_frames_and_joint_mapping_with_instance_names():
    cfg = config()
    cfg.scene.objs.robot.prefix = "motion_"
    cfg.scene.objs.robot.suffix = "_instance"
    with MotrixSimRuntime(cfg) as runtime:
        initial = motion(runtime.robot.spec)
        set_simulation_state(runtime, **vars(initial))

        def check_motion():
            state = runtime.robot.read_state(0.1)
            base = (runtime.scene.objs.robot.resolved_base_link_name,)
            position = runtime.model.compile_query({"position": mtx.query.LinkPosition(base)}).allocate(runtime.data)
            position.execute(runtime.data)
            np.testing.assert_allclose(position["position"].reshape(3), initial.root_position, atol=1e-6)
            np.testing.assert_allclose(state.base_orientation_xyzw, initial.root_orientation_xyzw, atol=1e-6)
            np.testing.assert_allclose(state.joint_position, initial.joint_position, atol=1e-6)
            np.testing.assert_allclose(state.joint_velocity, initial.joint_velocity, atol=1e-6)
            angle = 0.8
            rotation = np.asarray(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            np.testing.assert_allclose(
                state.base_angular_velocity, rotation.T @ initial.root_angular_velocity, atol=1e-6
            )
            sensor_name = runtime.scene.objs.robot.resolve_name(runtime.config.sensor_bindings.base_linear_velocity)
            sensor = next(sensor for sensor in runtime.world.sensors.frame if sensor.name == sensor_name)
            site_read = runtime.model.compile_query(
                {
                    "position": mtx.query.SitePosition((sensor.object_type.value,)),
                }
            ).allocate(runtime.data)
            site_read.execute(runtime.data)
            site_offset = site_read["position"].reshape(3) - initial.root_position
            np.testing.assert_allclose(
                state.base_linear_velocity,
                initial.root_linear_velocity + np.cross(initial.root_angular_velocity, site_offset),
                atol=1e-6,
            )
            base = (runtime.scene.objs.robot.resolved_base_link_name,)
            read = runtime.model.compile_query(
                {
                    "linear": mtx.query.LinkLinearVelocity(base),
                    "angular": mtx.query.LinkAngularVelocity(base),
                }
            ).allocate(runtime.data)
            read.execute(runtime.data)
            np.testing.assert_allclose(read["linear"].reshape(3), initial.root_linear_velocity, atol=1e-6)
            np.testing.assert_allclose(read["angular"].reshape(3), initial.root_angular_velocity, atol=1e-6)
            return state

        check_motion()


def test_linear_velocity_capability_is_independent_of_sensor_binding():
    cfg = config()
    capabilities = []
    for binding in (cfg.sensor_bindings.base_linear_velocity, None):
        bound_config = replace(cfg, sensor_bindings=replace(cfg.sensor_bindings, base_linear_velocity=binding))
        with MotrixSimRuntime(bound_config) as runtime:
            capabilities.append(runtime.robot.capabilities)
            assert "base_linear_velocity" in runtime.robot.capabilities.state_fields
            initial = motion(runtime.robot.spec)
            initial.root_angular_velocity.fill(0)
            set_simulation_state(runtime, **vars(initial))
            state = runtime.robot.read_state(0.1)
            if binding is None:
                assert state.base_linear_velocity is None
            else:
                np.testing.assert_allclose(state.base_linear_velocity, initial.root_linear_velocity, atol=1e-6)
    assert capabilities[0] == capabilities[1]


def test_software_pd_uses_command_gains_velocity_and_feedforward():
    with MotrixSimRuntime(config()) as runtime:
        assert all(isinstance(actuator, mtx.MotorActuator) for actuator in runtime.model.actuators)
        assert runtime.robot.capabilities.control_modes == (JointControlMode.SERVO, JointControlMode.TORQUE)
        assert runtime.robot.capabilities.stop_semantics == "zero_torque"
        initial = motion(runtime.robot.spec)
        set_simulation_state(runtime, **vars(initial))
        count = runtime.robot.spec.joint_count
        command = JointServoCommand(
            initial.joint_position + np.float32(0.01),
            np.full(count, 0.15, dtype=np.float32),
            np.full(count, 0.5, dtype=np.float32),
            np.full(count, 12, dtype=np.float32),
            np.full(count, 2, dtype=np.float32),
        )
        runtime.robot.write_command(command)
        runtime.robot.apply_control()
        expected = (
            command.kp * (command.joint_position - initial.joint_position)
            + command.kd * (command.joint_velocity - initial.joint_velocity)
            + command.feedforward_torque
        )
        np.testing.assert_allclose(runtime.model.get_actuator_ctrls(runtime.data).reshape(-1), expected, atol=1e-6)
        runtime.robot.write_command(JointTorqueCommand(np.ones(count, dtype=np.float32)))
        runtime.robot.apply_control()
        np.testing.assert_array_equal(runtime.model.get_actuator_ctrls(runtime.data), 1)
        runtime.reset()
        np.testing.assert_array_equal(runtime.model.get_actuator_ctrls(runtime.data), 0)
