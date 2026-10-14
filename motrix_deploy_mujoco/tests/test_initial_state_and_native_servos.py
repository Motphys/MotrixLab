# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Robot motion coordinates, sensor bindings and software PD contracts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from motrix_deploy.contracts import JointControlMode, JointServoCommand
from motrix_deploy_mujoco.runtime import MujocoRuntime


def _motion_fixture(spec):
    return SimpleNamespace(
        root_position=np.array([1.2, -0.5, 2.0], dtype=np.float32),
        root_orientation_xyzw=np.array([0, 0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32),
        root_linear_velocity=np.array([0.2, -0.4, 0.6], dtype=np.float32),
        root_angular_velocity=np.array([0.1, 0.3, -0.2], dtype=np.float32),
        joint_position=spec.default_joint_position + np.linspace(-0.01, 0.01, spec.joint_count, dtype=np.float32),
        joint_velocity=np.linspace(-0.3, 0.3, spec.joint_count, dtype=np.float32),
    )


def _set_motion_fixture(runtime, motion):
    """Write native MuJoCo data only to exercise robot state decoding in tests."""
    model, data = runtime.model, runtime.data
    cfg = runtime.scene.objs.robot
    root_joint = model.body_jntadr[model.body(cfg.resolved_base_link_name).id]
    qpos_address = model.jnt_qposadr[root_joint]
    dof_address = model.jnt_dofadr[root_joint]
    quaternion = motion.root_orientation_xyzw[[3, 0, 1, 2]].astype(np.float64)
    rotation = np.empty(9)
    runtime.mj.mju_quat2Mat(rotation, quaternion)
    data.qpos[qpos_address : qpos_address + 3] = motion.root_position
    data.qpos[qpos_address + 3 : qpos_address + 7] = quaternion
    data.qvel[dof_address : dof_address + 3] = motion.root_linear_velocity
    data.qvel[dof_address + 3 : dof_address + 6] = rotation.reshape(3, 3).T @ motion.root_angular_velocity
    for index, name in enumerate(cfg.key_pose.joint_names):
        joint = model.joint(cfg.resolve_name(name))
        data.qpos[joint.qposadr] = motion.joint_position[index]
        data.qvel[joint.dofadr] = motion.joint_velocity[index]
    runtime.mj.mj_forward(model, data)


def test_robot_state_maps_root_coordinates_gyro_world_velocity_and_canonical_joints(control_scene_config):
    # Reverse canonical order and prefix model names: neither vector may use raw MuJoCo order.
    cfg = control_scene_config.scene.objs.robot
    cfg.prefix = "test_"
    names = cfg.key_pose.joint_names
    cfg.key_pose.joint_names = tuple(reversed(names))
    cfg.key_pose.poses = {name: tuple(reversed(pose)) for name, pose in cfg.key_pose.poses.items()}
    with MujocoRuntime(control_scene_config, 0.02) as runtime:
        motion = _motion_fixture(runtime.robot.spec)
        _set_motion_fixture(runtime, motion)
        state = runtime.robot.read_state(0.1)
        np.testing.assert_allclose(runtime.data.xpos[runtime.robot._base_body_id], motion.root_position, atol=1e-6)
        np.testing.assert_allclose(state.base_orientation_xyzw, motion.root_orientation_xyzw, atol=1e-6)
        np.testing.assert_allclose(state.base_angular_velocity, [0.3, -0.1, -0.2], atol=1e-6)
        # The world-frame sensor includes omega cross offset for its attached site.
        sensor = runtime.model.sensor(cfg.resolve_name("global_linvel"))
        site_position = runtime.data.site_xpos[sensor.objid[0]]
        offset = site_position - runtime.data.xpos[runtime.robot._base_body_id]
        expected_velocity = motion.root_linear_velocity + np.cross(motion.root_angular_velocity, offset)
        np.testing.assert_allclose(state.base_linear_velocity, expected_velocity, atol=1e-6)
        np.testing.assert_array_equal(state.joint_position, motion.joint_position)
        np.testing.assert_array_equal(state.joint_velocity, motion.joint_velocity)


def test_linear_velocity_capability_is_independent_of_sensor_binding(control_scene_config):
    capabilities = []
    for binding in (control_scene_config.sensor_bindings.base_linear_velocity, None):
        config = replace(
            control_scene_config,
            sensor_bindings=replace(control_scene_config.sensor_bindings, base_linear_velocity=binding),
        )
        with MujocoRuntime(config) as runtime:
            capabilities.append(runtime.robot.capabilities)
            assert "base_linear_velocity" in runtime.robot.capabilities.state_fields
            motion = _motion_fixture(runtime.robot.spec)
            motion.root_angular_velocity.fill(0)
            _set_motion_fixture(runtime, motion)
            state = runtime.robot.read_state(0.1)
            if binding is None:
                assert state.base_linear_velocity is None
            else:
                np.testing.assert_allclose(state.base_linear_velocity, motion.root_linear_velocity, atol=1e-6)
    assert capabilities[0] == capabilities[1]


def test_software_pd_uses_all_command_fields_each_substep_and_zeroes_on_stop(control_scene_config, monkeypatch):
    with MujocoRuntime(control_scene_config, 0.02) as runtime:
        spec = runtime.robot.spec
        robot = runtime.robot
        indices = robot._actuator_indices
        assert robot.capabilities.control_modes == (JointControlMode.SERVO, JointControlMode.TORQUE)
        assert robot.capabilities.stop_semantics == "zero_torque"
        assert np.all(runtime.model.actuator_biastype[indices] == runtime.mj.mjtBias.mjBIAS_NONE)
        command = JointServoCommand(
            joint_position=spec.default_joint_position.copy(),
            joint_velocity=np.full(spec.joint_count, 0.2, dtype=np.float32),
            feedforward_torque=np.full(spec.joint_count, 0.3, dtype=np.float32),
            kp=np.full(spec.joint_count, 3.0, dtype=np.float32),
            kd=np.full(spec.joint_count, 0.7, dtype=np.float32),
        )
        robot.write_command(command)
        snapshots = []
        step = runtime.mj.mj_step

        def record_step(model, data):
            expected = np.clip(
                command.kp * (command.joint_position - data.qpos[robot._joint_qpos_indices])
                + command.kd * (command.joint_velocity - data.qvel[robot._joint_qvel_indices])
                + command.feedforward_torque,
                -spec.torque_limit,
                spec.torque_limit,
            )
            np.testing.assert_allclose(data.ctrl[indices], expected, atol=1e-7)
            snapshots.append(data.ctrl[indices].copy())
            step(model, data)

        monkeypatch.setattr(runtime.mj, "mj_step", record_step)
        runtime.advance_control_period()
        assert len(snapshots) == runtime.physics_substeps
        assert not np.array_equal(snapshots[0], snapshots[-1])
        robot.stop()
        np.testing.assert_array_equal(runtime.data.ctrl[indices], 0)
        robot.apply_control()
        np.testing.assert_array_equal(runtime.data.ctrl[indices], 0)
