# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Training/deployment observation, action and embedded motion parity."""

import io

import numpy as np
import pytest

import motrix_deploy_tasks
from motrix_deploy.contracts import RobotState
from motrix_deploy.runtime import ControlContext
from motrix_deploy_tasks.tasks.g1_wbt import G1WbtTaskSpec
from motrix_deploy_tasks.tasks.go2_walk import Go2WalkPolicyProcessor, Go2WalkTaskSpec
from motrix_env_core import registry
from motrix_env_core.input import PlanarVelocityCommand
from motrix_envs.deploy import build_deployment_profile


@pytest.mark.parametrize("env_name", ["go2-walk-flat", "go2-walk-rough"])
def test_go2_training_and_deployment_task_golden_probe(env_name: str) -> None:
    assert motrix_deploy_tasks.__name__
    profile = build_deployment_profile(env_name)
    env = registry.make(env_name, num_envs=1, mode="play")
    env.cfg.spawn_xy_range = 0.0
    env_state = env.init_state()
    probe = _make_env_probe(env)
    robot_state = _robot_state_from_env(env, env_state, probe)
    context = ControlContext(step=0, elapsed_time_s=0.0, command=PlanarVelocityCommand(env._commands), dt_s=0.02)
    task = Go2WalkPolicyProcessor(profile.task, profile.robot)
    assert isinstance(profile.task, Go2WalkTaskSpec)
    command_scale = np.asarray(profile.task.command_scale, dtype=np.float32)
    assert command_scale.shape == (3,)
    np.testing.assert_allclose(task.command_lower, env.cfg.commands.velocity.lower * command_scale)
    np.testing.assert_allclose(task.command_upper, env.cfg.commands.velocity.upper * command_scale)
    task.validate_command(context.command)
    task.reset(robot_state, context)
    assert profile.task.termination_min_up_z == env.cfg.termination_min_up_z
    for up_z in (-1.0, 0.0, 1.0):
        angle = np.arccos(up_z)
        robot_state.base_orientation_xyzw = np.array([np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)], dtype=np.float32)
        assert (task.check_termination(robot_state) is not None) == (up_z <= env.cfg.termination_min_up_z)
    robot_state = _robot_state_from_env(env, env_state, probe)

    np.testing.assert_allclose(
        task.build_observation(robot_state, context),
        env_state.obs.policy[0],
        atol=1e-6,
        rtol=1e-6,
    )
    raw_action = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
    command = task.process_action(raw_action)
    env.apply_action(raw_action[None, :], env_state)
    probe.execute()
    np.testing.assert_allclose(command.joint_position, probe["actuator_ctrls"][0], atol=1e-6, rtol=1e-6)

    stepped = env.step(raw_action[None, :])
    next_state = _robot_state_from_env(env, stepped, probe)
    next_context = ControlContext(
        step=1, elapsed_time_s=profile.control.period_s, command=PlanarVelocityCommand(env._commands), dt_s=0.02
    )
    task.validate_command(next_context.command)
    np.testing.assert_allclose(
        task.build_observation(next_state, next_context),
        stepped.obs.policy[0],
        atol=1e-5,
        rtol=1e-5,
    )


def test_g1_profile_uses_config_not_environment_and_embeds_named_motion(monkeypatch):
    pytest.importorskip("motrix_envs")
    from motrix_envs.deploy.g1_wbt import build_g1_wbt_profile
    from motrix_envs.motion.library import expand_motion_paths
    from motrix_envs.motion.loader import MotrixMotion

    def no_environment(*args, **kwargs):
        pytest.fail("profile compilation must not construct a training environment")

    monkeypatch.setattr(registry, "make", no_environment)
    profile = build_g1_wbt_profile("g1-wbt-dance")
    cfg = registry.make_env_config("g1-wbt-dance")
    motion = MotrixMotion(expand_motion_paths(cfg.commands.motion.motion_files)[0])
    restored = G1WbtTaskSpec.model_validate_json(profile.task.model_dump_json())
    indices = motion.joint_indices(list(profile.robot.joint_names))
    torso = motion.body_index(cfg.commands.motion.reference_body_name)
    payload = np.load(io.BytesIO(next(iter(profile.payloads.values()))), allow_pickle=False)
    np.testing.assert_array_equal(payload["joint_pos"], motion.joint_pos[:, indices])
    np.testing.assert_array_equal(payload["joint_vel"], motion.joint_vel[:, indices])
    np.testing.assert_array_equal(payload["reference_quaternion_xyzw"], motion.body_quat_w[:, torso])
    assert restored.motion_frames == len(motion.joint_pos)
    assert restored.termination_ref_orientation_threshold == cfg.terminations.bad_ref_ori.threshold
    assert restored.termination_joint_position_threshold == cfg.terminations.bad_dof_pos.threshold
    assert restored.termination_joint_velocity_threshold == cfg.terminations.bad_dof_vel.threshold
    kp = np.asarray(restored.kp)
    expected_scale = np.zeros_like(kp)
    np.divide(profile.robot.torque_limit, kp, out=expected_scale, where=kp != 0)
    expected_scale *= cfg.actions.joint_position.action_scale
    np.testing.assert_allclose(restored.action_scale, expected_scale)
    assert profile.observation_size == 154  # Actor interface, not task tuning.
    assert profile.action_size == profile.robot.joint_count == 29
    assert profile.control.period_s == restored.period_s


def _make_env_probe(env):
    """Side compiler reading the imu site pose and live actuator controls."""

    from motrix_env_core.sim import ActuatorCtrlQuery, SitePositionQuery, SiteQuaternionQuery

    return env.sim.compile_reads(
        {
            "imu_pos": SitePositionQuery(site="imu"),
            "imu_quat": SiteQuaternionQuery(site="imu"),
            "actuator_ctrls": ActuatorCtrlQuery(),
        }
    )


def _robot_state_from_env(env, state, probe) -> RobotState:
    probe.execute()
    imu_quat = probe["imu_quat"][0]
    return RobotState(
        sample_time_ns=int(state.episode_steps[0]) * 20_000_000,
        receive_time_ns=0,
        joint_position=env.get_dof_pos()[0],
        joint_velocity=env.get_dof_vel()[0],
        base_orientation_xyzw=imu_quat,
        base_angular_velocity=env.get_gyro()[0],
        base_linear_acceleration=np.zeros(3, dtype=np.float32),
    )
