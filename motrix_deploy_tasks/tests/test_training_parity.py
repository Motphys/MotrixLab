# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Golden parity between Go2 training and deployment observation/action semantics."""

import numpy as np
import pytest

import motrix_deploy_tasks
from motrix_deploy.contracts import RobotState
from motrix_deploy.runtime import PolicyContext
from motrix_deploy.task import create_task
from motrix_deploy_tasks.tasks.go2_walk import Go2WalkTaskSpec
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
    context = PolicyContext(
        step=0,
        elapsed_time_s=0.0,
        command=PlanarVelocityCommand(env._commands),
    )
    task = create_task(profile.task, profile.robot)
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
    next_context = PolicyContext(
        step=1,
        elapsed_time_s=profile.control.period_s,
        command=PlanarVelocityCommand(env._commands),
    )
    task.validate_command(next_context.command)
    np.testing.assert_allclose(
        task.build_observation(next_state, next_context),
        stepped.obs.policy[0],
        atol=1e-5,
        rtol=1e-5,
    )


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
