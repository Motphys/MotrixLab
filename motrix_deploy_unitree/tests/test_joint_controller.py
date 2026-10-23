# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic operator gates and single-joint trajectories, without SDK I/O."""

import numpy as np
import pytest

from motrix_deploy.contracts import JointServoCommand, RobotState
from motrix_deploy.errors import ControlFailure
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy_unitree.cli.joint_control.controller import Go2JointController
from motrix_deploy_unitree.config.go2 import Go2JointSettings, Go2StartupLimits, UnitreeGo2HardwareCfg
from motrix_deploy_unitree.remote import UnitreeRemoteGamePadDevice, UnitreeRemoteState


def _setup():
    robot = UnitreeGo2HardwareCfg().to_robot_spec()
    state = RobotState(
        sample_time_ns=0,
        receive_time_ns=0,
        joint_position=robot.default_joint_position.copy(),
        joint_velocity=np.zeros(12, dtype=np.float32),
        base_orientation_xyzw=np.array([0, 0, 0, 1], dtype=np.float32),
        base_angular_velocity=np.zeros(3, dtype=np.float32),
        base_linear_acceleration=np.zeros(3, dtype=np.float32),
    )
    remote = [UnitreeRemoteState()]
    device = UnitreeRemoteGamePadDevice(lambda: remote[0])
    settings = Go2JointSettings(
        kp=12,
        kd=0.5,
        transition_duration_s=0.04,
        move_duration_s=0.04,
        hold_duration_s=0.02,
        return_duration_s=0.04,
    )
    task = Go2JointController(
        robot,
        "FL_thigh_joint",
        float(robot.default_joint_position[1]) + 0.1,
        settings,
        gamepad=device,
        startup_limits=Go2StartupLimits(
            max_speed_rad_s=1.0, timeout_s=10.0, gain_ramp_s=0.02, kp=settings.kp, kd=settings.kd
        ),
    )
    task.reset(state)
    return task, state, remote


def _tick(task, state, step):
    decision = task.step(state, ControlContext(step, step * 0.02, None, 0.02))
    if decision.command is not None:
        state.joint_position[:] = decision.command.joint_position
    return decision


def test_operator_gates_single_joint_move_and_return():
    task, state, remote = _setup()
    reference = state.joint_position.copy()
    assert np.all(_tick(task, state, 0).command.kp == 0)
    remote[0] = UnitreeRemoteState(keys=(1 << 2) | (1 << 8))  # Start and premature A
    first = _tick(task, state, 1).command
    assert isinstance(first, JointServoCommand)
    np.testing.assert_array_equal(first.joint_position, reference)
    assert not np.any(first.kp) and not np.any(first.kd)
    for step in range(2, 7):
        _tick(task, state, step)
    assert task.phase == "wait_enable"  # held A cannot bypass the second gate
    remote[0] = UnitreeRemoteState()
    _tick(task, state, 7)
    remote[0] = UnitreeRemoteState(keys=1 << 8)
    half = _tick(task, state, 8)
    expected = reference.copy()
    expected[1] += np.float32(0.05)
    np.testing.assert_allclose(half.command.joint_position, expected)
    _tick(task, state, 9)
    for step in range(10, 16):
        decision = _tick(task, state, step)
        if decision.complete:
            break
    assert decision.complete and decision.success
    np.testing.assert_allclose(state.joint_position, reference)


@pytest.mark.parametrize("phase", ["wait_start", "transition", "wait_enable", "move", "hold"])
def test_b_stop_never_starts_an_unrequested_motion(phase):
    task, state, remote = _setup()
    task.phase = phase
    state.joint_position[1] += np.float32(0.02)
    remote[0] = UnitreeRemoteState(keys=1 << 9)
    decision = _tick(task, state, 0)
    assert task.phase in {"return", "complete"}
    assert not decision.policy_tick
    if phase in {"wait_start", "transition"}:
        assert decision.complete and not decision.success and decision.command is None


@pytest.mark.parametrize("moving", [False, True])
def test_a_rechecks_measured_readiness_instead_of_elapsed_ramp(moving):
    task, state, remote = _setup()
    task.phase = "wait_enable"
    if moving:
        state.joint_velocity[:] = 2.0
    else:
        state.joint_position[1] -= np.float32(0.5)
    remote[0] = UnitreeRemoteState(keys=1 << 8)
    with pytest.raises(ControlFailure) as error:
        _tick(task, state, 3)
    assert error.value.reason == "preparation_not_ready"
    assert task.phase == "wait_enable"


def test_joint_readiness_timeout_stops_without_starting_motion():
    task, state, _ = _setup()
    task.phase = "wait_enable"
    state.joint_position[1] -= np.float32(0.5)
    with pytest.raises(ControlFailure) as error:
        _tick(task, state, 400)
    assert error.value.reason == "preparation_timeout"


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_start_rejects_measured_pose_outside_model_bounds(side):
    task, state, remote = _setup()
    boundary = getattr(task.robot, f"position_{side}")[1]
    direction = -1 if side == "lower" else 1
    state.joint_position[1] = boundary + np.float32(direction * 0.1)
    initial = state.joint_position.copy()
    task.reset(state)
    remote[0] = UnitreeRemoteState(keys=1 << 2)
    with pytest.raises(ControlFailure) as error:
        _tick(task, state, 0)
    assert error.value.reason == "initial_state_out_of_range"
    assert task.phase == "wait_start"
    np.testing.assert_array_equal(state.joint_position, initial)


def test_task_rejects_missing_operator_and_invalid_target():
    task, _, _ = _setup()
    with pytest.raises(ValueError, match="gamepad"):
        Go2JointController(
            task.robot, "FL_thigh_joint", 0.9, task.settings, gamepad=None, startup_limits=task._startup_limits
        )
    with pytest.raises(ValueError, match="hardware position range"):
        Go2JointController(
            task.robot,
            "FL_thigh_joint",
            100,
            task.settings,
            gamepad=UnitreeRemoteGamePadDevice(UnitreeRemoteState),
            startup_limits=task._startup_limits,
        )
