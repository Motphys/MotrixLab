# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic preparation bounds, target speed and gain timing, without hardware I/O."""

from dataclasses import replace

import numpy as np
import pytest
from test_joint_controller import _setup

from motrix_deploy.contracts import JointServoCommand
from motrix_deploy.errors import ControlFailure
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy_unitree.cli.joint_control.controller import Go2JointController
from motrix_deploy_unitree.config.go2 import Go2StartupLimits
from motrix_deploy_unitree.remote import UnitreeRemoteState


def setup_startup():
    original, state, remote = _setup()
    limits = Go2StartupLimits(
        max_speed_rad_s=0.5,
        timeout_s=10,
        gain_ramp_s=2,
        kp=original.settings.kp,
        kd=original.settings.kd,
    )
    controller = Go2JointController(
        original.robot,
        "FL_thigh_joint",
        float(original.robot.default_joint_position[1]),
        original.settings,
        gamepad=original._device,
        startup_limits=limits,
    )
    state.joint_position[2::3] = original.robot.position_lower[2::3] + np.float32(0.06)
    controller.reset(state)
    return controller, state, remote, limits


def test_explicit_startup_is_continuous_rate_limited_and_returns_to_normal_commands():
    controller, state, remote, limits = setup_startup()
    initial = state.joint_position.copy()
    wait = controller.step(state, ControlContext(0, 0.0, None, 0.02))
    assert isinstance(wait.command, JointServoCommand)
    remote[0] = UnitreeRemoteState(keys=1 << 2)
    first = controller.step(state, ControlContext(1, 0.02, None, 0.02)).command
    assert isinstance(first, JointServoCommand)
    np.testing.assert_array_equal(first.joint_position, initial)
    assert not np.any(first.kp) and not np.any(first.kd)
    previous = first
    for step in range(2, 250):
        was_transition = controller.phase == "transition"
        command = controller.step(state, ControlContext(step, step * 0.02, None, 0.02)).command
        assert np.all(command.joint_position >= controller.robot.position_lower)
        assert np.all(command.joint_position <= controller.robot.position_upper)
        elapsed = (step - 1) * 0.02
        assert np.max(np.abs(command.joint_position - initial)) <= limits.max_speed_rad_s * elapsed + 1e-6
        assert np.all(command.kp <= limits.kp * min(elapsed / limits.gain_ramp_s, 1.0) + 1e-6)
        assert np.all(command.kd <= limits.kd * min(elapsed / limits.gain_ramp_s, 1.0) + 1e-6)
        assert np.max(np.abs(command.joint_position - previous.joint_position)) <= limits.max_speed_rad_s * 0.02 + 1e-6
        assert np.all(command.kp >= previous.kp - 1e-6)
        state.joint_position[:] = command.joint_position
        if not was_transition:
            break
        previous = command
    else:
        pytest.fail("startup never reached normal hold")
    np.testing.assert_allclose(command.joint_position, controller.robot.default_joint_position)
    assert controller.phase == "wait_enable"


@pytest.mark.parametrize("index,excess", [(2, 0.06), (0, 0.001)])
def test_startup_rechecks_model_bounds_after_reset(index, excess):
    controller, state, remote, _ = setup_startup()
    state.joint_position[index] = min(state.joint_position[index], controller.robot.position_lower[index]) - excess
    # Reset must not authorize an out-of-range starting pose.
    controller.reset(state)
    initial = state.joint_position.copy()
    remote[0] = UnitreeRemoteState(keys=1 << 2)
    with pytest.raises(ControlFailure, match=controller.robot.joint_names[index]):
        controller.step(state, ControlContext(0, 0.0, None, 0.02))
    np.testing.assert_array_equal(state.joint_position, initial)
    assert controller.phase == "wait_start"


def test_b_during_bounded_startup_cancels_without_return_trajectory():
    controller, state, remote, _ = setup_startup()
    remote[0] = UnitreeRemoteState(keys=1 << 2)
    controller.step(state, ControlContext(0, 0.0, None, 0.02))
    remote[0] = UnitreeRemoteState(keys=1 << 9)
    result = controller.step(state, ControlContext(1, 0.02, None, 0.02))
    assert result.complete and not result.success and result.command is None


def test_startup_refuses_trajectory_that_cannot_finish_before_deadline():
    controller, state, remote, limits = setup_startup()
    controller = Go2JointController(
        controller.robot,
        "FL_thigh_joint",
        float(controller.robot.default_joint_position[1]),
        controller.settings,
        gamepad=controller._device,
        startup_limits=replace(limits, timeout_s=1.0),
    )
    controller.reset(state)
    remote[0] = UnitreeRemoteState(keys=1 << 2)
    with pytest.raises(ControlFailure) as error:
        controller.step(state, ControlContext(0, 0.0, None, 0.02))
    assert error.value.reason == "startup_duration_exceeds_timeout"


def test_repeated_ticks_cannot_manufacture_startup_target_travel():
    controller, state, remote, limits = setup_startup()
    remote[0] = UnitreeRemoteState(keys=1 << 2)
    first = controller.step(state, ControlContext(0, 0.0, None, 0.02)).command
    previous = first
    for step in range(1, 100):
        now = min(step * 0.001, 0.01)
        command = controller.step(state, ControlContext(step, now, None, 0.02)).command
        assert np.max(np.abs(command.joint_position - first.joint_position)) <= limits.max_speed_rad_s * now + 1e-6
        if step > 10:
            np.testing.assert_array_equal(command.joint_position, previous.joint_position)
            np.testing.assert_array_equal(command.kp, previous.kp)
        previous = command


def test_startup_deadline_does_not_turn_delayed_tick_into_large_step():
    controller, state, remote, _ = setup_startup()
    remote[0] = UnitreeRemoteState(keys=1 << 2)
    controller.step(state, ControlContext(0, 0.0, None, 0.02))
    command = controller.step(state, ControlContext(1, 2.0, None, 0.02)).command
    assert np.max(np.abs(command.joint_position - state.joint_position)) <= 0.01 + 1e-6
    with pytest.raises(ControlFailure, match="timed out"):
        controller.step(state, ControlContext(2, 10.0, None, 0.02))


@pytest.mark.parametrize("field,value", [("max_speed_rad_s", 0.0), ("gain_ramp_s", float("nan")), ("kp", -1.0)])
def test_startup_limits_are_validated_before_control(field, value):
    controller, _, _, limits = setup_startup()
    with pytest.raises(ValueError, match=field):
        Go2JointController(
            controller.robot,
            "FL_thigh_joint",
            float(controller.robot.default_joint_position[1]),
            controller.settings,
            gamepad=controller._device,
            startup_limits=replace(limits, **{field: value}),
        )


def test_diagnostic_gains_cannot_exceed_startup_ceiling():
    controller, _, _, limits = setup_startup()
    with pytest.raises(ValueError, match="kp.*ceiling"):
        Go2JointController(
            controller.robot,
            "FL_thigh_joint",
            float(controller.robot.default_joint_position[1]),
            replace(controller.settings, kp=limits.kp + 1),
            gamepad=controller._device,
            startup_limits=limits,
        )
