# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Standard WBT deploy lifecycle and self-contained motion contracts."""

import numpy as np
import pytest
from pydantic import ValidationError as PydanticValidationError

from motrix_deploy.contracts import RobotSpec, RobotState
from motrix_deploy.errors import ValidationError
from motrix_deploy.policy import NoOpPolicy
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy_tasks.tasks.g1_wbt import G1WbtDeployTask, G1WbtMotion, G1WbtPolicyProcessor, G1WbtTaskSpec


def _motion(reference_quaternions=None):
    return G1WbtMotion(
        np.asarray([[0.0] * 29, [0.1] * 29], np.float32),
        np.asarray([[0.0] * 29, [0.2] * 29], np.float32),
        np.asarray(reference_quaternions or [[0, 0, 0, 1]] * 2, np.float32),
    )


def _task(
    reference_quaternions=None,
    *,
    controller=False,
    steps=None,
    automatic=False,
    preparation=None,
):
    motion = _motion(reference_quaternions)
    spec = G1WbtTaskSpec(
        action_scale=[0.5] * 29,
        kp=[10] * 29,
        kd=[1] * 29,
        motion_frames=len(motion),
        termination_ref_orientation_threshold=0.4,
        termination_joint_position_threshold=0.5,
        termination_joint_velocity_threshold=10,
        **(preparation or {}),
    )
    robot = RobotSpec(
        "pelvis",
        tuple(["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", *[f"joint_{i}" for i in range(26)]]),
        np.zeros(29, np.float32),
        np.full(29, -1, np.float32),
        np.ones(29, np.float32),
        np.full(29, 10, np.float32),
    )
    return (
        G1WbtDeployTask(spec, robot, NoOpPolicy(29), steps=steps, automatic_start=automatic, motion=motion)
        if controller
        else G1WbtPolicyProcessor(spec, robot, motion)
    )


def _state():
    return RobotState(
        0,
        0,
        np.zeros(29, np.float32),
        np.zeros(29, np.float32),
        np.array([0, 0, 0, 1], np.float32),
        np.array([1, 2, 3], np.float32),
        np.zeros(3, np.float32),
    )


# Ramp- and settle-free preparation for lifecycle tests that exercise handover timing.
_instant = {"preparation_ramp_duration_s": 0.0, "preparation_settle_duration_s": 0.0}


def test_standard_task_raw_history_servo_limits_frame_and_reset():
    task = _task()
    state = _state()
    context = ControlContext(1, 0.02, None, dt_s=0.02)
    task.validate_command(None)
    raw = np.full(29, 9, np.float32)
    command = task.process_action(raw)
    raw.fill(0)
    np.testing.assert_array_equal(command.joint_position, task.robot.position_upper)
    np.testing.assert_array_equal(command.kp, task.spec.kp)
    observation = task.build_observation(state, context)
    assert observation.shape == (154,)
    np.testing.assert_allclose(observation[:29], task.motion.joint_pos[1])
    np.testing.assert_array_equal(observation[64:67], state.base_angular_velocity)
    np.testing.assert_array_equal(observation[125:], 9)
    assert task.check_termination(state) is None
    task.reset(state, context)
    np.testing.assert_array_equal(task.build_observation(state, context)[125:], 0)
    assert G1WbtDeployTask.default_duration_s(task.spec) == 2 * task.spec.period_s
    with pytest.raises(ValidationError, match="context.step"):
        task.build_observation(state, ControlContext(2, 0.04, None, dt_s=0.02))


def test_orientation_termination_uses_latest_observed_reference_frame():
    task = _task(reference_quaternions=[[0, 0, 0, 1], [1, 0, 0, 0]])
    state = _state()
    task.build_observation(state, ControlContext(0, 0, None, dt_s=0.02))
    assert task.check_termination(state) is None
    task.build_observation(state, ControlContext(1, 0.02, None, dt_s=0.02))
    assert task.check_termination(state) == "bad_ref_ori"


def test_measured_base_and_waist_orientation_and_joint_safety_termination():
    task = _task()
    state = _state()
    state.base_orientation_xyzw[:] = [1, 0, 0, 0]
    assert task.check_termination(state) == "bad_ref_ori"
    # Torso FK sees waist pitch, not only the pelvis quaternion.
    state.base_orientation_xyzw[:] = [0, 0, 0, 1]
    state.joint_position[2] = 1
    assert task.check_termination(state) == "bad_ref_ori"
    state.joint_position.fill(0)
    state.joint_position[0] = task.robot.position_upper[0] + 1
    assert task.check_termination(state) == "bad_dof_pos"
    state.joint_position.fill(0)
    state.joint_velocity[0] = task.spec.termination_joint_velocity_threshold + 1
    assert task.check_termination(state) == "bad_dof_vel"


def test_spec_roundtrip_and_motion_shape_validation():
    task = _task()
    restored = G1WbtTaskSpec.model_validate_json(task.spec.model_dump_json())
    assert restored == task.spec
    with pytest.raises(ValidationError, match="motion.joint_vel"):
        G1WbtMotion(task.motion.joint_pos, task.motion.joint_vel[:1], task.motion.reference_quaternion_xyzw)
    state = _state()
    # Both actor input and safety checks use the same hardware-observable state.
    assert task.build_observation(state, ControlContext(0, 0, None, dt_s=0.02)).shape == (154,)
    assert task.check_termination(state) is None


def test_motion_validates_frame_counts_and_unit_quaternions():
    motion = _motion()
    with pytest.raises(ValidationError, match="motion.joint_vel"):
        G1WbtMotion(motion.joint_pos, motion.joint_vel[:1], motion.reference_quaternion_xyzw)
    with pytest.raises(ValidationError, match="motion.reference_quaternion_xyzw"):
        G1WbtMotion(motion.joint_pos, motion.joint_vel, motion.reference_quaternion_xyzw[:1])
    tilted = motion.reference_quaternion_xyzw.copy()
    tilted[0] = [0, 0, 0, 2]
    with pytest.raises(ValidationError, match="unit xyzw quaternions"):
        G1WbtMotion(motion.joint_pos, motion.joint_vel, tilted)


def test_preparation_commands_joint_posture_not_root_state_or_motion_velocity():
    task = _task(controller=True, preparation={"preparation_ramp_duration_s": 0.0})
    state = _state()
    ready = task.ready
    ready.reset(state)
    transition = ready.step(state, ControlContext(0, 0, None, dt_s=0.02))
    command = transition.command
    np.testing.assert_array_equal(command.joint_position, task.motion.joint_pos[0])
    np.testing.assert_array_equal(command.kp, task.spec.kp)
    np.testing.assert_array_equal(command.kd, task.spec.kd)
    np.testing.assert_array_equal(command.joint_velocity, 0)
    np.testing.assert_array_equal(command.feedforward_torque, 0)
    assert command.joint_position.dtype == np.float32


def test_preparation_ramps_from_first_step_measurement_then_requires_continuous_readiness():
    task = _task(
        controller=True,
        preparation={
            "preparation_ramp_duration_s": 0.4,
            "preparation_settle_duration_s": 0.2,
            "preparation_timeout_s": 2.0,
        },
    )
    state = _state()
    state.joint_position.fill(0.8)
    controller = task.ready
    state.joint_position.fill(0.6)
    controller.reset(state)
    first = controller.step(state, ControlContext(0, 0, None, dt_s=0.1))
    np.testing.assert_allclose(first.command.joint_position, 0.45)
    np.testing.assert_array_equal(first.command.kp, task.spec.kp)
    np.testing.assert_array_equal(first.command.kd, task.spec.kd)
    state.joint_position.fill(0)
    assert not controller.step(
        state, ControlContext(0, 0.3, None, dt_s=0.02)
    ).complete  # Ready joints do not bypass the ramp.
    assert not controller.step(state, ControlContext(0, 0.4, None, dt_s=0.02)).complete
    state.joint_velocity.fill(2)
    assert not controller.step(state, ControlContext(0, 0.5, None, dt_s=0.02)).complete
    state.joint_velocity.fill(0)
    assert not controller.step(state, ControlContext(0, 0.6, None, dt_s=0.02)).complete
    assert not controller.step(state, ControlContext(0, 0.7, None, dt_s=0.02)).complete
    ready = controller.step(state, ControlContext(0, 0.8, None, dt_s=0.02))
    assert ready.complete
    assert ready.command is None
    assert ready.error is None


def test_preparation_targets_standing_pose_and_clips_to_physical_joint_limits():
    base = _task(controller=True, preparation=_instant)
    motion = G1WbtMotion(  # Motion frame 0 must not drive preparation.
        np.full((2, 29), 2.0, np.float32),
        np.zeros((2, 29), np.float32),
        np.tile([0, 0, 0, 1], (2, 1)).astype(np.float32),
    )
    task = G1WbtDeployTask(base.spec, base.robot, NoOpPolicy(29), automatic_start=False, motion=motion)
    state = _state()
    controller = task.ready
    controller.reset(state)
    standing = controller.step(state, ControlContext(0, 0, None, dt_s=0.02))
    assert standing.complete  # Standing readiness ignores the motion frame.
    assert standing.command is None  # Completion carries no command; the task holds.
    hold = controller.hold_command()
    np.testing.assert_array_equal(hold.joint_position, task.robot.default_joint_position)
    np.testing.assert_array_equal(hold.kp, task.spec.kp)
    np.testing.assert_array_equal(hold.kd, task.spec.kd)
    np.testing.assert_array_equal(hold.joint_velocity, 0)
    np.testing.assert_array_equal(hold.feedforward_torque, 0)
    clipped = G1WbtDeployTask(
        task.spec.model_copy(update={"preparation_joint_position": [2] * 29}),
        task.robot,
        NoOpPolicy(29),
        motion=motion,
    )
    np.testing.assert_array_equal(clipped.ready.hold_command().joint_position, task.robot.position_upper)


def test_preparation_rejects_unsafe_tilt_and_times_out_without_readiness():
    task = _task(controller=True)
    state = _state()
    controller = task.ready
    controller.reset(state)
    state.base_orientation_xyzw[:] = [1, 0, 0, 0]
    unsafe = controller.step(state, ControlContext(0, 0, None, dt_s=0.02))
    assert "unsafe base tilt" in unsafe.error
    assert unsafe.command is None
    state.base_orientation_xyzw[:] = [0, 0, 0, 1]
    state.joint_velocity.fill(2)
    timeout = controller.step(state, ControlContext(0, 5, None, dt_s=0.02))
    assert "timed out" in timeout.error
    assert not timeout.complete
    assert timeout.command is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("preparation_timeout_s", 0),
        ("preparation_ramp_duration_s", -1),
        ("preparation_settle_duration_s", -0.5),
        ("preparation_timeout_s", float("nan")),
    ],
)
def test_spec_validates_preparation_timing(field, value):
    data = _task().spec.model_dump()
    data[field] = value
    with pytest.raises(PydanticValidationError, match=field):
        G1WbtTaskSpec.model_validate(data)


def test_normal_motion_stopping_is_finite_measured_damping_without_position_hold():
    task = _task(controller=True)
    state = _state()
    controller = task.damping
    state.joint_position.fill(0.4)
    state.joint_velocity.fill(0.7)
    state.base_orientation_xyzw[:] = [1, 0, 0, 0]  # Damping does not demand a ready posture.
    command = controller.step(state, ControlContext(0, 0, None, dt_s=0.02)).command
    np.testing.assert_array_equal(command.joint_position, state.joint_position)
    np.testing.assert_array_equal(command.kp, 0)
    np.testing.assert_array_equal(command.kd, task.spec.kd)
    np.testing.assert_array_equal(command.joint_velocity, 0)
    np.testing.assert_array_equal(command.feedforward_torque, 0)
    state.joint_position.fill(0.2)
    np.testing.assert_array_equal(
        controller.step(state, ControlContext(0, 0.2, None, dt_s=0.02)).command.joint_position, state.joint_position
    )
    state.joint_position[0] = task.robot.position_upper[0] + 0.1
    state.joint_position[1] = task.robot.position_lower[1] - 0.1
    clipped = controller.step(state, ControlContext(0, 0.25, None, dt_s=0.02)).command
    assert clipped.joint_position[0] == task.robot.position_upper[0]
    assert clipped.joint_position[1] == task.robot.position_lower[1]
    torque = (
        clipped.kp * (clipped.joint_position - state.joint_position)
        + clipped.kd * (clipped.joint_velocity - state.joint_velocity)
        + clipped.feedforward_torque
    )
    np.testing.assert_array_equal(torque, -clipped.kd * state.joint_velocity)
    complete = controller.step(state, ControlContext(0, 0.3, None, dt_s=0.02))
    assert complete.complete
    assert complete.command is None
    assert complete.error is None


def test_composite_handover_hold_playback_clocks_and_finite_damping():
    task = _task(
        controller=True, preparation={"preparation_ramp_duration_s": 0.0, "preparation_settle_duration_s": 0.02}
    )
    controller = task
    state = _state()
    seen = []
    observations = []
    observe = controller.policy_io.build_observation

    def observation(measurement, context):
        record = observe(measurement, context)
        seen.append((measurement, context.step, context.elapsed_time_s))
        observations.append(record)
        return record

    controller.policy_io.build_observation = observation
    controller.reset(state)
    assert controller.step(state, ControlContext(0, 3.0, None, dt_s=0.02)).command is not None
    assert controller.policy_controller.completed_steps == 0
    controller.request_policy_start()
    # Readiness completion and the first held model decision use the same sample and tick.
    assert controller.step(state, ControlContext(1, 3.02, None, dt_s=0.02)).command is not None
    assert controller.phase == "policy_hold"
    assert seen == [(state, 0, 0.0)]
    np.testing.assert_allclose(observations[0][:29], 0.0)  # Hold freezes reference frame 0.
    assert controller.transitions[0].complete
    assert controller.transitions[0].name == "prepare"
    controller.request_motion_start()
    assert controller.step(state, ControlContext(2, 3.04, None, dt_s=0.02)).command is not None
    assert controller.phase == "playback"
    # Inference history is preserved: the model clock continues from the hold origin.
    assert seen[-1][:2] == (state, 1)
    assert seen[-1][2] == pytest.approx(0.02)
    assert controller.playback_steps == 1
    assert controller.policy_controller.completed_steps == 2
    assert controller.step(state, ControlContext(3, 3.06, None, dt_s=0.02)).command is not None
    np.testing.assert_allclose(observations[2][:29], 0.1)  # Only playback advances the frame.
    assert controller.playback_steps == 2
    assert controller.policy_controller.completed_steps == 3
    # Budget completion starts damping immediately, not an empty interval.
    damping = controller.step(state, ControlContext(4, 3.08, None, dt_s=0.02))
    np.testing.assert_array_equal(damping.command.kp, 0)
    assert controller.policy_controller.completed_steps == 3
    assert controller.playback_steps == 2
    assert controller.transitions[-1].name == "damping"
    assert controller.transitions[-1].time_s == 0.02
    for step in range(5, 19):
        assert not controller.step(state, ControlContext(step, 3.0 + step * 0.02, None, dt_s=0.02)).complete
    assert controller.step(state, ControlContext(19, 3.38, None, dt_s=0.02)).complete
    assert controller.transitions[-1].complete


def test_composite_stop_request_during_ready_hold_enters_damping():
    controller = _task(controller=True, steps=2, preparation=_instant)
    state = _state()
    controller.reset(state)
    assert controller.step(state, ControlContext(0, 0.0, None, dt_s=0.02)).command is not None
    assert controller.preparation_ready
    assert controller.phase == "prepare"
    controller.request_stop()
    decision = controller.step(state, ControlContext(1, 0.02, None, dt_s=0.02))
    np.testing.assert_array_equal(decision.command.kp, 0)
    assert controller.phase == "damping"
    assert controller.policy_controller.completed_steps == 0  # Stopped before takeover.
    assert not controller.transitions[0].complete
    assert controller.transitions[-1].name == "damping"


def test_early_latched_requests_wait_for_measured_readiness_and_hold_once():
    controller = _task(
        controller=True,
        preparation={"preparation_ramp_duration_s": 0.1, "preparation_settle_duration_s": 0.1},
    )
    state = _state()
    controller.reset(state)
    controller.request_policy_start()
    controller.request_motion_start()
    for step in range(0, 10):
        decision = controller.step(state, ControlContext(step, step * 0.02, None, dt_s=0.02))
        assert controller.phase == "prepare"  # Latched intent cannot bypass ramp or settle.
        assert decision.command is not None
        assert controller.policy_controller.completed_steps == 0
        assert not controller.preparation_ready  # Settled readiness is not yet complete.
    # Settled readiness hands over on the same tick, but playback waits one hold interval.
    controller.step(state, ControlContext(10, 0.2, None, dt_s=0.02))
    assert controller.phase == "policy_hold"
    assert controller.policy_controller.completed_steps == 1
    assert controller.playback_steps == 0
    controller.step(state, ControlContext(11, 0.22, None, dt_s=0.02))
    assert controller.phase == "playback"
    assert controller.playback_steps == 1


def test_task_preparation_times_out_before_first_readiness():
    controller = _task(controller=True, steps=2, preparation=_instant)
    state = _state()
    state.joint_velocity.fill(2)  # Continuous readiness is never reached.
    controller.reset(state)
    controller.request_policy_start()  # Latched intent does not bypass the gate.
    assert controller.step(state, ControlContext(0, 0.0, None, dt_s=0.02)).command is not None
    failed = controller.step(state, ControlContext(250, 5.0, None, dt_s=0.02))  # Stage-local 5 s timeout.
    assert "timed out" in failed.error
    assert failed.command is None
    assert controller.phase == "prepare"
    assert controller.policy_controller.completed_steps == 0


def test_composite_checks_last_model_interval_before_damping():
    from motrix_deploy.errors import ControlFailure

    task = _task(controller=True, preparation=_instant)
    controller = G1WbtDeployTask(
        task.spec, task.robot, NoOpPolicy(29), steps=1, automatic_start=False, motion=task.motion
    )
    state = _state()
    controller.reset(state)
    controller.request_policy_start()
    controller.request_motion_start()
    assert controller.step(state, ControlContext(0, 0.0, None, dt_s=0.02)).command is not None
    assert controller.phase == "policy_hold"  # At least one hold interval first.
    assert controller.step(state, ControlContext(1, 0.02, None, dt_s=0.02)).command is not None
    assert controller.phase == "playback"
    assert controller.playback_steps == 1
    state.base_orientation_xyzw[:] = [1, 0, 0, 0]
    # Termination is checked on the measured state before budget completion can finish.
    with pytest.raises(ControlFailure, match="bad_ref_ori"):
        controller.step(state, ControlContext(2, 0.04, None, dt_s=0.02))
    assert len(controller.transitions) == 3  # Faults do not enter normal damping.
    assert controller.transitions[-1].name == "playback"


class _FakeKeyboard:
    """Deliver per-frame press edges exactly as an event-framed device reports them."""

    def __init__(self):
        self._pending = {}
        self._active = {}

    def press(self, *keys) -> None:
        self._pending = dict.fromkeys(keys, True)

    def poll(self) -> None:
        self._active, self._pending = self._pending, {}

    def is_key_down(self, key: str) -> bool:
        return self._active.get(key, False)


def test_attached_keyboard_press_edges_drive_phase_requests():
    task = _task(controller=True, steps=2, preparation=_instant)
    keyboard = _FakeKeyboard()
    task.attach_keyboard(keyboard)
    state = _state()
    task.reset(state)
    assert task.step(state, ControlContext(0, 0.0, None, dt_s=0.02)).command is not None
    assert task.preparation_ready and task.phase == "prepare"  # Ready, waiting for input.
    keyboard.press("p")
    decision = task.step(state, ControlContext(1, 0.02, None, dt_s=0.02))
    assert decision.command is not None and task.phase == "policy_hold"
    # Default zero hold target: the next held tick enters playback by itself.
    task.step(state, ControlContext(2, 0.04, None, dt_s=0.02))
    assert task.phase == "playback"


def test_attached_keyboard_stop_request_enters_damping():
    task = _task(controller=True, preparation=_instant)
    keyboard = _FakeKeyboard()
    task.attach_keyboard(keyboard)
    state = _state()
    task.reset(state)
    task.step(state, ControlContext(0, 0.0, None, dt_s=0.02))
    task.step(state, ControlContext(1, 0.02, None, dt_s=0.02))
    keyboard.press("o")
    decision = task.step(state, ControlContext(2, 0.04, None, dt_s=0.02))
    np.testing.assert_array_equal(decision.command.kp, 0)
    assert task.phase == "damping"
    assert task.policy_controller.completed_steps == 0


def test_automatic_start_gates_on_readiness_and_hold_ticks():
    task = _task(controller=True, steps=2, preparation=_instant)
    task.enable_automatic_start(hold_ticks=2)
    state = _state()
    task.reset(state)
    task.step(state, ControlContext(0, 0.0, None, dt_s=0.02))
    assert task.phase == "prepare" and task.preparation_ready
    task.step(state, ControlContext(1, 0.02, None, dt_s=0.02))
    assert task.phase == "policy_hold"  # Takeover happens on the tick after readiness.
    task.step(state, ControlContext(2, 0.04, None, dt_s=0.02))
    assert task.phase == "policy_hold"  # First held tick below the target.
    task.step(state, ControlContext(3, 0.06, None, dt_s=0.02))
    assert task.phase == "playback"


def test_operator_input_rejects_invalid_configuration():
    task = _task(controller=True)
    with pytest.raises(ValidationError, match="operator.keyboard"):
        task.attach_keyboard(None)


def test_default_automatic_start_progresses_after_measured_readiness():
    task = _task(controller=True, steps=2, automatic=True, preparation=_instant)
    state = _state()
    task.reset(state)
    assert task.step(state, ControlContext(0, 0.0, None, dt_s=0.02)).command is not None
    assert task.preparation_ready and task.phase == "prepare"
    # Takeover happens on the tick after readiness without any request call.
    task.step(state, ControlContext(1, 0.02, None, dt_s=0.02))
    assert task.phase == "policy_hold"
    # Default zero hold target: the next held tick enters playback by itself.
    task.step(state, ControlContext(2, 0.04, None, dt_s=0.02))
    assert task.phase == "playback"
