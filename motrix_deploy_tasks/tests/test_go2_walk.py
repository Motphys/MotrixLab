# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Direct Go2 walking deployment-task tests."""

from dataclasses import replace

import numpy as np
import pytest
from motrix_robots.unitree import UnitreeGo2Robot
from pydantic import ValidationError as PydanticValidationError

from motrix_deploy.artifact.schema import TaskSpec
from motrix_deploy.contracts import RobotSpec, RobotState
from motrix_deploy.errors import ControlFailure, ValidationError
from motrix_deploy.policy import NoOpPolicy
from motrix_deploy.runtime import ControlContext
from motrix_deploy.runtime.control import PolicyController
from motrix_deploy_tasks.tasks.go2_walk import (
    Go2WalkDeployTask,
    Go2WalkPolicyProcessor,
    Go2WalkPreparationSettings,
    Go2WalkTaskSpec,
)
from motrix_env_core.input import (
    BoundedGamePadPlanarVelocityBinding,
    ConstantPlanarVelocityBinding,
    GamePadDevice,
    KeyboardDevice,
    KeyboardPlanarVelocityBinding,
    PlanarVelocityCommand,
)


class FakeKeyboardDevice(KeyboardDevice):
    def poll(self) -> None: ...

    def is_key_down(self, key: str) -> bool:
        return False

    def is_key_up(self, key: str) -> bool:
        return False

    def is_pressing(self, key: str) -> bool:
        return False


class FakeGamePadDevice(GamePadDevice):
    def __init__(self) -> None:
        self.pressing: set[str] = set()
        self._previous: set[str] = set()
        self._polled: set[str] = set()
        self.poll_count = 0

    def poll(self) -> None:
        self.poll_count += 1
        self._previous = self._polled
        self._polled = self.pressing.copy()

    def axis_value(self, axis: str) -> float:
        return 0.0

    def is_button_down(self, button: str) -> bool:
        return button in self._polled and button not in self._previous

    def is_button_up(self, button: str) -> bool:
        return False

    def is_button_pressing(self, button: str) -> bool:
        return button in self.pressing


def _replace_spec(spec: Go2WalkTaskSpec, **overrides: object) -> Go2WalkTaskSpec:
    return type(spec).model_validate({**spec.model_dump(), **overrides})


def _context(step: int, elapsed_time_s: float, velocity: list[float] | np.ndarray) -> ControlContext:
    return ControlContext(
        step=step,
        elapsed_time_s=elapsed_time_s,
        command=PlanarVelocityCommand(np.asarray(velocity, dtype=np.float32)[None, :]),
        dt_s=0.02,
    )


def _robot() -> RobotSpec:
    return RobotSpec(
        base_link_name="base",
        joint_names=tuple(UnitreeGo2Robot().key_pose.joint_names),
        default_joint_position=np.zeros(12, dtype=np.float32),
        position_lower=np.full(12, -3.0, dtype=np.float32),
        position_upper=np.full(12, 2.0, dtype=np.float32),
        torque_limit=np.full(12, 24.0, dtype=np.float32),
    )


def _state(robot: RobotSpec) -> RobotState:
    return RobotState(
        sample_time_ns=0,
        receive_time_ns=0,
        joint_position=robot.default_joint_position + np.float32(0.1),
        joint_velocity=np.full(12, 0.2, dtype=np.float32),
        base_orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_angular_velocity=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        base_linear_acceleration=np.zeros(3, dtype=np.float32),
    )


def _task_spec() -> Go2WalkTaskSpec:
    return Go2WalkTaskSpec(
        action_scale=[0.25] * 12,
        command_lower=[-1.0, -1.0, -1.0],
        command_upper=[1.0, 1.0, 1.0],
        command_scale=[0.5, 0.5, 0.5],
        feet_phase_offsets=[0.0, 0.5, 0.5, 0.0],
        gait_frequency_hz=2.0,
        standing_threshold=0.05,
        termination_min_up_z=0.25,
        kp=[35.0] * 12,
        kd=[0.5] * 12,
        action_lower=[-1.0] * 12,
        action_upper=[1.0] * 12,
    )


def test_go2_preparation_is_always_part_of_the_task_lifecycle() -> None:
    robot = _robot()
    task = Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count), steps=1)

    task.prepare(simulation=True)
    task.reset(_state(robot))
    assert task.phase == "wait_stand"

    with pytest.raises(ValidationError, match="operator.gamepad"):
        task.prepare(simulation=False, spawn_pose="lie_down")
    assert task.phase == "wait_stand"

    task.prepare(simulation=True, spawn_pose="default")
    assert task.phase == "wait_stand"

    task.prepare(simulation=True, spawn_pose="lie_down")
    task.reset(_state(robot))
    assert task.phase == "wait_stand"


def test_simulated_standing_preparation_gates_policy_and_damps_after_budget() -> None:
    robot = _robot()
    state = _state(robot)
    state.joint_position[:] = -0.5
    state.joint_velocity[:] = 0.0
    task = Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count), steps=1)
    settings = Go2WalkPreparationSettings(
        ramp_duration_s=0.04, settle_duration_s=0.02, timeout_s=0.2, damping_duration_s=0.04
    )
    task.enable_simulation_preparation(settings)
    task.reset(state)
    assert task.phase == "wait_stand"
    first = task.step(state, _context(0, 0.0, [0.0, 0.0, 0.0]))
    assert task.phase == "prepare"
    np.testing.assert_allclose(first.command.joint_position, -0.25)
    assert task.policy_controller.completed_steps == 0
    task.step(state, _context(1, 0.02, [0.0, 0.0, 0.0]))
    state.joint_position[:] = 0.0
    ready = task.step(state, _context(2, 0.04, [0.0, 0.0, 0.0]))
    assert not ready.policy_tick and task.phase == "prepare"
    standing = task.step(state, _context(3, 0.06, [0.0, 0.0, 0.0]))
    assert not standing.policy_tick and task.phase == "wait_policy"
    takeover = task.step(state, _context(4, 0.08, [0.0, 0.0, 0.0]))
    assert takeover.policy_tick and task.phase == "policy"
    assert task.policy_controller.completed_steps == 1
    stop = task.step(state, _context(5, 0.10, [0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(stop.command.kp, 0.0)
    np.testing.assert_array_equal(stop.command.kd, settings.damping_kd)
    assert not stop.policy_tick and task.phase == "damping"
    task.step(state, _context(6, 0.12, [0.0, 0.0, 0.0]))
    assert task.step(state, _context(7, 0.14, [0.0, 0.0, 0.0])).complete
    assert task.phase == "complete"


def test_simulated_preparation_rejects_tilt_and_timeout() -> None:
    robot = _robot()
    state = _state(robot)
    task = Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count), steps=1)
    task.enable_simulation_preparation(
        Go2WalkPreparationSettings(ramp_duration_s=0.02, settle_duration_s=0.02, timeout_s=0.04)
    )
    task.reset(state)
    state.base_orientation_xyzw[:] = [1.0, 0.0, 0.0, 0.0]
    assert "tilt" in task.step(state, _context(0, 0.0, [0.0, 0.0, 0.0])).error
    state.base_orientation_xyzw[:] = [0.0, 0.0, 0.0, 1.0]
    state.joint_position[:] = 0.5
    assert "timed out" in task.step(state, _context(2, 0.04, [0.0, 0.0, 0.0])).error
    assert task.policy_controller.completed_steps == 0


@pytest.mark.parametrize("failure", ["position", "velocity", "settle"])
def test_preparation_timeout_describes_unmet_conditions_and_joint_values(failure) -> None:
    robot = _robot()
    state = _state(robot)
    settings = Go2WalkPreparationSettings(ramp_duration_s=0.02, settle_duration_s=0.02, timeout_s=0.04)
    task = Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count), steps=1)
    task.enable_simulation_preparation(settings)
    task.reset(state)
    if failure == "position":
        state.joint_position[2] = robot.default_joint_position[2] + 0.3
    elif failure == "velocity":
        state.joint_velocity[2] = -1.5
    task.step(state, _context(0, 0.0, [0.0] * 3))
    error = task.step(state, _context(2, 0.04, [0.0] * 3)).error
    assert "Go2 standing preparation timed out" in error
    assert "elapsed=0.040s" in error
    assert "stable=0.000/0.020s" in error
    assert "settle_duration" in error
    if failure == "settle":
        assert "unmet=[settle_duration]" in error
        assert "violating_joints=[]" in error
    else:
        assert robot.joint_names[2] in error
        assert "measured=" in error and "target=" in error
        assert ("joint_position_error" if failure == "position" else "joint_velocity") in error
        assert ("abs_error=0.300000" if failure == "position" else "velocity=-1.500000") in error
    assert task.policy_controller.completed_steps == 0


class _FakeKeyboard:
    def __init__(self) -> None:
        self.down: set[str] = set()
        self.poll_count = 0

    def poll(self) -> None:
        self.poll_count += 1

    def is_key_down(self, key: str) -> bool:
        return key in self.down


def test_viewer_requires_separate_stand_and_policy_button_edges() -> None:
    robot = _robot()
    state = _state(robot)
    state.joint_velocity[:] = 0.0
    task = Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count), steps=1)
    task.enable_simulation_preparation(
        Go2WalkPreparationSettings(ramp_duration_s=0.02, settle_duration_s=0.02, timeout_s=0.1)
    )
    keyboard = _FakeKeyboard()
    task.attach_keyboard(keyboard)
    task.reset(state)

    first = task.step(state, _context(0, 0.0, [0.0, 0.0, 0.0]))
    assert task.phase == "wait_stand" and not first.policy_tick
    np.testing.assert_array_equal(first.command.kp, 0.0)
    keyboard.down = {"p"}
    task.step(state, _context(0, 0.0, [0.0, 0.0, 0.0]))
    keyboard.down = {"r"}
    task.step(state, _context(1, 0.02, [0.0, 0.0, 0.0]))
    assert task.phase == "prepare"
    keyboard.down.clear()
    task.step(state, _context(2, 0.04, [0.0, 0.0, 0.0]))
    task.step(state, _context(3, 0.06, [0.0, 0.0, 0.0]))
    hold = task.step(state, _context(4, 0.08, [0.0, 0.0, 0.0]))
    assert task.phase == "wait_policy" and not hold.policy_tick
    assert task.policy_controller.completed_steps == 0
    keyboard.down.clear()
    task.step(state, _context(5, 0.10, [0.0, 0.0, 0.0]))
    assert task.phase == "wait_policy"
    keyboard.down = {"p"}
    takeover = task.step(state, _context(6, 0.12, [0.0, 0.0, 0.0]))
    assert takeover.policy_tick and task.phase == "policy"
    assert task.policy_controller.completed_steps == 1


def test_o_in_crouch_simulation_returns_to_spawn_before_damping_and_completion() -> None:
    robot = _robot()
    state = _state(robot)
    state.joint_position[:] = np.clip(robot.default_joint_position + 0.1, robot.position_lower, robot.position_upper)
    crouch = state.joint_position.copy()
    keyboard = FakeKeyboardDevice()
    keyboard.down = set()
    keyboard.is_key_down = lambda key: key in keyboard.down
    task = Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count))
    task.prepare(simulation=True, device=keyboard, spawn_pose="lie_down")
    task.reset(state)
    state.joint_position[:] = robot.default_joint_position
    start = state.joint_position.copy()
    task._stage = "policy"
    keyboard.down = {"o"}
    first = task.step(state, _context(0, 0.0, [0.0] * 3))
    assert task.phase == "shutdown" and not first.policy_tick
    np.testing.assert_allclose(first.command.joint_position, start + 0.01 * (crouch - start))
    keyboard.down.clear()
    last = task.step(state, _context(99, 1.98, [0.0] * 3))
    np.testing.assert_allclose(last.command.joint_position, crouch)
    damp = task.step(state, _context(100, 2.0, [0.0] * 3))
    assert task.phase == "damping" and not damp.complete
    finished = task.step(state, _context(150, 3.0, [0.0] * 3))
    assert finished.complete and finished.success
    assert task.policy_controller.completed_steps == 0


def test_o_keeps_damping_for_ten_seconds_before_closing() -> None:
    robot = _robot()
    state = _state(robot)
    task = Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count), steps=1)
    task.enable_simulation_preparation()
    keyboard = _FakeKeyboard()
    task.attach_keyboard(keyboard)
    task.reset(state)
    keyboard.down = {"o"}
    start = task.step(state, _context(0, 0.0, [0.0, 0.0, 0.0]))
    assert task.phase == "damping" and not start.complete
    np.testing.assert_array_equal(start.command.kp, 0.0)
    np.testing.assert_array_equal(start.command.kd, 8.0)
    keyboard.down.clear()
    before = task.step(state, _context(499, 9.98, [0.0, 0.0, 0.0]))
    assert task.phase == "damping" and not before.complete
    assert not before.policy_tick
    finished = task.step(state, _context(500, 10.0, [0.0, 0.0, 0.0]))
    assert finished.complete and not finished.success
    assert task.phase == "complete"


def test_shared_keyboard_velocity_binding_does_not_consume_operator_key_edges() -> None:
    robot = _robot()
    state = _state(robot)
    task = Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count), steps=1)
    task.enable_simulation_preparation()
    keyboard = _FakeKeyboard()
    task.attach_keyboard(keyboard, poll=False)
    task.reset(state)
    keyboard.down = {"r"}
    task.step(state, _context(0, 0.0, [0.0, 0.0, 0.0]))
    assert task.phase == "prepare"
    assert keyboard.poll_count == 0


def test_go2_missing_command_is_rejected_before_policy_and_action(monkeypatch):
    robot = _robot()
    task = Go2WalkPolicyProcessor(_task_spec(), robot)
    policy = NoOpPolicy(robot.joint_count)
    controller = PolicyController(task, policy, steps=1)
    calls = []
    monkeypatch.setattr(policy, "infer", lambda observation: calls.append("infer"))
    monkeypatch.setattr(task, "process_action", lambda action: calls.append("action"))
    state = _state(robot)
    controller.reset(state)
    with pytest.raises(ControlFailure) as failure:
        controller.step(state, ControlContext(0, 0.0, None, dt_s=0.02))
    assert failure.value.reason == "input_error"
    assert not calls


@pytest.mark.parametrize("up_z", [-1.0, 0.0, 0.75, 1.0])
def test_fall_orientation_uses_artifact_threshold(up_z: float) -> None:
    robot = _robot()
    state = _state(robot)
    angle = np.arccos(up_z)
    state.base_orientation_xyzw = np.array([np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)], dtype=np.float32)
    spec = _task_spec()
    task = Go2WalkPolicyProcessor(spec, robot)
    reason = task.check_termination(state)
    assert (reason == "fall_orientation") == (up_z <= spec.termination_min_up_z)


def test_fall_orientation_includes_threshold_boundary() -> None:
    spec = _task_spec()
    robot = _robot()
    task = Go2WalkPolicyProcessor(_replace_spec(spec, termination_min_up_z=1.0), robot)
    assert task.check_termination(_state(robot)) == "fall_orientation"


@pytest.mark.parametrize("key", ["termination_min_up_z", "standing_threshold"])
def test_missing_required_artifact_field_is_rejected_during_decoding(key: str) -> None:
    wire = _task_spec().to_dict()
    del wire["config"][key]
    with pytest.raises(ValidationError, match=key):
        TaskSpec.from_dict(wire)


def test_typed_task_spec_roundtrips_through_common_codec() -> None:
    spec = _task_spec()
    decoded = TaskSpec.from_dict(spec.to_dict())
    assert type(decoded) is Go2WalkTaskSpec
    assert decoded == spec
    assert decoded.name == "go2_walk/v1"


@pytest.mark.parametrize(
    ("key", "value"),
    [("termination_min_up_z", 2.0), ("termination_min_up_z", True)],
)
def test_termination_config_is_validated(key: str, value: object) -> None:
    with pytest.raises(PydanticValidationError, match=key):
        _replace_spec(_task_spec(), **{key: value})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("action_scale", [-0.1] * 12),
        ("kp", [-1.0] * 12),
        ("kd", [-1.0] * 12),
        ("action_lower", [2.0] * 12),
        ("action_upper", [-2.0] * 12),
        ("command_lower", [2.0] * 3),
        ("command_scale", [1.1] * 3),
        ("command_scale", [-0.1] * 3),
        ("gait_frequency_hz", 0.0),
        ("standing_threshold", -0.1),
        ("termination_min_up_z", -1.1),
        ("action_lower", [-1.0] * 11),
        ("action_upper", [1.0] * 11),
        ("action_lower", [float("nan")] * 12),
        ("action_upper", [float("inf")] * 12),
        ("action_scale", [0.25] * 11),
        ("action_scale", []),
        ("kp", [float("inf")] * 12),
        ("command_scale", [float("nan")] * 3),
        ("feet_phase_offsets", [0.0] * 3),
        ("feet_phase_offsets", [0.0] * 5),
        ("feet_phase_offsets", [float("inf")] * 4),
        ("command_lower", [-1.0] * 2),
        ("command_upper", [1.0] * 4),
        ("command_scale", [0.5] * 4),
        ("kp", []),
        ("kd", [0.5] * 11),
        ("gait_frequency_hz", float("inf")),
        ("standing_threshold", float("nan")),
    ],
)
def test_semantic_validation_runs_for_construction_and_wire_decoding(key: str, value: object) -> None:
    spec = _task_spec()
    with pytest.raises(PydanticValidationError):
        _replace_spec(spec, **{key: value})
    wire = spec.to_dict()
    wire["config"][key] = value
    with pytest.raises(ValidationError, match="task.config"):
        TaskSpec.from_dict(wire)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("action_scale", 0.25),
        ("action_lower", -1.0),
        ("action_upper", 1.0),
        ("action_lower", [[-1.0] * 12]),
        ("action_upper", [True] * 12),
        ("termination_min_up_z", True),
    ],
)
def test_invalid_wire_types_are_rejected_by_common_codec(key: str, value: object) -> None:
    wire = _task_spec().to_dict()
    wire["config"][key] = value
    with pytest.raises(ValidationError, match="task.config"):
        TaskSpec.from_dict(wire)


@pytest.mark.parametrize("key", ["gait_frequency_hz", "standing_threshold"])
def test_programmatic_numeric_fields_reject_booleans(key: str) -> None:
    with pytest.raises(PydanticValidationError, match=key):
        _replace_spec(_task_spec(), **{key: True})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("action_scale", (0.25,) * 12),
        ("kp", ["35.0"] * 12),
        ("kd", [True] * 12),
        ("command_lower", [[-1.0] * 3]),
        ("gait_frequency_hz", "2.0"),
        ("unknown_field", 1.0),
    ],
)
def test_programmatic_construction_is_strict(key: str, value: object) -> None:
    with pytest.raises(PydanticValidationError, match=key):
        _replace_spec(_task_spec(), **{key: value})


def test_task_spec_fields_are_frozen_but_lists_remain_public() -> None:
    spec = _task_spec()
    assert isinstance(spec.action_scale, list)
    with pytest.raises(PydanticValidationError, match="frozen_instance"):
        spec.standing_threshold = 0.1


@pytest.mark.parametrize("key", ["action_scale", "kp", "kd", "action_lower", "action_upper"])
def test_mutated_joint_vector_length_is_rejected_during_serialization(key: str) -> None:
    spec = _task_spec()
    getattr(spec, key).pop()
    with pytest.raises(ValidationError, match="task.config"):
        spec.to_dict()


@pytest.mark.parametrize(
    ("key", "value"),
    [("kp", -1.0), ("action_upper", -2.0), ("command_lower", 2.0), ("feet_phase_offsets", float("nan"))],
)
def test_mutated_list_values_are_rejected_during_serialization(key: str, value: float) -> None:
    spec = _task_spec()
    getattr(spec, key)[0] = value
    with pytest.raises(ValidationError, match="task.config"):
        spec.to_dict()


@pytest.mark.parametrize(
    "key",
    [
        "action_scale",
        "kp",
        "kd",
        "action_lower",
        "action_upper",
        "command_lower",
        "command_upper",
        "feet_phase_offsets",
    ],
)
@pytest.mark.parametrize("value", [-1e100, 1e100])
def test_vectors_reject_finite_values_outside_float32_range(key: str, value: float) -> None:
    # Runtime vectors are float32, so finite Python floats must not overflow on conversion.
    spec = _task_spec()
    vector = list(getattr(spec, key))
    vector[0] = value
    with pytest.raises(PydanticValidationError, match=key):
        _replace_spec(spec, **{key: vector})
    wire = spec.to_dict()
    wire["config"][key] = vector
    with pytest.raises(ValidationError, match=f"task.config.{key}.0"):
        TaskSpec.from_dict(wire)
    getattr(spec, key)[0] = value
    with pytest.raises(ValidationError, match=f"task.config.{key}.0"):
        spec.to_dict()


def test_runtime_checks_robot_joint_count() -> None:
    robot = _robot()
    short_robot = replace(
        robot,
        joint_names=robot.joint_names[:-1],
        default_joint_position=robot.default_joint_position[:-1],
        position_lower=robot.position_lower[:-1],
        position_upper=robot.position_upper[:-1],
        torque_limit=robot.torque_limit[:-1],
    )
    with pytest.raises(ValidationError, match="task.config.action_scale"):
        Go2WalkPolicyProcessor(_task_spec(), short_robot)

    joint_count = short_robot.joint_count
    spec = _replace_spec(
        _task_spec(),
        action_scale=[0.25] * joint_count,
        kp=[35.0] * joint_count,
        kd=[0.5] * joint_count,
        action_lower=[-1.0] * joint_count,
        action_upper=[1.0] * joint_count,
    )
    task = Go2WalkPolicyProcessor(spec, short_robot)
    command = task.process_action(np.zeros(joint_count, dtype=np.float32))
    assert command.joint_position.shape == (joint_count,)
    assert command.joint_velocity.shape == (joint_count,)
    with pytest.raises(ValidationError, match="action.raw.shape"):
        task.process_action(np.zeros(joint_count + 1, dtype=np.float32))


def test_go2_observation_has_exact_49_element_contract() -> None:
    robot = _robot()
    state = _state(robot)
    task = Go2WalkPolicyProcessor(_task_spec(), robot)
    context = _context(0, 0.0, [0.5, 0.0, -0.2])
    task.validate_command(context.command)
    task.reset(state, context)

    observation = task.build_observation(state, context)

    np.testing.assert_array_equal(observation[0:3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(observation[3:6], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(observation[6:18], 0.1)
    np.testing.assert_allclose(observation[18:30], 0.2)
    np.testing.assert_array_equal(observation[30:42], 0.0)
    np.testing.assert_allclose(observation[42:45], [0.5, 0.0, -0.2])
    np.testing.assert_array_equal(observation[45:49], 0.0)


def test_previous_action_and_trot_phase_advance_after_first_tick() -> None:
    robot = _robot()
    state = _state(robot)
    task = Go2WalkPolicyProcessor(_task_spec(), robot)
    reset_context = _context(0, 0.0, [0.5, 0.0, 0.0])
    task.validate_command(reset_context.command)
    task.reset(state, reset_context)
    task.process_action(np.ones(12, dtype=np.float32))

    context = _context(1, 0.02, [0.5, 0.0, 0.0])
    task.validate_command(context.command)
    observation = task.build_observation(state, context)

    np.testing.assert_array_equal(observation[30:42], 1.0)
    np.testing.assert_allclose(observation[45:49], [0.04, 0.54, 0.54, 0.04])


def test_gait_phase_restarts_after_standing_and_reset() -> None:
    robot = _robot()
    state = _state(robot)
    task = Go2WalkPolicyProcessor(_task_spec(), robot)
    moving = [0.5, 0.0, 0.0]
    standing = [0.0, 0.0, 0.0]
    task.reset(state, _context(0, 0.0, moving))

    phases = [
        task.build_observation(state, _context(step, step * 0.02, command))[45:49]
        for step, command in enumerate((moving, moving, standing, standing, moving, moving))
    ]
    np.testing.assert_allclose(phases[0], 0.0)
    np.testing.assert_allclose(phases[1], [0.04, 0.54, 0.54, 0.04])
    np.testing.assert_allclose(phases[2:4], 0.0)
    np.testing.assert_allclose(phases[4], phases[1])
    np.testing.assert_allclose(phases[5], [0.08, 0.58, 0.58, 0.08])

    task.reset(state, _context(0, 0.0, moving))
    np.testing.assert_allclose(task.build_observation(state, _context(1, 0.02, moving))[45:49], phases[1])


def test_standing_command_freezes_deployment_gait_phase() -> None:
    robot = _robot()
    task = Go2WalkPolicyProcessor(_task_spec(), robot)

    context = _context(10, 0.2, np.zeros(3, dtype=np.float32))
    task.validate_command(context.command)
    observation = task.build_observation(_state(robot), context)

    np.testing.assert_array_equal(observation[45:49], np.zeros(4, dtype=np.float32))


def test_action_processing_applies_clip_scale_and_position_limits() -> None:
    # Synthetic asymmetric action bounds and nonzero offsets distinguish policy
    # clipping from the subsequent robot-position clipping.
    robot = replace(
        _robot(),
        default_joint_position=np.full(12, 0.5, dtype=np.float32),
        position_lower=np.full(12, 0.25, dtype=np.float32),
        position_upper=np.full(12, 0.75, dtype=np.float32),
    )
    state = _state(robot)
    spec = _replace_spec(
        _task_spec(),
        action_lower=[-2.0] * 12,
        action_upper=[0.5] * 12,
    )
    task = Go2WalkPolicyProcessor(spec, robot)
    reset_context = _context(0, 0.0, np.zeros(3, dtype=np.float32))
    task.validate_command(reset_context.command)
    task.reset(state, reset_context)

    raw_action = np.tile(np.array([-3.0, 2.0], dtype=np.float32), 6)
    command = task.process_action(raw_action)
    next_context = _context(1, 0.02, np.zeros(3, dtype=np.float32))
    task.validate_command(next_context.command)
    next_observation = task.build_observation(state, next_context)

    np.testing.assert_allclose(command.joint_position, np.tile([0.25, 0.625], 6))
    np.testing.assert_array_equal(next_observation[30:42], np.tile([-2.0, 0.5], 6))
    np.testing.assert_array_equal(command.kp, 35.0)
    np.testing.assert_array_equal(command.kd, 0.5)


def test_go2_task_rejects_non_singleton_command_batch() -> None:
    task = Go2WalkPolicyProcessor(_task_spec(), _robot())

    with pytest.raises(ValidationError, match="command.batch_size"):
        task.validate_command(PlanarVelocityCommand(np.zeros((2, 3), dtype=np.float32)))


def test_go2_task_scales_command_range_for_deployment_input_mapping() -> None:
    task = Go2WalkPolicyProcessor(_task_spec(), _robot())

    np.testing.assert_array_equal(task.command_lower, [-0.5, -0.5, -0.5])
    np.testing.assert_array_equal(task.command_upper, [0.5, 0.5, 0.5])
    task.validate_command(PlanarVelocityCommand(np.array([[0.75, 0.0, 0.0]], dtype=np.float32)))


def test_go2_prepare_builds_binding_for_each_device_kind() -> None:
    robot = _robot()
    task = Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count), steps=1)

    keyboard_binding = task.prepare(simulation=True, device=FakeKeyboardDevice())
    assert isinstance(keyboard_binding, KeyboardPlanarVelocityBinding)

    gamepad_binding = task.prepare(simulation=True, device=FakeGamePadDevice())
    assert isinstance(gamepad_binding, BoundedGamePadPlanarVelocityBinding)

    constant_binding = task.prepare(simulation=True)
    assert isinstance(constant_binding, ConstantPlanarVelocityBinding)


def test_go2_prepare_attaches_shared_keyboard_and_exposes_operator_hint() -> None:
    robot = _robot()
    task = Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count), steps=1)

    assert task.operator_hint is None
    keyboard = FakeKeyboardDevice()
    task.prepare(simulation=True, device=keyboard, spawn_pose="lie_down")
    assert task.operator_hint is not None
    task.reset(_state(robot))
    assert task.phase == "wait_stand"


@pytest.mark.parametrize("measured_position", [0.0, 1.2])
def test_preparation_accepts_out_of_range_start_and_b_before_start_damps(measured_position) -> None:
    robot = _robot()
    state = _state(robot)
    state.joint_position[:] = measured_position
    state.joint_position[0] = robot.position_upper[0] + 0.2
    gamepad = FakeGamePadDevice()
    task = Go2WalkDeployTask(
        _task_spec(), robot, NoOpPolicy(robot.joint_count), settings={"preparation": {"damping_duration_s": 0.02}}
    )
    binding = task.prepare(simulation=False, device=gamepad)
    task.reset(state)
    gamepad.pressing = {"B"}
    binding.read_command()
    decision = task.step(state, _context(0, 0.0, [0.0] * 3))
    assert task.phase == "damping" and not decision.policy_tick
    np.testing.assert_array_equal(decision.command.kp, 0.0)
    assert task.step(state, _context(1, 0.02, [0.0] * 3)).success
    assert task.step(state, _context(2, 0.04, [0.0] * 3)).complete

    task.reset(state)
    gamepad.pressing = {"start"}
    binding.read_command()
    decision = task.step(state, _context(0, 0.0, [0.0] * 3))
    assert task.phase == "prepare" and decision.error is None
    assert np.all(decision.command.joint_position <= robot.position_upper)
    assert task.policy_controller.completed_steps == 0


@pytest.mark.parametrize("lost_readiness", ["position", "velocity", "tilt", "out_of_range"])
def test_hardware_policy_takeover_rechecks_measured_readiness(lost_readiness: str) -> None:
    robot = _robot()
    state = _state(robot)
    gamepad = FakeGamePadDevice()
    task = Go2WalkDeployTask(
        _task_spec(),
        robot,
        NoOpPolicy(robot.joint_count),
        settings={"preparation": {"ramp_duration_s": 0.02, "settle_duration_s": 0.02}},
    )
    binding = task.prepare(simulation=False, device=gamepad)
    task.reset(state)
    gamepad.pressing = {"start"}
    binding.read_command()
    task.step(state, _context(0, 0.0, [0.0] * 3))
    gamepad.pressing.clear()
    binding.read_command()
    task.step(state, _context(1, 0.02, [0.0] * 3))
    task.step(state, _context(2, 0.04, [0.0] * 3))
    assert task.phase == "wait_policy"
    if lost_readiness == "position":
        state.joint_position[0] = 0.5
    elif lost_readiness == "velocity":
        state.joint_velocity[0] = 2.0
    elif lost_readiness == "tilt":
        state.base_orientation_xyzw[:] = [1.0, 0.0, 0.0, 0.0]
    else:
        state.joint_position[0] = 3.0
    gamepad.pressing = {"A"}
    binding.read_command()
    with pytest.raises(ControlFailure) as failure:
        task.step(state, _context(3, 0.06, [0.0] * 3))
    assert failure.value.reason == (
        "preparation_initial_state_out_of_range" if lost_readiness == "out_of_range" else "preparation_not_ready"
    )
    assert task.phase == "wait_policy"
    assert task.policy_controller.completed_steps == 0


def test_hardware_gamepad_gates_policy_without_polling_and_preserves_policy_gains() -> None:
    robot = _robot()
    state = _state(robot)
    gamepad = FakeGamePadDevice()
    task = Go2WalkDeployTask(
        _task_spec(),
        robot,
        NoOpPolicy(robot.joint_count),
        settings={"preparation": {"ramp_duration_s": 0.02, "settle_duration_s": 0.02, "kp": 12.0}},
    )
    binding = task.prepare(simulation=False, device=gamepad)
    task.reset(state)
    assert all(button in task.operator_hint for button in ("Start", "A", "B", "Select"))

    def tick(step: int):
        binding.read_command()
        assert gamepad.poll_count == step + 1
        result = task.step(state, _context(step, step * 0.02, [0.0] * 3))
        assert gamepad.poll_count == step + 1
        return result

    gamepad.pressing = {"A"}
    hold = tick(0)
    assert task.phase == "wait_stand" and not hold.policy_tick
    gamepad.pressing = {"start", "A"}
    startup = tick(1)
    np.testing.assert_array_equal(startup.command.kp, 12.0)
    gamepad.pressing = {"A"}
    tick(2)
    tick(3)
    ready = tick(4)
    assert task.phase == "wait_policy" and not ready.policy_tick
    assert not tick(5).policy_tick  # Holding A across readiness is not a new confirmation.
    assert task.phase == "wait_policy"
    gamepad.pressing.clear()
    assert not tick(6).policy_tick
    gamepad.pressing = {"A"}
    takeover = tick(7)
    assert takeover.policy_tick
    np.testing.assert_array_equal(takeover.command.kp, task.spec.kp)
    np.testing.assert_array_equal(takeover.command.kd, task.spec.kd)
    gamepad.pressing = {"B"}
    stopped = tick(8)
    assert task.phase == "shutdown" and not stopped.policy_tick and stopped.success
    np.testing.assert_array_equal(stopped.command.kp, 12.0)


@pytest.mark.parametrize("phase", ["prepare", "wait_policy", "policy"])
@pytest.mark.parametrize("reordered", [False, True])
def test_hardware_b_returns_measured_pose_to_fixed_lie_down_then_finishes(phase, reordered) -> None:
    robot = _robot()
    if reordered:
        robot = replace(robot, joint_names=tuple(reversed(robot.joint_names)))
    state = _state(robot)
    gamepad = FakeGamePadDevice()
    task = Go2WalkDeployTask(
        _task_spec(),
        robot,
        NoOpPolicy(robot.joint_count),
        settings={"preparation": {"ramp_duration_s": 0.02, "settle_duration_s": 0.02, "damping_duration_s": 0.02}},
    )
    binding = task.prepare(simulation=False, device=gamepad)
    task.reset(state)
    gamepad.pressing = {"start"}
    binding.read_command()
    task.step(state, _context(0, 0.0, [0.0] * 3))
    if phase != "prepare":
        gamepad.pressing.clear()
        binding.read_command()
        task.step(state, _context(1, 0.02, [0.0] * 3))
        task.step(state, _context(2, 0.04, [0.0] * 3))
    if phase == "policy":
        gamepad.pressing = {"A"}
        binding.read_command()
        assert task.step(state, _context(3, 0.06, [0.0] * 3)).policy_tick
    assert task.phase == phase
    state.joint_position[:] = np.linspace(-0.1, 0.1, robot.joint_count)
    start = state.joint_position.copy()
    pose = UnitreeGo2Robot().key_pose
    expected = np.array([dict(zip(pose.joint_names, pose.poses["lie_down"]))[name] for name in robot.joint_names])
    gamepad.pressing = {"B"}
    binding.read_command()
    shutdown = task.step(state, _context(4, 0.08, [0.0] * 3))
    assert task.phase == "shutdown" and not shutdown.policy_tick
    np.testing.assert_allclose(shutdown.command.joint_position, start + 0.01 * (expected - start), atol=1e-7)
    gamepad.pressing.clear()
    last = task.step(state, _context(103, 2.06, [0.0] * 3))
    np.testing.assert_allclose(last.command.joint_position, expected, atol=1e-6)
    damp = task.step(state, _context(104, 2.08, [0.0] * 3))
    assert task.phase == "damping"
    np.testing.assert_array_equal(damp.command.kp, 0.0)
    finish = task.step(state, _context(105, 2.10, [0.0] * 3))
    assert finish.complete and finish.success


def test_hardware_shutdown_crouch_honors_artifact_limits_and_reports_restriction(caplog) -> None:
    robot = replace(_robot(), position_lower=np.full(12, -2.0, dtype=np.float32))
    state = _state(robot)
    gamepad = FakeGamePadDevice()
    task = Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count))
    binding = task.prepare(simulation=False, device=gamepad)
    task.reset(state)
    gamepad.pressing = {"start"}
    binding.read_command()
    task.step(state, _context(0, 0.0, [0.0] * 3))
    gamepad.pressing = {"B"}
    binding.read_command()
    task.step(state, _context(1, 0.02, [0.0] * 3))
    last = task.step(state, _context(100, 2.0, [0.0] * 3))
    assert np.all(last.command.joint_position >= robot.position_lower)
    assert np.all(last.command.joint_position <= robot.position_upper)
    assert "limited by artifact joint bounds" in caplog.text
    assert "does not guarantee body support" in caplog.text


def test_requested_stop_interrupts_b_shutdown_with_damping_not_lie_down() -> None:
    robot = _robot()
    state = _state(robot)
    gamepad = FakeGamePadDevice()
    task = Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count))
    binding = task.prepare(simulation=False, device=gamepad)
    task.reset(state)
    gamepad.pressing = {"start"}
    binding.read_command()
    task.step(state, _context(0, 0.0, [0.0] * 3))
    gamepad.pressing = {"B"}
    binding.read_command()
    assert task.step(state, _context(1, 0.02, [0.0] * 3)).command.kp[0] > 0
    task.request_stop()
    damping = task.step(state, _context(2, 0.04, [0.0] * 3))
    assert task.phase == "damping" and not damping.success
    np.testing.assert_array_equal(damping.command.kp, 0.0)


@pytest.mark.parametrize("device", [None, FakeKeyboardDevice()])
def test_hardware_requires_gamepad(device) -> None:
    robot = _robot()
    task = Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count))
    with pytest.raises(ValidationError, match="operator.gamepad"):
        task.prepare(simulation=False, device=device)


@pytest.mark.parametrize(
    "field", [name for name in Go2WalkPreparationSettings.__dataclass_fields__ if not name.endswith("_button")]
)
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True, "1"])
def test_preparation_numeric_settings_require_finite_positive_numbers(field: str, value) -> None:
    with pytest.raises(ValueError, match=field):
        Go2WalkPreparationSettings(**{field: value})


@pytest.mark.parametrize(
    "settings",
    [
        {"unknown": 1},
        {"preparation": []},
        {"preparation": {"unknown": 1}},
        {"preparation": {"start_button": ""}},
        {"preparation": {"start_button": "A"}},
        {"preparation": {"stop_button": "A"}},
        {"preparation": {"start_button": "select"}},
        {"preparation": {"policy_button": "Select"}},
    ],
)
def test_execution_settings_reject_invalid_input(settings: dict) -> None:
    robot = _robot()
    with pytest.raises((ValueError, TypeError, ValidationError)):
        Go2WalkDeployTask(_task_spec(), robot, NoOpPolicy(robot.joint_count), settings=settings)


def test_go2_root_task_executes_preparation_policy_budget_and_normal_stop():
    robot = _robot()
    task = Go2WalkDeployTask(
        _task_spec(),
        robot,
        NoOpPolicy(robot.joint_count),
        steps=2,
        settings={
            "preparation": {
                "ramp_duration_s": 0.02,
                "settle_duration_s": 0.02,
                "damping_duration_s": 0.02,
                "stop_damping_duration_s": 0.02,
            }
        },
    )
    state = _state(robot)
    state.joint_position[:] = robot.default_joint_position
    state.joint_velocity[:] = 0
    task.reset(state)
    decisions = []
    for step in range(20):
        decision = task.step(state, _context(step, step * 0.02, np.zeros(3, dtype=np.float32)))
        decisions.append(decision)
        if decision.complete:
            break
    assert not decisions[0].policy_tick
    assert decision.complete and decision.success
    assert sum(item.policy_tick for item in decisions) == 2
    assert task.policy_controller.completed_steps == 2

    task.reset(state)
    task.request_stop()
    first = task.step(state, _context(0, 0.0, np.zeros(3, dtype=np.float32)))
    assert not first.policy_tick
    stopped = task.step(state, _context(1, 0.02, np.zeros(3, dtype=np.float32)))
    assert stopped.complete and not stopped.success
