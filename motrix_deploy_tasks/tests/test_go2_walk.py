# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Direct Go2 walking deployment-task tests."""

from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError as PydanticValidationError

from motrix_deploy.artifact.schema import TaskSpec
from motrix_deploy.contracts import RobotSpec, RobotState
from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime import PolicyContext
from motrix_deploy_tasks.tasks.go2_walk import Go2WalkDeployTaskV1, Go2WalkTaskSpec
from motrix_env_core.input import PlanarVelocityCommand


def _replace_spec(spec: Go2WalkTaskSpec, **overrides: object) -> Go2WalkTaskSpec:
    return type(spec).model_validate({**spec.model_dump(), **overrides})


def _context(step: int, elapsed_time_s: float, velocity: list[float] | np.ndarray) -> PolicyContext:
    return PolicyContext(
        step=step,
        elapsed_time_s=elapsed_time_s,
        command=PlanarVelocityCommand(np.asarray(velocity, dtype=np.float32)[None, :]),
    )


def _robot() -> RobotSpec:
    return RobotSpec(
        base_link_name="base",
        joint_names=tuple(f"joint_{index}" for index in range(12)),
        default_joint_position=np.zeros(12, dtype=np.float32),
        position_lower=np.full(12, -2.0, dtype=np.float32),
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
        termination_min_base_height=None,
        kp=[35.0] * 12,
        kd=[0.5] * 12,
        action_lower=[-1.0] * 12,
        action_upper=[1.0] * 12,
    )


@pytest.mark.parametrize("up_z", [-1.0, 0.0, 0.75, 1.0])
def test_fall_orientation_uses_artifact_threshold(up_z: float) -> None:
    robot = _robot()
    state = _state(robot)
    angle = np.arccos(up_z)
    state.base_orientation_xyzw = np.array([np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)], dtype=np.float32)
    spec = _task_spec()
    task = Go2WalkDeployTaskV1(spec, robot)
    reason = task.check_termination(state)
    assert (reason == "fall_orientation") == (up_z <= spec.termination_min_up_z)


def test_fall_orientation_includes_threshold_boundary() -> None:
    spec = _task_spec()
    robot = _robot()
    task = Go2WalkDeployTaskV1(_replace_spec(spec, termination_min_up_z=1.0), robot)
    assert task.check_termination(_state(robot)) == "fall_orientation"


def test_enabled_fall_height_requires_position_and_uses_artifact_threshold() -> None:
    robot = _robot()
    state = _state(robot)
    spec = _task_spec()
    threshold = 0.25  # Exactly representable synthetic cutoff, not a tuned task default.
    task = Go2WalkDeployTaskV1(_replace_spec(spec, termination_min_base_height=threshold), robot)
    assert Go2WalkDeployTaskV1(spec, robot).check_termination(state) is None
    with pytest.raises(ValidationError, match="state.base_position.*enabled base-height termination"):
        task.check_termination(state)
    state.base_position = np.array([0.0, 0.0, threshold - 0.1], dtype=np.float32)
    assert task.check_termination(state) == "fall_height"
    assert Go2WalkDeployTaskV1(spec, robot).check_termination(state) is None
    state.base_position[2] = threshold
    assert task.check_termination(state) == "fall_height"
    state.base_position[2] = threshold + 0.1
    assert task.check_termination(state) is None


@pytest.mark.parametrize("key", ["termination_min_up_z", "termination_min_base_height", "standing_threshold"])
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
    [("termination_min_up_z", 2.0), ("termination_min_up_z", True), ("termination_min_base_height", float("nan"))],
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


@pytest.mark.parametrize("key", ["gait_frequency_hz", "standing_threshold", "termination_min_base_height"])
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
        Go2WalkDeployTaskV1(_task_spec(), short_robot)

    joint_count = short_robot.joint_count
    spec = _replace_spec(
        _task_spec(),
        action_scale=[0.25] * joint_count,
        kp=[35.0] * joint_count,
        kd=[0.5] * joint_count,
        action_lower=[-1.0] * joint_count,
        action_upper=[1.0] * joint_count,
    )
    task = Go2WalkDeployTaskV1(spec, short_robot)
    command = task.process_action(np.zeros(joint_count, dtype=np.float32))
    assert command.joint_position.shape == (joint_count,)
    assert command.joint_velocity.shape == (joint_count,)
    with pytest.raises(ValidationError, match="action.raw.shape"):
        task.process_action(np.zeros(joint_count + 1, dtype=np.float32))


def test_go2_observation_has_exact_49_element_contract() -> None:
    robot = _robot()
    state = _state(robot)
    task = Go2WalkDeployTaskV1(_task_spec(), robot)
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
    task = Go2WalkDeployTaskV1(_task_spec(), robot)
    reset_context = _context(0, 0.0, [0.5, 0.0, 0.0])
    task.validate_command(reset_context.command)
    task.reset(state, reset_context)
    task.process_action(np.ones(12, dtype=np.float32))

    context = _context(1, 0.02, [0.5, 0.0, 0.0])
    task.validate_command(context.command)
    observation = task.build_observation(state, context)

    np.testing.assert_array_equal(observation[30:42], 1.0)
    np.testing.assert_allclose(observation[45:49], [0.04, 0.54, 0.54, 0.04])


def test_standing_command_freezes_deployment_gait_phase() -> None:
    robot = _robot()
    task = Go2WalkDeployTaskV1(_task_spec(), robot)

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
    task = Go2WalkDeployTaskV1(spec, robot)
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
    task = Go2WalkDeployTaskV1(_task_spec(), _robot())

    with pytest.raises(ValidationError, match="command.batch_size"):
        task.validate_command(PlanarVelocityCommand(np.zeros((2, 3), dtype=np.float32)))


def test_go2_task_scales_command_range_for_deployment_input_mapping() -> None:
    task = Go2WalkDeployTaskV1(_task_spec(), _robot())

    np.testing.assert_array_equal(task.command_lower, [-0.5, -0.5, -0.5])
    np.testing.assert_array_equal(task.command_upper, [0.5, 0.5, 0.5])
    task.validate_command(PlanarVelocityCommand(np.array([[0.75, 0.0, 0.0]], dtype=np.float32)))
