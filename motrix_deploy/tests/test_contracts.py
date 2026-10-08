# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Tests for common robot and tensor contracts."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from motrix_deploy.contracts import (
    JointControlMode,
    JointServoCommand,
    JointTorqueCommand,
    RobotCapabilities,
    RobotSpec,
    TensorSpec,
)
from motrix_deploy.errors import ValidationError


def robot_spec(**overrides: object) -> RobotSpec:
    values = {
        "base_link_name": "base",
        "joint_names": ("a", "b"),
        "default_joint_position": np.array([0.0, 0.0], dtype=np.float32),
        "position_lower": np.array([-1.0, -1.0], dtype=np.float32),
        "position_upper": np.array([1.0, 1.0], dtype=np.float32),
        "torque_limit": np.array([3.0, 3.0], dtype=np.float32),
    }
    values.update(overrides)
    return RobotSpec(**values)


def test_contracts_preserve_valid_array_references() -> None:
    source = np.array([0.0, 0.0], dtype=np.float32)
    spec = robot_spec(default_joint_position=source)
    zero = np.zeros(2, dtype=np.float32)
    command = JointServoCommand(
        joint_position=source,
        joint_velocity=zero,
        feedforward_torque=zero,
        kp=zero,
        kd=zero,
    )

    assert spec.default_joint_position is source
    assert command.joint_position is source


def test_joint_command_modes_are_readonly() -> None:
    zero = np.zeros(2, dtype=np.float32)
    servo = JointServoCommand(zero, zero, zero, zero, zero)
    torque = JointTorqueCommand(zero)
    assert servo.mode is JointControlMode.SERVO
    assert torque.mode is JointControlMode.TORQUE
    assert torque.torque is zero
    for command in (servo, torque):
        with pytest.raises(AttributeError):
            command.mode = JointControlMode.SERVO


@pytest.mark.parametrize("modes", [("joint_servo",), ("joint_torque",), ("joint_servo", "joint_torque")])
def test_capabilities_convert_mode_strings_to_enums(modes: tuple[str, ...]) -> None:
    capabilities = RobotCapabilities(control_modes=modes, state_fields=frozenset())
    assert all(isinstance(mode, JointControlMode) for mode in capabilities.control_modes)
    assert tuple(mode.value for mode in capabilities.control_modes) == modes


@pytest.mark.parametrize("modes", [("unknown",), ("joint_servo", "joint_servo"), (), ["joint_servo"]])
def test_capabilities_reject_invalid_modes(modes: object) -> None:
    with pytest.raises(ValidationError, match="capabilities.control_modes"):
        RobotCapabilities(control_modes=modes, state_fields=frozenset())


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(2, dtype=np.float64),
        np.zeros((1, 2), dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
    ],
)
def test_torque_command_rejects_invalid_arrays(value: np.ndarray) -> None:
    with pytest.raises(ValidationError, match="command.torque"):
        JointTorqueCommand(value)


def test_robot_spec_compatibility_accepts_matching_contract_with_float_roundoff() -> None:
    expected = robot_spec()
    actual = robot_spec(torque_limit=expected.torque_limit + np.float32(2e-7))
    expected.validate_compatible(actual)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_link_name", "other"),
        ("joint_names", ("b", "a")),
        ("joint_names", ("a", "other")),
        ("default_joint_position", np.array([0.1, 0.0], dtype=np.float32)),
        ("position_lower", np.array([-0.9, -1.0], dtype=np.float32)),
        ("position_upper", np.array([0.9, 1.0], dtype=np.float32)),
        ("torque_limit", np.array([2.0, 3.0], dtype=np.float32)),
    ],
)
def test_robot_spec_compatibility_rejects_semantic_mismatch(field, value) -> None:
    with pytest.raises(ValidationError, match=f"robot.{field}"):
        robot_spec().validate_compatible(robot_spec(**{field: value}))


def test_robot_spec_fields_are_frozen() -> None:
    spec = robot_spec()

    with pytest.raises(FrozenInstanceError):
        spec.base_link_name = "other"


@pytest.mark.parametrize(
    ("field", "value", "error_path"),
    [
        ("position_lower", np.array([2.0, -1.0], dtype=np.float32), "robot.position_range"),
        ("torque_limit", np.array([3.0, np.inf], dtype=np.float32), "robot.torque_limit"),
        ("default_joint_position", np.array([0.0], dtype=np.float32), "robot.default_joint_position.shape"),
        ("default_joint_position", np.array([0.0, 0.0], dtype=np.float64), "robot.default_joint_position.dtype"),
    ],
)
def test_robot_spec_rejects_invalid_arrays(field: str, value: np.ndarray, error_path: str) -> None:
    with pytest.raises(ValidationError, match=error_path):
        robot_spec(**{field: value})


def test_tensor_spec_requires_float32_fixed_shape() -> None:
    with pytest.raises(ValidationError, match="tensor.dtype"):
        TensorSpec(name="input", shape=(1, 4), dtype="float64")


def test_robot_command_rejects_negative_gains() -> None:
    zero = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValidationError, match="command.gains"):
        JointServoCommand(
            joint_position=zero,
            joint_velocity=zero,
            feedforward_torque=zero,
            kp=np.array([1.0, -1.0], dtype=np.float32),
            kd=zero,
        )
