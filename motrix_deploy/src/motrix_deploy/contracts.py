# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Framework-independent deployment data contracts."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from motrix_deploy.errors import ValidationError

FloatArray = NDArray[np.float32]


def _validate_float32_array(value: object, *, path: str, shape: tuple[int, ...]) -> None:
    if not isinstance(value, np.ndarray):
        raise ValidationError(path, "a numpy array", type(value).__name__)
    if value.dtype != np.float32:
        raise ValidationError(f"{path}.dtype", "float32", str(value.dtype))
    if value.shape != shape:
        raise ValidationError(f"{path}.shape", str(shape), value.shape)
    if not np.all(np.isfinite(value)):
        raise ValidationError(path, "finite values", value.tolist())


def float32_array(value: object, *, path: str, shape: tuple[int, ...]) -> FloatArray:
    """Return an immutable finite float32 copy with the required shape."""
    array = np.asarray(value)
    _validate_float32_array(array, path=path, shape=shape)
    result = np.array(array, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


def _validate_names(value: object, *, path: str, allow_empty: bool = False) -> None:
    if not isinstance(value, tuple):
        raise ValidationError(path, "a tuple of names", type(value).__name__)
    if not allow_empty and not value:
        raise ValidationError(path, "at least one name", value)
    if any(not isinstance(name, str) or not name for name in value):
        raise ValidationError(path, "non-empty strings", value)
    if len(value) != len(set(value)):
        raise ValidationError(path, "unique names", value)


@dataclass(frozen=True)
class TensorSpec:
    """The name, shape and dtype of one policy tensor."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("tensor.name", "a non-empty string", self.name)
        if (
            not isinstance(self.shape, tuple)
            or not self.shape
            or any(not isinstance(size, int) or size <= 0 for size in self.shape)
        ):
            raise ValidationError("tensor.shape", "positive fixed dimensions", self.shape)
        if self.dtype != "float32":
            raise ValidationError("tensor.dtype", "float32", self.dtype)


@dataclass(frozen=True)
class RobotSpec:
    """Self-contained canonical robot contract stored in an artifact."""

    base_link_name: str
    joint_names: tuple[str, ...]
    default_joint_position: FloatArray
    position_lower: FloatArray
    position_upper: FloatArray
    torque_limit: FloatArray

    def __post_init__(self) -> None:
        if not self.base_link_name:
            raise ValidationError("robot.base_link_name", "a non-empty string", self.base_link_name)
        _validate_names(self.joint_names, path="robot.joint_names")
        shape = (len(self.joint_names),)
        _validate_float32_array(
            self.default_joint_position,
            path="robot.default_joint_position",
            shape=shape,
        )
        _validate_float32_array(self.position_lower, path="robot.position_lower", shape=shape)
        _validate_float32_array(self.position_upper, path="robot.position_upper", shape=shape)
        _validate_float32_array(self.torque_limit, path="robot.torque_limit", shape=shape)
        if np.any(self.position_lower > self.position_upper):
            raise ValidationError("robot.position_range", "lower <= upper for every joint", "reversed range")
        if np.any(self.default_joint_position < self.position_lower) or np.any(
            self.default_joint_position > self.position_upper
        ):
            raise ValidationError("robot.default_joint_position", "inside position range", "out-of-range value")
        if np.any(self.torque_limit <= 0):
            raise ValidationError("robot.torque_limit", "positive values", self.torque_limit.tolist())

    def validate_compatible(self, actual: "RobotSpec") -> None:
        """Require a runtime robot to match this policy/task contract in canonical order."""
        for name in ("base_link_name", "joint_names"):
            expected_value = getattr(self, name)
            actual_value = getattr(actual, name)
            if expected_value != actual_value:
                raise ValidationError(f"robot.{name}", expected_value, actual_value)
        for name in ("default_joint_position", "position_lower", "position_upper", "torque_limit"):
            expected_value = getattr(self, name)
            actual_value = getattr(actual, name)
            if not np.allclose(expected_value, actual_value, atol=1e-6, rtol=0.0):
                raise ValidationError(f"robot.{name}", expected_value.tolist(), actual_value.tolist())

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)


@dataclass
class RobotState:
    """One timestamped robot sample in canonical joint order and SI units."""

    sample_time_ns: int
    receive_time_ns: int
    joint_position: FloatArray
    joint_velocity: FloatArray
    base_orientation_xyzw: FloatArray
    base_angular_velocity: FloatArray
    base_linear_acceleration: FloatArray
    base_position: FloatArray | None = None
    base_linear_velocity: FloatArray | None = None
    extras: Mapping[str, FloatArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_time_ns < 0:
            raise ValidationError("state.sample_time_ns", "a non-negative integer", self.sample_time_ns)
        if self.receive_time_ns < 0:
            raise ValidationError("state.receive_time_ns", "a non-negative integer", self.receive_time_ns)
        if (
            not isinstance(self.joint_position, np.ndarray)
            or self.joint_position.ndim != 1
            or not self.joint_position.size
        ):
            actual = (
                self.joint_position.shape if isinstance(self.joint_position, np.ndarray) else type(self.joint_position)
            )
            raise ValidationError("state.joint_position.shape", "(joint_count,)", actual)
        shape = (self.joint_position.size,)
        _validate_float32_array(self.joint_position, path="state.joint_position", shape=shape)
        _validate_float32_array(self.joint_velocity, path="state.joint_velocity", shape=shape)
        _validate_float32_array(
            self.base_orientation_xyzw,
            path="state.base_orientation_xyzw",
            shape=(4,),
        )
        _validate_float32_array(
            self.base_angular_velocity,
            path="state.base_angular_velocity",
            shape=(3,),
        )
        _validate_float32_array(
            self.base_linear_acceleration,
            path="state.base_linear_acceleration",
            shape=(3,),
        )
        if self.base_position is not None:
            _validate_float32_array(self.base_position, path="state.base_position", shape=(3,))
        if self.base_linear_velocity is not None:
            _validate_float32_array(
                self.base_linear_velocity,
                path="state.base_linear_velocity",
                shape=(3,),
            )
        norm = np.linalg.norm(self.base_orientation_xyzw)
        if not np.isclose(norm, 1.0, atol=1e-4):
            raise ValidationError("state.base_orientation_xyzw", "unit quaternion", norm)
        for name, value in self.extras.items():
            if not isinstance(name, str) or not name:
                raise ValidationError("state.extras", "non-empty string keys", name)
            if not isinstance(value, np.ndarray):
                raise ValidationError(f"state.extras.{name}", "a numpy array", type(value).__name__)
            _validate_float32_array(value, path=f"state.extras.{name}", shape=value.shape)


class JointControlMode(str, Enum):
    """Canonical joint command modes shared by artifacts and backends."""

    SERVO = "joint_servo"
    TORQUE = "joint_torque"


@dataclass
class JointServoCommand:
    """Canonical joint servo plus feed-forward command in SI units."""

    joint_position: FloatArray
    joint_velocity: FloatArray
    feedforward_torque: FloatArray
    kp: FloatArray
    kd: FloatArray

    def __post_init__(self) -> None:
        if (
            not isinstance(self.joint_position, np.ndarray)
            or self.joint_position.ndim != 1
            or not self.joint_position.size
        ):
            actual = (
                self.joint_position.shape if isinstance(self.joint_position, np.ndarray) else type(self.joint_position)
            )
            raise ValidationError("command.joint_position.shape", "(joint_count,)", actual)
        shape = (self.joint_position.size,)
        _validate_float32_array(self.joint_position, path="command.joint_position", shape=shape)
        _validate_float32_array(self.joint_velocity, path="command.joint_velocity", shape=shape)
        _validate_float32_array(
            self.feedforward_torque,
            path="command.feedforward_torque",
            shape=shape,
        )
        _validate_float32_array(self.kp, path="command.kp", shape=shape)
        _validate_float32_array(self.kd, path="command.kd", shape=shape)
        if np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValidationError("command.gains", "non-negative values", "negative gain")

    @property
    def mode(self) -> JointControlMode:
        return JointControlMode.SERVO


@dataclass
class JointTorqueCommand:
    """Canonical direct joint torque command in SI units."""

    torque: FloatArray

    def __post_init__(self) -> None:
        if not isinstance(self.torque, np.ndarray) or self.torque.ndim != 1 or not self.torque.size:
            actual = self.torque.shape if isinstance(self.torque, np.ndarray) else type(self.torque)
            raise ValidationError("command.torque.shape", "(joint_count,)", actual)
        _validate_float32_array(self.torque, path="command.torque", shape=(self.torque.size,))

    @property
    def mode(self) -> JointControlMode:
        return JointControlMode.TORQUE


RobotCommand: TypeAlias = JointServoCommand | JointTorqueCommand


@dataclass(frozen=True)
class RobotCapabilities:
    """Static backend capabilities negotiated before control starts.

    Attributes:
        control_modes: Canonical joint command modes accepted by the backend.
        state_fields: ``RobotState`` fields populated by every state sample.
        privileged_state_fields: Subset of state_fields supplied as privileged simulator ground truth,
            not measurements guaranteed to be available on hardware.
        extra_sensors: Additional entries available in ``RobotState.extras``, mapped to their fixed shapes.
        supports_rendering: Whether the backend can present a live viewer.
        max_command_rate_hz: Highest command update rate accepted by the backend, or ``None`` when unconstrained.
        requires_enable: Whether ``RobotInterface.enable()`` must complete before policy commands are accepted.
        stop_semantics: Backend behavior after ``RobotInterface.stop()``, for example ``hold_position`` or ``damping``.
    """

    control_modes: tuple[JointControlMode, ...]
    state_fields: frozenset[str]
    extra_sensors: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    supports_rendering: bool = False
    max_command_rate_hz: float | None = None
    requires_enable: bool = False
    stop_semantics: str = "hold_position"
    privileged_state_fields: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _validate_names(self.control_modes, path="capabilities.control_modes")
        try:
            modes = tuple(JointControlMode(mode) for mode in self.control_modes)
        except ValueError as error:
            raise ValidationError(
                "capabilities.control_modes", "joint_servo or joint_torque", self.control_modes
            ) from error
        object.__setattr__(self, "control_modes", modes)
        if not isinstance(self.state_fields, frozenset) or any(
            not isinstance(name, str) or not name for name in self.state_fields
        ):
            raise ValidationError("capabilities.state_fields", "a frozenset of non-empty names", self.state_fields)
        if (
            not isinstance(self.privileged_state_fields, frozenset)
            or not self.privileged_state_fields <= self.state_fields
        ):
            raise ValidationError(
                "capabilities.privileged_state_fields",
                "a frozenset subset of state_fields",
                self.privileged_state_fields,
            )
        if self.max_command_rate_hz is not None and self.max_command_rate_hz <= 0:
            raise ValidationError("capabilities.max_command_rate_hz", "a positive value", self.max_command_rate_hz)
        if not self.stop_semantics:
            raise ValidationError("capabilities.stop_semantics", "a non-empty string", self.stop_semantics)
        for name, shape in self.extra_sensors.items():
            if not isinstance(name, str) or not name:
                raise ValidationError("capabilities.extra_sensors", "non-empty string keys", name)
            if not isinstance(shape, tuple) or any(not isinstance(size, int) or size <= 0 for size in shape):
                raise ValidationError(f"capabilities.extra_sensors.{name}", "a tuple of positive dimensions", shape)


@dataclass(frozen=True)
class HealthStatus:
    """Backend resource/communication health, not task success or fall detection.

    Task-owned termination is evaluated separately by DeployTask.check_termination.
    """

    healthy: bool
    reason: str
    last_successful_communication_ns: int

    def __post_init__(self) -> None:
        if self.last_successful_communication_ns < 0:
            raise ValidationError(
                "health.last_successful_communication_ns",
                "a non-negative integer",
                self.last_successful_communication_ns,
            )
        if not self.healthy and not self.reason:
            raise ValidationError("health.reason", "a non-empty failure reason", self.reason)
