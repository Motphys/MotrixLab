# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree Go2 hardware backend configuration."""

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from motrix_deploy.errors import ValidationError
from motrix_deploy_unitree.remote import BUTTON_NAMES

GO2_MOTOR_COUNT = 12
GO2_JOINT_NAME_TO_MOTOR_INDEX: Mapping[str, int] = MappingProxyType(
    {
        "FL_hip_joint": 3,
        "FL_thigh_joint": 4,
        "FL_calf_joint": 5,
        "FR_hip_joint": 0,
        "FR_thigh_joint": 1,
        "FR_calf_joint": 2,
        "RL_hip_joint": 9,
        "RL_thigh_joint": 10,
        "RL_calf_joint": 11,
        "RR_hip_joint": 6,
        "RR_thigh_joint": 7,
        "RR_calf_joint": 8,
    }
)
GO2_LIE_DOWN_JOINT_POSITION = (
    0.05175,
    1.238835,
    -2.74427,
    -0.0608,
    1.24118,
    -2.7375,
    0.31617,
    1.26637,
    -2.79547,
    -0.310495,
    1.266385,
    -2.80177,
)


@dataclass(frozen=True)
class UnitreeGo2BackendConfig:
    """Strict DDS, motor mapping, enable-transition, and stop settings."""

    network_interface: str
    joint_name_to_motor_index: Mapping[str, int] = field(default_factory=lambda: GO2_JOINT_NAME_TO_MOTOR_INDEX)
    domain_id: int = 0
    lowcmd_topic: str = "rt/lowcmd"
    lowstate_topic: str = "rt/lowstate"
    connect_timeout_s: float = 5.0
    default_transition_duration_s: float = 2.0
    damping_duration_s: float = 1.0
    damping_kd: float = 8.0
    kp: object | None = None
    kd: object | None = None
    wait_for_remote_buttons: bool = True
    start_button: str = "start"
    enable_button: str = "A"
    emergency_stop_button: str = "select"
    lie_down_button: str = "B"
    lie_down_duration_s: float = 2.0
    lie_down_joint_position: object = GO2_LIE_DOWN_JOINT_POSITION
    validate_crc: bool = True
    subscriber_queue_depth: int = 10
    motor_mode: int = 0x0A

    def __post_init__(self) -> None:
        if not isinstance(self.network_interface, str) or not self.network_interface:
            raise ValidationError("runtime.network_interface", "a non-empty interface name", self.network_interface)
        for name in ("wait_for_remote_buttons", "validate_crc"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValidationError(f"runtime.{name}", "a boolean", value)
        for name in ("lowcmd_topic", "lowstate_topic"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValidationError(f"runtime.{name}", "a non-empty DDS topic", value)
        for name in (
            "connect_timeout_s",
            "default_transition_duration_s",
            "damping_duration_s",
            "damping_kd",
            "lie_down_duration_s",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
                raise ValidationError(f"runtime.{name}", "a finite number", value)
        if self.connect_timeout_s <= 0:
            raise ValidationError("runtime.connect_timeout_s", "a positive duration", self.connect_timeout_s)
        if self.default_transition_duration_s <= 0:
            raise ValidationError(
                "runtime.default_transition_duration_s",
                "a positive duration",
                self.default_transition_duration_s,
            )
        if self.damping_duration_s < 0 or self.damping_kd < 0:
            raise ValidationError("runtime.damping", "non-negative duration and kd", "invalid value")
        if self.lie_down_duration_s <= 0:
            raise ValidationError("runtime.lie_down_duration_s", "a positive duration", self.lie_down_duration_s)
        self.lie_down_position()
        self._validate_gain_override("kp", self.kp)
        self._validate_gain_override("kd", self.kd)
        if not isinstance(self.domain_id, int) or isinstance(self.domain_id, bool) or self.domain_id < 0:
            raise ValidationError("runtime.domain_id", "a non-negative integer", self.domain_id)
        if (
            not isinstance(self.subscriber_queue_depth, int)
            or isinstance(self.subscriber_queue_depth, bool)
            or self.subscriber_queue_depth <= 0
        ):
            raise ValidationError(
                "runtime.subscriber_queue_depth",
                "a positive integer",
                self.subscriber_queue_depth,
            )
        if (
            not isinstance(self.motor_mode, int)
            or isinstance(self.motor_mode, bool)
            or not 0 <= self.motor_mode <= 0xFF
        ):
            raise ValidationError("runtime.motor_mode", "an integer in [0, 255]", self.motor_mode)
        for name in ("start_button", "enable_button", "emergency_stop_button", "lie_down_button"):
            value = getattr(self, name)
            if value not in BUTTON_NAMES:
                raise ValidationError(f"runtime.{name}", f"one of {sorted(BUTTON_NAMES)}", value)
        buttons = (self.start_button, self.enable_button, self.emergency_stop_button, self.lie_down_button)
        if len(set(buttons)) != len(buttons):
            raise ValidationError("runtime.remote_buttons", "four distinct buttons", "duplicate button")
        if not isinstance(self.joint_name_to_motor_index, Mapping):
            raise ValidationError(
                "runtime.joint_name_to_motor_index",
                "a joint-name to motor-index mapping",
                type(self.joint_name_to_motor_index).__name__,
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "UnitreeGo2BackendConfig":
        """Parse a strict Unitree backend section while honoring declared defaults."""
        values: dict[str, Any] = dict(value)
        values.pop("name", None)
        expected_fields = set(cls.__dataclass_fields__)
        required_fields = {
            name
            for name, field_definition in cls.__dataclass_fields__.items()
            if field_definition.default is MISSING and field_definition.default_factory is MISSING
        }
        actual_fields = set(values)
        if missing := required_fields - actual_fields:
            raise ValidationError("runtime", f"required fields {sorted(required_fields)}", f"missing={sorted(missing)}")
        if unknown := actual_fields - expected_fields:
            raise ValidationError("runtime", f"known fields {sorted(expected_fields)}", f"unknown={sorted(unknown)}")
        mapping = values.get("joint_name_to_motor_index")
        if isinstance(mapping, Mapping):
            values["joint_name_to_motor_index"] = dict(mapping)
        return cls(**values)

    @staticmethod
    def _validate_gain_override(name: str, value: object | None) -> None:
        if value is None:
            return
        if isinstance(value, bool):
            raise ValidationError(f"runtime.{name}", "a non-negative scalar or 12 values", value)
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValidationError(
                f"runtime.{name}",
                "a non-negative scalar or 12 values",
                value,
            ) from error
        if array.shape not in ((), (GO2_MOTOR_COUNT,)) or not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ValidationError(
                f"runtime.{name}",
                "a non-negative scalar or 12 finite non-negative values",
                value,
            )

    def lie_down_position(self) -> np.ndarray:
        """Return the backend-owned shutdown pose in canonical joint order."""
        try:
            position = np.asarray(self.lie_down_joint_position, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValidationError(
                "runtime.lie_down_joint_position",
                "12 finite canonical joint positions",
                self.lie_down_joint_position,
            ) from error
        if position.shape != (GO2_MOTOR_COUNT,) or not np.all(np.isfinite(position)):
            raise ValidationError(
                "runtime.lie_down_joint_position",
                "12 finite canonical joint positions",
                self.lie_down_joint_position,
            )
        return np.array(position, dtype=np.float32, copy=True)

    def gain_override(self, name: str, joint_count: int) -> np.ndarray | None:
        """Expand an optional scalar or canonical per-joint gain override."""
        value = getattr(self, name)
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32)
        if array.shape == ():
            return np.full(joint_count, array.item(), dtype=np.float32)
        if array.shape != (joint_count,):
            raise ValidationError(f"runtime.{name}", f"{joint_count} canonical joint gains", value)
        return np.array(array, dtype=np.float32, copy=True)

    def motor_indices(self, joint_names: tuple[str, ...]) -> np.ndarray:
        """Return motor indices in artifact canonical joint order."""
        expected = set(joint_names)
        actual = set(self.joint_name_to_motor_index)
        if actual != expected:
            raise ValidationError(
                "runtime.joint_name_to_motor_index",
                f"exactly joints {sorted(expected)}",
                f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}",
            )
        values = tuple(self.joint_name_to_motor_index[name] for name in joint_names)
        if any(not isinstance(index, int) or isinstance(index, bool) for index in values):
            raise ValidationError("runtime.motor_indices", "integer motor indices", values)
        if len(set(values)) != len(values) or any(not 0 <= index < GO2_MOTOR_COUNT for index in values):
            raise ValidationError(
                "runtime.motor_indices",
                f"unique values in [0, {GO2_MOTOR_COUNT - 1}]",
                values,
            )
        return np.asarray(values, dtype=np.int64)


__all__ = [
    "GO2_JOINT_NAME_TO_MOTOR_INDEX",
    "GO2_LIE_DOWN_JOINT_POSITION",
    "GO2_MOTOR_COUNT",
    "UnitreeGo2BackendConfig",
]
