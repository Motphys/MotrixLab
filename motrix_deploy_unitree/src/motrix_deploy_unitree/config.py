# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree hardware backend and robot-specific configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING as OC_MISSING

from motrix_deploy.config import HardwareRuntimeCfg
from motrix_deploy.errors import ValidationError
from motrix_deploy.operator import OperatorProfileCfg
from motrix_deploy.robot.g1_standing import G1_29DOF_STANDING_KD, G1_29DOF_STANDING_KP
from motrix_deploy_unitree.actuation import Actuation
from motrix_deploy_unitree.hardware import HardwareRobotCfg, UnitreeG1_29DOFHardwareCfg, UnitreeGo2HardwareCfg
from motrix_deploy_unitree.remote import ACCEPTED_BUTTON_NAMES, BUTTON_NAMES
from motrix_deploy_unitree.sensor import HardwareSensorBindings
from motrix_env_core.config.scene.base import KeyPoseCfg

GO2_MOTOR_COUNT = 12
G1_29DOF_MOTOR_COUNT = 29
G1_HG_MOTOR_SLOT_COUNT = 35
_DEFAULT_GO2_ROBOT = UnitreeGo2HardwareCfg()
_DEFAULT_G1_29DOF_ROBOT = UnitreeG1_29DOFHardwareCfg()
GO2_JOINT_NAME_TO_MOTOR_INDEX: Mapping[str, int] = MappingProxyType(
    {name: binding.index for name, binding in _DEFAULT_GO2_ROBOT.actuation.motors.items()}
)
G1_29DOF_JOINT_NAME_TO_MOTOR_ID: Mapping[str, int] = MappingProxyType(
    {name: binding.index for name, binding in _DEFAULT_G1_29DOF_ROBOT.actuation.motors.items()}
)


@dataclass
class UnitreeRobotCfg(HardwareRobotCfg):
    """Base for robot-specific Unitree hardware and control settings.

    Hardware fields are materialized by concrete recipes. Keeping the abstract
    schema missing avoids OmegaConf expanding frozen bindings before composition.
    """

    key_pose: Any = OC_MISSING
    sensors: Any = OC_MISSING
    actuation: Any = OC_MISSING


@dataclass
class _UnitreeServoRobotCfg(UnitreeRobotCfg):
    """Shared named-joint servo and safety configuration."""

    motor_slot_count: int = OC_MISSING
    joint_name_to_motor_index: dict[str, int] | None = None
    default_transition_duration_s: float = 2.0
    damping_duration_s: float = 1.0
    damping_kd: float = 8.0
    kp: Any = None
    kd: Any = None
    override_policy_gains: bool = True
    wait_for_remote_buttons: bool = True
    start_button: str = "start"
    enable_button: str = "A"
    emergency_stop_button: str = "select"
    lie_down_button: str = "B"
    lie_down_duration_s: float = 2.0
    lie_down_joint_position: Any = None
    motor_mode: int | None = None
    mode_pr: int | None = None
    expected_mode_machine: int | None = None
    command_publish_period_s: float | None = None
    command_timeout_s: float = 0.1
    allow_lie_down: bool = True
    startup_mode: str = "one_shot"

    def __post_init__(self) -> None:
        if self.startup_mode not in {"one_shot", "state_machine"}:
            raise ValidationError("runtime.robot.startup_mode", "one_shot or state_machine", self.startup_mode)
        if not isinstance(self.wait_for_remote_buttons, bool):
            raise ValidationError("runtime.robot.wait_for_remote_buttons", "a boolean", self.wait_for_remote_buttons)
        for name in (
            "default_transition_duration_s",
            "damping_duration_s",
            "damping_kd",
            "lie_down_duration_s",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
                raise ValidationError(f"runtime.robot.{name}", "a finite number", value)
        if (
            not isinstance(self.motor_slot_count, int)
            or isinstance(self.motor_slot_count, bool)
            or self.motor_slot_count <= 0
        ):
            raise ValidationError("runtime.robot.motor_slot_count", "a positive integer", self.motor_slot_count)
        if not isinstance(self.allow_lie_down, bool):
            raise ValidationError("runtime.robot.allow_lie_down", "a boolean", self.allow_lie_down)
        if not isinstance(self.override_policy_gains, bool):
            raise ValidationError("runtime.robot.override_policy_gains", "a boolean", self.override_policy_gains)
        if self.default_transition_duration_s <= 0:
            raise ValidationError(
                "runtime.robot.default_transition_duration_s", "a positive duration", self.default_transition_duration_s
            )
        if self.damping_duration_s < 0 or self.damping_kd < 0:
            raise ValidationError("runtime.robot.damping", "non-negative duration and kd", "invalid value")
        if self.lie_down_duration_s <= 0:
            raise ValidationError("runtime.robot.lie_down_duration_s", "a positive duration", self.lie_down_duration_s)
        if self.lie_down_joint_position is not None:
            self.lie_down_position()
        self._validate_gain_override("kp", self.kp)
        self._validate_gain_override("kd", self.kd)
        if self.motor_mode is not None and (
            not isinstance(self.motor_mode, int)
            or isinstance(self.motor_mode, bool)
            or not 0 <= self.motor_mode <= 0xFF
        ):
            raise ValidationError("runtime.robot.motor_mode", "an integer in [0, 255]", self.motor_mode)
        for name in ("mode_pr", "expected_mode_machine"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFF):
                raise ValidationError(f"runtime.robot.{name}", "an integer in [0, 255]", value)
        if self.command_publish_period_s is not None and (
            not isinstance(self.command_publish_period_s, (int, float))
            or isinstance(self.command_publish_period_s, bool)
            or not np.isfinite(self.command_publish_period_s)
            or self.command_publish_period_s <= 0
        ):
            raise ValidationError(
                "runtime.robot.command_publish_period_s",
                "a positive finite duration or null",
                self.command_publish_period_s,
            )
        if (
            not isinstance(self.command_timeout_s, (int, float))
            or isinstance(self.command_timeout_s, bool)
            or not np.isfinite(self.command_timeout_s)
            or self.command_timeout_s <= 0
        ):
            raise ValidationError(
                "runtime.robot.command_timeout_s", "a positive finite duration", self.command_timeout_s
            )
        for name in ("start_button", "enable_button", "emergency_stop_button", "lie_down_button"):
            value = getattr(self, name)
            if value not in BUTTON_NAMES:
                raise ValidationError(f"runtime.robot.{name}", f"one of {sorted(BUTTON_NAMES)}", value)
        buttons = (self.start_button, self.enable_button, self.emergency_stop_button, self.lie_down_button)
        if len(set(buttons)) != len(buttons):
            raise ValidationError("runtime.robot.remote_buttons", "four distinct buttons", "duplicate button")
        if self.joint_name_to_motor_index is not None and not isinstance(self.joint_name_to_motor_index, Mapping):
            raise ValidationError(
                "runtime.robot.joint_name_to_motor_index",
                "a joint-name to motor-index mapping",
                type(self.joint_name_to_motor_index).__name__,
            )

    def _validate_gain_override(self, name: str, value: object | None) -> None:
        if value is None:
            return
        if isinstance(value, Mapping):
            expected = set(self.actuation.motors)
            actual = set(value)
            if actual != expected:
                raise ValidationError(
                    f"runtime.robot.{name}",
                    f"exactly joint names {sorted(expected)}",
                    f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}",
                )
            if any(
                not isinstance(item, (int, float)) or isinstance(item, bool) or not np.isfinite(item) or item < 0
                for item in value.values()
            ):
                raise ValidationError(f"runtime.robot.{name}", "finite non-negative named gains", value)
            return
        if isinstance(value, bool):
            raise ValidationError(f"runtime.robot.{name}", "a non-negative scalar or joint vector", value)
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValidationError(f"runtime.robot.{name}", "a non-negative scalar or joint vector", value) from error
        joint_count = len(self.actuation.motors)
        if array.shape not in ((), (joint_count,)):
            raise ValidationError(
                f"runtime.robot.{name}", f"a non-negative scalar or {joint_count} joint values", value
            )
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ValidationError(f"runtime.robot.{name}", "finite non-negative gains", value)

    def lie_down_position(self) -> np.ndarray:
        """Return the named shutdown pose or an explicit canonical override."""
        if self.lie_down_joint_position is not None:
            try:
                position = np.asarray(self.lie_down_joint_position, dtype=np.float32)
            except (TypeError, ValueError) as error:
                raise ValidationError(
                    "runtime.robot.lie_down_joint_position",
                    f"{len(self.actuation.motors)} finite canonical joint positions",
                    self.lie_down_joint_position,
                ) from error
            if position.shape != (len(self.actuation.motors),) or not np.all(np.isfinite(position)):
                raise ValidationError(
                    "runtime.robot.lie_down_joint_position",
                    f"{len(self.actuation.motors)} finite canonical joint positions",
                    self.lie_down_joint_position,
                )
            return position.copy()
        try:
            pose = self.key_pose.poses["lie_down"]
        except KeyError as error:
            raise ValidationError("runtime.robot.key_pose", "a named lie_down pose", "missing") from error
        return np.asarray(pose, dtype=np.float32).copy()

    def gain_override(self, name: str, joint_names: tuple[str, ...]) -> np.ndarray | None:
        value = getattr(self, name)
        if value is None:
            return None
        if isinstance(value, Mapping):
            expected = set(joint_names)
            actual = set(value)
            if actual != expected:
                raise ValidationError(
                    f"runtime.robot.{name}",
                    f"exactly artifact joints {sorted(expected)}",
                    f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}",
                )
            return np.asarray([value[joint] for joint in joint_names], dtype=np.float32)
        array = np.asarray(value, dtype=np.float32)
        if array.shape == ():
            return np.full(len(joint_names), array.item(), dtype=np.float32)
        if array.shape != (len(joint_names),):
            raise ValidationError(f"runtime.robot.{name}", f"{len(joint_names)} canonical joint gains", value)
        return np.array(array, dtype=np.float32, copy=True)

    def motor_indices(self, joint_names: tuple[str, ...]) -> np.ndarray:
        mapping = self.joint_name_to_motor_index
        if mapping is None:
            mapping = {name: binding.index for name, binding in self.actuation.motors.items()}
        expected = set(joint_names)
        actual = set(mapping)
        if actual != expected:
            raise ValidationError(
                "runtime.robot.joint_name_to_motor_index",
                f"exactly joints {sorted(expected)}",
                f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}",
            )
        values = tuple(mapping[name] for name in joint_names)
        if any(not isinstance(index, int) or isinstance(index, bool) for index in values):
            raise ValidationError("runtime.robot.motor_indices", "integer motor indices", values)
        if len(set(values)) != len(values) or any(not 0 <= index < self.motor_slot_count for index in values):
            raise ValidationError(
                "runtime.robot.motor_indices", f"unique values in [0, {self.motor_slot_count - 1}]", values
            )
        return np.asarray(values, dtype=np.int64)


@dataclass
class UnitreeGo2RobotCfg(_UnitreeServoRobotCfg):
    """Go2-specific hardware wiring, gains, and safety settings."""

    key_pose: KeyPoseCfg = field(default_factory=lambda: UnitreeGo2HardwareCfg().key_pose)
    sensors: HardwareSensorBindings = field(default_factory=lambda: UnitreeGo2HardwareCfg().sensors)
    actuation: Actuation = field(default_factory=lambda: UnitreeGo2HardwareCfg().actuation)
    motor_slot_count: int = GO2_MOTOR_COUNT


@dataclass
class UnitreeG1_29DOFRobotCfg(_UnitreeServoRobotCfg):
    """G1 29-DOF HG wiring, mode contract, and high-rate publisher settings."""

    key_pose: KeyPoseCfg = field(default_factory=lambda: UnitreeG1_29DOFHardwareCfg().key_pose)
    sensors: HardwareSensorBindings = field(default_factory=lambda: UnitreeG1_29DOFHardwareCfg().sensors)
    actuation: Actuation = field(default_factory=lambda: UnitreeG1_29DOFHardwareCfg().actuation)
    motor_slot_count: int = G1_HG_MOTOR_SLOT_COUNT
    # Fixed FixStand gains, shared with the program and task deployment paths.
    kp: Any = field(default_factory=lambda: dict(G1_29DOF_STANDING_KP))
    kd: Any = field(default_factory=lambda: dict(G1_29DOF_STANDING_KD))
    mode_pr: int | None = 0
    expected_mode_machine: int | None = 5
    command_publish_period_s: float | None = 0.001
    command_timeout_s: float = 0.1
    default_transition_duration_s: float = 3.0
    damping_kd: float = 3.0
    motor_mode: int | None = 1
    allow_lie_down: bool = False
    override_policy_gains: bool = False


@dataclass
class UnitreeBackendConfig(HardwareRuntimeCfg):
    """Shared Unitree DDS transport configuration and selected robot recipe."""

    backend: str = "unitree"
    # Keep the Python SDK as the explicit compatibility default for existing
    # injected-transport tests and diagnostics. Hardware recipes that use the
    # native real-time path set ``transport: native_cpp``.
    transport: str = "python_sdk"
    network_interface: str = OC_MISSING
    domain_id: int = 0
    lowcmd_topic: str = "rt/lowcmd"
    lowstate_topic: str = "rt/lowstate"
    connect_timeout_s: float = 5.0
    validate_crc: bool = True
    subscriber_queue_depth: int = 10
    operator: OperatorProfileCfg | None = None
    robot: UnitreeRobotCfg = OC_MISSING

    def __post_init__(self) -> None:
        if self.transport not in {"python_sdk", "native_cpp"}:
            raise ValidationError("runtime.transport", "python_sdk or native_cpp", self.transport)
        if not isinstance(self.network_interface, str) or not self.network_interface:
            raise ValidationError("runtime.network_interface", "a non-empty interface name", self.network_interface)
        if not isinstance(self.validate_crc, bool):
            raise ValidationError("runtime.validate_crc", "a boolean", self.validate_crc)
        for name in ("lowcmd_topic", "lowstate_topic"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValidationError(f"runtime.{name}", "a non-empty DDS topic", value)
        if (
            not isinstance(self.connect_timeout_s, (int, float))
            or isinstance(self.connect_timeout_s, bool)
            or not np.isfinite(self.connect_timeout_s)
            or self.connect_timeout_s <= 0
        ):
            raise ValidationError("runtime.connect_timeout_s", "a positive duration", self.connect_timeout_s)
        if not isinstance(self.domain_id, int) or isinstance(self.domain_id, bool) or self.domain_id < 0:
            raise ValidationError("runtime.domain_id", "a non-negative integer", self.domain_id)
        if (
            not isinstance(self.subscriber_queue_depth, int)
            or isinstance(self.subscriber_queue_depth, bool)
            or self.subscriber_queue_depth <= 0
        ):
            raise ValidationError("runtime.subscriber_queue_depth", "a positive integer", self.subscriber_queue_depth)
        if not isinstance(self.robot, UnitreeRobotCfg):
            raise ValidationError("runtime.robot", "a UnitreeRobotCfg config", type(self.robot).__name__)
        if self.operator is not None:
            if not isinstance(self.operator, OperatorProfileCfg):
                raise ValidationError("runtime.operator", "an OperatorProfileCfg", type(self.operator).__name__)
            self._validate_operator_profile()

    def _validate_operator_profile(self) -> None:
        assert self.operator is not None
        for intent, gesture in self.operator.intents.items():
            buttons = [*gesture.held, *gesture.held_for_s]
            buttons.extend(name for name in (gesture.pressed, gesture.released) if name is not None)
            unknown = sorted(set(buttons) - set(ACCEPTED_BUTTON_NAMES))
            if unknown:
                raise ValidationError(
                    f"runtime.operator.intents.{intent}",
                    f"buttons in {sorted(ACCEPTED_BUTTON_NAMES)}",
                    unknown,
                )
        velocity = self.operator.planar_velocity
        if velocity is not None:
            axes = {velocity.linear_x_axis, velocity.linear_y_axis, velocity.yaw_axis}
            if not axes <= {"lx", "ly", "rx", "ry"}:
                raise ValidationError("runtime.operator.planar_velocity.axes", "Unitree joystick axes", sorted(axes))
            if velocity.deadman_button is not None and velocity.deadman_button not in ACCEPTED_BUTTON_NAMES:
                raise ValidationError(
                    "runtime.operator.planar_velocity.deadman_button",
                    f"one of {sorted(ACCEPTED_BUTTON_NAMES)}",
                    velocity.deadman_button,
                )
        if isinstance(self.robot, UnitreeG1_29DOFRobotCfg):
            required = (
                {"request_stand", "enable_policy", "enter_passive", "emergency_stop"}
                if self.robot.startup_mode == "one_shot"
                else {"request_fixstand", "request_walk", "request_wbt", "request_passive", "request_emergency"}
            )
            missing = sorted(required - set(self.operator.intents))
            if missing:
                raise ValidationError("runtime.operator.intents", f"G1 safety intents {sorted(required)}", missing)


def install_hardware_runtime_schema() -> None:
    """Install the Unitree hardware recipe schema into Hydra's ``runtime`` group."""
    store = ConfigStore.instance()
    store.store(group="runtime", name="_unitree_schema", node=UnitreeBackendConfig)
    store.store(group="unitree_robot", name="go2", node=UnitreeGo2RobotCfg)
    store.store(group="unitree_robot", name="g1-29dof", node=UnitreeG1_29DOFRobotCfg)


__all__ = [
    "GO2_JOINT_NAME_TO_MOTOR_INDEX",
    "GO2_MOTOR_COUNT",
    "G1_29DOF_MOTOR_COUNT",
    "G1_29DOF_JOINT_NAME_TO_MOTOR_ID",
    "G1_HG_MOTOR_SLOT_COUNT",
    "UnitreeBackendConfig",
    "UnitreeG1_29DOFRobotCfg",
    "UnitreeGo2RobotCfg",
    "UnitreeRobotCfg",
    "install_hardware_runtime_schema",
]
