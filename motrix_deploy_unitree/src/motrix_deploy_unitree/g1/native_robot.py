# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""G1 hardware adapter backed by the native Unitree SDK2 extension.

The extension owns DDS callbacks and the 1 kHz LowCmd writer.  This module
only translates the native state snapshot into the existing Motrix contracts
at policy-loop frequency.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from motrix_deploy.contracts import (
    HealthStatus,
    JointControlMode,
    JointServoCommand,
    JointTorqueCommand,
    RobotCapabilities,
    RobotCommand,
    RobotSpec,
    RobotState,
)
from motrix_deploy.errors import (
    EmergencyStopError,
    PassiveRequestedError,
    ValidationError,
)
from motrix_deploy.operator import OperatorIntentEvaluator, OperatorProfileCfg
from motrix_deploy.robot.interface import RobotInterface
from motrix_deploy_unitree.config.control import UnitreeControlConfig
from motrix_deploy_unitree.config.g1 import UnitreeG129DOFHardwareCfg
from motrix_deploy_unitree.config.runtime import (
    UNITREE_LOWCMD_TOPIC,
    UNITREE_LOWSTATE_TOPIC,
    UNITREE_SUBSCRIBER_QUEUE_DEPTH,
    UnitreeTransportConfig,
)
from motrix_deploy_unitree.g1.native import load_native_module
from motrix_deploy_unitree.remote import UnitreeRemoteGamePadDevice, UnitreeRemoteState, decode_wireless_remote
from motrix_env_core.input import GamePadDevice


@dataclass(frozen=True)
class NativeBackendHealth:
    healthy: bool
    reason: str
    last_successful_communication_ns: int


def _snapshot_field(snapshot: object, name: str) -> object:
    if isinstance(snapshot, dict):
        return snapshot[name]
    return getattr(snapshot, name)


class UnitreeG1NativeRobot(RobotInterface):
    """Map the native C++ G1 transport to the Motrix RobotInterface contract."""

    def __init__(
        self,
        transport: UnitreeTransportConfig,
        control: UnitreeControlConfig,
        *,
        spec: RobotSpec,
        control_period_s: float,
        state_timeout_s: float,
        operator: OperatorProfileCfg | None = None,
        native_backend: Any | None = None,
        native_factory: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if spec.joint_count != 29:
            raise ValidationError("robot.joint_count", 29, spec.joint_count)
        for path, value in (("control.period_s", control_period_s), ("control.state_timeout_s", state_timeout_s)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValidationError(path, "a positive finite number", value)

        self.transport = transport
        self.control = control
        self.hardware = UnitreeG129DOFHardwareCfg()
        self._spec = spec
        self._control_period_s = float(control_period_s)
        self._state_timeout_s = float(state_timeout_s)
        self._native_backend = native_backend
        self._native_factory = native_factory
        self._sleep = sleep
        self._motor_indices = np.empty(0, dtype=np.int64)
        self._native: Any | None = None
        self._latest_state: RobotState | None = None
        self._remote_sequence = -1
        self._remote = UnitreeRemoteState()
        self._gamepad = UnitreeRemoteGamePadDevice(lambda: self.remote)
        self._intent_evaluator = None if operator is None else OperatorIntentEvaluator(operator.intents)
        self._emergency_stop = False
        self._passive_requested = False
        self._opened = False
        self._enabled = False
        self._stopped = False
        self._closed = False
        self._capabilities: RobotCapabilities | None = None
        self._health_reason = "backend is not open"

    @property
    def spec(self) -> RobotSpec:
        return self._spec

    @property
    def capabilities(self) -> RobotCapabilities:
        if self._capabilities is None:
            raise RuntimeError("Unitree capabilities are available after open()")
        return self._capabilities

    @property
    def remote(self) -> UnitreeRemoteState:
        return self._remote

    def get_gamepad_device(self) -> GamePadDevice:
        return self._gamepad

    def _make_native_backend(self) -> Any:
        if self._native_backend is not None:
            return self._native_backend
        factory = self._native_factory
        if factory is None:
            factory = load_native_module().G1Backend
        robot = self.hardware
        period = self.control.command_publish_period_s or 0.001
        return factory(
            network_interface=self.transport.network_interface,
            domain_id=self.transport.domain_id,
            lowcmd_topic=UNITREE_LOWCMD_TOPIC,
            lowstate_topic=UNITREE_LOWSTATE_TOPIC,
            motor_indices=self._motor_indices.tolist(),
            mode_pr=-1 if robot.mode_pr is None else robot.mode_pr,
            expected_mode_machine=-1 if robot.expected_mode_machine is None else robot.expected_mode_machine,
            motor_mode=1 if robot.motor_mode is None else robot.motor_mode,
            damping_kd=self.control.damping_kd,
            command_period_s=period,
            command_timeout_s=self.control.command_timeout_s,
            state_timeout_s=self._state_timeout_s,
            validate_crc=True,
            subscriber_queue_depth=UNITREE_SUBSCRIBER_QUEUE_DEPTH,
        )

    def open(self) -> None:
        if self._opened and not self._closed:
            raise RuntimeError("native Unitree backend is already open")
        self._motor_indices = self.hardware.motor_indices(self._spec.joint_names)
        # Remote gesture evaluation runs once per sample, including after reopen.
        self._remote_sequence = -1
        self._latest_state = None
        self._remote = UnitreeRemoteState()
        self._emergency_stop = False
        self._passive_requested = False
        self._native = self._make_native_backend()
        try:
            self._native.open(self.transport.connect_timeout_s)
        except Exception:
            try:
                self._native.close()
            finally:
                self._closed = True
            raise
        self._opened = True
        self._closed = False
        self._enabled = False
        self._stopped = False
        self._health_reason = ""
        self._capabilities = RobotCapabilities(
            control_modes=(JointControlMode.SERVO, JointControlMode.TORQUE),
            state_fields=frozenset(
                {
                    "joint_position",
                    "joint_velocity",
                    "base_orientation_xyzw",
                    "base_angular_velocity",
                    "base_linear_acceleration",
                }
            ),
            supports_rendering=False,
            max_command_rate_hz=1.0 / self._control_period_s,
            requires_enable=True,
            stop_semantics="damping",
        )

    def read_state(self) -> RobotState:
        """Return the native cache with its original local arrival timestamp."""
        self._require_open()
        assert self._native is not None
        snapshot = self._native.read_state()
        sequence = int(_snapshot_field(snapshot, "sequence"))
        state = self._state_from_snapshot(snapshot)
        self._latest_state = state
        if sequence != self._remote_sequence:
            self._update_remote(snapshot, state.receive_time_ns)
            self._remote_sequence = sequence
        self._raise_if_remote_stop_requested()
        return state

    def _state_from_snapshot(self, snapshot: object) -> RobotState:
        def vector(name: str, shape: tuple[int, ...]) -> np.ndarray:
            value = np.asarray(_snapshot_field(snapshot, name), dtype=np.float32)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValidationError(f"state.{name}", f"finite float32 values with shape {shape}", value)
            return value

        quaternion = vector("quaternion_xyzw", (4,))
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1.0e-8:
            raise ValidationError("state.base_orientation_xyzw", "a non-zero quaternion", norm)
        quaternion = (quaternion / np.float32(norm)).astype(np.float32, copy=False)
        receive_time_ns = int(_snapshot_field(snapshot, "receive_time_ns"))
        return RobotState(
            sample_time_ns=receive_time_ns,
            receive_time_ns=receive_time_ns,
            joint_position=vector("joint_position", (29,)),
            joint_velocity=vector("joint_velocity", (29,)),
            base_orientation_xyzw=quaternion,
            base_angular_velocity=vector("gyro", (3,)),
            base_linear_acceleration=vector("accelerometer", (3,)),
        )

    def _update_remote(self, snapshot: object, receive_time_ns: int) -> None:
        payload = _snapshot_field(snapshot, "wireless_remote")
        self._remote = decode_wireless_remote(payload)
        if self._intent_evaluator is not None:
            intents = self._intent_evaluator.update(self._remote.pressed_buttons(), receive_time_ns)
            if "emergency_stop" in intents or "request_emergency" in intents:
                self._emergency_stop = True
            if "enter_passive" in intents and self.control.startup_mode == "one_shot":
                self._passive_requested = True
        if self._remote.pressed("select") or (
            self._intent_evaluator is None and self._remote.pressed(self.control.emergency_stop_button)
        ):
            self._emergency_stop = True

    def enable(self) -> None:
        """Allow command output without moving the robot or waiting for input."""
        self._require_open()
        self.read_state()
        health = self.health()
        if not health.healthy:
            raise RuntimeError(f"native Unitree backend is unhealthy: {health.reason}")
        assert self._native is not None
        self._native.set_enabled(True)
        self._enabled = True

    def write_command(self, command: RobotCommand) -> None:
        self._require_open()
        if not self._enabled:
            raise RuntimeError("Unitree command output is not enabled")
        self._raise_if_remote_stop_requested()
        assert self._native is not None
        self._validate_command(command)
        if isinstance(command, JointTorqueCommand):
            self._native.write_torque(command.torque)
        elif isinstance(command, JointServoCommand):
            self._write_servo(
                command.joint_position,
                command.joint_velocity,
                command.feedforward_torque,
                command.kp,
                command.kd,
            )
        else:
            raise ValidationError("command.type", "JointServoCommand or JointTorqueCommand", type(command).__name__)

    def _write_servo(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        torque: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        assert self._native is not None
        self._native.write_command(position, velocity, torque, kp, kd)

    def _validate_command(self, command: RobotCommand) -> None:
        """Enforce the same canonical limits as the Python DDS backend."""
        shape = (self._spec.joint_count,)
        if isinstance(command, JointTorqueCommand):
            if command.torque.shape != shape or command.torque.dtype != np.float32:
                raise ValidationError("command.torque", f"finite float32 shape {shape}", command.torque)
            if not np.all(np.isfinite(command.torque)) or np.any(np.abs(command.torque) > self._spec.torque_limit):
                raise ValidationError("command.torque", "inside RobotSpec torque limits", command.torque.tolist())
            return
        if not isinstance(command, JointServoCommand):
            raise ValidationError("command.type", "JointServoCommand or JointTorqueCommand", type(command).__name__)
        for name in ("joint_position", "joint_velocity", "feedforward_torque", "kp", "kd"):
            value = getattr(command, name)
            if value.shape != shape or value.dtype != np.float32 or not np.all(np.isfinite(value)):
                raise ValidationError(f"command.{name}", f"finite float32 shape {shape}", value)
        if np.any(command.kp < 0) or np.any(command.kd < 0):
            raise ValidationError("command.gains", "non-negative values", "negative gain")
        if np.any(command.joint_position < self._spec.position_lower) or np.any(
            command.joint_position > self._spec.position_upper
        ):
            raise ValidationError(
                "command.joint_position",
                "inside RobotSpec position range",
                command.joint_position.tolist(),
            )
        if np.any(np.abs(command.feedforward_torque) > self._spec.torque_limit):
            raise ValidationError(
                "command.feedforward_torque",
                "inside RobotSpec torque limits",
                command.feedforward_torque.tolist(),
            )

    def health(self) -> HealthStatus:
        if self._native is None:
            return HealthStatus(False, self._health_reason, 0)
        value = self._native.health()
        if isinstance(value, dict):
            return HealthStatus(
                healthy=bool(value["healthy"]),
                reason=str(value["reason"]),
                last_successful_communication_ns=int(value["last_successful_communication_ns"]),
            )
        return NativeBackendHealth(**value)

    def stop(self) -> None:
        if self._stopped:
            return
        self._enabled = False
        if self._native is not None:
            self._native.stop()
        self._stopped = True

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.stop()
        finally:
            if self._native is not None:
                self._native.close()
            self._closed = True

    def _raise_if_remote_stop_requested(self) -> None:
        if self._emergency_stop:
            raise EmergencyStopError(f"remote {self.control.emergency_stop_button!r} requested damping stop")
        if self._passive_requested:
            raise PassiveRequestedError("operator requested passive damping")

    def _require_open(self) -> None:
        if not self._opened or self._closed:
            raise RuntimeError("native Unitree backend is not open")


__all__ = ["NativeBackendHealth", "UnitreeG1NativeRobot"]
