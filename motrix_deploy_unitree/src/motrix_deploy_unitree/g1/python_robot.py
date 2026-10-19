# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree SDK2 DDS adapter for a physical G1 29-DOF."""

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
from motrix_deploy_unitree.remote import UnitreeRemoteGamePadDevice, UnitreeRemoteState, decode_wireless_remote
from motrix_env_core.input import GamePadDevice


@dataclass(frozen=True)
class UnitreeSdkBindings:
    """Late-bound SDK symbols, injectable for adapter contract tests."""

    channel_factory_initialize: Callable[[int, str], Any]
    channel_publisher: type
    channel_subscriber: type
    low_cmd_message_type: type
    low_state_message_type: type
    make_low_cmd: Callable[[], Any]
    crc_type: type


def _load_g1_29dof_sdk_bindings() -> UnitreeSdkBindings:
    """Load the HG messages used by a physical G1 29-DOF robot."""
    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHg
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHg
        from unitree_sdk2py.utils.crc import CRC
    except ImportError as error:
        raise RuntimeError(
            "Unitree G1 29-DOF support requires the Motphys unitree_sdk2py package; "
            "install https://github.com/motphys-developers/unitree_sdk2_python"
        ) from error
    return UnitreeSdkBindings(
        channel_factory_initialize=ChannelFactoryInitialize,
        channel_publisher=ChannelPublisher,
        channel_subscriber=ChannelSubscriber,
        low_cmd_message_type=LowCmdHg,
        low_state_message_type=LowStateHg,
        make_low_cmd=unitree_hg_msg_dds__LowCmd_,
        crc_type=CRC,
    )


class UnitreeG1_29DOFRobot(RobotInterface):
    """Map canonical G1 state/commands to SDK2 LowState/LowCmd DDS messages."""

    disabled_position = 0.0
    disabled_velocity = 0.0

    def __init__(
        self,
        transport: UnitreeTransportConfig,
        control: UnitreeControlConfig,
        *,
        spec: RobotSpec,
        control_period_s: float,
        state_timeout_s: float,
        operator: OperatorProfileCfg | None = None,
        sdk: UnitreeSdkBindings | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.transport = transport
        self.control = control
        self.hardware = UnitreeG129DOFHardwareCfg()
        for path, value in (
            ("control.period_s", control_period_s),
            ("control.state_timeout_s", state_timeout_s),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValidationError(path, "a positive finite number", value)
        self._control_period_s = float(control_period_s)
        self._state_timeout_s = float(state_timeout_s)
        self._sdk = sdk
        self._clock_ns = clock_ns
        self._spec = spec
        self._sleep = sleep
        self._condition = threading.Condition()
        self._write_lock = threading.Lock()
        self._writer_stop = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._latest_command_update_ns = 0
        self._publisher: Any = None
        self._subscriber: Any = None
        self._low_cmd: Any = None
        self._command_crc: Any = None
        self._state_crc: Any = None
        self._motor_indices = np.empty(0, dtype=np.int64)
        self._motor_modes: tuple[int, ...] = ()
        self._quaternion_indices = [1, 2, 3, 0]
        self._capabilities: RobotCapabilities | None = None
        self._latest_low_state: Any = None
        self._latest_receive_time_ns = 0
        self._remote = UnitreeRemoteState()
        self._gamepad = UnitreeRemoteGamePadDevice(lambda: self.remote)
        self._intent_evaluator = None if operator is None else OperatorIntentEvaluator(operator.intents)
        self._callback_error: Exception | None = None
        self._emergency_stop = False
        self._passive_requested = False
        self._health_reason = "backend is not open"
        self._last_communication_ns = 0
        self._opened = False
        self._enabled = False
        self._stopped = False
        self._closed = False

    def _load_sdk_bindings(self) -> UnitreeSdkBindings:
        return _load_g1_29dof_sdk_bindings()

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
        """Return the latest decoded remote sample."""
        with self._condition:
            return self._remote

    def get_gamepad_device(self) -> GamePadDevice:
        """Return the remote-backed gamepad device used for velocity commands."""
        return self._gamepad

    def _disable_high_level_motion_services(self, sdk: UnitreeSdkBindings) -> None:
        # G1 low-level mode is selected externally; verify HG mode fields on input.
        del sdk

    def open(self) -> None:
        if self._opened and not self._closed:
            raise RuntimeError("Unitree backend is already open")
        with self._condition:
            self._latest_low_state = None
            self._latest_receive_time_ns = 0
            self._callback_error = None
        self._resolve_hardware_bindings(self._spec)
        sdk = self._sdk or self._load_sdk_bindings()
        self._sdk = sdk
        try:
            sdk.channel_factory_initialize(self.transport.domain_id, self.transport.network_interface)
        except Exception as error:
            available_interfaces = sorted(path.name for path in Path("/sys/class/net").iterdir())
            available = ", ".join(available_interfaces) or "none detected"
            raise RuntimeError(
                f"Failed to initialize Unitree DDS on network interface "
                f"{self.transport.network_interface!r} (domain_id={self.transport.domain_id}). "
                f"The interface is unavailable or not visible to CycloneDDS. "
                f"Available interfaces: {available}. "
                f"Set runtime.transport.network_interface to the Ethernet interface connected to the robot."
            ) from error
        self._disable_high_level_motion_services(sdk)
        self._command_crc = sdk.crc_type()
        self._state_crc = sdk.crc_type()
        self._low_cmd = sdk.make_low_cmd()
        self._initialize_low_cmd()

        self._subscriber = sdk.channel_subscriber(UNITREE_LOWSTATE_TOPIC, sdk.low_state_message_type)
        self._subscriber.Init(self._low_state_handler, UNITREE_SUBSCRIBER_QUEUE_DEPTH)
        self._wait_for_first_state()
        assert self._latest_low_state is not None
        motor_slot_count = self.hardware.motor_slot_count
        if len(self._latest_low_state.motor_state) < motor_slot_count:
            raise ValidationError(
                "backend.lowstate.motor_state",
                f"at least {motor_slot_count} motors",
                len(self._latest_low_state.motor_state),
            )
        if len(self._low_cmd.motor_cmd) < motor_slot_count:
            raise ValidationError(
                "backend.lowcmd.motor_cmd",
                f"at least {motor_slot_count} motors",
                len(self._low_cmd.motor_cmd),
            )
        # A LowCmd publisher is created only after a CRC-valid LowState has
        # passed the robot-specific mode and motor-slot contract.
        self._publisher = sdk.channel_publisher(UNITREE_LOWCMD_TOPIC, sdk.low_cmd_message_type)
        self._publisher.Init()
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
        self._opened = True
        self._enabled = False
        self._stopped = False
        self._closed = False
        self._health_reason = ""
        self._last_communication_ns = self._latest_receive_time_ns
        if self.control.command_publish_period_s is not None:
            # Queue the robot-specific passive frame before the periodic writer starts.
            self._publish_pre_enable_command()
            self._start_command_writer()

    def _resolve_hardware_bindings(self, spec: RobotSpec) -> None:
        robot = self.hardware
        self._motor_indices = self.hardware.motor_indices(spec.joint_names)
        bindings = robot.actuation.motors
        if set(bindings) != set(spec.joint_names):
            raise ValidationError("hardware.actuation.motors", "exactly the canonical joints", set(bindings))
        self._motor_modes = tuple(
            bindings[name].enabled_mode if self.hardware.motor_mode is None else self.hardware.motor_mode
            for name in spec.joint_names
        )
        robot.key_pose.validate()

    def enable(self) -> None:
        """Allow command output without moving the robot or waiting for input."""
        self._require_open()
        self.read_state()
        health = self.health()
        if not health.healthy:
            raise RuntimeError(f"Unitree backend is unhealthy: {health.reason}")
        self._enabled = True

    def read_state(self) -> RobotState:
        """Return the latest valid snapshot without waiting for another DDS sample."""
        self._require_open()
        with self._condition:
            if self._callback_error is not None:
                raise RuntimeError(f"invalid Unitree LowState: {self._callback_error}") from self._callback_error
            self._raise_if_remote_stop_requested_locked()
            message = self._latest_low_state
            receive_time_ns = self._latest_receive_time_ns
        # DDS supplies a separate object per sample; published cache entries are
        # not mutated by the callback. Decode outside the cache lock.
        state = self._state_from_message(message, receive_time_ns)
        self._last_communication_ns = state.receive_time_ns
        return state

    def write_command(self, command: RobotCommand) -> None:
        self._require_open()
        if not self._enabled:
            raise RuntimeError("Unitree command output is not enabled")
        self._raise_if_remote_stop_requested()
        self._validate_command(command)
        self._publish_robot_command(command)

    def health(self) -> HealthStatus:
        with self._condition:
            emergency = self._emergency_stop
            passive = self._passive_requested
        reason = self._health_reason
        if emergency:
            reason = f"remote {self.control.emergency_stop_button!r} emergency stop is active"
        elif passive:
            reason = "operator requested passive damping"
        elif self._closed:
            reason = reason or "backend is closed"
        elif self._stopped:
            reason = reason or "backend is stopped"
        elif not self._opened:
            reason = reason or "backend is not open"
        return HealthStatus(
            healthy=self._opened and not self._closed and not self._stopped and not reason,
            reason=reason,
            last_successful_communication_ns=self._last_communication_ns,
        )

    def stop(self) -> None:
        if self._stopped:
            return
        self._enabled = False
        if self._publisher is not None and self._low_cmd is not None and self._command_crc is not None:
            steps = max(1, math.ceil(self.control.damping_duration_s / self._control_period_s))
            for step in range(steps):
                try:
                    self._publish_damping()
                except Exception as error:
                    self._health_reason = f"failed to publish damping command: {error}"
                    break
                if step + 1 < steps:
                    self._sleep(self._control_period_s)
        self._stopped = True

    def close(self) -> None:
        if self._closed:
            return
        errors: list[Exception] = []
        try:
            self.stop()
        except Exception as error:
            errors.append(error)
        try:
            self._stop_command_writer()
        except Exception as error:
            errors.append(error)
        for channel in (self._subscriber, self._publisher):
            if channel is None:
                continue
            close = getattr(channel, "Close", None) or getattr(channel, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as error:
                errors.append(error)
        self._closed = True
        if errors:
            raise RuntimeError(f"failed to close Unitree backend resources: {errors}") from errors[0]

    def _low_state_handler(self, message: Any) -> None:
        receive_time_ns = self._clock_ns()
        try:
            if self._state_crc.Crc(message) != message.crc:
                return
            remote = decode_wireless_remote(message.wireless_remote)
            self._validate_message_fields(message)
        except Exception as error:
            with self._condition:
                self._callback_error = error
                self._condition.notify_all()
            return
        with self._condition:
            self._latest_low_state = message
            self._latest_receive_time_ns = receive_time_ns
            self._remote = remote
            if self._intent_evaluator is not None:
                intents = self._intent_evaluator.update(remote.pressed_buttons(), receive_time_ns)
                if "emergency_stop" in intents or "request_emergency" in intents:
                    self._emergency_stop = True
                if "enter_passive" in intents and self.control.startup_mode == "one_shot":
                    self._passive_requested = True
            # Keep select/back as an unconditional, non-configurable fallback
            # emergency stop in addition to the configured semantic gesture.
            if remote.pressed("select") or (
                self._intent_evaluator is None and remote.pressed(self.control.emergency_stop_button)
            ):
                self._emergency_stop = True
            self._condition.notify_all()

    def _wait_for_first_state(self) -> None:
        deadline_ns = self._clock_ns() + round(self.transport.connect_timeout_s * 1e9)
        with self._condition:
            while self._latest_low_state is None and self._callback_error is None:
                remaining_s = (deadline_ns - self._clock_ns()) / 1e9
                if remaining_s <= 0:
                    raise TimeoutError(
                        f"no valid Unitree LowState received on {UNITREE_LOWSTATE_TOPIC!r} "
                        f"within {self.transport.connect_timeout_s:.3f}s"
                    )
                self._condition.wait(remaining_s)
            if self._callback_error is not None:
                raise RuntimeError(f"invalid Unitree LowState: {self._callback_error}") from self._callback_error

    def _state_from_message(self, message: Any, receive_time_ns: int) -> RobotState:
        position = np.asarray([message.motor_state[index].q for index in self._motor_indices], dtype=np.float32)
        velocity = np.asarray([message.motor_state[index].dq for index in self._motor_indices], dtype=np.float32)
        quaternion = np.asarray(message.imu_state.quaternion, dtype=np.float32)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValidationError("state.base_orientation_xyzw", "four finite quaternion values", quaternion)
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-8:
            raise ValidationError("state.base_orientation_xyzw", "a non-zero quaternion", norm)
        quaternion_xyzw = (quaternion[self._quaternion_indices] / np.float32(norm)).astype(np.float32)
        return RobotState(
            sample_time_ns=receive_time_ns,
            receive_time_ns=receive_time_ns,
            joint_position=position,
            joint_velocity=velocity,
            base_orientation_xyzw=quaternion_xyzw,
            base_angular_velocity=np.asarray(message.imu_state.gyroscope, dtype=np.float32),
            base_linear_acceleration=np.asarray(message.imu_state.accelerometer, dtype=np.float32),
        )

    def _validate_message_fields(self, message: Any) -> None:
        motor_slot_count = self.hardware.motor_slot_count
        if len(message.motor_state) < motor_slot_count:
            raise ValidationError(
                "state.motor_state",
                f"at least {motor_slot_count} motor states",
                len(message.motor_state),
            )
        expected_mode_pr = self.hardware.mode_pr
        if expected_mode_pr is not None:
            actual_mode_pr = getattr(message, "mode_pr", None)
            if actual_mode_pr != expected_mode_pr:
                raise ValidationError("state.mode_pr", expected_mode_pr, actual_mode_pr)
        expected_mode_machine = self.hardware.expected_mode_machine
        if expected_mode_machine is not None:
            actual_mode_machine = getattr(message, "mode_machine", None)
            if actual_mode_machine != expected_mode_machine:
                raise ValidationError("state.mode_machine", expected_mode_machine, actual_mode_machine)
        for role, expected, value in (
            ("base_orientation_xyzw", 4, message.imu_state.quaternion),
            ("base_angular_velocity", 3, message.imu_state.gyroscope),
            ("base_linear_acceleration", 3, message.imu_state.accelerometer),
        ):
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (expected,) or not np.all(np.isfinite(value)):
                raise ValidationError(f"state.{role}", f"{expected} finite values", value)

    def _validate_command(self, command: RobotCommand) -> None:
        expected = (self._spec.joint_count,)
        if not isinstance(command, (JointServoCommand, JointTorqueCommand)):
            raise ValidationError("command.type", "JointServoCommand or JointTorqueCommand", type(command).__name__)
        fields = (
            ("torque",)
            if isinstance(command, JointTorqueCommand)
            else ("joint_position", "joint_velocity", "feedforward_torque", "kp", "kd")
        )
        for name in fields:
            value = getattr(command, name)
            if value.shape != expected or value.dtype != np.float32 or not np.all(np.isfinite(value)):
                raise ValidationError(f"command.{name}", f"finite float32 shape {expected}", value)
        if isinstance(command, JointTorqueCommand):
            if np.any(np.abs(command.torque) > self._spec.torque_limit):
                raise ValidationError("command.torque", "inside RobotSpec torque limits", command.torque.tolist())
            return
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

    def _initialize_low_cmd(self) -> None:
        if hasattr(self._low_cmd, "head"):
            self._low_cmd.head[0] = 0xFE
            self._low_cmd.head[1] = 0xEF
            self._low_cmd.level_flag = 0xFF
            self._low_cmd.gpio = 0
        if self.hardware.mode_pr is not None:
            self._low_cmd.mode_pr = self.hardware.mode_pr
        if self.hardware.expected_mode_machine is not None:
            self._low_cmd.mode_machine = self.hardware.expected_mode_machine
        for motor in self._low_cmd.motor_cmd:
            motor.mode = 0
            motor.q = self.disabled_position
            motor.dq = self.disabled_velocity
            motor.kp = 0.0
            motor.kd = 0.0
            motor.tau = 0.0
        for motor_index, mode in zip(self._motor_indices, self._motor_modes):
            self._low_cmd.motor_cmd[int(motor_index)].mode = mode

    def _publish_robot_command(self, command: RobotCommand) -> None:
        if isinstance(command, JointTorqueCommand):
            with self._write_lock:
                for canonical_index, motor_index in enumerate(self._motor_indices):
                    motor = self._low_cmd.motor_cmd[int(motor_index)]
                    motor.mode = self._motor_modes[canonical_index]
                    motor.q = 0.0
                    motor.dq = 0.0
                    motor.kp = 0.0
                    motor.kd = 0.0
                    motor.tau = float(command.torque[canonical_index])
                self._write_low_cmd_locked()
            return
        self._publish_joint_fields(
            command.joint_position,
            command.joint_velocity,
            command.feedforward_torque,
            command.kp,
            command.kd,
        )

    def _publish_joint_fields(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        torque: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        with self._write_lock:
            for canonical_index, motor_index in enumerate(self._motor_indices):
                motor = self._low_cmd.motor_cmd[int(motor_index)]
                motor.mode = self._motor_modes[canonical_index]
                motor.q = float(position[canonical_index])
                motor.dq = float(velocity[canonical_index])
                motor.kp = float(kp[canonical_index])
                motor.kd = float(kd[canonical_index])
                motor.tau = float(torque[canonical_index])
            self._write_low_cmd_locked()

    def _publish_pre_enable_command(self) -> None:
        self._publish_damping()

    def _publish_damping(self) -> None:
        with self._write_lock:
            self._set_damping_fields_locked()
            self._write_low_cmd_locked()

    def _set_damping_fields_locked(self) -> None:
        state = self._latest_low_state
        for canonical_index, motor_index in enumerate(self._motor_indices):
            motor = self._low_cmd.motor_cmd[int(motor_index)]
            motor.mode = self._motor_modes[canonical_index]
            motor.q = 0.0 if state is None else float(state.motor_state[int(motor_index)].q)
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = self.control.damping_kd
            motor.tau = 0.0

    def _write_low_cmd_locked(self) -> None:
        self._latest_command_update_ns = self._clock_ns()
        if self.control.command_publish_period_s is not None:
            return
        self._publish_low_cmd_locked()

    def _publish_low_cmd_locked(self) -> None:
        self._low_cmd.crc = self._command_crc.Crc(self._low_cmd)
        self._publisher.Write(self._low_cmd)
        self._last_communication_ns = self._clock_ns()

    def _start_command_writer(self) -> None:
        if self._writer_thread is not None:
            raise RuntimeError("Unitree command writer is already running")
        self._writer_stop.clear()
        self._writer_thread = threading.Thread(
            target=self._command_writer_loop,
            name=f"{type(self).__name__}-lowcmd",
            daemon=True,
        )
        self._writer_thread.start()

    def _stop_command_writer(self) -> None:
        thread = self._writer_thread
        if thread is None:
            return
        if self._publisher is not None and self._low_cmd is not None and self._command_crc is not None:
            # stop() has already replaced the buffer with damping. Publish it
            # synchronously once so shutdown does not depend on thread timing.
            with self._write_lock:
                self._publish_low_cmd_locked()
        self._writer_stop.set()
        thread.join(timeout=1.0)
        if thread.is_alive():
            raise RuntimeError("Unitree command writer did not stop")
        self._writer_thread = None

    def _command_writer_loop(self) -> None:
        period_s = self.control.command_publish_period_s
        assert period_s is not None
        period_ns = max(1, round(period_s * 1e9))
        command_timeout_ns = round(self.control.command_timeout_s * 1e9)
        state_timeout_ns = round(self._state_timeout_s * 1e9)
        next_deadline_ns = self._clock_ns()
        try:
            while not self._writer_stop.is_set():
                now_ns = self._clock_ns()
                with self._write_lock:
                    state_stale = now_ns - self._latest_receive_time_ns > state_timeout_ns
                    command_stale = now_ns - self._latest_command_update_ns > command_timeout_ns
                    if state_stale or command_stale:
                        self._set_damping_fields_locked()
                        stale = "LowState" if state_stale else "policy command"
                        self._health_reason = f"{stale} is stale; publishing damping"
                    self._publish_low_cmd_locked()
                next_deadline_ns += period_ns
                remaining_s = (next_deadline_ns - self._clock_ns()) / 1e9
                if remaining_s <= 0:
                    next_deadline_ns = self._clock_ns()
                    continue
                self._writer_stop.wait(remaining_s)
        except Exception as error:
            with self._condition:
                self._callback_error = error
                self._health_reason = f"LowCmd writer failed: {error}"
                self._condition.notify_all()

    def _raise_if_remote_stop_requested(self) -> None:
        with self._condition:
            self._raise_if_remote_stop_requested_locked()

    def _raise_if_remote_stop_requested_locked(self) -> None:
        if self._emergency_stop:
            raise EmergencyStopError(f"remote {self.control.emergency_stop_button!r} requested damping stop")
        if self._passive_requested:
            raise PassiveRequestedError("operator requested passive damping")

    def _require_open(self) -> None:
        if not self._opened or self._closed:
            raise RuntimeError("Unitree backend is not open")
