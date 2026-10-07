# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree SDK2 DDS adapter for a physical Go2."""

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from operator import attrgetter
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
from motrix_deploy.errors import EmergencyStopError, LieDownRequestedError, ValidationError
from motrix_deploy.robot.interface import RobotInterface
from motrix_deploy_unitree.config import GO2_MOTOR_COUNT, UnitreeGo2BackendConfig
from motrix_deploy_unitree.remote import UnitreeRemoteGamePadDevice, UnitreeRemoteState, decode_wireless_remote
from motrix_env_core.input import GamePadDevice

_POSITION_STOP = 2.146e9
_VELOCITY_STOP = 16000.0


def _sdk_field_reader(field: str) -> Callable[[Any], Any]:
    """Resolve public dotted SDK attributes, never expressions or indexing."""
    if any(not part.isidentifier() or part.startswith("_") for part in field.split(".")):
        raise ValidationError("runtime.robot.sensors.hardware.field", "public dotted SDK attributes", field)
    return attrgetter(field)


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
    motion_switcher_client: type | None = None
    sport_client: type | None = None
    robot_state_client: type | None = None


def _load_sdk_bindings() -> UnitreeSdkBindings:
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
        from unitree_sdk2py.utils.crc import CRC
    except ImportError as error:
        raise RuntimeError(
            "Unitree Go2 support requires the Motphys unitree_sdk2py package; "
            "install https://github.com/motphys-developers/unitree_sdk2_python"
        ) from error
    return UnitreeSdkBindings(
        channel_factory_initialize=ChannelFactoryInitialize,
        channel_publisher=ChannelPublisher,
        channel_subscriber=ChannelSubscriber,
        low_cmd_message_type=LowCmdGo,
        low_state_message_type=LowStateGo,
        make_low_cmd=unitree_go_msg_dds__LowCmd_,
        crc_type=CRC,
        motion_switcher_client=MotionSwitcherClient,
        sport_client=SportClient,
        robot_state_client=RobotStateClient,
    )


class UnitreeGo2RobotInterface(RobotInterface):
    """Map canonical Go2 state/commands to SDK2 LowState/LowCmd DDS messages."""

    def __init__(
        self,
        config: UnitreeGo2BackendConfig,
        control_period_s: float,
        state_timeout_s: float,
        *,
        spec: RobotSpec,
        sdk: UnitreeSdkBindings | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
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
        self._publisher: Any = None
        self._subscriber: Any = None
        self._low_cmd: Any = None
        self._command_crc: Any = None
        self._state_crc: Any = None
        self._motor_indices = np.empty(0, dtype=np.int64)
        self._motor_modes: tuple[int, ...] = ()
        self._sensor_readers: dict[str, Callable[[Any], Any]] = {}
        self._quaternion_indices = [1, 2, 3, 0]
        self._lie_down_position = np.empty(0, dtype=np.float32)
        self._kp_override: np.ndarray | None = None
        self._kd_override: np.ndarray | None = None
        self._capabilities: RobotCapabilities | None = None
        self._latest_low_state: Any = None
        self._latest_receive_time_ns = 0
        self._state_sequence = 0
        self._last_read_sequence = 0
        self._remote = UnitreeRemoteState()
        self._gamepad = UnitreeRemoteGamePadDevice(lambda: self.remote)
        self._callback_error: Exception | None = None
        self._emergency_stop = False
        self._lie_down_requested = False
        self._last_position_kp = np.empty(0, dtype=np.float32)
        self._last_position_kd = np.empty(0, dtype=np.float32)
        self._health_reason = "backend is not open"
        self._last_communication_ns = 0
        self._opened = False
        self._enabled = False
        self._stopped = False
        self._closed = False

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
        """Release MCF/motion mode and disable the high-level sport service."""
        if sdk.motion_switcher_client is None or sdk.sport_client is None or sdk.robot_state_client is None:
            raise RuntimeError("Unitree SDK bindings lack MotionSwitcher/Sport/RobotState clients")

        motion_switcher = sdk.motion_switcher_client()
        motion_switcher.SetTimeout(self.config.connect_timeout_s)
        motion_switcher.Init()
        status, mode = motion_switcher.CheckMode()
        if status != 0:
            raise RuntimeError(f"MotionSwitcher CheckMode failed with code {status}")
        if mode and mode.get("name"):
            sport = sdk.sport_client()
            sport.SetTimeout(self.config.connect_timeout_s)
            sport.Init()
            stand_down_status = sport.StandDown()
            if stand_down_status != 0:
                raise RuntimeError(f"SportClient StandDown failed with code {stand_down_status}")
            release_status, _ = motion_switcher.ReleaseMode()
            if release_status != 0:
                raise RuntimeError(f"MotionSwitcher ReleaseMode failed with code {release_status}")
            status, mode = motion_switcher.CheckMode()
            if status != 0 or (mode and mode.get("name")):
                raise RuntimeError(f"MotionSwitcher mode remains active after release: {mode!r}")

        robot_state = sdk.robot_state_client()
        robot_state.SetTimeout(self.config.connect_timeout_s)
        robot_state.Init()
        service_status = robot_state.ServiceSwitch("sport_mode", False)
        if service_status != 0:
            raise RuntimeError(f"RobotState ServiceSwitch(sport_mode, false) failed with code {service_status}")

    def open(self) -> None:
        if self._opened and not self._closed:
            raise RuntimeError("Unitree backend is already open")
        self._resolve_hardware_bindings(self._spec)
        self._kp_override = self.config.gain_override("kp", self._spec.joint_count)
        self._kd_override = self.config.gain_override("kd", self._spec.joint_count)
        sdk = self._sdk or _load_sdk_bindings()
        self._sdk = sdk
        sdk.channel_factory_initialize(self.config.domain_id, self.config.network_interface)
        self._disable_high_level_motion_services(sdk)
        self._command_crc = sdk.crc_type()
        self._state_crc = sdk.crc_type()
        self._low_cmd = sdk.make_low_cmd()
        self._initialize_low_cmd()

        self._publisher = sdk.channel_publisher(self.config.lowcmd_topic, sdk.low_cmd_message_type)
        self._publisher.Init()
        self._subscriber = sdk.channel_subscriber(self.config.lowstate_topic, sdk.low_state_message_type)
        self._subscriber.Init(self._low_state_handler, self.config.subscriber_queue_depth)
        self._wait_for_first_state()
        assert self._latest_low_state is not None
        if len(self._latest_low_state.motor_state) < GO2_MOTOR_COUNT:
            raise ValidationError(
                "backend.lowstate.motor_state",
                f"at least {GO2_MOTOR_COUNT} motors",
                len(self._latest_low_state.motor_state),
            )
        if len(self._low_cmd.motor_cmd) < GO2_MOTOR_COUNT:
            raise ValidationError(
                "backend.lowcmd.motor_cmd",
                f"at least {GO2_MOTOR_COUNT} motors",
                len(self._low_cmd.motor_cmd),
            )
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

    def _resolve_hardware_bindings(self, spec: RobotSpec) -> None:
        robot = self.config.robot
        assert robot is not None
        self._motor_indices = self.config.motor_indices(spec.joint_names)
        bindings = robot.actuation.motors
        if set(bindings) != set(spec.joint_names):
            raise ValidationError("runtime.robot.actuation.motors", "exactly the canonical joints", set(bindings))
        self._motor_modes = tuple(
            bindings[name].enabled_mode if self.config.motor_mode is None else self.config.motor_mode
            for name in spec.joint_names
        )
        for role in ("base_orientation_xyzw", "base_angular_velocity", "base_linear_acceleration"):
            sensor = getattr(robot.sensors, role)
            if sensor is None:
                raise ValidationError(f"runtime.robot.sensors.{role}", "a hardware binding", "missing")
            self._sensor_readers[role] = _sdk_field_reader(sensor.field)
        order = robot.sensors.base_orientation_xyzw.quaternion_order
        if order not in ("wxyz", "xyzw"):
            raise ValidationError("runtime.robot.sensors.base_orientation_xyzw", "a declared quaternion order", order)
        self._quaternion_indices = [1, 2, 3, 0] if order == "wxyz" else [0, 1, 2, 3]
        robot.key_pose.validate()
        target = self.config.lie_down_position()
        if self.config.lie_down_joint_position is None:
            pose_by_joint = dict(zip(robot.key_pose.joint_names, target))
            if set(pose_by_joint) != set(spec.joint_names):
                raise ValidationError("runtime.robot.key_pose", "exactly the canonical joints", set(pose_by_joint))
            target = np.asarray([pose_by_joint[name] for name in spec.joint_names], dtype=np.float32)
        self._lie_down_position = target

    def enable(self) -> None:
        self._require_open()
        if self._enabled:
            return
        if self._kp_override is None or self._kd_override is None:
            raise ValidationError(
                "backend.enable",
                "explicit backend kp and kd safety gains for startup",
                "missing safety gains",
            )
        zeros = np.zeros(self._spec.joint_count, dtype=np.float32)
        initial_command = JointServoCommand(
            joint_position=self._spec.default_joint_position,
            joint_velocity=zeros,
            feedforward_torque=zeros,
            kp=self._kp_override,
            kd=self._kd_override,
        )
        self._validate_command(initial_command)
        kp, kd = self._effective_gains(initial_command)
        self._last_position_kp = np.array(kp, dtype=np.float32, copy=True)
        self._last_position_kd = np.array(kd, dtype=np.float32, copy=True)
        if self.config.wait_for_remote_buttons:
            self._wait_for_button(self.config.start_button, self._publish_zero_torque)
        start = self._latest_robot_state().joint_position
        steps = max(1, math.ceil(self.config.default_transition_duration_s / self._control_period_s))
        for step in range(1, steps + 1):
            self._raise_if_remote_stop_requested()
            alpha = np.float32(step / steps)
            position = ((np.float32(1.0) - alpha) * start + alpha * initial_command.joint_position).astype(np.float32)
            self._publish_joint_fields(
                position,
                initial_command.joint_velocity,
                initial_command.feedforward_torque,
                kp,
                kd,
            )
            self._wait_control_period()

        if self.config.wait_for_remote_buttons:
            self._wait_for_button(
                self.config.enable_button,
                lambda: self._publish_robot_command(initial_command),
            )
        self._raise_if_remote_stop_requested()
        self._enabled = True

    def read_state(self, timeout_s: float) -> RobotState:
        self._require_open()
        if timeout_s <= 0:
            raise TimeoutError(f"Unitree state timeout must be positive, got {timeout_s}")
        deadline_ns = self._clock_ns() + round(timeout_s * 1e9)
        with self._condition:
            while self._state_sequence <= self._last_read_sequence and self._callback_error is None:
                remaining_s = (deadline_ns - self._clock_ns()) / 1e9
                if remaining_s <= 0:
                    raise TimeoutError(f"no fresh Unitree LowState received within {timeout_s:.6f}s")
                self._condition.wait(remaining_s)
            if self._callback_error is not None:
                raise RuntimeError(f"invalid Unitree LowState: {self._callback_error}") from self._callback_error
            self._raise_if_remote_stop_requested_locked()
            state = self._state_from_message(self._latest_low_state, self._latest_receive_time_ns)
            self._last_read_sequence = self._state_sequence
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
            lie_down = self._lie_down_requested
        reason = self._health_reason
        if emergency:
            reason = f"remote {self.config.emergency_stop_button!r} emergency stop is active"
        elif lie_down and not reason:
            reason = f"remote {self.config.lie_down_button!r} lie-down shutdown is active"
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
        with self._condition:
            run_lie_down = self._lie_down_requested and not self._emergency_stop
        if run_lie_down:
            try:
                self._publish_lie_down_trajectory()
            except Exception as error:
                self._health_reason = f"failed to execute lie-down shutdown: {error}"
        if self._publisher is not None and self._low_cmd is not None and self._command_crc is not None:
            steps = max(1, math.ceil(self.config.damping_duration_s / self._control_period_s))
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
            if self.config.validate_crc and self._state_crc.Crc(message) != message.crc:
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
            self._state_sequence += 1
            self._remote = remote
            if remote.pressed(self.config.emergency_stop_button):
                self._emergency_stop = True
            if remote.pressed(self.config.lie_down_button):
                self._lie_down_requested = True
            self._condition.notify_all()

    def _wait_for_first_state(self) -> None:
        deadline_ns = self._clock_ns() + round(self.config.connect_timeout_s * 1e9)
        with self._condition:
            while self._latest_low_state is None and self._callback_error is None:
                remaining_s = (deadline_ns - self._clock_ns()) / 1e9
                if remaining_s <= 0:
                    raise TimeoutError(
                        f"no valid Unitree LowState received on {self.config.lowstate_topic!r} "
                        f"within {self.config.connect_timeout_s:.3f}s"
                    )
                self._condition.wait(remaining_s)
            if self._callback_error is not None:
                raise RuntimeError(f"invalid Unitree LowState: {self._callback_error}") from self._callback_error

    def _wait_for_button(self, button: str, publish: Callable[[], None]) -> None:
        while not self.remote.pressed(button):
            self._raise_if_remote_stop_requested()
            publish()
            self._wait_control_period()

    def _wait_control_period(self) -> None:
        self._sleep(self._control_period_s)
        with self._condition:
            age_s = (self._clock_ns() - self._latest_receive_time_ns) / 1e9
            callback_error = self._callback_error
        if callback_error is not None:
            raise RuntimeError(f"invalid Unitree LowState: {callback_error}") from callback_error
        if age_s > self._state_timeout_s:
            raise TimeoutError(f"Unitree LowState age {age_s:.6f}s exceeds {self._state_timeout_s:.6f}s during enable")

    def _latest_robot_state(self) -> RobotState:
        with self._condition:
            if self._latest_low_state is None:
                raise RuntimeError("Unitree LowState is unavailable")
            return self._state_from_message(self._latest_low_state, self._latest_receive_time_ns)

    def _state_from_message(self, message: Any, receive_time_ns: int) -> RobotState:
        position = np.asarray([message.motor_state[index].q for index in self._motor_indices], dtype=np.float32)
        velocity = np.asarray([message.motor_state[index].dq for index in self._motor_indices], dtype=np.float32)
        quaternion = np.asarray(self._sensor_readers["base_orientation_xyzw"](message), dtype=np.float32)
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
            base_angular_velocity=np.asarray(self._sensor_readers["base_angular_velocity"](message), dtype=np.float32),
            base_linear_acceleration=np.asarray(
                self._sensor_readers["base_linear_acceleration"](message), dtype=np.float32
            ),
        )

    def _validate_message_fields(self, message: Any) -> None:
        if len(message.motor_state) < GO2_MOTOR_COUNT:
            raise ValidationError(
                "state.motor_state",
                f"at least {GO2_MOTOR_COUNT} motor states",
                len(message.motor_state),
            )
        for role, expected in (
            ("base_orientation_xyzw", 4),
            ("base_angular_velocity", 3),
            ("base_linear_acceleration", 3),
        ):
            value = np.asarray(self._sensor_readers[role](message), dtype=np.float32)
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
        self._low_cmd.head[0] = 0xFE
        self._low_cmd.head[1] = 0xEF
        self._low_cmd.level_flag = 0xFF
        self._low_cmd.gpio = 0
        for motor in self._low_cmd.motor_cmd:
            motor.mode = 0
            motor.q = _POSITION_STOP
            motor.qd = _VELOCITY_STOP
            motor.kp = 0.0
            motor.kd = 0.0
            motor.tau = 0.0
        for motor_index, mode in zip(self._motor_indices, self._motor_modes):
            self._low_cmd.motor_cmd[int(motor_index)].mode = mode

    def _publish_robot_command(self, command: RobotCommand) -> None:
        if isinstance(command, JointTorqueCommand):
            # Do not replace the saved servo safety gains with torque-mode zeros.
            with self._write_lock:
                for canonical_index, motor_index in enumerate(self._motor_indices):
                    motor = self._low_cmd.motor_cmd[int(motor_index)]
                    motor.mode = self._motor_modes[canonical_index]
                    motor.q = 0.0
                    motor.qd = 0.0
                    motor.kp = 0.0
                    motor.kd = 0.0
                    motor.tau = float(command.torque[canonical_index])
                self._write_low_cmd_locked()
            return
        kp, kd = self._effective_gains(command)
        self._publish_joint_fields(
            command.joint_position,
            command.joint_velocity,
            command.feedforward_torque,
            kp,
            kd,
        )

    def _effective_gains(self, command: JointServoCommand) -> tuple[np.ndarray, np.ndarray]:
        kp = command.kp if self._kp_override is None else self._kp_override
        kd = command.kd if self._kd_override is None else self._kd_override
        return kp, kd

    def _publish_joint_fields(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        torque: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        self._last_position_kp = np.array(kp, dtype=np.float32, copy=True)
        self._last_position_kd = np.array(kd, dtype=np.float32, copy=True)
        with self._write_lock:
            for canonical_index, motor_index in enumerate(self._motor_indices):
                motor = self._low_cmd.motor_cmd[int(motor_index)]
                motor.mode = self._motor_modes[canonical_index]
                motor.q = float(position[canonical_index])
                motor.qd = float(velocity[canonical_index])
                motor.kp = float(kp[canonical_index])
                motor.kd = float(kd[canonical_index])
                motor.tau = float(torque[canonical_index])
            self._write_low_cmd_locked()

    def _publish_zero_torque(self) -> None:
        with self._write_lock:
            for canonical_index, motor_index in enumerate(self._motor_indices):
                motor = self._low_cmd.motor_cmd[int(motor_index)]
                motor.mode = self._motor_modes[canonical_index]
                motor.q = 0.0
                motor.qd = 0.0
                motor.kp = 0.0
                motor.kd = 0.0
                motor.tau = 0.0
            self._write_low_cmd_locked()

    def _publish_lie_down_trajectory(self) -> None:
        if self._last_position_kp.shape != (GO2_MOTOR_COUNT,) or self._last_position_kd.shape != (GO2_MOTOR_COUNT,):
            raise RuntimeError("position gains are unavailable for lie-down shutdown")
        start = self._latest_robot_state().joint_position
        target = self._lie_down_position
        zeros = np.zeros(GO2_MOTOR_COUNT, dtype=np.float32)
        steps = max(1, math.ceil(self.config.lie_down_duration_s / self._control_period_s))
        for step in range(1, steps + 1):
            with self._condition:
                if self._emergency_stop:
                    raise EmergencyStopError(f"remote {self.config.emergency_stop_button!r} requested damping stop")
            alpha = np.float32(step / steps)
            position = ((np.float32(1.0) - alpha) * start + alpha * target).astype(np.float32)
            self._publish_robot_command(
                JointServoCommand(
                    joint_position=position,
                    joint_velocity=zeros,
                    feedforward_torque=zeros,
                    kp=self._last_position_kp,
                    kd=self._last_position_kd,
                )
            )
            self._wait_control_period()

    def _publish_damping(self) -> None:
        with self._write_lock:
            for canonical_index, motor_index in enumerate(self._motor_indices):
                motor = self._low_cmd.motor_cmd[int(motor_index)]
                motor.mode = self._motor_modes[canonical_index]
                motor.q = 0.0
                motor.qd = 0.0
                motor.kp = 0.0
                motor.kd = self.config.damping_kd
                motor.tau = 0.0
            self._write_low_cmd_locked()

    def _write_low_cmd_locked(self) -> None:
        self._low_cmd.crc = self._command_crc.Crc(self._low_cmd)
        self._publisher.Write(self._low_cmd)
        self._last_communication_ns = self._clock_ns()

    def _raise_if_remote_stop_requested(self) -> None:
        with self._condition:
            self._raise_if_remote_stop_requested_locked()

    def _raise_if_remote_stop_requested_locked(self) -> None:
        if self._emergency_stop:
            raise EmergencyStopError(f"remote {self.config.emergency_stop_button!r} requested damping stop")
        if self._lie_down_requested:
            raise LieDownRequestedError(f"remote {self.config.lie_down_button!r} requested lie-down shutdown")

    def _require_open(self) -> None:
        if not self._opened or self._closed:
            raise RuntimeError("Unitree backend is not open")


__all__ = ["UnitreeGo2RobotInterface", "UnitreeSdkBindings"]
