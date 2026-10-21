# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree SDK2 DDS adapter for a physical Go2."""

import logging
import threading
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

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
from motrix_deploy.errors import EmergencyStopError, ValidationError
from motrix_deploy.robot.interface import RobotInterface
from motrix_deploy_unitree.config.control import UnitreeControlConfig
from motrix_deploy_unitree.config.go2 import (
    GO2_MOTOR_COUNT,
    UnitreeGo2HardwareCfg,
)
from motrix_deploy_unitree.config.runtime import (
    UNITREE_LOWCMD_TOPIC,
    UNITREE_LOWSTATE_TOPIC,
    UNITREE_SUBSCRIBER_QUEUE_DEPTH,
    UnitreeTransportConfig,
)
from motrix_deploy_unitree.go2.control_mode import (
    check_motion_mode,
    release_official_control,
    require_official_control_released,
)
from motrix_deploy_unitree.remote import UnitreeRemoteGamePadDevice, UnitreeRemoteState, decode_wireless_remote
from motrix_env_core.input import GamePadDevice

_LOGGER = logging.getLogger(__name__)

_POSITION_STOP = 2.146e9
_VELOCITY_STOP = 16000.0


class UnitreeGo2Robot(RobotInterface):
    """Map canonical Go2 state/commands to SDK2 LowState/LowCmd DDS messages."""

    def __init__(
        self,
        transport: UnitreeTransportConfig,
        control: UnitreeControlConfig,
        *,
        spec: RobotSpec,
        control_period_s: float,
        state_timeout_s: float,
    ) -> None:
        self.transport = transport
        self.hardware = UnitreeGo2HardwareCfg()
        self.control = control
        for path, value in (
            ("control.period_s", control_period_s),
            ("control.state_timeout_s", state_timeout_s),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValidationError(path, "a positive finite number", value)
        self._control_period_s = float(control_period_s)
        self._spec = spec
        self._damping_command = JointServoCommand(
            joint_position=np.zeros(spec.joint_count, dtype=np.float32),
            joint_velocity=np.zeros(spec.joint_count, dtype=np.float32),
            feedforward_torque=np.zeros(spec.joint_count, dtype=np.float32),
            kp=np.zeros(spec.joint_count, dtype=np.float32),
            kd=np.full(spec.joint_count, control.damping_kd, dtype=np.float32),
        )
        self._condition = threading.Condition()
        self._state_timeout_s = float(state_timeout_s)
        self._writer_thread: threading.Thread | None = None
        self._writer_shutdown = threading.Event()
        self._writer_wake = False
        self._writer_error: Exception | None = None
        self._pending_command: RobotCommand | None = None
        self._command_time_ns = 0
        self._damping_attempts = 0
        self._publisher: ChannelPublisher | None = None
        self._subscriber: ChannelSubscriber | None = None
        self._low_cmd: LowCmdGo | None = None
        self._crc: CRC | None = None
        self._motor_indices = np.empty(0, dtype=np.int64)
        self._motor_modes: tuple[int, ...] = ()
        self._capabilities: RobotCapabilities | None = None
        self._latest_state: RobotState | None = None
        self._latest_receive_time_ns = 0
        self._remote = UnitreeRemoteState()
        self._gamepad = UnitreeRemoteGamePadDevice(lambda: self.remote)
        self._callback_error: Exception | None = None
        self._emergency_stop = False
        self._health_reason = "backend is not open"
        self._last_communication_ns = 0
        self._opened = False
        self._enabled = False
        self._stopped = False
        self._closed = False
        self._resolve_hardware_bindings(spec)

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

    def acquire_control(self) -> None:
        """Explicitly hand official control over after interactive operator confirmation."""
        ChannelFactoryInitialize(self.transport.domain_id, self.transport.network_interface)
        client = MotionSwitcherClient()
        client.SetTimeout(self.transport.connect_timeout_s)
        client.Init()
        if not check_motion_mode(client):
            return
        print("WARNING: This may stand down the robot and remove built-in balance support.")
        print(
            "Prepare reliable support, clearance and an independent stop. "
            "Official control is not restored automatically."
        )
        try:
            confirmed = input("Type 'yes' to release official motion control: ").strip() == "yes"
        except (KeyboardInterrupt, EOFError) as error:
            raise RuntimeError("Go2 deploy handoff cancelled before StandDown; no LowCmd was sent.") from error
        if not confirmed:
            raise RuntimeError("Go2 deploy handoff cancelled before StandDown; no LowCmd was sent.")

        def observed_down() -> bool:
            print("StandDown was accepted, but that does not prove the movement has finished.")
            try:
                confirmed_down = (
                    input("Observe robot down, motion settled and body supported; type 'down' to release: ").strip()
                    == "down"
                )
                if confirmed_down:
                    print("Down confirmation received; releasing official control...", flush=True)
                return confirmed_down
            except (KeyboardInterrupt, EOFError):
                return False

        if not release_official_control(self.transport.connect_timeout_s, confirm_stand_down=observed_down):
            raise RuntimeError(
                "Go2 deploy handoff cancelled after StandDown; motion mode was not released, "
                "sport_mode was not disabled and no LowCmd was sent."
            )
        print(
            "Official control released and sport_mode disabled; handoff complete. "
            "Starting low-level communication; keep the robot supported.",
            flush=True,
        )

    def open(self) -> None:
        if self._writer_thread is not None and self._writer_thread.is_alive():
            raise RuntimeError("previous Go2 command writer is still running; close it before reopening")
        if self._opened and not self._closed:
            raise RuntimeError("Unitree backend is already open")
        self._resolve_hardware_bindings(self._spec)
        with self._condition:
            self._latest_state = None
            self._latest_receive_time_ns = 0
            self._callback_error = None
            self._pending_command = None
            self._command_time_ns = 0
            self._writer_error = None
            self._damping_attempts = 0
            self._last_communication_ns = 0
        try:
            ChannelFactoryInitialize(self.transport.domain_id, self.transport.network_interface)
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
        require_official_control_released(self.transport.connect_timeout_s)
        self._crc = CRC()
        self._low_cmd = unitree_go_msg_dds__LowCmd_()
        self._initialize_low_cmd()

        self._publisher = ChannelPublisher(UNITREE_LOWCMD_TOPIC, LowCmdGo)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber(UNITREE_LOWSTATE_TOPIC, LowStateGo)
        self._subscriber.Init(self._low_state_handler, UNITREE_SUBSCRIBER_QUEUE_DEPTH)
        self._wait_for_first_state()
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
        self._start_command_writer()
        _LOGGER.info("Go2 low-level communication ready; waiting for Start to prepare, then A after readiness.")

    def _resolve_hardware_bindings(self, spec: RobotSpec) -> None:
        robot = self.hardware
        hardware_spec = robot.to_robot_spec()
        if set(hardware_spec.joint_names) != set(spec.joint_names):
            raise ValidationError("hardware.key_pose", "exactly the canonical joints", hardware_spec.joint_names)
        hardware_indices = {name: index for index, name in enumerate(hardware_spec.joint_names)}
        order = [hardware_indices[name] for name in spec.joint_names]
        self._position_lower = np.maximum(spec.position_lower, hardware_spec.position_lower[order])
        self._position_upper = np.minimum(spec.position_upper, hardware_spec.position_upper[order])
        self._torque_limit = np.minimum(spec.torque_limit, hardware_spec.torque_limit[order])
        if np.any(self._position_lower > self._position_upper):
            raise ValidationError(
                "robot.position_range", "overlapping artifact and hardware limits", "empty intersection"
            )
        if np.any(spec.default_joint_position < self._position_lower) or np.any(
            spec.default_joint_position > self._position_upper
        ):
            raise ValidationError(
                "robot.default_joint_position",
                "inside artifact and hardware position range",
                spec.default_joint_position.tolist(),
            )
        bindings = robot.actuation.motors
        if set(bindings) != set(spec.joint_names):
            raise ValidationError("hardware.actuation.motors", "exactly the canonical joints", set(bindings))
        indices = tuple(bindings[name].index for name in spec.joint_names)
        if len(set(indices)) != len(indices) or any(not 0 <= index < GO2_MOTOR_COUNT for index in indices):
            raise ValidationError("hardware.actuation.motors", "unique Go2 motor indices in [0, 11]", indices)
        self._motor_indices = np.asarray(indices, dtype=np.int64)
        self._motor_modes = tuple(bindings[name].enabled_mode for name in spec.joint_names)

    def enable(self) -> None:
        self._require_open()
        with self._condition:
            self._raise_if_remote_stop_requested_locked()
            self._raise_if_writer_failed_locked()
            if self._stopped:
                raise RuntimeError("Unitree backend is stopped")
            self._enabled = True

    def read_state(self) -> RobotState:
        """Return the latest snapshot without waiting for a new frame.

        open() waits for the first valid frame. ControlSession checks age
        against the original receive timestamp, including on repeated reads.
        """
        self._require_open()
        with self._condition:
            if self._callback_error is not None:
                raise RuntimeError(f"invalid Unitree LowState: {self._callback_error}") from self._callback_error
            self._raise_if_remote_stop_requested_locked()
            self._raise_if_writer_failed_locked()
            state = self._latest_state
        assert state is not None  # Established by open().
        # Keep the validated cache private; consumers may mutate returned arrays.
        return deepcopy(state)

    def write_command(self, command: RobotCommand) -> None:
        self._require_open()
        # Own the submitted arrays: callers may reuse or mutate their command as
        # soon as this method returns. No transport work occurs on this path.
        owned = command.copy()
        if isinstance(owned, JointTorqueCommand):
            owned.validate_for(self._spec, torque_limit=self._torque_limit)
        elif isinstance(owned, JointServoCommand):
            owned.validate_for(
                self._spec,
                position_lower=self._position_lower,
                position_upper=self._position_upper,
                torque_limit=self._torque_limit,
            )
        with self._condition:
            self._raise_if_remote_stop_requested_locked()
            self._check_writer_health_locked(time.monotonic_ns())
            self._raise_if_writer_failed_locked()
            if not self._enabled or self._stopped:
                raise RuntimeError("Unitree command output is not enabled")
            self._pending_command = owned
            self._command_time_ns = time.monotonic_ns()

    def health(self) -> HealthStatus:
        with self._condition:
            emergency = self._emergency_stop
        reason = self._health_reason
        if emergency:
            reason = f"remote {self.control.emergency_stop_button!r} emergency stop is active"
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
        with self._condition:
            if self._stopped:
                return
            self._enabled = False
            self._pending_command = None
            self._stopped = True
            attempts = self._damping_attempts
            self._writer_wake = True
            self._condition.notify_all()
            writer = self._writer_thread
        if writer is None or not writer.is_alive():
            return
        # Waiting here never owns the cache lock. A blocked DDS write cannot be
        # cancelled; close() separately bounds thread joining and preserves channels.
        self._writer_shutdown.wait(self.control.damping_duration_s)
        with self._condition:
            attempted = self._condition.wait_for(
                lambda: self._damping_attempts > attempts or not writer.is_alive(),
                timeout=max(self.transport.connect_timeout_s, 0.1),
            )
            if not attempted or self._damping_attempts <= attempts:
                raise TimeoutError("Go2 writer did not attempt damping before stop timeout")

    def close(self) -> None:
        if self._closed:
            return
        errors: list[Exception] = []
        try:
            self.stop()
        except Exception as error:
            errors.append(error)
        self._writer_shutdown.set()
        with self._condition:
            self._condition.notify_all()
        writer = self._writer_thread
        if writer is not None:
            writer.join(timeout=max(self.transport.connect_timeout_s, 0.1))
            if writer.is_alive():
                error = TimeoutError("Go2 writer is still blocked; DDS channels remain open for a later close retry")
                with self._condition:
                    self._latch_writer_fault_locked(error)
                raise error
            self._writer_thread = None
        for channel in (self._subscriber, self._publisher):
            if channel is None:
                continue
            close = getattr(channel, "Close", None) or getattr(channel, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as error:
                    errors.append(error)
        self._closed = True
        if errors:
            raise RuntimeError(f"failed to close Unitree backend resources: {errors}") from errors[0]

    def _low_state_handler(self, message: LowStateGo) -> None:
        receive_time_ns = time.monotonic_ns()
        try:
            if self._crc.Crc(message) != message.crc:
                return
            remote = decode_wireless_remote(message.wireless_remote)
            state = self._state_from_message(message, receive_time_ns)
        except Exception as error:
            with self._condition:
                self._callback_error = error
                self._writer_wake = True
                self._condition.notify_all()
            return
        with self._condition:
            self._latest_state = state
            self._latest_receive_time_ns = receive_time_ns
            self._remote = remote
            if remote.pressed(self.control.emergency_stop_button):
                self._emergency_stop = True
                self._pending_command = None
                self._writer_wake = True
            self._condition.notify_all()

    def _wait_for_first_state(self) -> RobotState:
        deadline_ns = time.monotonic_ns() + round(self.transport.connect_timeout_s * 1e9)
        with self._condition:
            while self._latest_state is None and self._callback_error is None:
                remaining_s = (deadline_ns - time.monotonic_ns()) / 1e9
                if remaining_s <= 0:
                    raise TimeoutError(
                        f"no valid Unitree LowState received on {UNITREE_LOWSTATE_TOPIC!r} "
                        f"within {self.transport.connect_timeout_s:.3f}s"
                    )
                self._condition.wait(remaining_s)
            if self._callback_error is not None:
                raise RuntimeError(f"invalid Unitree LowState: {self._callback_error}") from self._callback_error
            assert self._latest_state is not None
            return self._latest_state

    def _state_from_message(self, message: LowStateGo, receive_time_ns: int) -> RobotState:
        position = np.asarray([message.motor_state[index].q for index in self._motor_indices], dtype=np.float32)
        velocity = np.asarray([message.motor_state[index].dq for index in self._motor_indices], dtype=np.float32)
        quaternion = np.asarray(message.imu_state.quaternion, dtype=np.float32)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValidationError("state.base_orientation_xyzw", "four finite quaternion values", quaternion)
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-8:
            raise ValidationError("state.base_orientation_xyzw", "a non-zero quaternion", norm)
        quaternion_xyzw = (quaternion[[1, 2, 3, 0]] / np.float32(norm)).astype(np.float32)
        return RobotState(
            sample_time_ns=receive_time_ns,
            receive_time_ns=receive_time_ns,
            joint_position=position,
            joint_velocity=velocity,
            base_orientation_xyzw=quaternion_xyzw,
            base_angular_velocity=np.array(message.imu_state.gyroscope, dtype=np.float32, copy=True),
            base_linear_acceleration=np.array(message.imu_state.accelerometer, dtype=np.float32, copy=True),
        )

    def _initialize_low_cmd(self) -> None:
        self._low_cmd.head[0] = 0xFE
        self._low_cmd.head[1] = 0xEF
        self._low_cmd.level_flag = 0xFF
        self._low_cmd.gpio = 0
        for motor in self._low_cmd.motor_cmd:
            motor.mode = 0
            motor.q = _POSITION_STOP
            motor.dq = _VELOCITY_STOP
            motor.kp = 0.0
            motor.kd = 0.0
            motor.tau = 0.0
        for motor_index, mode in zip(self._motor_indices, self._motor_modes):
            self._low_cmd.motor_cmd[int(motor_index)].mode = mode

    def _start_command_writer(self) -> None:
        self._writer_shutdown.clear()
        self._writer_wake = False
        self._writer_thread = threading.Thread(target=self._command_writer_loop, name="go2-lowcmd", daemon=True)
        self._writer_thread.start()

    def _latch_writer_fault_locked(self, error: Exception) -> None:
        if self._writer_error is None:
            self._writer_error = error
            self._health_reason = str(error)
        self._enabled = False
        self._pending_command = None

    def _raise_if_writer_failed_locked(self) -> None:
        if self._writer_error is not None:
            if isinstance(self._writer_error, TimeoutError):
                raise TimeoutError(str(self._writer_error)) from self._writer_error
            raise RuntimeError(f"Go2 command writer failed: {self._writer_error}") from self._writer_error

    def _check_writer_health_locked(self, now_ns: int) -> None:
        if not self._enabled or self._stopped or self._emergency_stop:
            return
        if self._callback_error is not None:
            self._latch_writer_fault_locked(RuntimeError(f"invalid Unitree LowState: {self._callback_error}"))
        elif (now_ns - self._latest_receive_time_ns) / 1e9 > self._state_timeout_s:
            self._latch_writer_fault_locked(TimeoutError("Go2 LowState expired during command publishing"))
        elif (
            self._pending_command is not None
            and (now_ns - self._command_time_ns) / 1e9 > self.control.command_timeout_s
        ):
            self._latch_writer_fault_locked(TimeoutError("Go2 command expired during command publishing"))

    def _command_writer_loop(self) -> None:
        period_ns = max(1, round(self.control.command_publish_period_s * 1e9))
        next_ns = time.monotonic_ns()
        while True:
            with self._condition:
                while not self._writer_shutdown.is_set() and not self._writer_wake:
                    remaining_s = (next_ns - time.monotonic_ns()) / 1e9
                    if remaining_s <= 0:
                        break
                    self._condition.wait(remaining_s)
                shutting_down = self._writer_shutdown.is_set()
                self._writer_wake = False
            # A shutdown requested during an in-flight active write still gets a
            # damping attempt after that write returns, before the thread exits.
            self._publish_pending_once()
            if shutting_down:
                return
            next_ns += period_ns
            now_ns = time.monotonic_ns()
            if next_ns <= now_ns:
                # Missed deadlines are skipped, never replayed as catch-up bursts.
                next_ns = now_ns + period_ns

    def _publish_pending_once(self) -> None:
        """Publish one snapshot; called only by the periodic writer."""
        with self._condition:
            self._check_writer_health_locked(time.monotonic_ns())
            command = (
                self._pending_command
                if self._enabled and not self._stopped and not self._emergency_stop and self._writer_error is None
                else None
            )
        try:
            # Internal damping bypasses position checks: kp=0 makes q irrelevant.
            self._publish_robot_command(self._damping_command if command is None else command)
        except Exception as error:
            with self._condition:
                self._latch_writer_fault_locked(error)
        finally:
            if command is None:
                with self._condition:
                    self._damping_attempts += 1
                    self._condition.notify_all()

    def _publish_robot_command(self, command: RobotCommand) -> None:
        # Only the periodic writer owns the mutable DDS command message.
        for canonical_index, motor_index in enumerate(self._motor_indices):
            motor = self._low_cmd.motor_cmd[int(motor_index)]
            motor.mode = self._motor_modes[canonical_index]
            if isinstance(command, JointTorqueCommand):
                motor.q = motor.dq = motor.kp = motor.kd = 0.0
                motor.tau = float(command.torque[canonical_index])
            else:
                motor.q = float(command.joint_position[canonical_index])
                motor.dq = float(command.joint_velocity[canonical_index])
                motor.kp = float(command.kp[canonical_index])
                motor.kd = float(command.kd[canonical_index])
                motor.tau = float(command.feedforward_torque[canonical_index])
        self._write_low_cmd()

    def _write_low_cmd(self) -> None:
        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        if not self._publisher.Write(self._low_cmd):
            raise RuntimeError("failed to publish Unitree LowCmd")
        with self._condition:
            self._last_communication_ns = time.monotonic_ns()

    def _raise_if_remote_stop_requested_locked(self) -> None:
        if self._emergency_stop:
            raise EmergencyStopError(f"remote {self.control.emergency_stop_button!r} requested damping stop")

    def _require_open(self) -> None:
        if not self._opened or self._closed:
            raise RuntimeError("Unitree backend is not open")


__all__ = ["UnitreeGo2Robot"]
