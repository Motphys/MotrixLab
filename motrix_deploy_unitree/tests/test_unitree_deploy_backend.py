# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree Go2 backend contract tests with an injected SDK2 transport."""

import copy
import struct
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from motrix_deploy.artifact import ControlSpec
from motrix_deploy.contracts import JointServoCommand, JointTorqueCommand, RobotSpec
from motrix_deploy.errors import EmergencyStopError, ValidationError
from motrix_deploy.runtime.factory import RuntimeCreateContext, create_hardware_runtime
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_deploy_unitree.cli.app import _parser, main
from motrix_deploy_unitree.config.control import UnitreeControlConfig
from motrix_deploy_unitree.config.go2 import UnitreeGo2HardwareCfg
from motrix_deploy_unitree.config.runtime import UnitreeRuntimeConfig, UnitreeTransportConfig
from motrix_deploy_unitree.go2.robot import UnitreeGo2Robot
from motrix_deploy_unitree.plugin import create_runtime
from motrix_deploy_unitree.remote import UnitreeRemoteGamePadDevice, decode_wireless_remote

JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
MOTOR_MAPPING = {
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


class _Motor:
    # Match SDK fields strictly: misspellings must not become ignored attributes.
    __slots__ = ("mode", "q", "dq", "kp", "kd", "tau")

    def __init__(self, index: int = 0) -> None:
        self.mode = 0
        self.q = float(index)
        self.dq = float(-index)
        self.kp = 0.0
        self.kd = 0.0
        self.tau = 0.0


class _LowCmd:
    def __init__(self) -> None:
        self.head = [0, 0]
        self.level_flag = 0
        self.gpio = 0
        self.motor_cmd = [_Motor() for _ in range(12)]
        self.crc = 0


class _LowState:
    def __init__(self) -> None:
        self.motor_state = [_Motor(index) for index in range(12)]
        self.imu_state = SimpleNamespace(
            quaternion=[1.0, 0.0, 0.0, 0.0],
            gyroscope=[0.1, 0.2, 0.3],
            accelerometer=[0.0, 0.0, 9.81],
        )
        self.wireless_remote = _remote_payload()
        self.crc = 123


class _CRC:
    def Crc(self, message: object) -> int:
        del message
        return 123


class _MotionSwitcherClient:
    def __init__(self, bus: "_Bus") -> None:
        self.bus = bus

    def SetTimeout(self, timeout_s: float) -> None:
        self.bus.service_calls.append(("motion_timeout", timeout_s))

    def Init(self) -> None:
        self.bus.service_calls.append(("motion_init", None))

    def CheckMode(self):
        self.bus.service_calls.append(("check_mode", self.bus.motion_mode))
        return 0, {"name": self.bus.motion_mode}

    def ReleaseMode(self):
        self.bus.service_calls.append(("release_mode", self.bus.motion_mode))
        self.bus.motion_mode = ""
        return 0, None


class _SportClient:
    def __init__(self, bus: "_Bus") -> None:
        self.bus = bus

    def SetTimeout(self, timeout_s: float) -> None:
        self.bus.service_calls.append(("sport_timeout", timeout_s))

    def Init(self) -> None:
        self.bus.service_calls.append(("sport_init", None))

    def StandDown(self) -> int:
        self.bus.service_calls.append(("stand_down", None))
        return self.bus.stand_down_status


class _RobotStateClient:
    def __init__(self, bus: "_Bus") -> None:
        self.bus = bus

    def SetTimeout(self, timeout_s: float) -> None:
        self.bus.service_calls.append(("robot_state_timeout", timeout_s))

    def Init(self) -> None:
        self.bus.service_calls.append(("robot_state_init", None))

    def ServiceSwitch(self, name: str, enabled: bool) -> int:
        self.bus.service_calls.append(("service_switch", (name, enabled)))
        return self.bus.service_switch_status


class _Publisher:
    def __init__(self, bus: "_Bus", topic: str, message_type: type) -> None:
        self.bus = bus
        self.topic = topic
        self.message_type = message_type
        self.initialized = False
        self.closed = False

    def Init(self) -> None:
        self.initialized = True

    def Write(self, message: _LowCmd) -> bool:
        snapshot = copy.deepcopy(message)
        with self.bus.command_ready:
            self.bus.commands.append(snapshot)
            self.bus.command_ready.notify_all()
        write_count = len(self.bus.commands)
        if write_count in self.bus.remote_keys_after_writes:
            self.bus.state.wireless_remote = _remote_payload(self.bus.remote_keys_after_writes[write_count])
        for motor_index, motor in enumerate(snapshot.motor_cmd):
            if motor.kp > 0:
                self.bus.state.motor_state[motor_index].q = motor.q
                self.bus.state.motor_state[motor_index].dq = motor.dq
        if self.bus.emit_on_write:
            self.bus.emit()
        return True

    def Close(self) -> None:
        self.closed = True


class _Subscriber:
    def __init__(self, bus: "_Bus", topic: str, message_type: type) -> None:
        self.bus = bus
        self.topic = topic
        self.message_type = message_type
        self.closed = False

    def Init(self, callback, queue_depth: int) -> None:
        self.bus.callback = callback
        self.bus.queue_depth = queue_depth
        if self.bus.emit_on_write:
            self.bus.emit()
        return True

    def Close(self) -> None:
        self.closed = True


class _Bus:
    def __init__(self) -> None:
        self.state = _LowState()
        self.commands: list[_LowCmd] = []
        self.command_ready = threading.Condition()
        self.emit_on_write = True
        self.callback = None
        self.queue_depth = 0
        self.factory_calls: list[tuple[int, str]] = []
        self.remote_keys_after_writes: dict[int, int] = {}
        self.motion_mode = ""
        self.stand_down_status = 0
        self.service_switch_status = 0
        self.service_calls: list[tuple[str, object]] = []

    def wait_for(self, predicate, *, after=0):
        with self.command_ready:
            assert self.command_ready.wait_for(
                lambda: any(predicate(command) for command in self.commands[after:]), timeout=2
            ), "writer did not publish the expected command"
            return next(command for command in reversed(self.commands[after:]) if predicate(command))

    def emit(self) -> None:
        if self.callback is not None:
            self.callback(copy.deepcopy(self.state))

    def install(self, monkeypatch):
        """Replace native SDK symbols without constructing DDS channels."""
        for module in ("go2.robot", "cli.release_official_control"):
            monkeypatch.setattr(
                f"motrix_deploy_unitree.{module}.ChannelFactoryInitialize",
                lambda domain, interface: self.factory_calls.append((domain, interface)),
            )
        monkeypatch.setattr(
            "motrix_deploy_unitree.go2.robot.time",
            SimpleNamespace(monotonic_ns=time.monotonic_ns, sleep=lambda _: None),
        )
        symbols = {
            "ChannelPublisher": lambda topic, message_type: _Publisher(self, topic, message_type),
            "ChannelSubscriber": lambda topic, message_type: _Subscriber(self, topic, message_type),
            "LowCmdGo": _LowCmd,
            "LowStateGo": _LowState,
            "unitree_go_msg_dds__LowCmd_": _LowCmd,
            "CRC": _CRC,
        }
        for name, value in symbols.items():
            monkeypatch.setattr(f"motrix_deploy_unitree.go2.robot.{name}", value)
        monkeypatch.setattr("motrix_deploy_unitree.go2.robot.MotionSwitcherClient", lambda: _MotionSwitcherClient(self))
        for name, client in (
            ("MotionSwitcherClient", _MotionSwitcherClient),
            ("SportClient", _SportClient),
            ("RobotStateClient", _RobotStateClient),
        ):
            monkeypatch.setattr(f"motrix_deploy_unitree.go2.control_mode.{name}", lambda client=client: client(self))
        return self


@pytest.fixture(autouse=True)
def _close_opened_robots(monkeypatch):
    opened = []
    original = UnitreeGo2Robot.open

    def open_robot(robot):
        result = original(robot)
        opened.append(robot)
        return result

    monkeypatch.setattr(UnitreeGo2Robot, "open", open_robot)
    yield
    for robot in reversed(opened):
        robot.close()


def _is_damping(message):
    return all(motor.kp == 0 and motor.kd == 8 for motor in message.motor_cmd)


def _servo_published(bus, command, joint_names=JOINT_NAMES):
    return bus.wait_for(
        lambda message: all(
            message.motor_cmd[MOTOR_MAPPING[name]].q == float(command.joint_position[index])
            and message.motor_cmd[MOTOR_MAPPING[name]].kp == float(command.kp[index])
            for index, name in enumerate(joint_names)
        )
    )


def _remote_payload(keys: int = 0, *, lx: float = 0.0, ly: float = 0.0, rx: float = 0.0, ry: float = 0.0) -> bytes:
    payload = bytearray(40)
    struct.pack_into("<H", payload, 2, keys)
    struct.pack_into("<f", payload, 4, lx)
    struct.pack_into("<f", payload, 8, rx)
    struct.pack_into("<f", payload, 12, ry)
    struct.pack_into("<f", payload, 20, ly)
    return bytes(payload)


def _button(name: str) -> int:
    names = ("R1", "L1", "start", "select", "R2", "L2", "F1", "F2", "A", "B", "X", "Y")
    return 1 << names.index(name)


def _spec() -> RobotSpec:
    return UnitreeGo2HardwareCfg().to_robot_spec()


def _config(**overrides) -> UnitreeRuntimeConfig:
    control = UnitreeControlConfig(damping_duration_s=overrides.pop("damping_duration_s", 0.0))
    return UnitreeRuntimeConfig(
        transport=UnitreeTransportConfig(network_interface=overrides.pop("network_interface", "enp3s0"), **overrides),
        robot="go2",
        control=control,
    )


def _backend(
    config: UnitreeRuntimeConfig,
    bus: _Bus,
) -> UnitreeGo2Robot:
    return UnitreeGo2Robot(
        config.transport,
        control=config.control,
        spec=_spec(),
        control_period_s=0.02,
        state_timeout_s=0.1,
    )


def _command(position: np.ndarray) -> JointServoCommand:
    zeros = np.zeros(12, dtype=np.float32)
    return JointServoCommand(
        joint_position=np.asarray(position, dtype=np.float32),
        joint_velocity=zeros,
        feedforward_torque=zeros,
        kp=np.full(12, 35.0, dtype=np.float32),
        kd=np.full(12, 0.5, dtype=np.float32),
    )


def test_unitree_interface_provides_remote_gamepad_device(monkeypatch) -> None:
    backend = _backend(_config(), _Bus().install(monkeypatch))
    assert backend.get_gamepad_device() is backend.get_gamepad_device()


def test_remote_payload_decoding_uses_explicit_button_and_axis_layout() -> None:
    remote = decode_wireless_remote(_remote_payload(_button("start") | _button("A"), lx=0.25, ly=-0.5, rx=0.75))

    assert remote.pressed("start")
    assert remote.pressed("A")
    assert not remote.pressed("select")
    assert remote.lx == pytest.approx(0.25)
    assert remote.ly == pytest.approx(-0.5)
    assert remote.rx == pytest.approx(0.75)


def test_remote_gamepad_adapter_exposes_axes_and_button_edges() -> None:
    states = [
        decode_wireless_remote(_remote_payload(lx=0.25, ly=-0.5, rx=0.75)),
        decode_wireless_remote(_remote_payload(_button("L1"), lx=0.5, ly=0.25, rx=-0.5)),
    ]
    device = UnitreeRemoteGamePadDevice(lambda: states.pop(0))
    device.poll()
    assert device.axis_value("ly") == pytest.approx(-0.5)
    assert not device.is_button_pressing("L1")
    device.poll()
    assert device.axis_value("ly") == pytest.approx(0.25)
    assert device.is_button_down("L1")
    assert device.is_button_pressing("L1")
    assert not device.is_button_up("L1")


def test_open_only_checks_released_ownership(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    backend = _backend(_config(), bus)
    backend.open()
    assert [name for name, _ in bus.service_calls] == ["motion_timeout", "motion_init", "check_mode"]
    bus.wait_for(_is_damping)
    backend.close()


def test_active_official_control_aborts_before_lowcmd_channels(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    bus.motion_mode = "sport"
    backend = _backend(_config(), bus)
    with pytest.raises(RuntimeError, match="release-control"):
        backend.open()
    assert bus.motion_mode == "sport"
    assert bus.callback is None
    assert bus.commands == []
    assert not any(name in {"stand_down", "release_mode", "service_switch"} for name, _ in bus.service_calls)


@pytest.mark.parametrize("confirmed", [False, True])
def test_release_waits_for_observed_stand_down_before_releasing(confirmed, monkeypatch) -> None:
    from motrix_deploy_unitree.go2.control_mode import release_official_control

    bus = _Bus().install(monkeypatch)
    bus.motion_mode = "sport"

    def observe():
        assert ("stand_down", None) in bus.service_calls
        assert bus.motion_mode == "sport"
        assert not any(name == "service_switch" for name, _ in bus.service_calls)
        return confirmed

    assert release_official_control(5.0, confirm_stand_down=observe) == confirmed
    assert bus.motion_mode == ("" if confirmed else "sport")
    assert (("service_switch", ("sport_mode", False)) in bus.service_calls) == confirmed
    assert bus.commands == []


@pytest.mark.parametrize("failure", ["stand_down", "service_switch"])
def test_release_propagates_rpc_failure(failure, monkeypatch) -> None:
    from motrix_deploy_unitree.go2.control_mode import release_official_control

    bus = _Bus().install(monkeypatch)
    bus.motion_mode = "sport"
    if failure == "stand_down":
        bus.stand_down_status = 42
    else:
        bus.service_switch_status = 42
    with pytest.raises(RuntimeError, match="42"):
        release_official_control(5.0, confirm_stand_down=lambda: True)
    if failure == "stand_down":
        assert bus.motion_mode == "sport"
        assert not any(name == "service_switch" for name, _ in bus.service_calls)
    assert bus.commands == []


@pytest.mark.parametrize("status,mode", [(42, {"name": ""}), (0, None), (0, {}), (0, {"name": 0})])
def test_ownership_query_failure_never_opens_command_channel(status, mode, monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    backend = _backend(_config(), bus)
    client = _MotionSwitcherClient(bus)
    client.CheckMode = lambda: (status, mode)
    monkeypatch.setattr("motrix_deploy_unitree.go2.control_mode.MotionSwitcherClient", lambda: client)
    with pytest.raises(RuntimeError, match="CheckMode|invalid mode"):
        backend.open()
    assert bus.callback is None
    assert bus.commands == []


@pytest.mark.parametrize("answer,expected", [("down", 0), ("no", 1)])
def test_release_cli_requires_second_confirmation_after_stand_down(monkeypatch, answer, expected) -> None:
    bus = _Bus().install(monkeypatch)
    bus.motion_mode = "sport"
    answers = iter(["yes", answer])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert main(["release-control", "eth0", "--timeout-s", "2"]) == expected
    assert bus.factory_calls == [(0, "eth0")]
    assert bus.motion_mode == ("" if answer == "down" else "sport")
    assert bus.commands == []


def test_release_cli_cancel_does_not_initialize_sdk(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _: "no")
    monkeypatch.setattr(
        "motrix_deploy_unitree.cli.release_official_control.ChannelFactoryInitialize",
        lambda *args: pytest.fail("must not initialize DDS"),
    )
    assert main(["release-control", "eth0"]) == 0


def test_lowstate_and_lowcmd_are_mapped_by_canonical_joint_name(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    spec = _spec()
    backend = _backend(_config(), bus)
    backend.open()

    state = backend.read_state()
    expected_position = np.asarray([MOTOR_MAPPING[name] for name in JOINT_NAMES], dtype=np.float32)
    np.testing.assert_array_equal(state.joint_position, expected_position)
    np.testing.assert_array_equal(state.base_orientation_xyzw, [0.0, 0.0, 0.0, 1.0])

    backend.enable()
    target = spec.default_joint_position + np.linspace(-0.01, 0.01, 12, dtype=np.float32)
    command = _command(target)
    backend.write_command(command)
    published = _servo_published(bus, command)

    for canonical_index, name in enumerate(JOINT_NAMES):
        motor = published.motor_cmd[MOTOR_MAPPING[name]]
        assert motor.q == pytest.approx(command.joint_position[canonical_index])
        assert motor.dq == pytest.approx(0.0)
        assert motor.kp == pytest.approx(35.0)
        assert motor.kd == pytest.approx(0.5)
    assert published.crc == 123

    backend.stop()
    bus.wait_for(_is_damping, after=1)
    for motor_index in MOTOR_MAPPING.values():
        motor = bus.commands[-1].motor_cmd[motor_index]
        assert motor.kp == 0.0
        assert motor.kd == 8.0
        assert motor.dq == 0.0
        assert motor.tau == 0.0
    backend.close()
    assert not backend.health().healthy
    assert backend.health().reason == "backend is closed"


@pytest.mark.parametrize("mode", ["servo", "torque"])
def test_enable_requires_explicit_command_and_commands_preserve_requested_fields(mode: str, monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    backend = _backend(_config(), bus)
    backend.open()
    with pytest.raises(RuntimeError, match="not enabled"):
        backend.write_command(JointTorqueCommand(np.zeros(12, dtype=np.float32)))
    backend.enable()
    if mode == "servo":
        command = _command(_spec().default_joint_position)
        command.kp = np.arange(12, dtype=np.float32) + 1
        command.kd = np.arange(12, dtype=np.float32) / 10
        command.joint_velocity = np.linspace(-0.1, 0.1, 12, dtype=np.float32)
        command.feedforward_torque = np.linspace(-1, 1, 12, dtype=np.float32)
        expected = (command.joint_position, command.joint_velocity, command.kp, command.kd, command.feedforward_torque)
    else:
        command = JointTorqueCommand(np.linspace(-1, 1, 12, dtype=np.float32))
        zeros = np.zeros(12, dtype=np.float32)
        expected = (zeros, zeros, zeros, zeros, command.torque)
    backend.write_command(command)
    if mode == "servo":
        published = _servo_published(bus, command)
    else:
        published = bus.wait_for(
            lambda msg: all(
                msg.motor_cmd[MOTOR_MAPPING[name]].tau == float(command.torque[index])
                for index, name in enumerate(JOINT_NAMES)
            )
        )
    actual = [published.motor_cmd[MOTOR_MAPPING[name]] for name in JOINT_NAMES]
    for field, values in zip(("q", "dq", "kp", "kd", "tau"), expected):
        np.testing.assert_array_equal([getattr(motor, field) for motor in actual], values)
    backend.close()


def test_failed_publish_latches_fault_without_updating_communication_time(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    failed = threading.Event()

    def fail_write(self, message):
        failed.set()
        return False

    monkeypatch.setattr(_Publisher, "Write", fail_write)
    backend = _backend(_config(), bus)
    backend.open()
    assert failed.wait(1)
    with backend._condition:
        assert backend._condition.wait_for(lambda: backend._writer_error is not None, timeout=1)
    assert not backend.health().healthy
    assert backend.health().last_successful_communication_ns == 0
    with pytest.raises(RuntimeError, match="failed to publish Unitree LowCmd"):
        backend.enable()
    backend.close()


@pytest.mark.integration
def test_velocity_survives_sdk_lowcmd_serialization_without_dds(monkeypatch) -> None:
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

    bus = _Bus().install(monkeypatch)
    backend = _backend(_config(damping_duration_s=0.02), bus)
    monkeypatch.setattr("motrix_deploy_unitree.go2.robot.unitree_go_msg_dds__LowCmd_", unitree_go_msg_dds__LowCmd_)
    backend.open()
    message_type = type(backend._low_cmd)

    def wire_message():
        return message_type.deserialize(backend._low_cmd.serialize())

    backend.enable()
    command = _command(_spec().default_joint_position)
    command.joint_velocity = np.linspace(-0.6, 0.6, 12, dtype=np.float32)
    backend.write_command(command)
    _servo_published(bus, command)
    decoded = wire_message()
    for index, name in enumerate(JOINT_NAMES):
        assert decoded.motor_cmd[MOTOR_MAPPING[name]].dq == float(command.joint_velocity[index])

    backend.write_command(JointTorqueCommand(np.ones(12, dtype=np.float32)))
    bus.wait_for(
        lambda msg: all(
            msg.motor_cmd[index].tau == 1.0 and msg.motor_cmd[index].dq == 0 for index in MOTOR_MAPPING.values()
        )
    )
    backend.write_command(command)
    backend.stop()
    decoded = wire_message()
    for index in MOTOR_MAPPING.values():
        assert decoded.motor_cmd[index].dq == 0.0
        assert decoded.motor_cmd[index].kp == 0.0
        assert decoded.motor_cmd[index].kd == backend.control.damping_kd
    backend.close()


@pytest.mark.parametrize("button", ["start", "A", "B"])
def test_task_buttons_are_available_without_stopping_backend(button: str, monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    backend = _backend(_config(), bus)
    backend.open()
    backend.enable()
    bus.state.wireless_remote = _remote_payload(_button(button))
    bus.emit()
    backend.read_state()
    device = backend.get_gamepad_device()
    device.poll()
    assert device.is_button_pressing(button)
    backend.write_command(_command(_spec().default_joint_position))
    assert backend.health().healthy
    backend.close()


@pytest.mark.parametrize("field", ["joint_position", "feedforward_torque", "torque"])
@pytest.mark.parametrize("widen_artifact", [False, True])
def test_command_limits_intersect_hardware_by_name_before_open(field: str, widen_artifact: bool, monkeypatch) -> None:
    hardware = UnitreeGo2HardwareCfg()
    native = hardware.to_robot_spec()
    order = np.arange(12)[::-1]
    factor = 2 if widen_artifact else 0.5
    # Reverse artifact names and use a valid center pose, independently scaling
    # each interval about that center to test both sides of the intersection.
    center = native.default_joint_position[order]
    lower = center + factor * (native.position_lower[order] - center)
    upper = center + factor * (native.position_upper[order] - center)
    spec = replace(
        native,
        joint_names=tuple(native.joint_names[i] for i in order),
        default_joint_position=center.copy(),
        position_lower=lower,
        position_upper=upper,
        torque_limit=native.torque_limit[order] * factor,
    )
    bus = _Bus().install(monkeypatch)
    backend = UnitreeGo2Robot(
        _config().transport, UnitreeControlConfig(), control_period_s=0.02, state_timeout_s=0.1, spec=spec
    )
    assert backend.spec is spec
    bounds = dict(
        position_lower=backend._position_lower,
        position_upper=backend._position_upper,
        torque_limit=backend._torque_limit,
    )
    _command(center.copy()).validate_for(spec, **bounds)
    for index in range(12):
        if field == "torque":
            command = JointTorqueCommand(np.zeros(12, dtype=np.float32))
        else:
            command = _command(center.copy())
        values = getattr(command, field)
        if field == "joint_position":
            values[index] = min(upper[index], native.position_upper[order[index]]) + np.float32(0.01)
        else:
            values[index] = min(spec.torque_limit[index], native.torque_limit[order[index]]) + np.float32(0.01)
        with pytest.raises(ValidationError, match=f"command.{field}"):
            if isinstance(command, JointTorqueCommand):
                command.validate_for(spec, torque_limit=backend._torque_limit)
            else:
                command.validate_for(spec, **bounds)
    assert bus.factory_calls == []


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_measured_pose_outside_hardware_bounds_cannot_be_commanded_even_with_zero_gains(side, monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    robot = _backend(_config(), bus)
    for name, position in zip(robot.spec.joint_names, robot.spec.default_joint_position):
        bus.state.motor_state[MOTOR_MAPPING[name]].q = float(position)
    joint = robot.spec.joint_names[2]
    boundary = getattr(robot.spec, f"position_{side}")[2]
    bus.state.motor_state[MOTOR_MAPPING[joint]].q = float(boundary + (-0.01 if side == "lower" else 0.01))
    robot.open()
    try:
        robot.enable()
        measured = robot.read_state()
        command = _command(measured.joint_position)
        command.kp.fill(0)
        command.kd.fill(0)
        with pytest.raises(ValidationError, match=joint):
            robot.write_command(command)
        assert not any(any(motor.kp > 0 for motor in msg.motor_cmd) for msg in bus.commands)
    finally:
        robot.close()


def test_default_pose_outside_hardware_limits_is_rejected_before_connection(monkeypatch) -> None:
    hardware = UnitreeGo2HardwareCfg()
    native = hardware.to_robot_spec()
    position = native.default_joint_position.copy()
    upper = native.position_upper.copy()
    upper[0] += np.float32(0.2)
    position[0] = native.position_upper[0] + np.float32(0.1)
    # Valid for the artifact alone, but not for the selected physical hardware.
    spec = replace(native, default_joint_position=position, position_upper=upper)
    bus = _Bus().install(monkeypatch)
    with pytest.raises(ValidationError, match="robot.default_joint_position"):
        UnitreeGo2Robot(
            _config().transport, UnitreeControlConfig(), control_period_s=0.02, state_timeout_s=0.1, spec=spec
        )
    np.testing.assert_array_equal(spec.default_joint_position, position)
    np.testing.assert_array_equal(spec.position_upper, upper)
    assert bus.factory_calls == []
    assert bus.commands == []


def test_cached_state_reads_do_not_wait_or_refresh_timestamp(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    backend = _backend(_config(), bus)
    monkeypatch.setattr(backend, "_start_command_writer", lambda: None)
    backend.open()
    first = backend.read_state()
    # Advancing the host clock must not make an old sample appear newly received.
    monkeypatch.setattr(
        "motrix_deploy_unitree.go2.robot.time.monotonic_ns", lambda: first.receive_time_ns + 1_000_000_000
    )
    cached = backend.read_state()
    assert cached.receive_time_ns == first.receive_time_ns
    assert cached.sample_time_ns == first.sample_time_ns
    np.testing.assert_array_equal(cached.joint_position, first.joint_position)
    bus.state.motor_state[3].q += 0.1
    bus.emit()
    updated = backend.read_state()
    assert updated.receive_time_ns > cached.receive_time_ns
    assert updated.joint_position[0] != cached.joint_position[0]
    np.testing.assert_array_equal(first.joint_position, cached.joint_position)
    backend.close()


def test_lowstate_crc_rejects_corruption_without_refreshing_state_and_accepts_recovery(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    monkeypatch.setattr(UnitreeGo2Robot, "_start_command_writer", lambda self: None)
    backend = _backend(_config(), bus)
    backend.open()
    first = backend.read_state()
    bus.state.crc = 0
    bus.state.motor_state[3].q += 0.1
    bus.emit()
    cached = backend.read_state()
    np.testing.assert_array_equal(cached.joint_position, first.joint_position)
    assert cached.receive_time_ns == first.receive_time_ns
    bus.state.crc = 123
    bus.emit()
    recovered = backend.read_state()
    assert recovered.joint_position[0] != first.joint_position[0]
    assert recovered.receive_time_ns > first.receive_time_ns
    backend.close()


def test_invalid_state_cleanup_publishes_only_damping_and_closes_channels(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    backend = _backend(_config(damping_duration_s=0.02), bus)
    backend.open()
    backend.enable()
    backend.read_state()
    bus.state.imu_state.gyroscope = [float("nan")] * 3
    bus.emit()
    with pytest.raises(RuntimeError, match="invalid Unitree LowState"):
        backend.read_state()
    backend.close()
    backend.close()
    assert bus.commands
    for motor in bus.commands[-1].motor_cmd:
        assert (motor.kp, motor.kd, motor.tau) == (0.0, backend.control.damping_kd, 0.0)
    assert backend._subscriber.closed and backend._publisher.closed


def test_select_remote_button_raises_emergency_stop_and_stop_publishes_damping(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    backend = _backend(
        _config(damping_duration_s=0.02),
        bus,
    )
    backend.open()
    backend.read_state()
    backend.enable()
    commands_before_stop = len(bus.commands)

    bus.state.wireless_remote = _remote_payload(_button("select"))
    bus.emit()

    with pytest.raises(EmergencyStopError, match="requested damping stop"):
        backend.read_state()
    assert not backend.health().healthy
    bus.state.wireless_remote = _remote_payload()
    bus.emit()
    with pytest.raises(EmergencyStopError):
        backend.enable()
    with pytest.raises(EmergencyStopError):
        backend.write_command(_command(_spec().default_joint_position))

    backend.stop()
    assert len(bus.commands) > commands_before_stop
    assert all(bus.commands[-1].motor_cmd[index].kd == 8.0 for index in MOTOR_MAPPING.values())
    backend.close()


def test_trusted_recipe_instantiates_robot_config() -> None:
    config = OmegaConf.to_object(OmegaConf.structured(_config()))
    runtime = create_runtime(config, _context())
    assert isinstance(runtime, HardwareRuntime)
    assert isinstance(runtime.robot, UnitreeGo2Robot)
    assert isinstance(runtime.robot.hardware, UnitreeGo2HardwareCfg)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quaternion", [1.0, 0.0, 0.0]),
        ("quaternion", [float("nan"), 0.0, 0.0, 0.0]),
        ("gyroscope", [0.0, 0.0]),
        ("accelerometer", [0.0, 0.0, float("inf")]),
    ],
)
def test_invalid_sdk_sensor_shape_or_finiteness_aborts_open(field: str, value: list[float], monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    setattr(bus.state.imu_state, field, value)
    with pytest.raises(RuntimeError, match="invalid Unitree LowState"):
        _backend(_config(), bus).open()
    assert bus.commands == []


def test_zero_quaternion_cannot_establish_initial_valid_state(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    bus.state.imu_state.quaternion = [0.0] * 4
    backend = _backend(_config(), bus)
    try:
        with pytest.raises(RuntimeError, match="non-zero quaternion"):
            backend.open()
        assert not bus.commands
    finally:
        backend.close()


def test_cached_state_is_isolated_from_message_and_returned_array_mutations(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    monkeypatch.setattr(UnitreeGo2Robot, "_start_command_writer", lambda self: None)
    backend = _backend(_config(), bus)
    backend.open()
    first = backend.read_state()
    expected = first.joint_position.copy()
    first.joint_position.fill(100)
    bus.state.motor_state[3].q = 200
    second = backend.read_state()
    np.testing.assert_array_equal(second.joint_position, expected)
    assert second.receive_time_ns == first.receive_time_ns


def _context() -> RuntimeCreateContext:
    return RuntimeCreateContext(robot=_spec(), control=ControlSpec(period_s=0.02, state_timeout_s=0.1))


def test_open_reports_available_interfaces_when_dds_interface_is_invalid(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    backend = _backend(_config(network_interface="missing0"), bus)

    def fail_factory(domain_id, network_interface):
        raise RuntimeError(f"{network_interface}: does not match an available interface")

    monkeypatch.setattr("motrix_deploy_unitree.go2.robot.ChannelFactoryInitialize", fail_factory)
    with pytest.raises(RuntimeError, match="Available interfaces:.*runtime.transport.network_interface"):
        backend.open()


def test_plugin_factory_creates_robot_without_sdk_or_operator_actions(monkeypatch) -> None:

    bus = _Bus().install(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("factory must not prompt"))
    config = _config()
    context = _context()

    runtime = create_runtime(config, context)
    assert isinstance(runtime, HardwareRuntime)
    robot = runtime.robot
    assert isinstance(robot, UnitreeGo2Robot)
    assert robot.transport is config.transport
    assert isinstance(robot.hardware, UnitreeGo2HardwareCfg)
    assert robot.spec is context.robot
    assert runtime.robot is robot
    assert runtime.realtime is True
    assert runtime.gamepad is robot.get_gamepad_device()
    assert not runtime.opened
    assert bus.factory_calls == []
    assert bus.service_calls == []
    assert bus.commands == []


def _handoff_runtime(config):
    runtime = create_runtime(config, _context())
    runtime.realtime = False
    return runtime


def test_go2_runtime_handoff_precedes_lowlevel_session_start(monkeypatch, capsys) -> None:
    from motrix_deploy.runtime.control import ControlSession
    from motrix_deploy.runtime.lifecycle import ControllerStep

    bus = _Bus().install(monkeypatch)
    bus.motion_mode = "sport"
    answers = iter(["yes", "down"])

    def answer(prompt):
        response = next(answers)
        bus.service_calls.append(("confirm", response))
        return response

    class CompleteController:
        def reset(self, state):
            pass

        def step(self, state, context):
            return ControllerStep(complete=True)

        def request_stop(self):
            pass

    original_open = UnitreeGo2Robot.open

    def open_robot(robot):
        assert bus.motion_mode == ""
        bus.service_calls.append(("lowlevel_start", None))
        original_open(robot)

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr(UnitreeGo2Robot, "open", open_robot)
    config = _config(domain_id=23, network_interface="robot-test0")
    runtime = _handoff_runtime(config)
    runtime.bind_session(
        ControlSession(
            robot=runtime.robot,
            controller=CompleteController(),
            period_s=0.02,
            state_timeout_s=0.1,
        )
    )
    assert runtime.run().success
    actions = [
        event
        for event in bus.service_calls
        if event[0] in {"confirm", "stand_down", "release_mode", "service_switch", "lowlevel_start"}
    ]
    assert actions == [
        ("confirm", "yes"),
        ("stand_down", None),
        ("confirm", "down"),
        ("release_mode", "sport"),
        ("service_switch", ("sport_mode", False)),
        ("lowlevel_start", None),
    ]
    assert bus.factory_calls == [(23, "robot-test0"), (23, "robot-test0")]
    assert bus.commands and all(_is_damping(message) for message in bus.commands)
    output = capsys.readouterr().out
    assert "support" in output
    assert "Down confirmation received; releasing official control..." in output
    assert "Official control released and sport_mode disabled; handoff complete." in output
    assert not runtime.opened


@pytest.mark.parametrize("cancel_at", [0, 1])
@pytest.mark.parametrize("answer", ["no", EOFError, KeyboardInterrupt])
def test_go2_deploy_cancel_never_releases_or_starts_lowlevel(cancel_at, answer, monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    bus.motion_mode = "sport"
    calls = 0

    def confirm(prompt):
        nonlocal calls
        index = calls
        calls += 1
        if index < cancel_at:
            return "yes"
        if isinstance(answer, str):
            return answer
        raise answer()

    monkeypatch.setattr("builtins.input", confirm)
    monkeypatch.setattr(UnitreeGo2Robot, "open", lambda *args: pytest.fail("must not open lowlevel robot"))
    runtime = _handoff_runtime(_config())
    runtime.session = SimpleNamespace(period_s=0.02)
    with pytest.raises(RuntimeError, match="cancelled"):
        runtime.run()
    assert sum(name == "stand_down" for name, _ in bus.service_calls) == cancel_at
    assert not any(name in {"release_mode", "service_switch"} for name, _ in bus.service_calls)
    assert bus.motion_mode == "sport"
    assert bus.callback is None
    assert not bus.commands
    assert not runtime.opened


def test_go2_deploy_already_released_skips_prompts_and_service_changes(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    robot = _backend(_config(), bus)
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("released control needs no prompt"))
    robot.acquire_control()
    assert bus.factory_calls == [(0, "enp3s0")]
    assert not any(name in {"stand_down", "release_mode", "service_switch"} for name, _ in bus.service_calls)
    assert bus.commands == []


def test_active_go2_acquisition_without_interactive_input_fails_before_motion(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    bus.motion_mode = "sport"
    robot = _backend(_config(), bus)

    def no_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_input)
    with pytest.raises(RuntimeError, match="cancelled before StandDown"):
        robot.acquire_control()
    assert not any(name in {"stand_down", "release_mode", "service_switch"} for name, _ in bus.service_calls)
    assert bus.commands == []


@pytest.mark.parametrize("failure", ["stand_down", "release", "verify", "service_switch"])
def test_go2_deploy_handoff_failure_never_starts_lowlevel(failure, monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    bus.motion_mode = "sport"
    if failure == "stand_down":
        bus.stand_down_status = 42
    elif failure == "release":
        monkeypatch.setattr(_MotionSwitcherClient, "ReleaseMode", lambda self: (42, None))
    elif failure == "verify":
        monkeypatch.setattr(_MotionSwitcherClient, "ReleaseMode", lambda self: (0, None))
    else:
        bus.service_switch_status = 42
    answers = iter(["yes", "down"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    monkeypatch.setattr(UnitreeGo2Robot, "open", lambda *args: pytest.fail("must not open lowlevel robot"))
    runtime = _handoff_runtime(_config())
    runtime.session = SimpleNamespace(period_s=0.02)
    with pytest.raises(RuntimeError, match="42|remains active"):
        runtime.run()
    if failure == "stand_down":
        assert not any(name == "release_mode" for name, _ in bus.service_calls)
    if failure != "service_switch":
        assert not any(name == "service_switch" for name, _ in bus.service_calls)
    assert bus.callback is None
    assert bus.commands == []


@pytest.mark.parametrize("target_flag", ["--target", "--offset"])
def test_joint_control_dispatches_hardware_options(monkeypatch, target_flag: str) -> None:
    calls = []
    monkeypatch.setattr("motrix_deploy_unitree.cli.app.run_joint_control", lambda **kwargs: calls.append(kwargs))
    assert (
        main(
            [
                "joint-control",
                "enp5s0",
                "FL_thigh_joint",
                target_flag,
                "0.1",
                "--kp",
                "42",
                "--kd",
                "0.8",
                "--hold-pose",
                "lie_down",
                "--control-period-s",
                "0.01",
                "--state-timeout-s",
                "0.2",
            ]
        )
        == 0
    )
    assert calls[0]["target_position"] == (0.1 if target_flag == "--target" else None)
    assert calls[0]["offset"] == (0.1 if target_flag == "--offset" else None)
    assert calls[0]["kp"] == 42.0
    assert calls[0]["kd"] == 0.8
    assert calls[0]["hold_pose"] == "lie_down"
    assert calls[0]["control_period_s"] == 0.01
    assert calls[0]["state_timeout_s"] == 0.2


@pytest.mark.parametrize("flags", [[], ["--target", "0.9", "--offset", "0.1"]])
def test_joint_control_requires_exactly_one_target(
    flags,
) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["joint-control", "eth0", "FL_thigh_joint", "--kp", "1", "--kd", "1", *flags])


@pytest.mark.parametrize("flag", ["--kp", "--kd"])
def test_joint_control_requires_each_task_gain(
    flag: str,
) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["joint-control", "eth0", "FL_thigh_joint", "--target", "0.9", flag, "1"])


def test_read_lowstate_uses_unified_cli_defaults() -> None:
    args = _parser().parse_args(["read-lowstate", "enp5s0"])

    assert args.command == "read-lowstate"
    assert args.network_interface == "enp5s0"
    assert args.topic == "rt/lowstate"
    assert args.duration_s == pytest.approx(10.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"joint_name": "unknown"},
        {"hold_pose": "unknown"},
        {"target_position": 100.0},
        {"target_position": float("nan")},
        {"control_period_s": 0.0},
        {"kp": -1.0},
        {"move_duration": 0.0},
    ],
)
def test_joint_control_rejects_invalid_requests_before_connection(monkeypatch, kwargs) -> None:
    from motrix_deploy_unitree.cli.joint_control.runner import run_joint_control

    bus = _Bus().install(monkeypatch)
    options = dict(network_interface="eth0", joint_name="FL_thigh_joint", target_position=0.9, kp=12.0, kd=0.5)
    options.update(kwargs)
    with pytest.raises(ValueError):
        run_joint_control(**options)
    assert bus.factory_calls == []


def test_joint_control_declining_confirmation_does_not_connect(monkeypatch) -> None:
    from motrix_deploy_unitree.cli.joint_control.runner import run_joint_control

    bus = _Bus().install(monkeypatch)
    monkeypatch.setattr("builtins.input", lambda _: "no")
    run_joint_control(network_interface="eth0", joint_name="FL_thigh_joint", offset=0.1, kp=12.0, kd=0.5)
    assert bus.factory_calls == []


def test_joint_controller_runs_through_shared_session_and_fault_stop(monkeypatch) -> None:
    from motrix_deploy.runtime.control import ControlSession
    from motrix_deploy_unitree.cli.joint_control.controller import Go2JointController
    from motrix_deploy_unitree.config.go2 import Go2JointSettings

    bus = _Bus().install(monkeypatch)
    spec = _spec()
    for index, name in enumerate(spec.joint_names):
        bus.state.motor_state[MOTOR_MAPPING[name]].q = float(spec.default_joint_position[index])
    from motrix_deploy_unitree.config.go2 import Go2StartupLimits

    startup = Go2StartupLimits(max_speed_rad_s=1, timeout_s=2, gain_ramp_s=0.02, kp=12, kd=0.5)

    robot = UnitreeGo2Robot(
        _config().transport,
        UnitreeControlConfig(),
        control_period_s=0.02,
        state_timeout_s=0.1,
        spec=spec,
    )
    original_read = robot.read_state
    reads = 0

    def fresh_read():
        nonlocal reads
        reads += 1
        # Session startup consumes two samples before the first task tick.
        keys = _button("start") if reads == 4 else _button("A") if reads == 8 else 0
        bus.state.wireless_remote = _remote_payload(keys)
        bus.emit()
        return original_read()

    robot.read_state = fresh_read
    settings = Go2JointSettings(
        kp=12,
        kd=0.5,
        transition_duration_s=0.04,
        move_duration_s=0.04,
        hold_duration_s=0.02,
        return_duration_s=0.04,
    )
    task = Go2JointController(
        robot.spec,
        "FL_thigh_joint",
        0.9,
        settings,
        gamepad=robot.get_gamepad_device(),
        startup_limits=startup,
    )
    runtime = HardwareRuntime(robot, realtime=True)
    runtime.bind_session(ControlSession(robot=robot, controller=task, period_s=0.02, state_timeout_s=0.1))
    result = runtime.run()
    assert result.success, result.error
    assert task.phase == "complete"
    assert robot.health().reason == "backend is closed"
    assert all(motor.kp == 0 for motor in bus.commands[-1].motor_cmd)
    assert any(cmd.motor_cmd[4].q == pytest.approx(0.9) for cmd in bus.commands)


@pytest.mark.integration
def test_plugin_runtime_uses_production_interface_with_injected_sdk(monkeypatch) -> None:
    bus = _Bus().install(monkeypatch)
    runtime = create_hardware_runtime(
        _config(damping_duration_s=0.0),
        _context(),
    )
    robot = runtime.robot
    assert runtime.gamepad is robot.get_gamepad_device()
    try:
        robot.open()
        robot.read_state()
        robot.enable()
        robot.write_command(_command(robot.spec.default_joint_position))
        assert robot.health().healthy
        assert bus.factory_calls == [(0, "enp3s0")]
        assert _servo_published(bus, _command(robot.spec.default_joint_position)).crc == 123
    finally:
        robot.close()
        runtime.close()
    assert robot.health().reason == "backend is closed"


def test_slow_dds_does_not_block_latest_command_submission_or_queue_old_commands(monkeypatch):
    bus = _Bus().install(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    original_write = _Publisher.Write
    blocked = [False]

    def slow_write(publisher, message):
        if any(motor.kp > 0 for motor in message.motor_cmd) and not blocked[0]:
            blocked[0] = True
            entered.set()
            assert release.wait(2)
        return original_write(publisher, message)

    monkeypatch.setattr(_Publisher, "Write", slow_write)
    config = _config()
    config.control.command_timeout_s = 2
    robot = UnitreeGo2Robot(config.transport, config.control, spec=_spec(), control_period_s=0.02, state_timeout_s=2)
    robot.open()
    robot.enable()
    first = _command(robot.spec.default_joint_position.copy())
    robot.write_command(first)
    assert entered.wait(1)
    communication_before = robot.health().last_successful_communication_ns
    latest = _command(robot.spec.default_joint_position.copy())
    latest.joint_position[0] += np.float32(0.02)
    expected_latest = latest.joint_position.copy()
    submitted = threading.Event()

    def submit():
        intermediate = _command(robot.spec.default_joint_position.copy())
        intermediate.joint_position[0] += np.float32(0.01)
        robot.write_command(intermediate)
        robot.write_command(latest)
        latest.joint_position.fill(100)
        latest.kp.fill(999)
        submitted.set()

    submitter = threading.Thread(target=submit)
    submitter.start()
    try:
        assert submitted.wait(0.5), "command submission waited for the blocked DDS write"
        assert robot.health().last_successful_communication_ns == communication_before
    finally:
        release.set()
        submitter.join(1)
    expected = _command(expected_latest)
    _servo_published(bus, expected)
    servo = [message for message in bus.commands if any(motor.kp > 0 for motor in message.motor_cmd)]
    assert servo[0].motor_cmd[3].q == float(first.joint_position[0])
    assert all(message.motor_cmd[3].q == float(expected_latest[0]) for message in servo[1:])
    assert all(message.motor_cmd[3].kp == 35 for message in servo)
    robot.close()


@pytest.mark.parametrize("expired", ["command", "state"])
def test_writer_watchdog_damps_and_latches_without_policy_reads(expired, monkeypatch):
    bus = _Bus().install(monkeypatch)
    config = _config()
    config.control.command_timeout_s = 0.04 if expired == "command" else 1
    robot = UnitreeGo2Robot(
        config.transport,
        config.control,
        spec=_spec(),
        control_period_s=0.02,
        state_timeout_s=0.04 if expired == "state" else 1,
    )
    robot.open()
    robot.enable()
    command = _command(robot.spec.default_joint_position.copy())
    robot.write_command(command)
    _servo_published(bus, command)
    if expired == "state":
        bus.emit_on_write = False
    start = len(bus.commands)
    bus.wait_for(_is_damping, after=start)
    assert not robot.health().healthy
    assert ("command expired" if expired == "command" else "LowState expired") in robot.health().reason
    bus.emit()
    with pytest.raises(TimeoutError, match="expired"):
        robot.write_command(command)
    robot.close()


def test_remote_emergency_damps_without_another_policy_read(monkeypatch):
    bus = _Bus().install(monkeypatch)
    robot = _backend(_config(), bus)
    robot.open()
    robot.enable()
    command = _command(robot.spec.default_joint_position.copy())
    robot.write_command(command)
    _servo_published(bus, command)
    before = len(bus.commands)
    bus.state.wireless_remote = _remote_payload(_button("select"))
    bus.emit()
    bus.wait_for(_is_damping, after=before)
    assert not robot.health().healthy
    bus.state.wireless_remote = _remote_payload()
    bus.emit()
    with pytest.raises(EmergencyStopError):
        robot.write_command(command)
    robot.close()


def test_blocked_writer_close_preserves_channels_until_join_succeeds(monkeypatch):
    bus = _Bus().install(monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    original_write = _Publisher.Write

    def blocked_write(publisher, message):
        if any(motor.kp > 0 for motor in message.motor_cmd):
            entered.set()
            assert release.wait(2)
        return original_write(publisher, message)

    monkeypatch.setattr(_Publisher, "Write", blocked_write)
    robot = _backend(_config(connect_timeout_s=0.01), bus)
    robot.open()
    robot.enable()
    robot.write_command(_command(robot.spec.default_joint_position.copy()))
    assert entered.wait(1)
    writer = robot._writer_thread
    try:
        with pytest.raises(TimeoutError, match="still blocked"):
            robot.close()
        assert not robot._publisher.closed and not robot._subscriber.closed
    finally:
        release.set()
        robot.close()
    assert robot._publisher.closed and robot._subscriber.closed
    assert robot._writer_thread is None
    assert not writer.is_alive()


def test_stop_clears_command_slot_and_cannot_resume_outputs(monkeypatch):
    bus = _Bus().install(monkeypatch)
    robot = _backend(_config(), bus)
    robot.open()
    robot.enable()
    command = _command(robot.spec.default_joint_position.copy())
    robot.write_command(command)
    _servo_published(bus, command)
    robot.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        robot.enable()
    with pytest.raises(RuntimeError, match="not enabled|stopped"):
        robot.write_command(command)
    after = len(bus.commands)
    bus.wait_for(_is_damping, after=after)
    assert all(_is_damping(message) for message in bus.commands[after:])
    robot.close()
