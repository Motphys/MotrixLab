# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree Go2 backend contract tests with an injected SDK2 transport."""

import copy
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from motrix_deploy.artifact import ControlSpec, TaskSpec
from motrix_deploy.contracts import JointServoCommand, JointTorqueCommand, RobotSpec
from motrix_deploy.errors import EmergencyStopError, LieDownRequestedError, ValidationError
from motrix_deploy.profile import DeploymentProfile
from motrix_deploy.robot.interface import GamePadDeviceProvider
from motrix_deploy.runtime.factory import RuntimeCreateContext
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_deploy_unitree import (
    UnitreeGo2BackendConfig,
    UnitreeGo2DirectInterface,
    UnitreeGo2RobotInterface,
    UnitreeRemoteGamePadDevice,
    UnitreeSdkBindings,
    decode_wireless_remote,
)
from motrix_deploy_unitree.cli import _parser, main
from motrix_deploy_unitree.plugin import create_runtime

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

    def Write(self, message: _LowCmd) -> None:
        snapshot = copy.deepcopy(message)
        self.bus.commands.append(snapshot)
        write_count = len(self.bus.commands)
        if write_count in self.bus.remote_keys_after_writes:
            self.bus.state.wireless_remote = _remote_payload(self.bus.remote_keys_after_writes[write_count])
        for motor_index, motor in enumerate(snapshot.motor_cmd):
            if motor.kp > 0:
                self.bus.state.motor_state[motor_index].q = motor.q
                self.bus.state.motor_state[motor_index].dq = motor.qd
        self.bus.emit()

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
        self.bus.emit()

    def Close(self) -> None:
        self.closed = True


class _Bus:
    def __init__(self) -> None:
        self.state = _LowState()
        self.commands: list[_LowCmd] = []
        self.callback = None
        self.queue_depth = 0
        self.factory_calls: list[tuple[int, str]] = []
        self.remote_keys_after_writes: dict[int, int] = {}
        self.motion_mode = "sport"
        self.stand_down_status = 0
        self.service_switch_status = 0
        self.service_calls: list[tuple[str, object]] = []

    def emit(self) -> None:
        if self.callback is not None:
            self.callback(copy.deepcopy(self.state))

    def sdk(self) -> UnitreeSdkBindings:
        return UnitreeSdkBindings(
            channel_factory_initialize=lambda domain, interface: self.factory_calls.append((domain, interface)),
            channel_publisher=lambda topic, message_type: _Publisher(self, topic, message_type),
            channel_subscriber=lambda topic, message_type: _Subscriber(self, topic, message_type),
            low_cmd_message_type=_LowCmd,
            low_state_message_type=_LowState,
            make_low_cmd=_LowCmd,
            crc_type=_CRC,
            motion_switcher_client=lambda: _MotionSwitcherClient(self),
            sport_client=lambda: _SportClient(self),
            robot_state_client=lambda: _RobotStateClient(self),
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
    return RobotSpec(
        base_link_name="base",
        joint_names=JOINT_NAMES,
        default_joint_position=np.array(
            [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8],
            dtype=np.float32,
        ),
        position_lower=np.tile(np.array([-0.9472, -1.4, -2.6227], dtype=np.float32), 4),
        position_upper=np.tile(np.array([0.9472, 2.5, -0.84776], dtype=np.float32), 4),
        torque_limit=np.full(12, 24.0, dtype=np.float32),
    )


def _config(**overrides) -> UnitreeGo2BackendConfig:
    values = {
        "network_interface": "enp3s0",
        "default_transition_duration_s": 0.02,
        "damping_duration_s": 0.04,
        "wait_for_remote_buttons": False,
        "kp": 35.0,
        "kd": 0.5,
    }
    values.update(overrides)
    return UnitreeGo2BackendConfig(**values)


def _backend(
    config: UnitreeGo2BackendConfig,
    bus: _Bus,
) -> UnitreeGo2RobotInterface:
    return UnitreeGo2RobotInterface(
        config,
        control_period_s=0.02,
        state_timeout_s=0.1,
        spec=_spec(),
        sdk=bus.sdk(),
        sleep=lambda _: None,
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


def test_unitree_interface_provides_remote_gamepad_device() -> None:
    backend = _backend(_config(), _Bus())
    assert isinstance(backend, GamePadDeviceProvider)
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


def test_open_releases_mcf_mode_and_disables_sport_service_before_lowcmd() -> None:
    bus = _Bus()
    backend = _backend(_config(), bus)

    backend.open()

    call_names = [name for name, _ in bus.service_calls]
    assert call_names.index("stand_down") < call_names.index("release_mode")
    assert call_names.index("release_mode") < call_names.index("service_switch")
    assert ("service_switch", ("sport_mode", False)) in bus.service_calls
    assert bus.motion_mode == ""


def test_sport_service_shutdown_failure_aborts_before_lowcmd_channels() -> None:
    bus = _Bus()
    bus.service_switch_status = 42
    backend = _backend(_config(), bus)

    with pytest.raises(RuntimeError, match="ServiceSwitch.*code 42"):
        backend.open()

    assert bus.callback is None
    assert bus.commands == []


def test_lowstate_and_lowcmd_are_mapped_by_canonical_joint_name() -> None:
    bus = _Bus()
    spec = _spec()
    backend = _backend(_config(), bus)
    backend.open()

    state = backend.read_state(0.1)
    expected_position = np.asarray([MOTOR_MAPPING[name] for name in JOINT_NAMES], dtype=np.float32)
    np.testing.assert_array_equal(state.joint_position, expected_position)
    np.testing.assert_array_equal(state.base_orientation_xyzw, [0.0, 0.0, 0.0, 1.0])

    backend.enable()
    target = spec.default_joint_position + np.linspace(-0.01, 0.01, 12, dtype=np.float32)
    command = _command(target)
    backend.write_command(command)
    published = copy.deepcopy(bus.commands[-1])

    for canonical_index, name in enumerate(JOINT_NAMES):
        motor = published.motor_cmd[MOTOR_MAPPING[name]]
        assert motor.q == pytest.approx(command.joint_position[canonical_index])
        assert motor.qd == pytest.approx(0.0)
        assert motor.kp == pytest.approx(35.0)
        assert motor.kd == pytest.approx(0.5)
    assert published.crc == 123

    backend.stop()
    assert len(bus.commands) >= 4
    for motor_index in MOTOR_MAPPING.values():
        motor = bus.commands[-1].motor_cmd[motor_index]
        assert motor.kp == 0.0
        assert motor.kd == 8.0
        assert motor.tau == 0.0
    backend.close()
    assert not backend.health().healthy
    assert backend.health().reason == "backend is closed"


def test_enable_requires_explicit_hardware_safety_gains_before_publishing() -> None:
    bus = _Bus()
    backend = _backend(_config(kp=None, kd=None), bus)
    backend.open()

    with pytest.raises(ValidationError, match="safety gains"):
        backend.enable()
    assert bus.commands == []
    backend.close()


def test_servo_only_interface_rejects_torque_commands_before_publishing() -> None:
    bus = _Bus()
    backend = _backend(_config(), bus)
    backend.open()
    backend.enable()
    command_count = len(bus.commands)

    with pytest.raises(ValidationError, match="joint_servo"):
        backend.write_command(JointTorqueCommand(torque=np.zeros(12, dtype=np.float32)))
    assert len(bus.commands) == command_count
    backend.close()


def test_runtime_gain_overrides_apply_to_transition_and_policy_commands() -> None:
    bus = _Bus()
    spec = _spec()
    kd = np.linspace(0.2, 1.3, 12, dtype=np.float32)
    backend = _backend(_config(kp=25.0, kd=kd.tolist()), bus)
    backend.open()

    backend.enable()
    transition = copy.deepcopy(bus.commands[-1])
    backend.write_command(_command(spec.default_joint_position))
    policy = copy.deepcopy(bus.commands[-1])

    for published in (transition, policy):
        for canonical_index, name in enumerate(JOINT_NAMES):
            motor = published.motor_cmd[MOTOR_MAPPING[name]]
            assert motor.kp == pytest.approx(25.0)
            assert motor.kd == pytest.approx(kd[canonical_index])

    backend.stop()
    for motor_index in MOTOR_MAPPING.values():
        assert bus.commands[-1].motor_cmd[motor_index].kp == 0.0
        assert bus.commands[-1].motor_cmd[motor_index].kd == pytest.approx(8.0)


def test_direct_interface_reads_and_sends_through_production_backend() -> None:
    bus = _Bus()
    spec = _spec()
    backend = _backend(_config(), bus)
    direct = UnitreeGo2DirectInterface(
        robot=spec,
        task_config={"kp": [35.0] * 12, "kd": [0.5] * 12},
        control_period_s=0.02,
        state_timeout_s=0.1,
        backend=backend,
    )
    direct.open()

    state = direct.read_data()
    expected_position = np.asarray([MOTOR_MAPPING[name] for name in JOINT_NAMES], dtype=np.float32)
    np.testing.assert_array_equal(state.joint_position, expected_position)

    with pytest.raises(RuntimeError, match="not enabled"):
        direct.send_joint_command(spec.default_joint_position)

    direct.enable_command_output()
    target = spec.default_joint_position + np.linspace(-0.01, 0.01, 12, dtype=np.float32)
    command = direct.send_joint_command(target)
    published = bus.commands[-1]

    np.testing.assert_array_equal(command.joint_position, target)
    assert published.motor_cmd[MOTOR_MAPPING["FL_hip_joint"]].q == pytest.approx(target[0])
    assert published.motor_cmd[MOTOR_MAPPING["RR_calf_joint"]].q == pytest.approx(target[-1])

    direct.close()
    assert direct.health().reason == "backend is closed"


def test_enable_waits_for_start_transitions_then_waits_for_a() -> None:
    bus = _Bus()
    bus.remote_keys_after_writes = {
        1: _button("start"),
        2: _button("start"),
        3: _button("start") | _button("A"),
    }
    backend = _backend(
        UnitreeGo2BackendConfig(
            network_interface="enp3s0",
            default_transition_duration_s=0.02,
            damping_duration_s=0.0,
            kp=35.0,
            kd=0.5,
        ),
        bus,
    )
    backend.open()
    backend.read_state(0.1)

    backend.enable()

    assert len(bus.commands) == 3
    assert all(motor.kp == 0.0 and motor.kd == 0.0 for motor in bus.commands[0].motor_cmd)
    assert any(motor.kp == 35.0 for motor in bus.commands[1].motor_cmd)
    assert any(motor.kp == 35.0 for motor in bus.commands[2].motor_cmd)
    backend.close()


def test_select_remote_button_raises_emergency_stop_and_stop_publishes_damping() -> None:
    bus = _Bus()
    backend = _backend(
        _config(damping_duration_s=0.02),
        bus,
    )
    backend.open()
    backend.read_state(0.1)
    backend.enable()
    commands_before_stop = len(bus.commands)

    bus.state.wireless_remote = _remote_payload(_button("select"))
    bus.emit()

    with pytest.raises(EmergencyStopError, match="requested damping stop"):
        backend.read_state(0.1)
    assert not backend.health().healthy

    backend.stop()
    assert len(bus.commands) == commands_before_stop + 1
    assert all(bus.commands[-1].motor_cmd[index].kd == 8.0 for index in MOTOR_MAPPING.values())
    backend.close()


def test_b_button_lies_down_then_enters_damping() -> None:
    bus = _Bus()
    config = _config(lie_down_duration_s=0.04, damping_duration_s=0.02)
    backend = _backend(config, bus)
    backend.open()
    backend.read_state(0.1)
    backend.enable()
    commands_before_stop = len(bus.commands)

    bus.state.wireless_remote = _remote_payload(_button("B"))
    bus.emit()

    with pytest.raises(LieDownRequestedError, match="lie-down shutdown"):
        backend.read_state(0.1)
    backend.stop()

    shutdown_commands = bus.commands[commands_before_stop:]
    assert len(shutdown_commands) == 3
    target = config.lie_down_position()
    for canonical_index, name in enumerate(JOINT_NAMES):
        motor_index = MOTOR_MAPPING[name]
        final_position = shutdown_commands[-2].motor_cmd[motor_index]
        assert final_position.q == pytest.approx(target[canonical_index])
        assert final_position.kp == pytest.approx(35.0)
        assert final_position.kd == pytest.approx(0.5)
        damping = shutdown_commands[-1].motor_cmd[motor_index]
        assert damping.kp == 0.0
        assert damping.kd == pytest.approx(8.0)
    backend.close()


def test_select_has_priority_over_simultaneous_lie_down_request() -> None:
    bus = _Bus()
    backend = _backend(_config(damping_duration_s=0.02), bus)
    backend.open()
    backend.read_state(0.1)
    backend.enable()
    commands_before_stop = len(bus.commands)

    bus.state.wireless_remote = _remote_payload(_button("select") | _button("B"))
    bus.emit()

    with pytest.raises(EmergencyStopError, match="damping stop"):
        backend.read_state(0.1)
    backend.stop()

    shutdown_commands = bus.commands[commands_before_stop:]
    assert len(shutdown_commands) == 1
    assert all(shutdown_commands[0].motor_cmd[index].kd == 8.0 for index in MOTOR_MAPPING.values())
    backend.close()


def test_lie_down_config_rejects_invalid_pose() -> None:
    with pytest.raises(ValidationError, match="runtime.lie_down_joint_position"):
        _config(lie_down_joint_position=[0.0] * 11)


def test_go2_hardware_defaults_define_motor_mapping_and_lie_down_pose() -> None:
    config = UnitreeGo2BackendConfig(network_interface="enp3s0")

    assert config.joint_name_to_motor_index == MOTOR_MAPPING
    assert config.lie_down_position().shape == (12,)


def test_motor_mapping_must_exactly_cover_artifact_joint_names() -> None:
    mapping = dict(MOTOR_MAPPING)
    mapping.pop("FL_hip_joint")
    config = _config(joint_name_to_motor_index=mapping)

    with pytest.raises(ValidationError, match="missing"):
        config.motor_indices(JOINT_NAMES)


def _context() -> RuntimeCreateContext:
    return RuntimeCreateContext(
        robot=_spec(),
        control=ControlSpec(period_s=0.02, state_timeout_s=0.1),
    )


def test_plugin_factory_creates_realtime_hardware_runtime_and_strict_config() -> None:
    config = {
        "name": "unitree_go2",
        "network_interface": "enp3s0",
    }

    runtime = create_runtime(config, _context())
    assert isinstance(runtime, HardwareRuntime)
    assert runtime.realtime is True
    assert isinstance(runtime.robot, UnitreeGo2RobotInterface)
    assert runtime.robot.spec.joint_names == JOINT_NAMES

    with pytest.raises(ValidationError, match="unknown"):
        UnitreeGo2BackendConfig.from_mapping({**config, "unexpected": 1})


def test_plugin_runtime_uses_production_interface_with_injected_sdk(monkeypatch) -> None:
    bus = _Bus()
    monkeypatch.setattr("motrix_deploy_unitree.interface._load_sdk_bindings", bus.sdk)
    runtime = create_runtime(
        {
            "network_interface": "enp3s0",
            "kp": 35.0,
            "kd": 0.5,
            "default_transition_duration_s": 0.02,
            "damping_duration_s": 0.0,
            "wait_for_remote_buttons": False,
        },
        _context(),
    )
    robot = runtime.robot
    try:
        robot.open()
        robot.read_state(0.1)
        robot.enable()
        robot.write_command(_command(robot.spec.default_joint_position))
        assert robot.health().healthy
        assert bus.factory_calls == [(0, "enp3s0")]
        assert bus.commands[-1].crc == 123
    finally:
        robot.close()
        runtime.close()
    assert robot.health().reason == "backend is closed"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("kp", -1.0),
        ("kd", True),
        ("kp", [1.0] * 11),
        ("kd", [0.5] * 11 + [float("nan")]),
    ],
)
def test_backend_config_rejects_invalid_gain_overrides(name: str, value: object) -> None:
    with pytest.raises(ValidationError, match=f"runtime.{name}"):
        _config(**{name: value})


class _JointControlTaskSpec(TaskSpec):
    task_name = "unitree_joint_control/v1"
    kp: list[float]
    kd: list[float]


def _profile() -> DeploymentProfile:
    return DeploymentProfile(
        robot=_spec(),
        task=_JointControlTaskSpec(kp=[35.0] * 12, kd=[0.5] * 12),
        control=ControlSpec(period_s=0.02, state_timeout_s=0.1),
        observation_size=1,
        action_size=12,
    )


def test_direct_interface_builds_from_profile_without_artifact() -> None:
    bus = _Bus()
    direct = UnitreeGo2DirectInterface.from_profile(
        _profile(),
        network_interface="enp5s0",
        backend_options={
            "default_transition_duration_s": 0.02,
            "damping_duration_s": 0.0,
            "wait_for_remote_buttons": False,
        },
        sdk=bus.sdk(),
        sleep=lambda _: None,
    )

    assert direct.control_period_s == pytest.approx(0.02)
    assert direct.state_timeout_s == pytest.approx(0.1)
    assert direct.backend.config.network_interface == "enp5s0"

    direct.open()
    direct.read_data()
    direct.enable_command_output()
    direct.send_joint_command(direct.robot.default_joint_position)
    direct.close()

    assert bus.factory_calls == [(0, "enp5s0")]
    assert bus.commands


def test_joint_control_cli_dispatches_artifact_and_motion_defaults(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("motrix_deploy_unitree.cli.run_joint_control", lambda **kwargs: calls.append(kwargs))

    assert main(["joint-control", "enp5s0", "FL_thigh_joint", "0.9", "--artifact", "example.deploy"]) == 0

    assert len(calls) == 1
    assert str(calls[0]["artifact"]) == "example.deploy"
    assert calls[0]["network_interface"] == "enp5s0"
    assert calls[0]["joint_name"] == "FL_thigh_joint"
    assert calls[0]["target_position"] == pytest.approx(0.9)
    assert calls[0]["move_duration"] == pytest.approx(2.0)
    assert calls[0]["hold_duration"] == pytest.approx(1.0)
    assert calls[0]["return_duration"] == pytest.approx(2.0)


def test_joint_control_requires_artifact() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["joint-control", "enp5s0", "FL_thigh_joint", "0.9"])


def test_read_lowstate_uses_unified_cli_defaults() -> None:
    args = _parser().parse_args(["read-lowstate", "enp5s0"])

    assert args.command == "read-lowstate"
    assert args.network_interface == "enp5s0"
    assert args.topic == "rt/lowstate"
    assert args.duration_s == pytest.approx(10.0)


def test_joint_control_rejects_environment_option() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "joint-control",
                "enp5s0",
                "FL_thigh_joint",
                "0.9",
                "--artifact",
                "example.deploy",
                "--env",
                "go2-walk-rough",
            ]
        )
