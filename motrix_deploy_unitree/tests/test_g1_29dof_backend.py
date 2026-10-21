# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""G1 29-DOF HG message, mode, and named-joint mapping contracts."""

import copy
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from motrix_deploy.contracts import JointServoCommand, RobotSpec
from motrix_deploy.errors import EmergencyStopError, PassiveRequestedError
from motrix_deploy.operator import ButtonGestureCfg, OperatorProfileCfg
from motrix_deploy.runtime.factory import RuntimeCreateContext
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_deploy_unitree.config.control import load_control_config
from motrix_deploy_unitree.config.g1 import (
    G1_29DOF_JOINT_NAMES,
    UnitreeG129DOFHardwareCfg,
)
from motrix_deploy_unitree.config.runtime import UnitreeRuntimeConfig, UnitreeTransportConfig
from motrix_deploy_unitree.g1.python_robot import UnitreeG1_29DOFRobot, UnitreeSdkBindings
from motrix_deploy_unitree.plugin import create_runtime


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
        self.mode_pr = 0
        self.mode_machine = 0
        self.motor_cmd = [_Motor() for _ in range(35)]
        self.crc = 0


class _LowState:
    def __init__(self, *, mode_pr: int = 0, mode_machine: int = 5) -> None:
        self.mode_pr = mode_pr
        self.mode_machine = mode_machine
        self.motor_state = [_Motor(index) for index in range(35)]
        self.imu_state = SimpleNamespace(
            quaternion=[1.0, 0.0, 0.0, 0.0],
            gyroscope=[0.1, 0.2, 0.3],
            accelerometer=[0.0, 0.0, 9.81],
        )
        self.wireless_remote = bytes(40)
        self.crc = 123


class _CRC:
    def Crc(self, message: object) -> int:
        del message
        return 123


class _Publisher:
    def __init__(self, bus: "_Bus", topic: str, message_type: type) -> None:
        self.bus = bus
        self.topic = topic
        self.message_type = message_type

    def Init(self) -> None:
        pass

    def Write(self, message: _LowCmd) -> None:
        self.bus.commands.append(copy.deepcopy(message))
        if self.bus.callback is not None:
            self.bus.callback(copy.deepcopy(self.bus.state))

    def Close(self) -> None:
        pass


class _Subscriber:
    def __init__(self, bus: "_Bus", topic: str, message_type: type) -> None:
        self.bus = bus
        self.topic = topic
        self.message_type = message_type

    def Init(self, callback, queue_depth: int) -> None:
        self.bus.callback = callback
        self.bus.queue_depth = queue_depth
        callback(copy.deepcopy(self.bus.state))

    def Close(self) -> None:
        pass


class _Bus:
    def __init__(self, *, mode_pr: int = 0, mode_machine: int = 5) -> None:
        self.state = _LowState(mode_pr=mode_pr, mode_machine=mode_machine)
        self.commands: list[_LowCmd] = []
        self.callback = None
        self.queue_depth = 0

    def sdk(self) -> UnitreeSdkBindings:
        return UnitreeSdkBindings(
            channel_factory_initialize=lambda domain, interface: None,
            channel_publisher=lambda topic, message_type: _Publisher(self, topic, message_type),
            channel_subscriber=lambda topic, message_type: _Subscriber(self, topic, message_type),
            low_cmd_message_type=_LowCmd,
            low_state_message_type=_LowState,
            make_low_cmd=_LowCmd,
            crc_type=_CRC,
        )


def _spec(joint_names: tuple[str, ...] = G1_29DOF_JOINT_NAMES) -> RobotSpec:
    count = len(joint_names)
    return RobotSpec(
        base_link_name="torso_link",
        joint_names=joint_names,
        default_joint_position=np.zeros(count, dtype=np.float32),
        position_lower=np.full(count, -10.0, dtype=np.float32),
        position_upper=np.full(count, 10.0, dtype=np.float32),
        torque_limit=np.full(count, 100.0, dtype=np.float32),
    )


def _config(**overrides) -> UnitreeRuntimeConfig:
    robot_values = {
        "damping_duration_s": 0.0,
        "command_publish_period_s": None,
    }
    robot_values.update(overrides)
    return UnitreeRuntimeConfig(
        transport=UnitreeTransportConfig(network_interface="eth0"),
        robot="g1-29dof",
        control=replace(load_control_config("g1"), **robot_values),
    )


def _remote_payload(*buttons: str) -> bytes:
    from motrix_deploy_unitree.remote import BUTTON_NAMES, canonical_button_name

    keys = sum(1 << BUTTON_NAMES.index(canonical_button_name(button)) for button in buttons)
    payload = bytearray(40)
    payload[2:4] = keys.to_bytes(2, "little")
    return bytes(payload)


def _backend(config: UnitreeRuntimeConfig, bus: _Bus, *, spec: RobotSpec | None = None) -> UnitreeG1_29DOFRobot:
    return UnitreeG1_29DOFRobot(
        transport=config.transport,
        control=config.control,
        operator=config.operator,
        control_period_s=0.02,
        state_timeout_s=0.1,
        spec=_spec() if spec is None else spec,
        sdk=bus.sdk(),
        sleep=lambda _: None,
    )


def test_g1_hardware_uses_explicit_joint_name_to_motor_id_mapping() -> None:
    hardware = UnitreeG129DOFHardwareCfg()
    hardware.validate()
    assert tuple(hardware.key_pose.joint_names) == G1_29DOF_JOINT_NAMES
    assert {name: binding.index for name, binding in hardware.actuation.motors.items()} == {
        name: motor_id for motor_id, name in enumerate(G1_29DOF_JOINT_NAMES)
    }
    assert all(binding.enabled_mode == 1 for binding in hardware.actuation.motors.values())


def test_g1_state_and_command_follow_artifact_joint_names_not_policy_indices() -> None:
    joint_names = G1_29DOF_JOINT_NAMES[7:] + G1_29DOF_JOINT_NAMES[:7]
    bus = _Bus()
    backend = _backend(_config(), bus, spec=_spec(joint_names))
    backend.open()
    state = backend.read_state()
    expected_motor_ids = [G1_29DOF_JOINT_NAMES.index(name) for name in joint_names]
    assert state.joint_position.tolist() == pytest.approx(expected_motor_ids)

    backend.enable()
    target = np.arange(29, dtype=np.float32) / 100.0
    zeros = np.zeros(29, dtype=np.float32)
    policy_kp = np.full(29, 17.0, dtype=np.float32)
    policy_kd = np.full(29, 0.7, dtype=np.float32)
    backend.write_command(
        JointServoCommand(
            joint_position=target,
            joint_velocity=zeros,
            feedforward_torque=zeros,
            kp=policy_kp,
            kd=policy_kd,
        )
    )
    command = bus.commands[-1]
    assert command.mode_pr == 0
    assert command.mode_machine == 5
    for policy_index, motor_id in enumerate(expected_motor_ids):
        assert command.motor_cmd[motor_id].q == pytest.approx(target[policy_index])
        assert command.motor_cmd[motor_id].kp == pytest.approx(policy_kp[policy_index])
        assert command.motor_cmd[motor_id].kd == pytest.approx(policy_kd[policy_index])
    for motor in command.motor_cmd[29:]:
        assert (motor.mode, motor.q, motor.dq, motor.kp, motor.kd, motor.tau) == (0, 0.0, 0.0, 0.0, 0.0, 0.0)
    backend.close()


@pytest.mark.parametrize("startup_mode", ["one_shot", "state_machine"])
def test_enable_allows_output_without_startup_motion_or_confirmation(startup_mode) -> None:
    bus = _Bus()
    backend = _backend(_config(startup_mode=startup_mode), bus)
    backend.open()
    try:
        assert bus.commands == []
        backend.enable()
        assert bus.commands == []
        zeros = np.zeros(29, dtype=np.float32)
        backend.write_command(JointServoCommand(zeros, zeros, zeros, zeros, np.full(29, 3.0, dtype=np.float32)))
        assert len(bus.commands) == 1
        assert all(motor.kp == 0.0 and motor.kd == pytest.approx(3.0) for motor in bus.commands[-1].motor_cmd[:29])
    finally:
        backend.close()


def test_enable_rejects_remote_emergency_stop_without_motion() -> None:
    bus = _Bus()
    backend = _backend(_config(), bus)
    backend.open()
    try:
        bus.state.wireless_remote = _remote_payload("select")
        bus.callback(copy.deepcopy(bus.state))
        with pytest.raises(EmergencyStopError):
            backend.enable()
        assert bus.commands == []
    finally:
        backend.close()


def test_stop_publishes_only_passive_damping_after_policy_output() -> None:
    bus = _Bus()
    config = _config(damping_duration_s=0.04, damping_kd=2.5)
    backend = _backend(config, bus)
    backend.open()
    backend.enable()
    zeros = np.zeros(29, dtype=np.float32)
    backend.write_command(
        JointServoCommand(
            joint_position=np.full(29, 0.2, dtype=np.float32),
            joint_velocity=zeros,
            feedforward_torque=zeros,
            kp=np.full(29, 17, dtype=np.float32),
            kd=np.ones(29, dtype=np.float32),
        )
    )
    before = len(bus.commands)
    backend.stop()
    assert len(bus.commands) > before
    for command in bus.commands[before:]:
        for motor in command.motor_cmd[:29]:
            assert (motor.dq, motor.tau, motor.kp, motor.kd) == (0.0, 0.0, 0.0, 2.5)
    backend.close()


def test_operator_alias_gesture_requests_passive_damping() -> None:
    profile = OperatorProfileCfg(
        intents={
            "request_stand": ButtonGestureCfg(held=["LT"], pressed="up"),
            "enable_policy": ButtonGestureCfg(held=["RB"], pressed="X"),
            "enter_passive": ButtonGestureCfg(held=["LT"], pressed="B"),
            "emergency_stop": ButtonGestureCfg(pressed="select"),
        }
    )
    config = _config()
    config.operator = profile
    config.__post_init__()
    bus = _Bus()
    backend = _backend(config, bus)
    backend.open()
    bus.state.wireless_remote = _remote_payload("LT")
    bus.callback(copy.deepcopy(bus.state))
    bus.state.wireless_remote = _remote_payload("LT", "B")
    bus.callback(copy.deepcopy(bus.state))

    with pytest.raises(PassiveRequestedError):
        backend.read_state()
    backend.close()


def test_select_remains_an_unconditional_emergency_stop() -> None:
    profile = OperatorProfileCfg(
        intents={
            "request_stand": ButtonGestureCfg(held=["LT"], pressed="up"),
            "enable_policy": ButtonGestureCfg(held=["RB"], pressed="X"),
            "enter_passive": ButtonGestureCfg(held=["LT"], pressed="B"),
            "emergency_stop": ButtonGestureCfg(pressed="A"),
        }
    )
    config = _config()
    config.operator = profile
    config.__post_init__()
    bus = _Bus()
    backend = _backend(config, bus)
    backend.open()
    bus.state.wireless_remote = _remote_payload("select")
    bus.callback(copy.deepcopy(bus.state))

    with pytest.raises(EmergencyStopError):
        backend.read_state()
    backend.close()


@pytest.mark.parametrize(("mode_pr", "mode_machine"), [(1, 5), (0, 4)])
def test_g1_rejects_lowstate_mode_mismatch_before_enable(mode_pr: int, mode_machine: int) -> None:
    backend = _backend(_config(), _Bus(mode_pr=mode_pr, mode_machine=mode_machine))
    with pytest.raises(RuntimeError, match="mode_pr|mode_machine"):
        backend.open()
    backend.close()


def test_g1_plugin_constructs_explicit_robot_type() -> None:
    context = RuntimeCreateContext(robot=_spec(), control=SimpleNamespace(period_s=0.02, state_timeout_s=0.1))
    runtime = create_runtime(_config(), context)
    assert isinstance(runtime, HardwareRuntime)
    assert type(runtime.robot) is UnitreeG1_29DOFRobot


def test_g1_periodic_writer_repeats_latest_command() -> None:
    bus = _Bus()
    backend = _backend(
        _config(command_publish_period_s=0.001, command_timeout_s=0.1),
        bus,
    )
    backend.open()
    deadline = time.monotonic() + 0.05
    while len(bus.commands) < 3 and time.monotonic() < deadline:
        time.sleep(0.001)
    passive = bus.commands[0]
    for motor_id in range(29):
        assert passive.motor_cmd[motor_id].mode == 1
        assert passive.motor_cmd[motor_id].q == pytest.approx(bus.state.motor_state[motor_id].q)
        assert passive.motor_cmd[motor_id].kp == 0.0
        assert passive.motor_cmd[motor_id].kd == 3.0
    backend.close()
    assert len(bus.commands) >= 3
    assert all(command.crc == 123 for command in bus.commands)


def test_g1_cached_reads_preserve_arrival_time_and_update_on_callback(monkeypatch) -> None:
    bus = _Bus()
    backend = _backend(_config(), bus)
    now = [1_000_000_000]
    backend._clock_ns = lambda: now[0]
    backend.open()
    try:
        first = backend.read_state()
        monkeypatch.setattr(backend._condition, "wait", lambda *_: pytest.fail("cached reads must not wait"))
        now[0] += 1_000_000_000
        cached = backend.read_state()
        assert cached.receive_time_ns == first.receive_time_ns
        assert cached.sample_time_ns == first.sample_time_ns
        np.testing.assert_array_equal(cached.joint_position, first.joint_position)
        bus.state.motor_state[0].q += 0.1
        bus.callback(copy.deepcopy(bus.state))
        updated = backend.read_state()
        assert updated.receive_time_ns == now[0]
        assert updated.joint_position[0] != first.joint_position[0]
    finally:
        backend.close()


def test_g1_open_wait_is_bounded_without_valid_first_frame(monkeypatch) -> None:
    config = _config()
    config.transport.connect_timeout_s = 0.001
    bus = _Bus()
    bus.state.crc = 0
    backend = _backend(config, bus)
    now = [0]
    backend._clock_ns = lambda: now[0]
    monkeypatch.setattr(backend._condition, "wait", lambda _: now.__setitem__(0, 2_000_000))
    with pytest.raises(TimeoutError, match="no valid Unitree LowState"):
        backend.open()
    assert bus.commands == []
    backend.close()
