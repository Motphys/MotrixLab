# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the Python adapter around the native G1 transport."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from motrix_deploy.artifact.schema import ControlSpec
from motrix_deploy.contracts import JointServoCommand, RobotSpec
from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime.factory import RuntimeCreateContext
from motrix_deploy_unitree.config.control import load_control_config
from motrix_deploy_unitree.config.g1 import G1_29DOF_JOINT_NAMES
from motrix_deploy_unitree.config.runtime import UnitreeRuntimeConfig, UnitreeTransportConfig
from motrix_deploy_unitree.g1.native_robot import UnitreeG1NativeRobot
from motrix_deploy_unitree.plugin import create_runtime


class _NativeFake:
    def __init__(self) -> None:
        self.sequence = 0
        self.enabled = False
        self.commands: list[tuple[str, tuple[np.ndarray, ...]]] = []
        self.closed = False

    def open(self, timeout_s: float) -> None:
        assert timeout_s > 0

    def read_state(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "receive_time_ns": 1_000_000 + self.sequence,
            "joint_position": np.arange(29, dtype=np.float32),
            "joint_velocity": -np.arange(29, dtype=np.float32),
            "quaternion_xyzw": np.asarray([0.0, 0.0, 0.0, 2.0], dtype=np.float32),
            "gyro": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "accelerometer": np.asarray([0.0, 0.0, 9.81], dtype=np.float32),
            "wireless_remote": bytes(40),
        }

    def write_command(self, *values: np.ndarray) -> None:
        self.commands.append(("servo", tuple(np.asarray(value) for value in values)))

    def write_torque(self, torque: np.ndarray) -> None:
        self.commands.append(("torque", (np.asarray(torque),)))

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def stop(self) -> None:
        self.enabled = False

    def close(self) -> None:
        self.closed = True

    def health(self) -> dict[str, object]:
        return {
            "healthy": not self.closed,
            "reason": "" if not self.closed else "backend is closed",
            "last_successful_communication_ns": 1_000_001,
        }


def _spec() -> RobotSpec:
    return RobotSpec(
        base_link_name="pelvis",
        joint_names=G1_29DOF_JOINT_NAMES,
        default_joint_position=np.zeros(29, dtype=np.float32),
        position_lower=np.full(29, -10.0, dtype=np.float32),
        position_upper=np.full(29, 10.0, dtype=np.float32),
        torque_limit=np.full(29, 100.0, dtype=np.float32),
    )


def _config() -> UnitreeRuntimeConfig:
    return UnitreeRuntimeConfig(
        transport=UnitreeTransportConfig(implementation="native_cpp", network_interface="eth0"),
        robot="g1-29dof",
        control=replace(
            load_control_config("g1"),
            startup_mode="state_machine",
        ),
    )


def test_native_adapter_normalizes_snapshot_and_forwards_command() -> None:
    native = _NativeFake()
    config = _config()
    robot = UnitreeG1NativeRobot(
        config.transport,
        config.control,
        control_period_s=0.02,
        state_timeout_s=0.1,
        spec=_spec(),
        operator=config.operator,
        native_backend=native,
        sleep=lambda _: None,
    )
    robot.open()
    state = robot.read_state()
    assert state.joint_position.shape == (29,)
    np.testing.assert_array_equal(state.joint_position[:3], np.asarray([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(state.base_orientation_xyzw, [0.0, 0.0, 0.0, 1.0])

    robot.enable()
    zeros = np.zeros(29, dtype=np.float32)
    robot.write_command(JointServoCommand(zeros, zeros, zeros, np.ones(29, np.float32), np.ones(29, np.float32)))
    assert native.enabled
    assert native.commands[-1][0] == "servo"
    # The backend forwards the command gains without applying startup overrides.
    np.testing.assert_array_equal(native.commands[-1][1][3], np.ones(29, np.float32))
    np.testing.assert_array_equal(native.commands[-1][1][4], np.ones(29, np.float32))
    robot.close()
    assert native.closed


def test_native_factory_receives_configured_damping_gain() -> None:
    """The damping kd must come from the runtime config, not a hardcoded value."""

    recorded: dict[str, object] = {}

    class _Recorder(_NativeFake):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            recorded.update(kwargs)

    config = _config()
    robot = UnitreeG1NativeRobot(
        config.transport,
        config.control,
        control_period_s=0.02,
        state_timeout_s=0.1,
        spec=_spec(),
        operator=config.operator,
        native_factory=_Recorder,
        sleep=lambda _: None,
    )
    robot.open()
    robot.close()
    assert recorded["lowcmd_topic"] == "rt/lowcmd"
    assert recorded["lowstate_topic"] == "rt/lowstate"
    assert recorded["validate_crc"] is True
    assert recorded["subscriber_queue_depth"] == 10
    assert recorded["damping_kd"] == config.control.damping_kd
    assert recorded["command_period_s"] == config.control.command_publish_period_s
    assert recorded["command_timeout_s"] == config.control.command_timeout_s


def test_native_transport_selects_g1_native_robot() -> None:
    context = RuntimeCreateContext(
        control=ControlSpec(period_s=0.02, state_timeout_s=0.1),
        robot=_spec(),
    )
    runtime = create_runtime(_config(), context)
    assert isinstance(runtime.robot, UnitreeG1NativeRobot)


def test_native_adapter_enforces_robot_command_limits() -> None:
    native = _NativeFake()
    config = _config()
    robot = UnitreeG1NativeRobot(
        config.transport,
        config.control,
        control_period_s=0.02,
        state_timeout_s=0.1,
        spec=_spec(),
        operator=config.operator,
        native_backend=native,
        sleep=lambda _: None,
    )
    robot.open()
    robot.enable()
    zeros = np.zeros(29, dtype=np.float32)
    with pytest.raises(ValidationError):
        robot.write_command(
            JointServoCommand(
                joint_position=np.full(29, 11.0, dtype=np.float32),
                joint_velocity=zeros,
                feedforward_torque=zeros,
                kp=np.ones(29, dtype=np.float32),
                kd=np.ones(29, dtype=np.float32),
            )
        )
    robot.close()


def test_native_cached_reads_preserve_time_and_process_gestures_once() -> None:
    native = _NativeFake()
    config = _config()
    robot = UnitreeG1NativeRobot(
        config.transport,
        config.control,
        spec=_spec(),
        control_period_s=0.02,
        state_timeout_s=0.1,
        native_backend=native,
        sleep=lambda _: None,
    )
    seen = []
    robot._intent_evaluator = SimpleNamespace(update=lambda buttons, stamp: seen.append(stamp) or set())
    robot.open()
    try:
        first = robot.read_state()
        cached = robot.read_state()
        assert first.receive_time_ns == cached.receive_time_ns
        assert first.sample_time_ns == cached.sample_time_ns
        np.testing.assert_array_equal(first.joint_position, cached.joint_position)
        assert seen == [first.receive_time_ns]
        native.sequence += 1
        updated = robot.read_state()
        assert updated.receive_time_ns > first.receive_time_ns
        assert seen == [first.receive_time_ns, updated.receive_time_ns]
    finally:
        robot.close()


@pytest.mark.parametrize("startup_mode", ["one_shot", "state_machine"])
def test_native_enable_allows_output_without_startup_motion_or_confirmation(startup_mode) -> None:
    config = _config()
    native = _NativeFake()
    robot = UnitreeG1NativeRobot(
        config.transport,
        config.control,
        spec=_spec(),
        control_period_s=0.02,
        state_timeout_s=0.1,
        native_backend=native,
        sleep=lambda _: pytest.fail("enable must not wait for startup"),
    )
    robot.open()
    try:
        robot.enable()
        assert native.enabled
        assert native.commands == []
        zeros = np.zeros(29, dtype=np.float32)
        robot.write_command(JointServoCommand(zeros, zeros, zeros, zeros, zeros))
        assert len(native.commands) == 1
    finally:
        robot.close()
