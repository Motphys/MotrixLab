"""Contract tests for the Python adapter around the native G1 transport."""

from __future__ import annotations

import numpy as np
import pytest

from motrix_deploy.artifact.schema import ControlSpec
from motrix_deploy.contracts import JointServoCommand, RobotSpec
from motrix_deploy.errors import ValidationError
from motrix_deploy.runtime.factory import RuntimeCreateContext
from motrix_deploy_unitree.config import UnitreeBackendConfig, UnitreeG1_29DOFRobotCfg
from motrix_deploy_unitree.hardware import G1_29DOF_JOINT_NAMES
from motrix_deploy_unitree.native_interface import UnitreeG1NativeRobot
from motrix_deploy_unitree.plugin import create_runtime


class _NativeFake:
    def __init__(self) -> None:
        self.sequence = 0
        self.enabled = False
        self.commands: list[tuple[str, tuple[np.ndarray, ...]]] = []
        self.closed = False

    def open(self, timeout_s: float) -> None:
        assert timeout_s > 0

    def read_state(self, timeout_s: float, after_sequence: int) -> dict[str, object]:
        assert timeout_s > 0
        self.sequence = max(self.sequence + 1, after_sequence + 1)
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


def _config() -> UnitreeBackendConfig:
    return UnitreeBackendConfig(
        transport="native_cpp",
        network_interface="eth0",
        robot=UnitreeG1_29DOFRobotCfg(
            wait_for_remote_buttons=False,
            startup_mode="state_machine",
            kp=20.0,
            kd=1.0,
        ),
    )


def test_native_adapter_normalizes_snapshot_and_forwards_command() -> None:
    native = _NativeFake()
    robot = UnitreeG1NativeRobot(_config(), 0.02, 0.1, spec=_spec(), native_backend=native, sleep=lambda _: None)
    robot.open()
    state = robot.read_state(0.1)
    assert state.joint_position.shape == (29,)
    np.testing.assert_array_equal(state.joint_position[:3], np.asarray([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(state.base_orientation_xyzw, [0.0, 0.0, 0.0, 1.0])

    robot.enable()
    zeros = np.zeros(29, dtype=np.float32)
    robot.write_command(JointServoCommand(zeros, zeros, zeros, np.ones(29, np.float32), np.ones(29, np.float32)))
    assert native.enabled
    assert native.commands[-1][0] == "servo"
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
    robot = UnitreeG1NativeRobot(config, 0.02, 0.1, spec=_spec(), native_factory=_Recorder, sleep=lambda _: None)
    robot.open()
    robot.close()
    assert recorded["damping_kd"] == config.robot.damping_kd
    assert recorded["command_period_s"] == config.robot.command_publish_period_s
    assert recorded["command_timeout_s"] == config.robot.command_timeout_s


def test_native_transport_selects_g1_native_robot() -> None:
    context = RuntimeCreateContext(
        control=ControlSpec(period_s=0.02, state_timeout_s=0.1),
        robot=_spec(),
    )
    runtime = create_runtime(_config(), context)
    assert isinstance(runtime.robot, UnitreeG1NativeRobot)


def test_native_adapter_enforces_robot_command_limits() -> None:
    native = _NativeFake()
    robot = UnitreeG1NativeRobot(_config(), 0.02, 0.1, spec=_spec(), native_backend=native, sleep=lambda _: None)
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
