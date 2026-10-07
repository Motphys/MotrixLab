# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Direct state and command API backed by the production Unitree deploy adapter."""

import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from motrix_deploy.artifact import read_artifact
from motrix_deploy.contracts import HealthStatus, JointServoCommand, RobotCommand, RobotSpec, RobotState
from motrix_deploy.errors import ValidationError
from motrix_deploy.profile import DeploymentProfile
from motrix_deploy_unitree.config import UnitreeGo2BackendConfig
from motrix_deploy_unitree.interface import UnitreeGo2RobotInterface, UnitreeSdkBindings


@runtime_checkable
class _JointGainSpec(Protocol):
    """Structural task contract needed by direct joint control, not a task runtime."""

    @property
    def kp(self) -> list[float]: ...

    @property
    def kd(self) -> list[float]: ...


def _joint_gains(task: object) -> _JointGainSpec:
    if not isinstance(task, _JointGainSpec):
        raise ValidationError("task", "a typed task spec with kp and kd joint vectors", type(task).__name__)
    return task


class UnitreeGo2DirectInterface:
    """Read state and send canonical commands without running a policy.

    Construction uses either an immutable deployment artifact or an explicit
    deployment profile. DDS, CRC, joint mapping, remote enable, emergency stop,
    validation, damping, and cleanup all delegate to the production
    UnitreeGo2RobotInterface used by policy deployment.
    """

    def __init__(
        self,
        *,
        kp: list[float],
        kd: list[float],
        control_period_s: float,
        state_timeout_s: float,
        backend: UnitreeGo2RobotInterface,
    ) -> None:
        for path, value in (
            ("control.period_s", control_period_s),
            ("control.state_timeout_s", state_timeout_s),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValidationError(path, "a positive finite number", value)
        self.robot = backend.spec
        self._kp = self._command_array(kp, "kp")
        self._kd = self._command_array(kd, "kd")
        self.control_period_s = float(control_period_s)
        self.state_timeout_s = float(state_timeout_s)
        self.backend = backend
        self._opened = False
        self._closed = False
        # Direct callers provide explicit typed gains; preserve the main direct API's
        # startup fallback without changing policy runtime gain requirements.
        backend.config = replace(
            backend.config,
            kp=self._kp.copy() if backend.config.kp is None else backend.config.kp,
            kd=self._kd.copy() if backend.config.kd is None else backend.config.kd,
        )

    @classmethod
    def from_profile(
        cls,
        profile: DeploymentProfile,
        *,
        network_interface: str,
        joint_name_to_motor_index: Mapping[str, int] | None = None,
        backend_options: Mapping[str, Any] | None = None,
        sdk: UnitreeSdkBindings | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "UnitreeGo2DirectInterface":
        """Build the direct API from an explicit robot, gain, and timing profile."""
        gains = _joint_gains(profile.task)
        return cls._from_contract(
            robot=profile.robot,
            kp=gains.kp,
            kd=gains.kd,
            control_period_s=profile.control.period_s,
            state_timeout_s=profile.control.state_timeout_s,
            network_interface=network_interface,
            joint_name_to_motor_index=joint_name_to_motor_index,
            backend_options=backend_options,
            sdk=sdk,
            clock_ns=clock_ns,
            sleep=sleep,
        )

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        network_interface: str,
        joint_name_to_motor_index: Mapping[str, int] | None = None,
        backend_options: Mapping[str, Any] | None = None,
        sdk: UnitreeSdkBindings | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "UnitreeGo2DirectInterface":
        """Build the direct API from the same artifact and settings as deployment."""
        artifact = read_artifact(artifact_path)
        manifest = artifact.manifest
        gains = _joint_gains(manifest.task)
        return cls._from_contract(
            robot=manifest.robot,
            kp=gains.kp,
            kd=gains.kd,
            control_period_s=manifest.control.period_s,
            state_timeout_s=manifest.control.state_timeout_s,
            network_interface=network_interface,
            joint_name_to_motor_index=joint_name_to_motor_index,
            backend_options=backend_options,
            sdk=sdk,
            clock_ns=clock_ns,
            sleep=sleep,
        )

    @classmethod
    def _from_contract(
        cls,
        *,
        robot: RobotSpec,
        kp: list[float],
        kd: list[float],
        control_period_s: float,
        state_timeout_s: float,
        network_interface: str,
        joint_name_to_motor_index: Mapping[str, int] | None,
        backend_options: Mapping[str, Any] | None,
        sdk: UnitreeSdkBindings | None,
        clock_ns: Callable[[], int],
        sleep: Callable[[float], None],
    ) -> "UnitreeGo2DirectInterface":
        """Construct the production backend from validated control contracts."""
        options = dict(backend_options or {})
        options["network_interface"] = network_interface
        if joint_name_to_motor_index is not None:
            options["joint_name_to_motor_index"] = dict(joint_name_to_motor_index)
        config = UnitreeGo2BackendConfig.from_mapping(options)
        backend = UnitreeGo2RobotInterface(
            config,
            control_period_s=control_period_s,
            state_timeout_s=state_timeout_s,
            spec=robot,
            sdk=sdk,
            clock_ns=clock_ns,
            sleep=sleep,
        )
        return cls(
            kp=kp,
            kd=kd,
            control_period_s=control_period_s,
            state_timeout_s=state_timeout_s,
            backend=backend,
        )

    def open(self) -> None:
        """Open DDS and wait for the first valid LowState sample."""
        if self._closed:
            raise RuntimeError("direct Unitree interface cannot be reopened after close()")
        if self._opened:
            raise RuntimeError("direct Unitree interface is already open")
        try:
            self.backend.open()
        except Exception:
            try:
                self.backend.close()
            finally:
                self._closed = True
            raise
        self._opened = True

    def read_data(self, timeout_s: float | None = None) -> RobotState:
        """Read one fresh canonical state sample without running policy inference."""
        self._require_open()
        return self.backend.read_state(self.state_timeout_s if timeout_s is None else timeout_s)

    def default_pose_command(self) -> JointServoCommand:
        """Build the zero-action default-pose command stored by the control contract."""
        zeros = np.zeros(self.robot.joint_count, dtype=np.float32)
        return JointServoCommand(
            joint_position=np.array(self.robot.default_joint_position, copy=True),
            joint_velocity=zeros,
            feedforward_torque=zeros,
            kp=self._kp.copy(),
            kd=self._kd.copy(),
        )

    def enable_command_output(self) -> None:
        """Run the production Start -> default pose -> A hardware enable sequence."""
        self._require_open()
        self.backend.enable()

    def make_joint_command(
        self,
        joint_position: object,
        *,
        joint_velocity: object | None = None,
        feedforward_torque: object | None = None,
        kp: object | None = None,
        kd: object | None = None,
    ) -> JointServoCommand:
        """Construct one canonical float32 command using configured gains by default."""
        zeros = np.zeros(self.robot.joint_count, dtype=np.float32)
        return JointServoCommand(
            joint_position=self._command_array(joint_position, "joint_position"),
            joint_velocity=(zeros if joint_velocity is None else self._command_array(joint_velocity, "joint_velocity")),
            feedforward_torque=(
                zeros if feedforward_torque is None else self._command_array(feedforward_torque, "feedforward_torque")
            ),
            kp=self._kp.copy() if kp is None else self._command_array(kp, "kp"),
            kd=self._kd.copy() if kd is None else self._command_array(kd, "kd"),
        )

    def send_command(self, command: RobotCommand) -> None:
        """Send one canonical command through the production LowCmd writer."""
        self._require_open()
        self.backend.write_command(command)

    def send_joint_command(
        self,
        joint_position: object,
        *,
        joint_velocity: object | None = None,
        feedforward_torque: object | None = None,
        kp: object | None = None,
        kd: object | None = None,
    ) -> JointServoCommand:
        """Construct and send one command, returning the exact command sent."""
        command = self.make_joint_command(
            joint_position,
            joint_velocity=joint_velocity,
            feedforward_torque=feedforward_torque,
            kp=kp,
            kd=kd,
        )
        self.send_command(command)
        return command

    def health(self) -> HealthStatus:
        """Return production backend health."""
        return self.backend.health()

    def stop(self) -> None:
        """Disable output and publish the configured damping stop."""
        if self._opened:
            self.backend.stop()

    def close(self) -> None:
        """Stop command output and close both DDS channels."""
        if self._closed:
            return
        try:
            if self._opened:
                self.backend.close()
        finally:
            self._opened = False
            self._closed = True

    def __enter__(self) -> "UnitreeGo2DirectInterface":
        self.open()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        self.close()

    def _command_array(self, value: object, name: str) -> np.ndarray:
        try:
            array = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValidationError(
                f"command.{name}",
                f"{self.robot.joint_count} numeric values",
                value,
            ) from error
        expected = (self.robot.joint_count,)
        if array.shape != expected or not np.all(np.isfinite(array)):
            raise ValidationError(f"command.{name}", f"finite shape {expected}", array)
        return np.array(array, dtype=np.float32, copy=True)

    def _require_open(self) -> None:
        if not self._opened or self._closed:
            raise RuntimeError("direct Unitree interface is not open")


__all__ = ["UnitreeGo2DirectInterface"]
