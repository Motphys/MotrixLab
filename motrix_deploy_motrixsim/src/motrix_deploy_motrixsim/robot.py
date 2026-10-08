# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Scene-derived robot I/O without simulation lifecycle ownership."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import motrixsim as mtx
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
from motrix_deploy.errors import ValidationError
from motrix_deploy.robot.interface import RobotInterface
from motrix_env_core.config.scene.base import RobotCfg

if TYPE_CHECKING:
    from motrix_deploy_motrixsim.runtime import MotrixSimRuntime


class Robot(RobotInterface):
    """Bind canonical I/O to a compiled scene without owning or stepping it.

    The runtime converts actuators before construction, and calls ``bind`` on
    every open (including reopen with newly allocated model/data). It invokes
    ``apply_control`` before *every* physics step, not just each policy tick.
    """

    def __init__(self, runtime: MotrixSimRuntime, spec: RobotSpec, actuator_names: tuple[str, ...]) -> None:
        self._runtime = runtime
        self._spec = spec
        self._robot: RobotCfg = runtime.scene.objs.robot
        self._command: RobotCommand | None = None
        self._opened = False
        self._closed = False
        self._last_communication_ns = 0
        self._health_reason = "simulation is not open"
        self._read: mtx.query.QueryProgram | None = None
        self._read_fields: tuple[str, ...] = ()
        self._joint_read: mtx.query.QueryProgram | None = None
        self._write: mtx.write.WriteProgram | None = None
        self._joint_names = tuple(self._robot.resolve_name(name) for name in spec.joint_names)
        self._sensor_names: dict[str, str] = {}
        self._actuator_names = actuator_names
        self._joint_dof_pos_indices = np.empty(0, dtype=np.intp)
        self._capabilities = RobotCapabilities(
            control_modes=(JointControlMode.SERVO, JointControlMode.TORQUE),
            state_fields=frozenset(
                {
                    "joint_position",
                    "joint_velocity",
                    "base_orientation_xyzw",
                    "base_angular_velocity",
                    "base_linear_acceleration",
                    "base_linear_velocity",
                }
            ),
            supports_rendering=True,
            max_command_rate_hz=1.0 / runtime.config.physics.dt,
            stop_semantics="zero_torque",
        )

    @property
    def spec(self) -> RobotSpec:
        return self._spec

    @property
    def capabilities(self) -> RobotCapabilities:
        return self._capabilities

    @property
    def joint_dof_pos_indices(self) -> np.ndarray:
        """Canonical joint position addresses in the currently bound model."""
        return self._joint_dof_pos_indices

    def _bind_model(self) -> None:
        """Bind final-model addresses; the runtime validated the source model."""
        model = self._runtime.model
        self._joint_dof_pos_indices = np.asarray(
            [model.get_joint(name).dof_pos_index for name in self._joint_names], dtype=np.intp
        )
        self._validate_sensors()

    def _validate_sensors(self) -> None:
        sensors = {sensor.name: sensor for sensor in self._runtime.world.sensors.frame}
        self._sensor_names.clear()
        for role, sensor_type, reference in (
            ("base_angular_velocity", mtx.msd.FrameSensorType.FrameAngVel, "local"),
            ("base_linear_acceleration", mtx.msd.FrameSensorType.FrameLinAcc, "local"),
            ("base_linear_velocity", mtx.msd.FrameSensorType.FrameLinVel, "world"),
        ):
            path = f"backend.sensor_bindings.{role}"
            local_name = getattr(self._runtime.config.sensor_bindings, role)
            if local_name is None and role == "base_linear_velocity":
                continue
            if local_name is None:
                raise ValidationError(path, "an explicit model-local sensor name", local_name)
            name = self._robot.resolve_name(local_name)
            sensor = sensors.get(name)
            if sensor is None or sensor.sensor_type != sensor_type or sensor.ref_frame.variant != reference:
                raise ValidationError(path, f"{sensor_type} in the {reference} frame", name)
            target = sensor.object_type
            if reference == "local" or target.variant == "site":
                site = self._runtime.model.get_site(target.value) if target.variant == "site" else None
                if (
                    site is None
                    or site.parent_link is None
                    or site.parent_link.name != self._robot.resolved_base_link_name
                    or not np.allclose(np.asarray(site.local_quat).reshape(4)[:3], 0, atol=1e-7, rtol=0)
                ):
                    raise ValidationError(path, "a base-body site aligned with the body frame", name)
            elif target.variant != "link" or target.value != self._robot.resolved_base_link_name:
                raise ValidationError(path, "the base link or an aligned base-body site", name)
            self._sensor_names[role] = name

    def bind(self) -> None:
        """Rebuild model-bound programs, then reset through the owning runtime."""
        self._bind_model()
        self._runtime.reset()
        self.refresh_reads()
        self._command = None
        self._opened = True
        self._closed = False
        self.stop()
        self._health_reason = ""
        self._last_communication_ns = time.monotonic_ns()

    def refresh_reads(self) -> None:
        """Reallocate query/write programs after runtime-owned data replacement."""
        model, data = self._runtime.model, self._runtime.data
        joint_queries = {
            "joint_position": mtx.query.JointPosition(self._joint_names),
            "joint_velocity": mtx.query.JointVelocity(self._joint_names),
        }
        self._joint_read = model.compile_query(joint_queries).allocate(data)
        queries = {
            **joint_queries,
            "base_orientation_xyzw": mtx.query.LinkRotation((self._robot.resolved_base_link_name,)),
        }
        queries.update({role: mtx.query.SensorValues((name,)) for role, name in self._sensor_names.items()})
        self._read = model.compile_query(queries).allocate(data)
        self._read_fields = tuple(queries)
        self._write = model.compile_write({"torque": mtx.write.ActuatorCtrls(self._actuator_names)}).allocate(data)

    def open(self) -> None:
        """Check activation without opening the runtime-owned simulation."""
        self._require_open()

    def read_state(self, timeout_s: float) -> RobotState:
        self._require_open()
        if not np.isfinite(timeout_s) or timeout_s <= 0:
            raise TimeoutError(f"MotrixSim state timeout must be positive and finite, got {timeout_s}")
        assert self._read is not None
        self._read.execute(self._runtime.data)
        self._last_communication_ns = time.monotonic_ns()
        # Native programs reuse their buffers; public samples must own snapshots.
        # RobotState validates the canonical shapes, dtype and finite values.
        fields = {
            name: np.array(self._read[name].reshape(-1), dtype=np.float32, copy=True) for name in self._read_fields
        }
        return RobotState(
            sample_time_ns=round(self._runtime.simulation_time_s * 1e9),
            receive_time_ns=self._last_communication_ns,
            **fields,
        )

    def write_command(self, command: RobotCommand) -> None:
        self._require_open()
        shape = (self._spec.joint_count,)
        if isinstance(command, JointServoCommand):
            snapshot = JointServoCommand(
                joint_position=command.joint_position.copy(),
                joint_velocity=command.joint_velocity.copy(),
                feedforward_torque=command.feedforward_torque.copy(),
                kp=command.kp.copy(),
                kd=command.kd.copy(),
            )
            if snapshot.joint_position.shape != shape:
                raise ValidationError("command.joint_position.shape", shape, snapshot.joint_position.shape)
            if np.any(snapshot.joint_position < self._spec.position_lower) or np.any(
                snapshot.joint_position > self._spec.position_upper
            ):
                raise ValidationError(
                    "command.joint_position", "inside RobotSpec position range", snapshot.joint_position
                )
            if np.any(np.abs(snapshot.feedforward_torque) > self._spec.torque_limit):
                raise ValidationError(
                    "command.feedforward_torque", "inside RobotSpec torque limits", snapshot.feedforward_torque
                )
            self._command = snapshot
        elif isinstance(command, JointTorqueCommand):
            snapshot = JointTorqueCommand(torque=command.torque.copy())
            if snapshot.torque.shape != shape:
                raise ValidationError("command.torque.shape", shape, snapshot.torque.shape)
            if np.any(np.abs(snapshot.torque) > self._spec.torque_limit):
                raise ValidationError("command.torque", "inside RobotSpec torque limits", snapshot.torque)
            self._command = snapshot
        else:
            raise ValidationError("command", "JointServoCommand or JointTorqueCommand", type(command).__name__)
        self._last_communication_ns = time.monotonic_ns()

    def apply_control(self) -> None:
        """Recompute PD + feedforward from current physics state and clip torque."""
        if not self._opened or self._closed:
            return
        assert self._joint_read is not None and self._write is not None
        command = self._command
        if command is None:
            torque = 0.0
        elif isinstance(command, JointTorqueCommand):
            torque = command.torque
        else:
            self._joint_read.execute(self._runtime.data)
            position = self._joint_read["joint_position"].reshape(-1)
            velocity = self._joint_read["joint_velocity"].reshape(-1)
            torque = (
                command.kp * (command.joint_position - position)
                + command.kd * (command.joint_velocity - velocity)
                + command.feedforward_torque
            )
        self._write["torque"][:] = np.clip(torque, -self._spec.torque_limit, self._spec.torque_limit)
        self._write.execute(self._runtime.data)

    def health(self) -> HealthStatus:
        healthy = self._opened and not self._closed and self._runtime.opened and not self._health_reason
        return HealthStatus(
            healthy=healthy,
            reason=self._health_reason if healthy or self._health_reason else "simulation is not open",
            last_successful_communication_ns=self._last_communication_ns,
        )

    def stop(self) -> None:
        """Clear the command and apply zero torque immediately."""
        self._command = None
        if self._opened and not self._closed and self._write is not None:
            self._write["torque"][:] = 0
            self._write.execute(self._runtime.data)

    def close(self) -> None:
        """Detach I/O only; the simulation, renderer and clock remain runtime-owned."""
        self.stop()
        self._closed = True
        self._health_reason = "robot port is closed"

    def invalidate(self) -> None:
        self._command = None
        self._read = None
        self._joint_read = None
        self._write = None
        self._opened = False
        self._closed = True
        self._health_reason = "simulation is not open"

    def _require_open(self) -> None:
        if not self._opened or self._closed or not self._runtime.opened:
            raise RuntimeError("MotrixSim robot port is not open")
