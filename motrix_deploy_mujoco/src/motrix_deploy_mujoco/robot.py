# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Robot I/O adapter bound to an externally owned MuJoCo simulation."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
from motrix_deploy_mujoco.transform import convert_actuators_to_motors
from motrix_env_core.config.scene.base import RobotCfg

if TYPE_CHECKING:
    import mujoco

    from motrix_deploy_mujoco.runtime import MujocoRuntime

logger = logging.getLogger(__name__)


def wxyz_to_xyzw(quaternion: ArrayLike) -> NDArray[np.float32]:
    """Convert MuJoCo's explicit ``wxyz`` order to the public ``xyzw`` order."""
    return np.asarray(quaternion, dtype=np.float32)[[1, 2, 3, 0]]


def xyzw_to_wxyz(quaternion: ArrayLike) -> NDArray[np.generic]:
    """Convert the public ``xyzw`` order to MuJoCo's ``wxyz`` order, preserving dtype."""
    return np.asarray(quaternion)[[3, 0, 1, 2]]


class Robot(RobotInterface):
    """Robot I/O and joint control in an externally owned MuJoCo simulation."""

    def __init__(self, runtime: MujocoRuntime, spec: RobotSpec) -> None:
        self.config = runtime.config
        self._runtime = runtime
        self._cfg = cast(RobotCfg, self._runtime.scene.objs.robot)
        self._command: RobotCommand | None = None
        self._spec = spec
        self._capabilities: RobotCapabilities | None = None
        self._joint_qpos_indices: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._joint_qvel_indices: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._actuator_indices: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._base_body_id = -1
        self._sensor_slices: dict[str, slice] = {}
        self._last_communication_ns = 0
        self._health_reason = "backend is not open"
        self._opened = False
        self._closed = False

    @property
    def spec(self) -> RobotSpec:
        return self._spec

    @property
    def capabilities(self) -> RobotCapabilities:
        if self._capabilities is None:
            raise RuntimeError("MuJoCo capabilities are available after open()")
        return self._capabilities

    def bind(self) -> None:
        spec = self._spec
        self._bind_base()
        self._bind_joints_and_actuators(spec)
        self._validate_torque_actuator_contract(spec)
        self._bind_sensors()
        self._runtime.reset(self._joint_qpos_indices, spec.default_joint_position)
        self._command = None
        self._capabilities = RobotCapabilities(
            control_modes=(JointControlMode.SERVO, JointControlMode.TORQUE),
            state_fields=frozenset(
                {
                    "joint_position",
                    "joint_velocity",
                    "base_orientation_xyzw",
                    "base_angular_velocity",
                    "base_linear_acceleration",
                    "base_position",
                    "base_linear_velocity",
                }
            ),
            supports_rendering=True,
            max_command_rate_hz=1.0 / self._runtime.sim.dt,
            stop_semantics="zero_torque",
        )
        self._opened = True
        self._closed = False
        self._last_communication_ns = time.monotonic_ns()
        self._health_reason = ""

    def read_state(self, timeout_s: float) -> RobotState:
        self._require_open()
        if timeout_s <= 0:
            raise TimeoutError(f"MuJoCo state timeout must be positive, got {timeout_s}")
        self._runtime.check_viewer_running()
        self._last_communication_ns = time.monotonic_ns()
        return self._state()

    def write_command(self, command: RobotCommand) -> None:
        self._require_open()
        assert self._runtime.model is not None and self._runtime.data is not None
        joint_count = self._spec.joint_count
        fields: tuple[str, ...]
        if isinstance(command, JointServoCommand):
            fields = ("joint_position", "joint_velocity", "feedforward_torque", "kp", "kd")
        elif isinstance(command, JointTorqueCommand):
            fields = ("torque",)
        else:
            raise ValidationError("command", "JointServoCommand or JointTorqueCommand", type(command).__name__)
        for field_name in fields:
            value = getattr(command, field_name)
            if value.shape != (joint_count,) or not np.isfinite(value).all():
                raise ValidationError(f"command.{field_name}", f"finite shape ({joint_count},)", value)
        if isinstance(command, JointServoCommand):
            if np.any(command.joint_position < self._spec.position_lower) or np.any(
                command.joint_position > self._spec.position_upper
            ):
                raise ValidationError(
                    "command.joint_position", "inside RobotSpec position range", command.joint_position
                )
            if np.any(command.kp < 0) or np.any(command.kd < 0):
                raise ValidationError("command.gains", "non-negative kp/kd", (command.kp, command.kd))
            torque = command.feedforward_torque
            torque_field = "feedforward_torque"
        else:
            torque = command.torque
            torque_field = "torque"
        if np.any(np.abs(torque) > self._spec.torque_limit):
            raise ValidationError(f"command.{torque_field}", "inside RobotSpec torque limits", torque)

        # Snapshot the command: the caller may reuse or mutate its arrays afterwards.
        self._command = type(command)(**{name: getattr(command, name).copy() for name in fields})
        self._last_communication_ns = time.monotonic_ns()

    def apply_control(self) -> None:
        assert self._runtime.data is not None
        command = self._command
        if command is None:
            self._runtime.data.ctrl[self._actuator_indices] = 0.0
            return
        if isinstance(command, JointTorqueCommand):
            self._runtime.data.ctrl[self._actuator_indices] = command.torque
            return
        position = self._runtime.data.qpos[self._joint_qpos_indices]
        velocity = self._runtime.data.qvel[self._joint_qvel_indices]
        torque = (
            command.kp * (command.joint_position - position)
            + command.kd * (command.joint_velocity - velocity)
            + command.feedforward_torque
        )
        self._runtime.data.ctrl[self._actuator_indices] = np.clip(
            torque, -self._spec.torque_limit, self._spec.torque_limit
        )

    def health(self) -> HealthStatus:
        return HealthStatus(
            healthy=self._opened and not self._closed and not self._health_reason,
            reason=self._health_reason,
            last_successful_communication_ns=self._last_communication_ns,
        )

    def stop(self) -> None:
        self._command = None
        if not self._opened or self._closed or self._runtime.data is None:
            return
        if self._joint_qpos_indices.size:
            self._runtime.data.ctrl[self._actuator_indices] = 0.0

    def open(self) -> None:
        """Check that the owning runtime has prepared and bound robot I/O."""
        self._require_open()

    def close(self) -> None:
        """Detach robot I/O without closing the externally owned simulation."""
        self.stop()
        self._closed = True
        self._health_reason = "robot interface is closed"

    def invalidate(self) -> None:
        self._command = None
        self._opened = False
        self._closed = True
        self._health_reason = "simulation is not open"

    def _bind_base(self) -> None:
        assert self._runtime.model is not None
        body_name = self._cfg.resolved_base_link_name
        body_id = self._name_id(self._runtime.mj.mjtObj.mjOBJ_BODY, body_name, "base body")
        joint_address = int(self._runtime.model.body_jntadr[body_id])
        if joint_address < 0 or self._runtime.model.jnt_type[joint_address] != self._runtime.mj.mjtJoint.mjJNT_FREE:
            raise ValidationError("backend.robot.base_link_name", "a body with a free joint", body_name)
        self._base_body_id = body_id

    def prepare(self, source_model: mujoco.MjModel, model_spec: mujoco.MjSpec) -> None:
        robot = self._spec
        self._bind_joints_and_actuators(robot)
        self._validate_actuator_transmission()
        motors = source_model.actuator_biastype[self._actuator_indices] == self._runtime.mj.mjtBias.mjBIAS_NONE
        if np.all(motors):
            self._validate_torque_actuator_contract(robot)
            joint_ids = source_model.actuator_trnid[self._actuator_indices, 0]
            if not np.all(source_model.jnt_limited[joint_ids]) or not np.allclose(
                source_model.jnt_range[joint_ids],
                np.column_stack((robot.position_lower, robot.position_upper)),
                atol=1e-6,
                rtol=0.0,
            ):
                raise ValidationError("backend.joints.position_range", "RobotSpec position limits", "mismatch")
            return
        if np.any(motors):
            raise ValidationError("backend.actuators", "uniform position servos or torque motors", "mixed actuators")
        self._validate_position_actuator_contract(robot)

        actuator_names = convert_actuators_to_motors(
            model_spec,
            source_model,
            self._actuator_indices,
            robot,
        )
        logger.info(
            "Converting %d actuators from position servos to torque motors with MjSpec: %s",
            len(actuator_names),
            ", ".join(actuator_names),
        )

    def _bind_joints_and_actuators(self, spec: RobotSpec) -> None:
        assert self._runtime.model is not None
        qpos_indices: list[int] = []
        qvel_indices: list[int] = []
        actuator_indices: list[int] = []
        for name in spec.joint_names:
            joint_id = self._name_id(self._runtime.mj.mjtObj.mjOBJ_JOINT, self._cfg.resolve_name(name), "joint")
            if self._runtime.model.jnt_type[joint_id] != self._runtime.mj.mjtJoint.mjJNT_HINGE:
                raise ValidationError(
                    f"backend.joints.{name}", "a one-DoF hinge joint", self._runtime.model.jnt_type[joint_id]
                )
            matches = np.flatnonzero(self._runtime.model.actuator_trnid[:, 0] == joint_id)
            if matches.size != 1:
                raise ValidationError(f"backend.actuators.{name}", "exactly one actuator", matches.tolist())
            qpos_indices.append(int(self._runtime.model.jnt_qposadr[joint_id]))
            qvel_indices.append(int(self._runtime.model.jnt_dofadr[joint_id]))
            actuator_indices.append(int(matches[0]))
        self._joint_qpos_indices = np.asarray(qpos_indices, dtype=np.int64)
        self._joint_qvel_indices = np.asarray(qvel_indices, dtype=np.int64)
        self._actuator_indices = np.asarray(actuator_indices, dtype=np.int64)

    def _validate_actuator_transmission(self) -> None:
        assert self._runtime.model is not None
        model = self._runtime.model
        indices = self._actuator_indices
        if not np.all(model.actuator_trntype[indices] == self._runtime.mj.mjtTrn.mjTRN_JOINT):
            raise ValidationError("backend.actuators.transmission", "direct joint transmission", "unsupported")
        gear = np.zeros((indices.size, 6))
        gear[:, 0] = 1.0
        if not np.allclose(model.actuator_gear[indices], gear):
            raise ValidationError("backend.actuators.gear", "unit joint transmission", model.actuator_gear[indices])
        for field, expected in (
            ("dyntype", self._runtime.mj.mjtDyn.mjDYN_NONE),
            ("gaintype", self._runtime.mj.mjtGain.mjGAIN_FIXED),
            ("ctrllimited", True),
            ("forcelimited", True),
        ):
            actual = getattr(model, f"actuator_{field}")[indices]
            if not np.all(actual == expected):
                raise ValidationError(f"backend.actuators.{field}", str(expected), actual)

    def _validate_position_actuator_contract(self, spec: RobotSpec) -> None:
        assert self._runtime.model is not None
        indices = self._actuator_indices
        ctrl_range = self._runtime.model.actuator_ctrlrange[indices]
        force_range = self._runtime.model.actuator_forcerange[indices]
        kp = self._runtime.model.actuator_gainprm[indices, 0]
        stiffness = -self._runtime.model.actuator_biasprm[indices, 1]
        kd = -self._runtime.model.actuator_biasprm[indices, 2]
        checks = (
            ("ctrl_lower", ctrl_range[:, 0], spec.position_lower),
            ("ctrl_upper", ctrl_range[:, 1], spec.position_upper),
            ("force_lower", force_range[:, 0], -spec.torque_limit),
            ("force_upper", force_range[:, 1], spec.torque_limit),
        )
        for name, actual, expected in checks:
            if not np.allclose(actual, expected, atol=1e-6, rtol=0.0):
                raise ValidationError(f"backend.actuators.{name}", expected.tolist(), actual.tolist())
        if not np.allclose(kp, stiffness, atol=1e-7, rtol=0.0):
            raise ValidationError("backend.actuators.kp", "gain equals stiffness", (kp.tolist(), stiffness.tolist()))
        if not np.all(self._runtime.model.actuator_biastype[indices] == self._runtime.mj.mjtBias.mjBIAS_AFFINE):
            raise ValidationError("backend.actuators.bias", "position servo affine bias", "unsupported")
        if not np.allclose(self._runtime.model.actuator_biasprm[indices, 0], 0.0):
            raise ValidationError("backend.actuators.bias", "no constant force bias", "unsupported")
        if not np.isfinite(kp).all() or not np.isfinite(kd).all() or np.any(kp < 0) or np.any(kd < 0):
            raise ValidationError("backend.actuators.gains", "finite non-negative kp/kd", (kp.tolist(), kd.tolist()))

    def _validate_torque_actuator_contract(self, spec: RobotSpec) -> None:
        assert self._runtime.model is not None
        indices = self._actuator_indices
        checks = (
            ("ctrl_lower", self._runtime.model.actuator_ctrlrange[indices, 0], -spec.torque_limit),
            ("ctrl_upper", self._runtime.model.actuator_ctrlrange[indices, 1], spec.torque_limit),
            ("force_lower", self._runtime.model.actuator_forcerange[indices, 0], -spec.torque_limit),
            ("force_upper", self._runtime.model.actuator_forcerange[indices, 1], spec.torque_limit),
            ("gain", self._runtime.model.actuator_gainprm[indices, 0], np.ones(spec.joint_count)),
            ("bias", self._runtime.model.actuator_biastype[indices], np.zeros(spec.joint_count)),
        )
        for name, actual, expected in checks:
            if not np.allclose(actual, expected, atol=1e-6, rtol=0.0):
                raise ValidationError(f"backend.torque_actuators.{name}", str(expected.tolist()), actual.tolist())

    def _bind_sensors(self) -> None:
        assert self._runtime.model is not None
        self._sensor_slices.clear()
        for role, expected_type in (
            ("base_angular_velocity", self._runtime.mj.mjtSensor.mjSENS_GYRO),
            ("base_linear_acceleration", self._runtime.mj.mjtSensor.mjSENS_ACCELEROMETER),
            ("base_linear_velocity", self._runtime.mj.mjtSensor.mjSENS_FRAMELINVEL),
        ):
            path = f"runtime.sensor_bindings.{role}"
            local_name = getattr(self.config.sensor_bindings, role)
            if local_name is None:
                raise ValidationError(path, "an explicit model-local sensor name", "missing")
            sensor_name = self._cfg.resolve_name(local_name)
            sensor_id = self._name_id(self._runtime.mj.mjtObj.mjOBJ_SENSOR, sensor_name, f"sensor_bindings.{role}")
            actual_dimension = int(self._runtime.model.sensor_dim[sensor_id])
            if actual_dimension != 3:
                raise ValidationError(path, "a 3D sensor", actual_dimension)
            actual_type = self._runtime.model.sensor_type[sensor_id]
            if actual_type != expected_type:
                raise ValidationError(path, expected_type.name, int(actual_type))
            if role in ("base_angular_velocity", "base_linear_acceleration"):
                site_id = int(self._runtime.model.sensor_objid[sensor_id])
                if (
                    self._runtime.model.sensor_objtype[sensor_id] != self._runtime.mj.mjtObj.mjOBJ_SITE
                    or self._runtime.model.site_bodyid[site_id] != self._base_body_id
                    or not np.allclose(self._runtime.model.site_quat[site_id, 1:], 0.0, atol=1e-7, rtol=0.0)
                ):
                    raise ValidationError(path, "a site on the base body aligned with its body frame", sensor_name)
            else:
                model = self._runtime.model
                object_type = model.sensor_objtype[sensor_id]
                object_id = int(model.sensor_objid[sensor_id])
                body_id = (
                    int(model.site_bodyid[object_id])
                    if object_type == self._runtime.mj.mjtObj.mjOBJ_SITE
                    else object_id
                )
                if (
                    model.sensor_refid[sensor_id] != -1
                    or object_type not in (self._runtime.mj.mjtObj.mjOBJ_SITE, self._runtime.mj.mjtObj.mjOBJ_BODY)
                    or body_id != self._base_body_id
                ):
                    raise ValidationError(path, "a world-frame linear velocity sensor on the base body", sensor_name)
            address = int(self._runtime.model.sensor_adr[sensor_id])
            self._sensor_slices[role] = slice(address, address + actual_dimension)

    def _state(self) -> RobotState:
        assert self._runtime.model is not None and self._runtime.data is not None
        orientation_xyzw = wxyz_to_xyzw(self._runtime.data.xquat[self._base_body_id])
        return RobotState(
            sample_time_ns=round(self._runtime.data.time * 1e9),
            receive_time_ns=self._last_communication_ns,
            joint_position=np.asarray(self._runtime.data.qpos[self._joint_qpos_indices], dtype=np.float32),
            joint_velocity=np.asarray(self._runtime.data.qvel[self._joint_qvel_indices], dtype=np.float32),
            base_orientation_xyzw=orientation_xyzw,
            base_angular_velocity=self._sensor("base_angular_velocity"),
            base_linear_acceleration=self._sensor("base_linear_acceleration"),
            base_position=np.asarray(self._runtime.data.xpos[self._base_body_id], dtype=np.float32),
            base_linear_velocity=self._sensor("base_linear_velocity"),
        )

    def _sensor(self, role: str) -> NDArray[np.float32]:
        assert self._runtime.data is not None
        return np.array(self._runtime.data.sensordata[self._sensor_slices[role]], dtype=np.float32, copy=True)

    def _name_id(self, object_type: mujoco.mjtObj, name: str, description: str) -> int:
        assert self._runtime.model is not None
        object_id = int(self._runtime.mj.mj_name2id(self._runtime.model, object_type, name))
        if object_id < 0:
            raise ValidationError(f"backend.{description}", f"existing name {name!r}", "missing")
        return object_id

    def _require_open(self) -> None:
        if not self._opened or self._closed or self._runtime.model is None or self._runtime.data is None:
            raise RuntimeError("MuJoCo backend is not open")
