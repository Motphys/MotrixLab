# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixSim compiler and executable program for declarative sim writes."""

from dataclasses import dataclass

import motrixsim as mtx
import numpy as np
from motrixsim import write as mtx_write

from motrix_env_core.sim.write import (
    ActuatorDampingWrite,
    ActuatorKpWrite,
    BodyAngularVelocityWrite,
    BodyComWrite,
    BodyJointPositionWrite,
    BodyJointVelocityWrite,
    BodyLinearVelocityWrite,
    BodyMassWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    CtrlTargetsWrite,
    GeomFrictionWrite,
    JointAngularVelocityWrite,
    JointPositionWrite,
    JointQuaternionWrite,
    JointVelocityWrite,
    KinematicBodyPositionWrite,
    KinematicBodyRotationWrite,
    SimWriteCompiler,
    WriteProgram,
)


@dataclass(frozen=True)
class _CompiledWrite:
    pos_indices: np.ndarray | None = None
    vel_indices: np.ndarray | None = None
    refresh_kinematics: bool = False
    native: mtx_write.WriteSource | None = None
    ctrl_indices: np.ndarray | None = None


class MotrixSimWriteCompiler(SimWriteCompiler):
    """Compile neutral write declarations against one MotrixSim model and data batch.

    Every write compiles into one native write plan; reset and
    forward-kinematics behavior are baked into the plan at compile time.
    """

    def __init__(self, model: mtx.SceneModel, data: mtx.SceneData) -> None:
        self._model = model
        self._data = data
        self._pending: list[tuple[str, _CompiledWrite]] = []

    def _begin_compile(self) -> None:
        self._pending = []

    def _build_program(self, *, reset: bool, forward_kinematics: bool) -> WriteProgram:
        native_fields: dict[str, mtx_write.WriteSource] = {}
        ctrl_owners: dict[int, str] = {}
        claimed_pos: dict[int, str] = {}
        claimed_vel: dict[int, str] = {}
        refresh_kinematics = False
        for name, compiled in self._pending:
            if compiled.ctrl_indices is not None:
                self._claim_ctrl_targets(name, compiled.ctrl_indices, ctrl_owners)
            if compiled.pos_indices is not None:
                self._claim(name, compiled.pos_indices, claimed_pos, "position")
            if compiled.vel_indices is not None:
                self._claim(name, compiled.vel_indices, claimed_vel, "velocity")
            native_fields[name] = compiled.native
            refresh_kinematics |= compiled.refresh_kinematics
        buffers: dict[str, np.ndarray] = {}
        native_program = None
        if native_fields:
            # Reset (restore defaults, then apply the writes) and the FK pass
            # are baked into the native plan at compile time.
            refresh = forward_kinematics and (reset or refresh_kinematics)
            native_program = self._model.compile_write(native_fields, reset=reset, forward_kinematic=refresh).allocate(
                self._data
            )
            for name in native_fields:
                buffers[name] = native_program[name]
        return _MotrixSimWriteProgram(self._data, buffers, native_program)

    def compile_body_joint_position(self, name: str, write: BodyJointPositionWrite) -> None:
        body = _named_body(self._model, write.body)
        joints = list(body.joints)
        indices = np.asarray(body.get_dof_pos_indices(False), dtype=np.int64)
        self._require_single_dof_body(name, write.body, joints, "position")
        self._pending.append(
            (
                name,
                _CompiledWrite(
                    native=mtx_write.BodyJointPosition([joint.name for joint in joints]),
                    pos_indices=indices,
                    refresh_kinematics=True,
                ),
            )
        )

    def compile_body_joint_velocity(self, name: str, write: BodyJointVelocityWrite) -> None:
        body = _named_body(self._model, write.body)
        joints = list(body.joints)
        indices = np.asarray(body.get_dof_vel_indices(False), dtype=np.int64)
        self._require_single_dof_body(name, write.body, joints, "velocity")
        self._pending.append(
            (
                name,
                _CompiledWrite(
                    native=mtx_write.BodyJointVelocity([joint.name for joint in joints]),
                    vel_indices=indices,
                ),
            )
        )

    def compile_joint_position(self, name: str, write: JointPositionWrite) -> None:
        joints = self._joints(name, write.joints)
        indices = np.asarray([joint.dof_pos_index for joint in joints], dtype=np.int64)
        self._pending.append(
            (
                name,
                _CompiledWrite(
                    native=mtx_write.BodyJointPosition([joint.name for joint in joints]),
                    pos_indices=indices,
                    refresh_kinematics=True,
                ),
            )
        )

    def compile_joint_velocity(self, name: str, write: JointVelocityWrite) -> None:
        joints = self._joints(name, write.joints)
        indices = np.asarray([joint.dof_vel_index for joint in joints], dtype=np.int64)
        self._pending.append(
            (
                name,
                _CompiledWrite(
                    native=mtx_write.BodyJointVelocity([joint.name for joint in joints]),
                    vel_indices=indices,
                ),
            )
        )

    def compile_joint_quaternion(self, name: str, write: JointQuaternionWrite) -> None:
        self._ball_joints(name, write.joints)
        self._pending.append(
            (name, _CompiledWrite(native=mtx_write.JointQuaternion(list(write.joints)), refresh_kinematics=True))
        )

    def compile_joint_angular_velocity(self, name: str, write: JointAngularVelocityWrite) -> None:
        self._ball_joints(name, write.joints)
        self._pending.append((name, _CompiledWrite(native=mtx_write.JointAngularVelocity(list(write.joints)))))

    def compile_ctrl_targets(self, name: str, write: CtrlTargetsWrite) -> None:
        if write.actuators is None:
            indices = np.arange(self._model.num_actuators, dtype=np.int64)
            selection = None
        else:
            if not write.actuators:
                raise ValueError(f"CtrlTargetsWrite {name!r} actuator names must not be empty.")
            if len(set(write.actuators)) != len(write.actuators):
                raise ValueError(f"CtrlTargetsWrite {name!r} actuator names must be unique.")
            indices = np.asarray(
                [_named_actuator(self._model, actuator).index for actuator in write.actuators], dtype=np.int64
            )
            selection = list(write.actuators)
        self._pending.append((name, _CompiledWrite(native=mtx_write.ActuatorCtrls(selection), ctrl_indices=indices)))

    def compile_body_position(self, name: str, write: BodyPositionWrite) -> None:
        bases = self._floating_bases(name, write.bodies, type(write).__name__)
        indices = np.asarray([base.dof_pos_indices[:3] for base in bases], dtype=np.int64)
        self._pending.append(
            (
                name,
                _CompiledWrite(
                    pos_indices=indices.ravel(),
                    refresh_kinematics=True,
                    native=mtx_write.BodyPosition(list(write.bodies)),
                ),
            )
        )

    def compile_body_rotation(self, name: str, write: BodyRotationWrite) -> None:
        bases = self._floating_bases(name, write.bodies, type(write).__name__)
        indices = np.asarray([base.dof_pos_indices[3:] for base in bases], dtype=np.int64)
        self._pending.append(
            (
                name,
                _CompiledWrite(
                    pos_indices=indices.ravel(),
                    refresh_kinematics=True,
                    native=mtx_write.BodyRotation(list(write.bodies)),
                ),
            )
        )

    def compile_body_linear_velocity(self, name: str, write: BodyLinearVelocityWrite) -> None:
        bases = self._floating_bases(name, write.bodies, type(write).__name__)
        indices = np.asarray([base.dof_vel_indices[:3] for base in bases], dtype=np.int64)
        self._pending.append(
            (
                name,
                _CompiledWrite(
                    vel_indices=indices.ravel(),
                    native=mtx_write.BodyLinearVelocity(list(write.bodies)),
                ),
            )
        )

    def compile_body_angular_velocity(self, name: str, write: BodyAngularVelocityWrite) -> None:
        bases = self._floating_bases(name, write.bodies, type(write).__name__)
        indices = np.asarray([base.dof_vel_indices[3:] for base in bases], dtype=np.int64)
        self._pending.append(
            (
                name,
                _CompiledWrite(
                    vel_indices=indices.ravel(),
                    native=mtx_write.BodyAngularVelocity(list(write.bodies)),
                ),
            )
        )

    def compile_kinematic_body_position(self, name: str, write: KinematicBodyPositionWrite) -> None:
        self._mocaps(name, write.bodies)
        self._pending.append(
            (name, _CompiledWrite(native=mtx_write.BodyPosition(list(write.bodies)), refresh_kinematics=True))
        )

    def compile_kinematic_body_rotation(self, name: str, write: KinematicBodyRotationWrite) -> None:
        self._mocaps(name, write.bodies)
        self._pending.append(
            (name, _CompiledWrite(native=mtx_write.BodyRotation(list(write.bodies)), refresh_kinematics=True))
        )

    def compile_actuator_kp(self, name: str, write: ActuatorKpWrite) -> None:
        self._targets(name, write.actuators, "actuator", _named_actuator)
        self._pending.append((name, _CompiledWrite(native=mtx_write.ActuatorKpOverride(list(write.actuators)))))

    def compile_actuator_damping(self, name: str, write: ActuatorDampingWrite) -> None:
        self._targets(name, write.actuators, "actuator", _named_actuator)
        self._pending.append((name, _CompiledWrite(native=mtx_write.ActuatorDampingOverride(list(write.actuators)))))

    def compile_body_mass(self, name: str, write: BodyMassWrite) -> None:
        self._targets(name, write.links, "link", _named_link)
        self._pending.append((name, _CompiledWrite(native=mtx_write.LinkMassOverride(list(write.links)))))

    def compile_body_com(self, name: str, write: BodyComWrite) -> None:
        self._targets(name, write.links, "link", _named_link)
        self._pending.append((name, _CompiledWrite(native=mtx_write.LinkCenterOfMassOverride(list(write.links)))))

    def compile_geom_friction(self, name: str, write: GeomFrictionWrite) -> None:
        self._targets(name, write.geoms, "geom", _named_geom)
        self._pending.append((name, _CompiledWrite(native=mtx_write.GeomFrictionOverride(list(write.geoms)))))

    def _joints(self, name: str, joint_names: tuple[str, ...]):
        if not joint_names:
            raise ValueError(f"Simulator write {name!r} must declare at least one joint.")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError(f"Simulator write {name!r} contains duplicate joint names.")
        joints = []
        for joint_name in joint_names:
            joint = self._model.get_joint(joint_name)
            if joint is None:
                raise KeyError(f"Unknown joint {joint_name!r} in write {name!r}.")
            if joint.num_dof_pos != 1 or joint.num_dof_vel != 1:
                raise ValueError(f"Simulator write joint {joint_name!r} must have one position and velocity DOF.")
            joints.append(joint)
        return joints

    def _ball_joints(self, name: str, joint_names: tuple[str, ...]) -> None:
        if not joint_names:
            raise ValueError(f"Simulator write {name!r} must declare at least one joint.")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError(f"Simulator write {name!r} contains duplicate joint names.")
        for joint_name in joint_names:
            joint = self._model.get_joint(joint_name)
            if joint is None:
                raise KeyError(f"Unknown joint {joint_name!r} in write {name!r}.")
            if joint.num_dof_pos != 4 or joint.num_dof_vel != 3:
                raise ValueError(f"Simulator write joint {joint_name!r} in {name!r} must be a ball joint.")

    def _require_single_dof_body(self, name: str, body_name: str, joints, channel: str) -> None:
        if joints and all(joint.num_dof_pos == 1 and joint.num_dof_vel == 1 for joint in joints):
            return
        multi = [joint.name for joint in joints if joint.num_dof_pos != 1 or joint.num_dof_vel != 1]
        orientation = "JointQuaternionWrite" if channel == "position" else "JointAngularVelocityWrite"
        raise ValueError(
            f"Simulator write {name!r} body {body_name!r} has multi-DoF joints {multi}; declare per-quantity "
            f"writes instead (JointPositionWrite/{orientation} for position, "
            "JointVelocityWrite/JointAngularVelocityWrite for velocity)."
        )

    def _floating_bases(self, name: str, body_names: tuple[str, ...], write_type: str):
        bodies = self._targets(name, body_names, "body", _named_body)
        bases = []
        for body_name, body in zip(body_names, bodies):
            if body.floatingbase is None:
                raise ValueError(f"{write_type} {name!r} body {body_name!r} has no floating base.")
            bases.append(body.floatingbase)
        return bases

    def _mocaps(self, name: str, body_names: tuple[str, ...]) -> None:
        for body_name in body_names:
            if _named_body(self._model, body_name).mocap is None:
                raise ValueError(f"Kinematic body write target {body_name!r} in {name!r} is not a mocap body.")

    def _targets(self, name: str, names: tuple[str, ...], target_type: str, resolver):
        if not names:
            raise ValueError(f"Simulator write {name!r} must declare at least one {target_type}.")
        if len(set(names)) != len(names):
            raise ValueError(f"Simulator write {name!r} contains duplicate {target_type} names.")
        return [resolver(self._model, target_name) for target_name in names]

    @staticmethod
    def _claim(name: str, indices: np.ndarray, claimed: dict[int, str], channel: str) -> None:
        for index in indices:
            previous = claimed.get(int(index))
            if previous is not None:
                raise ValueError(
                    f"Simulator writes {previous!r} and {name!r} conflict on DOF {channel} index {int(index)}."
                )
            claimed[int(index)] = name

    def _claim_ctrl_targets(self, name: str, indices: np.ndarray, owners: dict[int, str]) -> None:
        for index in indices:
            existing = owners.get(int(index))
            if existing is not None:
                target = self._model.actuators[int(index)].name
                raise ValueError(f"CtrlTargetsWrite {existing!r} and {name!r} both target actuator {target!r}.")
            owners[int(index)] = name


class _MotrixSimWriteProgram(WriteProgram):
    """Compiled MotrixSim writes with program-owned value buffers."""

    def __init__(
        self,
        data: mtx.SceneData,
        buffers: dict[str, np.ndarray],
        native_program: mtx_write.WriteProgram | None = None,
    ) -> None:
        self._data = data
        self._buffers = buffers
        self._native_program = native_program

    def buffer(self, name: str) -> np.ndarray:
        return self._buffers[name]

    def execute(self, env_ids: np.ndarray | None = None) -> None:
        if env_ids is not None:
            if not isinstance(env_ids, np.ndarray) or env_ids.dtype != np.int64 or env_ids.ndim != 1:
                raise TypeError("Simulator write env_ids must be a one-dimensional int64 ndarray.")
            if np.any(env_ids < 0) or np.any(env_ids >= self._data.shape[0]):
                raise IndexError("Simulator write env_ids are out of range.")
            if np.unique(env_ids).size != env_ids.size:
                raise ValueError("Simulator write env_ids must not contain duplicates.")
            if env_ids.size == 0:
                return
        if self._native_program is not None:
            self._native_program.execute(self._data, env_ids=None if env_ids is None else np.sort(env_ids))


def _named_body(model: mtx.SceneModel, body_name: str):
    body = model.get_body(body_name)
    if body is None:
        raise KeyError(f"Unknown body {body_name!r}.")
    return body


def _named_actuator(model: mtx.SceneModel, actuator_name: str):
    for actuator in model.actuators:
        if actuator.name == actuator_name:
            return actuator
    raise KeyError(f"Unknown actuator {actuator_name!r}.")


def _named_link(model: mtx.SceneModel, link_name: str):
    link = model.get_link(link_name)
    if link is None:
        raise KeyError(f"Unknown link {link_name!r}.")
    return link


def _named_geom(model: mtx.SceneModel, geom_name: str):
    geom = model.get_geom(geom_name)
    if geom is None:
        raise KeyError(f"Unknown geom {geom_name!r}.")
    return geom


__all__ = ["MotrixSimWriteCompiler"]
