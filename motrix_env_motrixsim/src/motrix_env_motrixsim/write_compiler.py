# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixSim compiler and executable program for declarative sim writes."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

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
    DofPositionWrite,
    DofVelocityWrite,
    GeomFrictionWrite,
    JointPositionWrite,
    JointVelocityWrite,
    MocapPoseWrite,
    SimWriteCompiler,
    WriteProgram,
)


class _WriteOp(Protocol):
    def alloc(self, num_envs: int) -> np.ndarray: ...

    def __call__(self, buffers, idx: np.ndarray | slice, rows: mtx.SceneData) -> None: ...


class _ResetPatchOp(Protocol):
    def apply(self, dof_pos, dof_vel, buffers, env_ids) -> None: ...


@dataclass(frozen=True)
class _CompiledWrite:
    op: _WriteOp | None = None
    reset_op: _ResetPatchOp | None = None
    pos_indices: np.ndarray | None = None
    vel_indices: np.ndarray | None = None
    refresh_kinematics: bool = False
    native: mtx_write.WriteSource | None = None
    ctrl_indices: np.ndarray | None = None


class MotrixSimWriteCompiler(SimWriteCompiler):
    """Compile neutral write declarations against one MotrixSim model and data batch."""

    def __init__(
        self,
        model: mtx.SceneModel,
        data: mtx.SceneData,
        masked_rows: Callable[[np.ndarray], mtx.SceneData],
    ) -> None:
        self._model = model
        self._data = data
        self._masked_rows = masked_rows
        self._pending: list[tuple[str, _CompiledWrite]] = []

    def _begin_compile(self) -> None:
        self._pending = []

    def _build_program(self, *, reset: bool, forward_kinematics: bool) -> WriteProgram:
        buffers: dict[str, np.ndarray] = {}
        ops: list[tuple[_CompiledWrite, np.ndarray]] = []
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
            if compiled.native is not None:
                native_fields[name] = compiled.native
            else:
                sub = compiled.op.alloc(self._data.shape[0])
                ops.append((compiled, sub))
                buffers[name] = sub
            refresh_kinematics |= compiled.refresh_kinematics
        native_program = None
        if native_fields:
            native_program = self._model.compile_write(native_fields).allocate(self._data)
            for name in native_fields:
                buffers[name] = native_program[name]
        return _MotrixSimWriteProgram(
            self._model,
            self._data,
            self._masked_rows,
            buffers,
            ops,
            reset=reset,
            refresh_kinematics=forward_kinematics and (reset or refresh_kinematics),
            native_program=native_program,
        )

    def compile_dof_position(self, name: str, write: DofPositionWrite) -> None:
        del write
        indices = np.arange(self._model.num_dof_pos, dtype=np.int64)
        op = _DofChannelOp(indices)
        self._pending.append((name, _CompiledWrite(op, op, pos_indices=indices, refresh_kinematics=True)))

    def compile_dof_velocity(self, name: str, write: DofVelocityWrite) -> None:
        del write
        indices = np.arange(self._model.num_dof_vel, dtype=np.int64)
        op = _DofChannelOp(indices, velocity=True)
        self._pending.append((name, _CompiledWrite(op, op, vel_indices=indices)))

    def compile_body_joint_position(self, name: str, write: BodyJointPositionWrite) -> None:
        indices = np.asarray(_named_body(self._model, write.body).get_dof_pos_indices(False), dtype=np.int64)
        op = _DofChannelOp(indices)
        self._pending.append((name, _CompiledWrite(op, op, pos_indices=indices, refresh_kinematics=True)))

    def compile_body_joint_velocity(self, name: str, write: BodyJointVelocityWrite) -> None:
        indices = np.asarray(_named_body(self._model, write.body).get_dof_vel_indices(False), dtype=np.int64)
        op = _DofChannelOp(indices, velocity=True)
        self._pending.append((name, _CompiledWrite(op, op, vel_indices=indices)))

    def compile_joint_position(self, name: str, write: JointPositionWrite) -> None:
        indices = np.asarray([joint.dof_pos_index for joint in self._joints(name, write.joints)], dtype=np.int64)
        op = _DofChannelOp(indices)
        self._pending.append((name, _CompiledWrite(op, op, pos_indices=indices, refresh_kinematics=True)))

    def compile_joint_velocity(self, name: str, write: JointVelocityWrite) -> None:
        indices = np.asarray([joint.dof_vel_index for joint in self._joints(name, write.joints)], dtype=np.int64)
        op = _DofChannelOp(indices, velocity=True)
        self._pending.append((name, _CompiledWrite(op, op, vel_indices=indices)))

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
                    reset_op=_DofComponentPatchOp(indices),
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
                    reset_op=_DofComponentPatchOp(indices),
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
                    reset_op=_DofComponentPatchOp(indices, velocity=True),
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
                    reset_op=_DofComponentPatchOp(indices, velocity=True),
                    vel_indices=indices.ravel(),
                    native=mtx_write.BodyAngularVelocity(list(write.bodies)),
                ),
            )
        )

    def compile_mocap_pose(self, name: str, write: MocapPoseWrite) -> None:
        bodies = self._targets(name, write.bodies, "body", _named_body)
        mocaps = []
        for body_name, body in zip(write.bodies, bodies):
            if body.mocap is None:
                raise ValueError(f"MocapPoseWrite body {body_name!r} is not a mocap body.")
            mocaps.append(body.mocap)
        self._pending.append((name, _CompiledWrite(_MultiTargetOp(mocaps, "set_pose", 7), refresh_kinematics=True)))

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

    def _floating_bases(self, name: str, body_names: tuple[str, ...], write_type: str):
        bodies = self._targets(name, body_names, "body", _named_body)
        bases = []
        for body_name, body in zip(body_names, bodies):
            if body.floatingbase is None:
                raise ValueError(f"{write_type} {name!r} body {body_name!r} has no floating base.")
            bases.append(body.floatingbase)
        return bases

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
        model: mtx.SceneModel,
        data: mtx.SceneData,
        masked_rows: Callable[[np.ndarray], mtx.SceneData],
        buffers: dict[str, np.ndarray],
        ops: list[tuple[_CompiledWrite, np.ndarray]],
        *,
        reset: bool,
        refresh_kinematics: bool,
        native_program: mtx_write.WriteProgram | None = None,
    ) -> None:
        self._model = model
        self._data = data
        self._masked_rows = masked_rows
        self._buffers = buffers
        self._ops = ops
        self._reset = reset
        self._refresh_kinematics = refresh_kinematics
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
        selected_ids = np.arange(self._data.shape[0], dtype=np.int64) if env_ids is None else np.sort(env_ids)
        idx = slice(None) if env_ids is None else selected_ids
        if self._reset:
            self._execute_reset(selected_ids, idx)
            return
        needs_rows = bool(self._ops) or (self._refresh_kinematics and self._native_program is None)
        rows = (self._data if env_ids is None else self._masked_rows(selected_ids)) if needs_rows else None
        if rows is not None:
            for compiled, sub_buffers in self._ops:
                compiled.op(sub_buffers, idx, rows)
            if self._refresh_kinematics and self._native_program is None:
                self._model.forward_kinematic(rows)
        if self._native_program is not None:
            # Native programs reject masked row views, so selection goes through
            # env_ids against the full batch; FK then runs after legacy ops too.
            self._native_program.execute(
                self._data,
                env_ids=None if env_ids is None else selected_ids,
                forward_kinematic=self._refresh_kinematics,
            )

    def _execute_reset(self, env_ids: np.ndarray, idx: np.ndarray | slice) -> None:
        full_batch = isinstance(idx, slice)
        rows = self._data if full_batch else self._masked_rows(env_ids)
        default_dof_pos = np.asarray(self._model.compute_init_dof_pos(), dtype=np.float32)
        dof_pos = np.broadcast_to(default_dof_pos, (env_ids.size, self._model.num_dof_pos)).copy()
        dof_vel = np.zeros((env_ids.size, self._model.num_dof_vel), dtype=np.float32)
        post_reset_ops = []
        native_reset_ids = None if full_batch else env_ids
        for compiled, buffers in self._ops:
            if compiled.reset_op is None:
                post_reset_ops.append((compiled.op, buffers))
            else:
                compiled.reset_op.apply(dof_pos, dof_vel, buffers, env_ids)
        # Fused reset FK only covers pure-legacy programs; anything else that
        # still needs a refresh gets one explicit pass after all writes.
        fk_fused = self._refresh_kinematics and not post_reset_ops and self._native_program is None
        kwargs = {"forward_kinematic": fk_fused}
        if self._model.num_dof_pos:
            kwargs["dof_pos"] = np.ascontiguousarray(dof_pos)
        if self._model.num_dof_vel:
            kwargs["dof_vel"] = np.ascontiguousarray(dof_vel)
        rows.reset(self._model, **kwargs)
        if self._native_program is not None:
            self._native_program.execute(self._data, env_ids=native_reset_ids, forward_kinematic=False)
        for op, buffers in post_reset_ops:
            op(buffers, idx, rows)
        if self._refresh_kinematics and (post_reset_ops or self._native_program is not None):
            self._model.forward_kinematic(rows)


class _DofChannelOp:
    def __init__(self, indices: np.ndarray, *, velocity: bool = False) -> None:
        self._indices = indices
        self._velocity = velocity

    def alloc(self, num_envs: int) -> np.ndarray:
        return np.zeros((num_envs, self._indices.size), dtype=np.float32)

    def __call__(self, buffers, idx: np.ndarray | slice, rows: mtx.SceneData) -> None:
        target = rows.dof_vel if self._velocity else rows.dof_pos
        target[:, self._indices] = buffers[idx]

    def apply(self, dof_pos, dof_vel, buffers, env_ids) -> None:
        target = dof_vel if self._velocity else dof_pos
        target[:, self._indices] = buffers[env_ids]


class _DofComponentPatchOp:
    def __init__(self, indices: np.ndarray, *, velocity: bool = False) -> None:
        self._indices = indices
        self._velocity = velocity

    def apply(self, dof_pos, dof_vel, buffers, env_ids) -> None:
        target = dof_vel if self._velocity else dof_pos
        values = buffers[env_ids]
        for target_index, indices in enumerate(self._indices):
            target[:, indices] = values[:, target_index]


class _MultiTargetOp:
    """Apply one fixed-width property to targets in declared order."""

    def __init__(self, targets, setter_name: str, width: int) -> None:
        self._setters = [getattr(target, setter_name) for target in targets]
        self._width = width

    def alloc(self, num_envs: int) -> np.ndarray:
        shape = (num_envs, len(self._setters)) if self._width == 1 else (num_envs, len(self._setters), self._width)
        return np.zeros(shape, dtype=np.float32)

    def __call__(self, buffers, idx: np.ndarray | slice, rows: mtx.SceneData) -> None:
        values = buffers[idx]
        for target_index, setter in enumerate(self._setters):
            setter(rows, values[:, target_index])


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
