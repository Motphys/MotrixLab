# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixSim compiler and executable program for declarative sim writes."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import motrixsim as mtx
import numpy as np

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
    """One declared write: either native plan fields or a numpy scatter op."""

    op: _WriteOp | None = None
    reset_op: _ResetPatchOp | None = None
    fields: tuple[tuple[str, mtx.write.WriteSource], ...] = ()
    # Logical buffer shape synthesized across a multi-field declaration; the
    # fields interleave per leading axis so one contiguous slice views them.
    synthesized_shape: tuple[int, ...] | None = None
    pos_indices: np.ndarray | None = None
    vel_indices: np.ndarray | None = None
    ctrl_indices: np.ndarray | None = None
    refresh_kinematics: bool = False


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
        fields: dict[str, mtx.write.WriteSource] = {}
        ops: list[tuple[_CompiledWrite, np.ndarray]] = []
        buffers: dict[str, np.ndarray] = {}
        ctrl_owners: dict[int, str] = {}
        claimed_pos: dict[int, str] = {}
        claimed_vel: dict[int, str] = {}
        lead_op_count = 0
        any_kinematics = False
        numpy_kinematics = False
        for name, compiled in self._pending:
            if compiled.ctrl_indices is not None:
                self._claim_ctrl_targets(name, compiled.ctrl_indices, ctrl_owners)
            if compiled.pos_indices is not None:
                self._claim(name, compiled.pos_indices, claimed_pos, "position")
            if compiled.vel_indices is not None:
                self._claim(name, compiled.vel_indices, claimed_vel, "velocity")
            any_kinematics |= compiled.refresh_kinematics
            if compiled.fields:
                if not fields:
                    lead_op_count = len(ops)
                for field_name, source in compiled.fields:
                    fields[field_name] = source
                continue
            numpy_kinematics |= compiled.refresh_kinematics
            sub = compiled.op.alloc(self._data.shape[0])
            ops.append((compiled, sub))
            buffers[name] = sub
        native = None
        manual_fk = False
        needs_kinematics = forward_kinematics and (reset or any_kinematics)
        synthesized: dict[str, tuple[int, tuple[int, ...]]] = {}
        if fields:
            # Forward kinematics runs inside the native execute unless numpy
            # scatter ops must still land before the refresh; either way each
            # execute refreshes at most once.
            native = self._model.compile_write(
                fields, reset=reset, forward_kinematic=needs_kinematics and not numpy_kinematics
            ).allocate(self._data)
            manual_fk = needs_kinematics and numpy_kinematics
            offsets = {field.name: field.offset for field in native.fields}
            for name, compiled in self._pending:
                if compiled.synthesized_shape is not None:
                    first_field = compiled.fields[0][0]
                    synthesized[name] = (offsets[first_field], compiled.synthesized_shape)
        else:
            manual_fk = needs_kinematics
        return _MotrixSimWriteProgram(
            self._model,
            self._data,
            self._masked_rows,
            buffers,
            ops,
            native=native,
            reset=reset,
            refresh_kinematics=manual_fk,
            lead_op_count=lead_op_count,
            synthesized=synthesized,
        )

    def compile_dof_position(self, name: str, write: DofPositionWrite) -> None:
        del write
        indices = np.arange(self._model.num_dof_pos, dtype=np.int64)
        op = _DofChannelOp(self._model, indices)
        self._pending.append((name, _CompiledWrite(op, op, pos_indices=indices, refresh_kinematics=True)))

    def compile_dof_velocity(self, name: str, write: DofVelocityWrite) -> None:
        del write
        indices = np.arange(self._model.num_dof_vel, dtype=np.int64)
        op = _DofChannelOp(self._model, indices, velocity=True)
        self._pending.append((name, _CompiledWrite(op, op, vel_indices=indices)))

    def compile_body_joint_position(self, name: str, write: BodyJointPositionWrite) -> None:
        indices = np.asarray(_named_body(self._model, write.body).get_dof_pos_indices(False), dtype=np.int64)
        op = _DofChannelOp(self._model, indices)
        self._pending.append((name, _CompiledWrite(op, op, pos_indices=indices, refresh_kinematics=True)))

    def compile_body_joint_velocity(self, name: str, write: BodyJointVelocityWrite) -> None:
        indices = np.asarray(_named_body(self._model, write.body).get_dof_vel_indices(False), dtype=np.int64)
        op = _DofChannelOp(self._model, indices, velocity=True)
        self._pending.append((name, _CompiledWrite(op, op, vel_indices=indices)))

    def compile_joint_position(self, name: str, write: JointPositionWrite) -> None:
        indices = np.asarray([joint.dof_pos_index for joint in self._joints(name, write.joints)], dtype=np.int64)
        op = _DofChannelOp(self._model, indices)
        self._pending.append((name, _CompiledWrite(op, op, pos_indices=indices, refresh_kinematics=True)))

    def compile_joint_velocity(self, name: str, write: JointVelocityWrite) -> None:
        indices = np.asarray([joint.dof_vel_index for joint in self._joints(name, write.joints)], dtype=np.int64)
        op = _DofChannelOp(self._model, indices, velocity=True)
        self._pending.append((name, _CompiledWrite(op, op, vel_indices=indices)))

    def compile_ctrl_targets(self, name: str, write: CtrlTargetsWrite) -> None:
        if write.actuators is None:
            indices = np.arange(self._model.num_actuators, dtype=np.int64)
        else:
            if not write.actuators:
                raise ValueError(f"CtrlTargetsWrite {name!r} actuator names must not be empty.")
            if len(set(write.actuators)) != len(write.actuators):
                raise ValueError(f"CtrlTargetsWrite {name!r} actuator names must be unique.")
            indices = np.asarray(
                [_named_actuator(self._model, actuator).index for actuator in write.actuators], dtype=np.int64
            )
        self._pending.append(
            (name, _CompiledWrite(fields=((name, mtx.write.ActuatorCtrls(indices)),), ctrl_indices=indices))
        )

    def compile_body_position(self, name: str, write: BodyPositionWrite) -> None:
        bases = self._floating_bases(name, write.bodies, type(write).__name__)
        indices = np.asarray([base.dof_pos_indices[:3] for base in bases], dtype=np.int64)
        self._pending.append(
            (
                name,
                _CompiledWrite(
                    fields=((name, mtx.write.BodyPosition(list(write.bodies))),),
                    pos_indices=indices.ravel(),
                    refresh_kinematics=True,
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
                    fields=((name, mtx.write.BodyRotation(list(write.bodies))),),
                    pos_indices=indices.ravel(),
                    refresh_kinematics=True,
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
                    fields=((name, mtx.write.BodyLinearVelocity(list(write.bodies))),),
                    vel_indices=indices.ravel(),
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
                    fields=((name, mtx.write.BodyAngularVelocity(list(write.bodies))),),
                    vel_indices=indices.ravel(),
                ),
            )
        )

    def compile_mocap_pose(self, name: str, write: MocapPoseWrite) -> None:
        bodies = self._targets(name, write.bodies, "body", _named_body)
        for body_name, body in zip(write.bodies, bodies):
            if body.mocap is None:
                raise ValueError(f"MocapPoseWrite body {body_name!r} is not a mocap body.")
        # Per-body interleaved position/rotation fields keep the whole group
        # contiguous in the program buffer, so buffer(name) synthesizes the
        # neutral (N, B, 7) pose layout as one zero-copy slice.
        fields = []
        for index, body_name in enumerate(write.bodies):
            fields.append((f"{name}.pos.{index}", mtx.write.BodyPosition((body_name,))))
            fields.append((f"{name}.rot.{index}", mtx.write.BodyRotation((body_name,))))
        self._pending.append(
            (
                name,
                _CompiledWrite(
                    fields=tuple(fields),
                    synthesized_shape=(len(write.bodies), 7),
                    refresh_kinematics=True,
                ),
            )
        )

    def compile_actuator_kp(self, name: str, write: ActuatorKpWrite) -> None:
        targets = self._targets(name, write.actuators, "actuator", _named_actuator)
        indices = np.asarray([target.index for target in targets], dtype=np.int64)
        self._pending.append((name, _CompiledWrite(fields=((name, mtx.write.ActuatorKpOverride(indices)),))))

    def compile_actuator_damping(self, name: str, write: ActuatorDampingWrite) -> None:
        targets = self._targets(name, write.actuators, "actuator", _named_actuator)
        indices = np.asarray([target.index for target in targets], dtype=np.int64)
        self._pending.append((name, _CompiledWrite(fields=((name, mtx.write.ActuatorDampingOverride(indices)),))))

    def compile_body_mass(self, name: str, write: BodyMassWrite) -> None:
        targets = self._targets(name, write.links, "link", _named_link)
        self._pending.append(
            (name, _CompiledWrite(fields=((name, mtx.write.LinkMassOverride([link.name for link in targets])),)))
        )

    def compile_body_com(self, name: str, write: BodyComWrite) -> None:
        targets = self._targets(name, write.links, "link", _named_link)
        self._pending.append(
            (
                name,
                _CompiledWrite(fields=((name, mtx.write.LinkCenterOfMassOverride([link.name for link in targets])),)),
            )
        )

    def compile_geom_friction(self, name: str, write: GeomFrictionWrite) -> None:
        targets = self._targets(name, write.geoms, "geom", _named_geom)
        self._pending.append(
            (name, _CompiledWrite(fields=((name, mtx.write.GeomFrictionOverride([geom.name for geom in targets])),)))
        )

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
    """Compiled MotrixSim writes: one native FFI plan plus numpy scatter ops."""

    def __init__(
        self,
        model: mtx.SceneModel,
        data: mtx.SceneData,
        masked_rows: Callable[[np.ndarray], mtx.SceneData],
        buffers: dict[str, np.ndarray],
        ops: list[tuple[_CompiledWrite, np.ndarray]],
        *,
        native: mtx.write.WriteProgram | None = None,
        reset: bool,
        refresh_kinematics: bool,
        lead_op_count: int = 0,
        synthesized: dict[str, tuple[int, tuple[int, ...]]] | None = None,
    ) -> None:
        self._model = model
        self._data = data
        self._masked_rows = masked_rows
        self._buffers = buffers
        self._ops = ops
        self._native = native
        self._reset = reset
        self._refresh_kinematics = refresh_kinematics
        self._lead_op_count = lead_op_count
        self._synthesized = synthesized or {}

    def buffer(self, name: str) -> np.ndarray:
        if name in self._buffers:
            return self._buffers[name]
        if name in self._synthesized:
            offset, shape = self._synthesized[name]
            view = self._native.buffer[:, offset : offset + math.prod(shape)]
            return view.reshape(*self._data.shape, *shape)
        return self._native[name]

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
        rows = self._data if env_ids is None else self._masked_rows(selected_ids)
        idx = slice(None) if env_ids is None else selected_ids
        if self._native is None:
            self._execute_numpy(rows, selected_ids, idx)
            return
        ids = None if env_ids is None else selected_ids
        # Native fields apply before numpy ops declared after them; reset
        # programs always run the native reset-and-fields pass first.
        start = 0 if self._reset else self._lead_op_count
        for compiled, sub_buffers in self._ops[:start]:
            compiled.op(sub_buffers, idx, rows)
        self._native.execute(self._data, ids)
        for compiled, sub_buffers in self._ops[start:]:
            compiled.op(sub_buffers, idx, rows)
        if self._refresh_kinematics:
            self._model.forward_kinematic(rows)

    def _execute_numpy(self, rows: mtx.SceneData, env_ids: np.ndarray, idx: np.ndarray | slice) -> None:
        if self._reset:
            self._execute_reset(rows, env_ids)
            return
        for compiled, sub_buffers in self._ops:
            compiled.op(sub_buffers, idx, rows)
        if self._refresh_kinematics:
            self._model.forward_kinematic(rows)

    def _execute_reset(self, rows: mtx.SceneData, env_ids: np.ndarray) -> None:
        # Every numpy op is a fused dof-channel patch: fold the declared
        # values into the reset state and restore it in one native call.
        default_dof_pos = np.asarray(self._model.compute_init_dof_pos(), dtype=np.float32)
        dof_pos = np.broadcast_to(default_dof_pos, (env_ids.size, self._model.num_dof_pos)).copy()
        dof_vel = np.zeros((env_ids.size, self._model.num_dof_vel), dtype=np.float32)
        for compiled, buffers in self._ops:
            compiled.reset_op.apply(dof_pos, dof_vel, buffers, env_ids)
        kwargs = {"forward_kinematic": self._refresh_kinematics}
        if self._model.num_dof_pos:
            kwargs["dof_pos"] = np.ascontiguousarray(dof_pos)
        if self._model.num_dof_vel:
            kwargs["dof_vel"] = np.ascontiguousarray(dof_vel)
        rows.reset(self._model, **kwargs)


class _DofChannelOp:
    def __init__(self, model: mtx.SceneModel, indices: np.ndarray, *, velocity: bool = False) -> None:
        self._model = model
        self._indices = indices
        self._velocity = velocity

    def alloc(self, num_envs: int) -> np.ndarray:
        return np.zeros((num_envs, self._indices.size), dtype=np.float32)

    def __call__(self, buffers, idx: np.ndarray | slice, rows: mtx.SceneData) -> None:
        if not self._indices.size:
            return
        # SceneData property projections are read-only copies; channel patches
        # reach the sim state only through the explicit dof setters.
        if self._velocity:
            values = rows.dof_vel
            values[:, self._indices] = buffers[idx]
            rows.set_dof_vel(values)
        else:
            values = rows.dof_pos
            values[:, self._indices] = buffers[idx]
            rows.set_dof_pos(values, self._model)

    def apply(self, dof_pos, dof_vel, buffers, env_ids) -> None:
        target = dof_vel if self._velocity else dof_pos
        target[:, self._indices] = buffers[env_ids]


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
