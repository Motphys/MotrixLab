# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Bind resolved MotrixSim write fields to persistent CUDA storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from motrixsim import write

from motrix_env_core.sim.write import WriteProgram
from motrix_env_motrixsim._gpu_primitives import joint_actuator_slots
from motrix_env_motrixsim.write_compiler import MotrixWriteCompiler, ResolvedWrite

if TYPE_CHECKING:
    from motrix_env_motrixsim.gpu_runtime import MotrixCudaBackend


@dataclass
class _Channel:
    slot: int
    target: torch.Tensor
    columns: torch.Tensor | None
    quaternion: bool


@dataclass
class _LinkOverride:
    slot: int
    table: torch.Tensor
    columns: torch.Tensor
    positive: bool


@dataclass
class _FrictionOverride:
    slot: int
    spin_roll: torch.Tensor
    collider_friction: torch.Tensor
    source_cols: torch.Tensor
    collider_cols: torch.Tensor


class CudaWriteCompiler(MotrixWriteCompiler):
    def __init__(self, backend: MotrixCudaBackend, *, direct_ctrl: bool = False):
        super().__init__(backend.host_model)
        self.backend = backend
        self.direct_ctrl = direct_ctrl

    def allocate(self, *, reset: bool, refresh: bool) -> WriteProgram:
        with self.backend.physics_stream():
            return CudaWrites(self.backend, self._pending, reset, refresh, direct_ctrl=self.direct_ctrl)


class CudaWrites(WriteProgram):
    def __init__(
        self,
        backend: MotrixCudaBackend,
        fields: list[tuple[str, ResolvedWrite]],
        reset: bool,
        refresh: bool,
        *,
        direct_ctrl: bool = False,
    ):
        self._full_batch_only = direct_ctrl
        self._backend = backend
        self._reset = reset
        self._refresh = refresh
        self._buffers: dict[str, torch.Tensor] = {}
        self._channels: list[_Channel] = []
        mocaps: dict[str, list] = {}
        self._gains: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        self._link_overrides: list[_LinkOverride] = []
        self._friction_overrides: list[_FrictionOverride] = []
        for slot, (name, field) in enumerate(fields):
            direct = None
            for indices, target in (
                (field.ctrl_indices, backend.control),
                (field.pos_indices, backend.position),
                (field.vel_indices, backend.velocity),
            ):
                if indices is None:
                    continue
                shape = indices.shape
                flat = indices.ravel()
                quaternion = isinstance(field.native, write.JointQuaternion)
                if flat.size and np.array_equal(flat, np.arange(flat[0], flat[0] + flat.size)):
                    selected = target[:, int(flat[0]) : int(flat[-1]) + 1]
                    if direct_ctrl and field.ctrl_indices is not None:
                        direct = selected
                    else:
                        self._channels.append(_Channel(slot, selected, None, quaternion))
                else:
                    self._channels.append(
                        _Channel(slot, target, torch.as_tensor(flat, device=backend.device), quaternion)
                    )
                break
            else:
                source = field.native
                if isinstance(source, (write.BodyPosition, write.BodyRotation)):
                    rotation = isinstance(source, write.BodyRotation)
                    shape = (len(source.selection), 4 if rotation else 3)
                    components = slice(3, 7) if rotation else slice(3)
                    for column, body in enumerate(source.selection):
                        mocaps.setdefault(body, []).append((slot, column, components))
                else:
                    shape = self._bind_parameter(slot, source)
            self._buffers[name] = (
                direct if direct is not None else backend.position.new_zeros((backend.data.num_worlds, *shape))
            )
        self._mocap_fields = tuple((backend.model.get_body(body).mocap, fields) for body, fields in mocaps.items())
        self._recompute_inertia = bool(self._link_overrides)
        self._values = tuple(self._buffers.values())
        self._quaternion_identity = (
            backend.position.new_tensor((0, 0, 0, 1)) if any(channel.quaternion for channel in self._channels) else None
        )

    def _bind_parameter(self, slot: int, source: write.WriteSource) -> tuple[int, ...]:
        backend = self._backend
        model = backend.model
        names = source.selection
        if isinstance(source, (write.ActuatorKpOverride, write.ActuatorDampingOverride)):
            actuators = [model.get_actuator(name) for name in names]
            slots = joint_actuator_slots(model)
            for a in actuators:
                if a.typ != "position" or a.target_type not in ("joint", "floating_base") or a.index not in slots:
                    raise ValueError(
                        "MotrixSim CUDA 1.0.0 supports gain overrides only for joint or floating-base "
                        f"position actuators, got {a.name!r} ({a.typ}/{a.target_type})"
                    )
            columns = torch.tensor([slots[a.index] for a in actuators], device=backend.device, dtype=torch.long)
            attr = (
                "actuator_kp_overrides"
                if isinstance(source, write.ActuatorKpOverride)
                else "actuator_damping_overrides"
            )
            self._gains.append((slot, backend.gain_table(attr), columns))
            return (len(names),)

        if isinstance(source, (write.LinkMassOverride, write.LinkCenterOfMassOverride)):
            mass_table, com_table = backend.overrides.bind_link_inertia()
            columns = torch.tensor(
                [model.get_link(name).index for name in names], device=backend.device, dtype=torch.long
            )
            if isinstance(source, write.LinkMassOverride):
                self._link_overrides.append(_LinkOverride(slot, mass_table, columns, positive=True))
                return (len(names),)
            self._link_overrides.append(_LinkOverride(slot, com_table, columns, positive=False))
            return (len(names), 3)

        if isinstance(source, write.GeomFrictionOverride):
            geoms = [backend.host_model.get_geom(name) for name in names]
            geom_indices = [g.index for g in geoms]
            spin_roll = backend.position.new_tensor(np.stack([g.friction[1:] for g in geoms]))
            collider_friction, source_cols, collider_cols = backend.overrides.bind_geom_friction(geom_indices)
            self._friction_overrides.append(
                _FrictionOverride(slot, spin_roll, collider_friction, source_cols, collider_cols)
            )
            return (len(names), 3)

        raise NotImplementedError(f"Unsupported CUDA parameter {type(source).__name__}")

    def buffer(self, name: str) -> torch.Tensor:
        return self._buffers[name]

    def execute(self, env_ids: torch.Tensor | None = None) -> None:
        self._execute(env_ids, self._values, compact=False)

    def execute_compact(self, env_ids: torch.Tensor, buffers: tuple[torch.Tensor, ...]) -> None:
        self._execute(env_ids, buffers, compact=True)

    def execute_masked(self, world_mask: torch.Tensor) -> None:
        self._execute(None, self._values, compact=False, world_mask=world_mask)

    def _execute(
        self,
        ids: torch.Tensor | None,
        buffers: tuple[torch.Tensor, ...],
        *,
        compact: bool,
        world_mask: torch.Tensor | None = None,
    ) -> None:
        backend = self._backend
        if self._full_batch_only and (ids is not None or world_mask is not None or compact):
            raise RuntimeError("This action write program supports full-batch execute() only.")
        channels = self._channels
        if ids is not None:
            if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
                raise TypeError("Simulator write env_ids must be a one-dimensional integer tensor.")
            if ids.numel() == 0:
                return
            ids = ids.to(backend.device, dtype=torch.long)
        if not (
            self._reset
            or self._refresh
            or channels
            or self._gains
            or self._link_overrides
            or self._friction_overrides
            or self._mocap_fields
        ):
            return
        rows = slice(None) if ids is None else ids
        source_rows = slice(0, ids.numel()) if compact else rows

        with backend.physics_stream():
            mask = world_mask if world_mask is not None else backend.world_mask(ids)
            warp_mask = wp.from_torch(mask) if self._reset or self._recompute_inertia or self._mocap_fields else None
            for friction in self._friction_overrides:
                raw = buffers[friction.slot][source_rows]
                supported = ((raw[..., 1:] == friction.spin_roll) | ~raw[..., 0:1].isfinite()).all(dim=(1, 2))
                if world_mask is not None:
                    supported = supported | ~mask
                torch._assert_async(supported.all(), "MotrixSim CUDA 1.0.0 supports only sliding-friction overrides")
            if self._reset:
                backend.reset_inputs(mask, warp_mask, recompute_inertia=not self._recompute_inertia)
            for channel in channels:
                values = buffers[channel.slot][source_rows]
                if channel.quaternion:
                    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
                    values = torch.where(norm == 0, self._quaternion_identity, values / norm)
                values = values.flatten(1)
                self._update_columns(channel.target, channel.columns, ids, world_mask, values)
            for slot, table, columns in self._gains:
                row_index = rows if ids is None else ids[:, None]
                previous = table[row_index, columns]
                values = buffers[slot][source_rows]
                values = torch.where(values.isfinite() & (values >= 0), values, previous)
                self._update_columns(table, columns, ids, world_mask, values)
            for override in self._link_overrides:
                row_index = rows if ids is None else ids[:, None]
                previous = override.table[row_index, override.columns]
                values = buffers[override.slot][source_rows]
                if values.ndim == 3:
                    valid = values.isfinite().all(dim=-1, keepdim=True)
                else:
                    valid = values.isfinite() & ((values > 0) if override.positive else (values >= 0))
                values = torch.where(valid, values, previous)
                self._update_columns(override.table, override.columns, ids, world_mask, values)
            if self._recompute_inertia:
                backend.overrides.commit_link_inertia(warp_mask)
            for friction in self._friction_overrides:
                if friction.collider_cols.numel() == 0:
                    continue
                row_index = rows if ids is None else ids[:, None]
                previous = friction.collider_friction[row_index, friction.collider_cols, 0]
                picked = buffers[friction.slot][source_rows][:, friction.source_cols]
                sliding = picked[..., 0]
                valid = picked.isfinite().all(dim=-1) & (sliding >= 0)
                values = torch.where(valid, sliding, previous)
                if ids is not None:
                    friction.collider_friction[ids[:, None], friction.collider_cols, 0] = values
                else:
                    if world_mask is not None:
                        values = torch.where(world_mask[:, None], values, previous)
                    friction.collider_friction[:, friction.collider_cols, 0] = values
            for mocap, fields in self._mocap_fields:
                poses = wp.to_torch(mocap.get_pose(backend.data))
                for slot, column, components in fields:
                    self._update_columns(
                        poses[:, components], None, ids, world_mask, buffers[slot][source_rows, column]
                    )
                mocap.set_pose(backend.data, wp.from_torch(poses), world_mask=warp_mask)
            if self._refresh:
                # SDK 1.0.0 only exposes a whole-batch kinematics refresh.
                backend.model.forward_kinematic(backend.data)

    @staticmethod
    def _update_columns(target, columns, ids, mask, values) -> None:
        mask_view = None if mask is None else mask.reshape((-1,) + (1,) * (values.ndim - 1))
        if columns is None:
            if ids is not None:
                target[ids] = values
            elif mask_view is not None:
                target.copy_(torch.where(mask_view, values, target))
            else:
                target.copy_(values)
        elif ids is not None:
            target[ids[:, None], columns] = values
        else:
            if mask_view is not None:
                values = torch.where(mask_view, values, target[:, columns])
            target.index_copy_(1, columns, values)
