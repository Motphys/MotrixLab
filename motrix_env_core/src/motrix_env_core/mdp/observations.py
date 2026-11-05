# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable robot observation terms for manager-based environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.observations import ObservationTermCfg, ObsTerm
from motrix_env_core.sim import (
    BodyJointPositionQuery,
    BodyJointVelocityQuery,
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkQuaternionQuery,
)

if TYPE_CHECKING:
    from motrix_env_core.sim.model import BodyModel


@configclass(kw_only=True)
class UniformNoiseCfg:
    amplitude: float = 0.0


def _body(ctx: BuildContext, name: str) -> BodyModel:
    """One scene body's compiled model by its ``SceneObjsCfg`` field name."""
    body_cfg = ctx.cfg.scene.objs[name] if ctx.cfg.scene is not None else None
    if not isinstance(body_cfg, RobotCfg):
        raise ValueError(
            f"Framework robot observation terms derive their queries from scene.objs.{name} "
            f"(RobotCfg); configure a scene robot to derive the defaults."
        )
    return ctx.model.bodies[name]


def add_uniform_noise(ctx: ManagerContext, out: torch.Tensor, amplitude: float) -> None:
    """Consume one random value per output component and add scaled uniform noise in place."""
    if amplitude != 0.0:
        out += ctx.rand.next_uniform(out.shape[1]) * amplitude


def _rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor, out: torch.Tensor) -> None:
    """Rotate ``(N, 3)`` vectors by the inverse of ``(N, 4)`` ``[x, y, z, w]`` quaternions."""
    x, y, z, w = quaternion.unbind(dim=1)
    vx, vy, vz = vector.unbind(dim=1)
    cx = y * vz - z * vy - w * vx
    cy = z * vx - x * vz - w * vy
    cz = x * vy - y * vx - w * vz
    out[:, 0] = vx + 2.0 * (y * cz - z * cy)
    out[:, 1] = vy + 2.0 * (z * cx - x * cz)
    out[:, 2] = vz + 2.0 * (x * cy - y * cx)


def body_joint_vel_obs(
    ctx: ManagerContext, out: torch.Tensor, dof_vel: torch.Tensor, scale: float, noise_amplitude: float
) -> None:
    out.copy_(dof_vel)
    out.mul_(scale)
    add_uniform_noise(ctx, out, noise_amplitude)


def actions_obs(ctx: ManagerContext, out: torch.Tensor, action_name: str) -> None:
    out.copy_(ctx.actions[action_name].current())


@configclass(kw_only=True)
class ActionsObsCfg(ObservationTermCfg):
    """Echo the current actions of one named action term."""

    action_name: str = "joint_position"

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        action = ctx.action_terms[self.action_name].state
        return ObsTerm(action.action_queue.shape[2], actions_obs, self.action_name)


def command_obs(ctx: ManagerContext, out: torch.Tensor, command_name: str) -> None:
    out.copy_(ctx.commands[command_name].command)


@configclass(kw_only=True)
class CommandObsCfg(ObservationTermCfg):
    """Emit one named command term's goal vector (``CommandTerm.command``)."""

    command_name: str

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        command = ctx.command_terms[self.command_name].command
        return ObsTerm(command.shape[1], command_obs, self.command_name)


@configclass(kw_only=True)
class BodyJointVelObsCfg(ObservationTermCfg):
    """Joint-velocity observation of one scene body in its own frame.

    The term passes its query as an argument; the resolver registers the
    query and the dispatch receives the query's batched view in that
    position. Equal queries across terms fold into one physical read at the
    backend.
    """

    body: str = "robot"
    scale: float = 1.0
    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        body = _body(ctx, self.body)
        return ObsTerm(
            len(body.joint_names),
            body_joint_vel_obs,
            BodyJointVelocityQuery(body=body.base_link_name),
            np.float32(self.scale),
            np.float32(self.noise.amplitude),
        )


def body_linear_velocity_obs(
    ctx: ManagerContext,
    out: torch.Tensor,
    base_quat: torch.Tensor,
    linear_velocity: torch.Tensor,
    scale: float,
    noise_amplitude: float,
) -> None:
    _rotate_inverse(base_quat, linear_velocity, out)
    out.mul_(scale)
    add_uniform_noise(ctx, out, noise_amplitude)


@configclass(kw_only=True)
class BodyLinearVelocityObsCfg(ObservationTermCfg):
    """Linear velocity observation of one scene body in its own frame."""

    body: str = "robot"
    scale: float = 1.0
    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        body = _body(ctx, self.body)
        return ObsTerm(
            3,
            body_linear_velocity_obs,
            LinkQuaternionQuery(link=body.base_link_name),
            LinkLinearVelocityQuery(link=body.base_link_name),
            np.float32(self.scale),
            np.float32(self.noise.amplitude),
        )


def body_angular_velocity_obs(
    ctx: ManagerContext,
    out: torch.Tensor,
    base_quat: torch.Tensor,
    angular_velocity: torch.Tensor,
    scale: float,
    noise_amplitude: float,
) -> None:
    _rotate_inverse(base_quat, angular_velocity, out)
    out.mul_(scale)
    add_uniform_noise(ctx, out, noise_amplitude)


@configclass(kw_only=True)
class BodyAngularVelocityObsCfg(ObservationTermCfg):
    """Angular velocity observation of one scene body in its own frame."""

    body: str = "robot"
    scale: float = 1.0
    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        body = _body(ctx, self.body)
        return ObsTerm(
            3,
            body_angular_velocity_obs,
            LinkQuaternionQuery(link=body.base_link_name),
            LinkAngularVelocityQuery(link=body.base_link_name),
            np.float32(self.scale),
            np.float32(self.noise.amplitude),
        )


def body_projected_gravity_obs(
    ctx: ManagerContext, out: torch.Tensor, base_quat: torch.Tensor, noise_amplitude: float
) -> None:
    gravity = out.new_zeros(out.shape)
    gravity[:, 2] = -1.0
    _rotate_inverse(base_quat, gravity, out)
    add_uniform_noise(ctx, out, noise_amplitude)


@configclass(kw_only=True)
class BodyProjectedGravityObsCfg(ObservationTermCfg):
    """Gravity direction of one scene body in its own frame."""

    body: str = "robot"
    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        body = _body(ctx, self.body)
        return ObsTerm(
            3,
            body_projected_gravity_obs,
            LinkQuaternionQuery(link=body.base_link_name),
            np.float32(self.noise.amplitude),
        )


def body_joint_pos_rel_obs(
    ctx: ManagerContext,
    out: torch.Tensor,
    joint_pos: torch.Tensor,
    defaults: torch.Tensor,
    scale: float,
    noise_amplitude: float,
) -> None:
    out.copy_((joint_pos - defaults.to(out.dtype)) * scale)
    add_uniform_noise(ctx, out, noise_amplitude)


@configclass(kw_only=True)
class BodyJointPosRelObsCfg(ObservationTermCfg):
    """Joint positions of one scene body relative to its init joint angles.

    The defaults come from the body's :class:`~motrix_env_core.sim.model.BodyModel`
    init snapshot (key-pose joint angles permuted into the body's joint-DOF
    order), so this is the standard "deviation from the nominal stance"
    observation. The joint-position query is passed as an argument.
    """

    body: str = "robot"
    scale: float = 1.0
    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        body = _body(ctx, self.body)
        return ObsTerm(
            int(body.init_joint_pos.shape[0]),
            body_joint_pos_rel_obs,
            BodyJointPositionQuery(body=body.base_link_name),
            body.init_joint_pos,
            np.float32(self.scale),
            np.float32(self.noise.amplitude),
        )


__all__ = [
    "ActionsObsCfg",
    "BodyAngularVelocityObsCfg",
    "BodyJointPosRelObsCfg",
    "BodyJointVelObsCfg",
    "BodyLinearVelocityObsCfg",
    "BodyProjectedGravityObsCfg",
    "CommandObsCfg",
    "UniformNoiseCfg",
]
