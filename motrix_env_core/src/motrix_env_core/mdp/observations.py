# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable robot observation terms for manager-based environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import literally

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.mdp.noise import add_uniform_noise
from motrix_env_core.numba.manager.context import BuildContext, ManagerContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.numba.manager.observations import ObservationTermCfg, ObsTerm
from motrix_env_core.numba.math.quaternion import rotate_inverse
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


@dispatch
def body_joint_vel_obs(
    ctx: ManagerContext, out: np.ndarray, dof_vel: np.ndarray, scale: np.float32, noise_amplitude: np.float32
) -> None:
    out[:] = dof_vel
    out *= scale
    add_uniform_noise(out, noise_amplitude, ctx.rand.state)


@dispatch
def actions_obs(ctx: ManagerContext, out: np.ndarray, action_name: str) -> None:
    action_name = literally(action_name)
    action = ctx.actions[action_name]
    out[:] = action.current


@configclass(kw_only=True)
class ActionsObsCfg(ObservationTermCfg):
    """Echo the current actions of one named action term."""

    action_name: str = "joint_position"

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        action = ctx.action_terms[self.action_name]
        return ObsTerm(action.current.shape[1], actions_obs, self.action_name)


@dispatch
def command_obs(ctx: ManagerContext, out: np.ndarray, command_name: str) -> None:
    command_name = literally(command_name)
    command = ctx.commands[command_name].command
    out[:] = command


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

    The term passes its query as an argument; the compiler registers the
    query and the dispatch receives the query's lane view in that position.
    Equal queries across terms fold into one physical read at the backend.
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


@dispatch
def body_linear_velocity_obs(
    ctx: ManagerContext,
    out: np.ndarray,
    base_quat: np.ndarray,
    linear_velocity: np.ndarray,
    scale: np.float32,
    noise_amplitude: np.float32,
) -> None:
    rotate_inverse(base_quat, linear_velocity, out)
    out *= scale
    add_uniform_noise(out, noise_amplitude, ctx.rand.state)


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


@dispatch
def body_angular_velocity_obs(
    ctx: ManagerContext,
    out: np.ndarray,
    base_quat: np.ndarray,
    angular_velocity: np.ndarray,
    scale: np.float32,
    noise_amplitude: np.float32,
) -> None:
    rotate_inverse(base_quat, angular_velocity, out)
    out *= scale
    add_uniform_noise(out, noise_amplitude, ctx.rand.state)


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


@dispatch
def body_projected_gravity_obs(ctx: ManagerContext, out: np.ndarray, base_quat: np.ndarray) -> None:
    rotate_inverse(base_quat, (0.0, 0.0, -1.0), out)


@configclass(kw_only=True)
class BodyProjectedGravityObsCfg(ObservationTermCfg):
    """Gravity direction of one scene body in its own frame."""

    body: str = "robot"

    def __call__(self, ctx: BuildContext) -> ObsTerm:
        body = _body(ctx, self.body)
        return ObsTerm(3, body_projected_gravity_obs, LinkQuaternionQuery(link=body.base_link_name))


@dispatch
def body_joint_pos_rel_obs(
    ctx: ManagerContext,
    out: np.ndarray,
    joint_pos: np.ndarray,
    defaults: np.ndarray,
    scale: np.float32,
    noise_amplitude: np.float32,
) -> None:
    out[:] = joint_pos
    out -= defaults
    out *= scale
    add_uniform_noise(out, noise_amplitude, ctx.rand.state)


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
