# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable reward terms for manager-based environments."""

import math

import numpy as np
from numba import literally

from motrix_env_core.config import configclass
from motrix_env_core.manager import RewardTerm, RewardTermCfg
from motrix_env_core.numba.manager.context import BuildContext, ManagerContext
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.numba.math.quaternion import rotate_inverse_components
from motrix_env_core.sim import (
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkQuaternionQuery,
)


@dispatch
def alive_reward(_ctx: ManagerContext) -> float:
    return 1.0


@configclass(kw_only=True)
class AliveRewardCfg(RewardTermCfg):
    """Constant alive bonus; zero weight disables it."""

    def __call__(self, ctx) -> RewardTerm:
        del ctx
        return RewardTerm(alive_reward)


@dispatch
def action_rate_reward(ctx: ManagerContext, action_name: str) -> float:
    action_name = literally(action_name)
    action = ctx.actions[action_name]
    delta = action.current - action.previous
    return float(np.dot(delta, delta))


@configclass(kw_only=True)
class ActionRateRewardCfg(RewardTermCfg):
    """L2 penalty on the per-step change of one named action term."""

    action_name: str = "joint_position"

    def __call__(self, ctx) -> RewardTerm:
        del ctx
        return RewardTerm(action_rate_reward, self.action_name)


@dispatch
def tracking_lin_vel_xy_reward(
    ctx: ManagerContext,
    sigma: np.float32,
    base_quat: np.ndarray,
    base_lin_vel: np.ndarray,
    command_name: str,
) -> float:
    command_name = literally(command_name)
    command = ctx.commands[command_name].command
    vx, vy, _ = rotate_inverse_components(base_quat, base_lin_vel)
    error = (command[0] - vx) * (command[0] - vx) + (command[1] - vy) * (command[1] - vy)
    return math.exp(-error / sigma)


@configclass(kw_only=True)
class TrackingLinVelXyRewardCfg(RewardTermCfg):
    """Exponential reward for tracking a command's xy linear velocity in the base frame."""

    command_name: str
    sigma: float = 0.25
    body: str = "robot"

    def __call__(self, ctx: BuildContext) -> RewardTerm:
        link = ctx.model.bodies[self.body].base_link_name
        return RewardTerm(
            tracking_lin_vel_xy_reward,
            np.float32(self.sigma),
            LinkQuaternionQuery(link=link),
            LinkLinearVelocityQuery(link=link),
            self.command_name,
        )


@dispatch
def tracking_ang_vel_z_reward(
    ctx: ManagerContext,
    sigma: np.float32,
    base_quat: np.ndarray,
    base_ang_vel: np.ndarray,
    command_name: str,
) -> float:
    command_name = literally(command_name)
    command = ctx.commands[command_name].command
    _, _, wz = rotate_inverse_components(base_quat, base_ang_vel)
    error = (command[2] - wz) * (command[2] - wz)
    return math.exp(-error / sigma)


@configclass(kw_only=True)
class TrackingAngVelZRewardCfg(RewardTermCfg):
    """Exponential reward for tracking a command's yaw angular velocity in the base frame."""

    command_name: str
    sigma: float = 0.25
    body: str = "robot"

    def __call__(self, ctx: BuildContext) -> RewardTerm:
        link = ctx.model.bodies[self.body].base_link_name
        return RewardTerm(
            tracking_ang_vel_z_reward,
            np.float32(self.sigma),
            LinkQuaternionQuery(link=link),
            LinkAngularVelocityQuery(link=link),
            self.command_name,
        )


__all__ = [
    "ActionRateRewardCfg",
    "AliveRewardCfg",
    "TrackingAngVelZRewardCfg",
    "TrackingLinVelXyRewardCfg",
    "action_rate_reward",
    "alive_reward",
    "tracking_ang_vel_z_reward",
    "tracking_lin_vel_xy_reward",
]
