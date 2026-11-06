# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable reward terms for manager-based environments."""

import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager import RewardTerm, RewardTermCfg
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.math.quaternion_array import rotate_inverse_components
from motrix_env_core.sim import (
    LinkAngularVelocityQuery,
    LinkLinearVelocityQuery,
    LinkQuaternionQuery,
)


def alive_reward(ctx: ManagerContext) -> torch.Tensor:
    return torch.ones_like(ctx.env_id, dtype=torch.float32)


@configclass(kw_only=True)
class AliveRewardCfg(RewardTermCfg):
    """Constant alive bonus; zero weight disables it."""

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(alive_reward)


def action_rate_reward(ctx: ManagerContext, action_name: str) -> torch.Tensor:
    action = ctx.actions[action_name]
    delta = action.current() - action.previous()
    return (delta * delta).sum(dim=1)


@configclass(kw_only=True)
class ActionRateRewardCfg(RewardTermCfg):
    """L2 penalty on the per-step change of one named action term."""

    action_name: str = "joint_position"

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(action_rate_reward, self.action_name)


def tracking_lin_vel_xy_reward(
    ctx: ManagerContext,
    sigma: float,
    base_quat: torch.Tensor,
    base_lin_vel: torch.Tensor,
    command_name: str,
) -> torch.Tensor:
    command = ctx.commands[command_name].command
    vx, vy, _ = rotate_inverse_components(base_quat, base_lin_vel)
    error = (command[:, 0] - vx) * (command[:, 0] - vx) + (command[:, 1] - vy) * (command[:, 1] - vy)
    return torch.exp(-error / sigma)


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
            float(self.sigma),
            LinkQuaternionQuery(link=link),
            LinkLinearVelocityQuery(link=link),
            self.command_name,
        )


def tracking_ang_vel_z_reward(
    ctx: ManagerContext,
    sigma: float,
    base_quat: torch.Tensor,
    base_ang_vel: torch.Tensor,
    command_name: str,
) -> torch.Tensor:
    command = ctx.commands[command_name].command
    _, _, wz = rotate_inverse_components(base_quat, base_ang_vel)
    error = (command[:, 2] - wz) * (command[:, 2] - wz)
    return torch.exp(-error / sigma)


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
            float(self.sigma),
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
