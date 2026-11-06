# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reward terms for the manager-based ball-balance task."""

from dataclasses import dataclass

import numpy as np
import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, RewardTerm, RewardTermCfg
from motrix_env_core.math.quaternion_array import rotate_inverse as rotate_inverse_batch


def upright_reward(ctx: ManagerContext, sigma: float) -> torch.Tensor:
    base_quat = ctx.sim["robot_base_quat"]
    gravity = base_quat.new_zeros(base_quat.shape[0], 3)
    gravity[:, 2] = -1.0
    gravity = rotate_inverse_batch(base_quat, gravity)
    error_sq = gravity[:, 0] * gravity[:, 0] + gravity[:, 1] * gravity[:, 1]
    aligned = gravity[:, 2] + 1.0
    error_sq += aligned * aligned
    return torch.exp(-error_sq / (sigma * sigma))


@configclass(kw_only=True)
class UprightRewardCfg(RewardTermCfg):
    """Reward keeping the base upright (base-frame gravity pointing down)."""

    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(upright_reward, np.float32(self.sigma))


def base_height_reward(ctx: ManagerContext, target_z: float, sigma: float) -> torch.Tensor:
    error = ctx.sim["robot_base_pos"][:, 2] - target_z
    return torch.exp(-(error * error) / (sigma * sigma))


@configclass(kw_only=True)
class BaseHeightRewardCfg(RewardTermCfg):
    """Reward keeping the base at its balanced on-ball height."""

    target_z: float
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(base_height_reward, np.float32(self.target_z), np.float32(self.sigma))


def ball_under_feet_reward(ctx: ManagerContext, sigma: float) -> torch.Tensor:
    feet_pos = ctx.sim["feet_pos"]
    ball_pos = ctx.sim["ball_pos"]
    mid_x = 0.5 * (feet_pos[:, 0, 0] + feet_pos[:, 1, 0])
    mid_y = 0.5 * (feet_pos[:, 0, 1] + feet_pos[:, 1, 1])
    error_x = ball_pos[:, 0] - mid_x
    error_y = ball_pos[:, 1] - mid_y
    error_sq = error_x * error_x + error_y * error_y
    return torch.exp(-error_sq / (sigma * sigma))


@configclass(kw_only=True)
class BallUnderFeetRewardCfg(RewardTermCfg):
    """Reward keeping the ball under the midpoint of the feet."""

    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(ball_under_feet_reward, np.float32(self.sigma))


@dataclass(frozen=True, slots=True)
class DofDefaultParams:
    reference: torch.Tensor
    sigma: float


def dof_default_reward(ctx: ManagerContext, params: DofDefaultParams) -> torch.Tensor:
    error = ctx.sim["robot_dof_pos"] - params.reference
    error_sq = (error * error).sum(dim=1) / error.shape[1]
    return torch.exp(-error_sq / (params.sigma * params.sigma))


@configclass(kw_only=True)
class DofDefaultRewardCfg(RewardTermCfg):
    """Reward staying close to the default posture."""

    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        robot = ctx.cfg.scene.objs.robot
        joint_names = ctx.model.bodies["robot"].joint_names
        default = dict(
            zip(
                (robot.resolve_name(name) for name in robot.key_pose.joint_names),
                robot.key_pose.poses["default"],
                strict=True,
            )
        )
        missing = sorted(set(joint_names).difference(default))
        if missing:
            raise KeyError(f"key pose 'default' is missing joints: {missing}")
        reference = np.asarray([default[name] for name in joint_names], dtype=np.float32)
        params = DofDefaultParams(torch.from_numpy(reference), float(self.sigma))
        return RewardTerm(dof_default_reward, params)
