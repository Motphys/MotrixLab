# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reward terms for the manager-based ball-balance task."""

import math

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, ManagerEnv, RewardTerm, RewardTermCfg
from motrix_env_core.manager.math.quaternion import rotate_inverse
from motrix_env_core.numba.kernel_data import SharedArray, kernel_data
from motrix_env_core.numba.manager.dispatch import dispatch


@dispatch
def alive_reward(ctx: ManagerContext) -> float:
    return 1.0


@configclass(kw_only=True)
class AliveRewardCfg(RewardTermCfg):
    """Constant per-step survival bonus.

    Dense positive rewards already reward surviving implicitly; this term
    bottoms out the return so early policies with large action-rate penalties
    cannot profit from terminating quickly.
    """

    def __call__(self, env: ManagerEnv) -> RewardTerm:
        del env
        return RewardTerm(alive_reward)


@dispatch
def upright_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    base_quat = ctx.sim["robot_base_quat"]
    gravity = np.empty(3, dtype=np.float32)
    rotate_inverse(base_quat, (0.0, 0.0, -1.0), gravity)
    error_sq = gravity[0] * gravity[0] + gravity[1] * gravity[1]
    aligned = gravity[2] + 1.0
    error_sq += aligned * aligned
    return math.exp(-error_sq / (sigma * sigma))


@configclass(kw_only=True)
class UprightRewardCfg(RewardTermCfg):
    """Reward keeping the base upright (base-frame gravity pointing down)."""

    sigma: float

    def __call__(self, env: ManagerEnv) -> RewardTerm:
        del env
        return RewardTerm(upright_reward, np.float32(self.sigma))


@dispatch
def base_height_reward(ctx: ManagerContext, target_z: np.float32, sigma: np.float32) -> float:
    base_pos = ctx.sim["robot_base_pos"]
    error = base_pos[2] - target_z
    return math.exp(-(error * error) / (sigma * sigma))


@configclass(kw_only=True)
class BaseHeightRewardCfg(RewardTermCfg):
    """Reward keeping the base at its balanced on-ball height."""

    target_z: float
    sigma: float

    def __call__(self, env: ManagerEnv) -> RewardTerm:
        del env
        return RewardTerm(base_height_reward, np.float32(self.target_z), np.float32(self.sigma))


@dispatch
def ball_under_feet_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    feet_pos = ctx.sim["feet_pos"]
    ball_pos = ctx.sim["ball_pos"]
    mid_x = 0.5 * (feet_pos[0, 0] + feet_pos[1, 0])
    mid_y = 0.5 * (feet_pos[0, 1] + feet_pos[1, 1])
    error_x = ball_pos[0] - mid_x
    error_y = ball_pos[1] - mid_y
    error_sq = error_x * error_x + error_y * error_y
    return math.exp(-error_sq / (sigma * sigma))


@configclass(kw_only=True)
class BallUnderFeetRewardCfg(RewardTermCfg):
    """Reward keeping the ball under the midpoint of the feet."""

    sigma: float

    def __call__(self, env: ManagerEnv) -> RewardTerm:
        del env
        return RewardTerm(ball_under_feet_reward, np.float32(self.sigma))


@kernel_data
class DofDefaultParams:
    reference: SharedArray
    sigma: np.float32


@dispatch
def dof_default_reward(ctx: ManagerContext, params: DofDefaultParams) -> float:
    dof_pos = ctx.sim["robot_dof_pos"]
    error_sq = 0.0
    for joint_id in range(dof_pos.shape[0]):
        error = dof_pos[joint_id] - params.reference[joint_id]
        error_sq += error * error
    error_sq /= dof_pos.shape[0]
    return math.exp(-error_sq / (params.sigma * params.sigma))


@configclass(kw_only=True)
class DofDefaultRewardCfg(RewardTermCfg):
    """Reward staying close to the default posture."""

    sigma: float

    def __call__(self, env: ManagerEnv) -> RewardTerm:
        query = env.sim_data.query("robot_dof_pos")
        robot = env.cfg.scene.objs.robot
        default = dict(
            zip(
                (robot.resolve_name(name) for name in robot.key_pose.joint_names),
                robot.key_pose.poses["default"],
                strict=True,
            )
        )
        reference = np.asarray([default[name] for name in query.joints], dtype=np.float32)
        params = DofDefaultParams(reference, np.float32(self.sigma))
        return RewardTerm(dof_default_reward, params)
