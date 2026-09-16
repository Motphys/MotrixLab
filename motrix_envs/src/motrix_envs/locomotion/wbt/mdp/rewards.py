# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""WBT command-backed reward terms."""

import math
from typing import cast

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, RewardTerm, RewardTermCfg
from motrix_env_core.manager.math.quaternion import rotation_distance
from motrix_env_core.numba.kernel_data import SharedArray, kernel_data
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_envs.locomotion.wbt.mdp.action import WbtJointPositionAction
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand


@dispatch
def global_ref_position_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    tracked_body_pos = ctx.sim["tracked_body_pos"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    error = motion.clip.reference_body_pos_w[motion.steps[0]] - tracked_body_pos[motion.reference_index]
    error_sq = float(np.dot(error, error))
    return math.exp(-error_sq / (sigma * sigma))


@configclass(kw_only=True)
class GlobalRefPositionRewardCfg(RewardTermCfg):
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        del ctx
        return RewardTerm(global_ref_position_reward, np.float32(self.sigma))


@dispatch
def global_ref_orientation_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    tracked_body_quat = ctx.sim["tracked_body_quat"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    distance = rotation_distance(
        motion.clip.reference_body_quat_w[motion.steps[0]],
        tracked_body_quat[motion.reference_index],
    )
    return math.exp(-(distance * distance) / (sigma * sigma))


@configclass(kw_only=True)
class GlobalRefOrientationRewardCfg(RewardTermCfg):
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        del ctx
        return RewardTerm(global_ref_orientation_reward, np.float32(self.sigma))


@dispatch
def relative_body_position_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    tracked_body_pos = ctx.sim["tracked_body_pos"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    diff = motion.target_body_position_relative - tracked_body_pos
    error_sq = float(np.sum(diff * diff))
    return math.exp(-(error_sq / tracked_body_pos.shape[0]) / (sigma * sigma))


@configclass(kw_only=True)
class RelativeBodyPositionRewardCfg(RewardTermCfg):
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        del ctx
        return RewardTerm(relative_body_position_reward, np.float32(self.sigma))


@dispatch
def relative_body_orientation_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    tracked_body_quat = ctx.sim["tracked_body_quat"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    error_sq = 0.0
    for body_id in range(tracked_body_quat.shape[0]):
        distance = rotation_distance(
            motion.target_body_orientation_relative[body_id],
            tracked_body_quat[body_id],
        )
        error_sq += distance * distance
    return math.exp(-(error_sq / tracked_body_quat.shape[0]) / (sigma * sigma))


@configclass(kw_only=True)
class RelativeBodyOrientationRewardCfg(RewardTermCfg):
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        del ctx
        return RewardTerm(relative_body_orientation_reward, np.float32(self.sigma))


@dispatch
def global_body_linear_velocity_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    tracked_body_linear_velocity = ctx.sim["tracked_body_linear_velocity"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    diff = motion.clip.tracked_bodies_lin_vel_w[motion.steps[0]] - tracked_body_linear_velocity
    error_sq = float(np.sum(diff * diff))
    return math.exp(-(error_sq / tracked_body_linear_velocity.shape[0]) / (sigma * sigma))


@configclass(kw_only=True)
class GlobalBodyLinearVelocityRewardCfg(RewardTermCfg):
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        del ctx
        return RewardTerm(global_body_linear_velocity_reward, np.float32(self.sigma))


@dispatch
def global_body_angular_velocity_reward(ctx: ManagerContext, sigma: np.float32) -> float:
    tracked_body_angular_velocity = ctx.sim["tracked_body_angular_velocity"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    diff = motion.clip.tracked_bodies_ang_vel_w[motion.steps[0]] - tracked_body_angular_velocity
    error_sq = float(np.sum(diff * diff))
    return math.exp(-(error_sq / tracked_body_angular_velocity.shape[0]) / (sigma * sigma))


@configclass(kw_only=True)
class GlobalBodyAngularVelocityRewardCfg(RewardTermCfg):
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        del ctx
        return RewardTerm(global_body_angular_velocity_reward, np.float32(self.sigma))


@kernel_data
class DofLimitParams:
    midpoint: SharedArray
    half_range: SharedArray
    soft_limit: np.float32
    cap: np.float32


@dispatch
def dof_limit_reward(ctx: ManagerContext, params: DofLimitParams) -> float:
    dof_pos = ctx.sim["robot_dof_pos"]
    violation = np.abs(dof_pos - params.midpoint) - params.half_range * params.soft_limit
    return min(float(np.sum(np.maximum(violation, 0.0))), params.cap)


@configclass(kw_only=True)
class DofLimitRewardCfg(RewardTermCfg):
    soft_limit: float
    cap: float

    def __call__(self, ctx) -> RewardTerm:
        action = cast(WbtJointPositionAction, ctx.action_terms["joint_position"])
        params = DofLimitParams(
            midpoint=(action.joint_lower + action.joint_upper) * 0.5,
            half_range=(action.joint_upper - action.joint_lower) * 0.5,
            soft_limit=np.float32(self.soft_limit),
            cap=np.float32(self.cap),
        )
        return RewardTerm(dof_limit_reward, params)


@dispatch
def undesired_contacts_reward(ctx: ManagerContext, threshold: np.float32) -> float:
    contact_forces = ctx.sim["undesired_contact_forces"]
    count = 0.0
    for link_id in range(contact_forces.shape[0]):
        fx, fy, fz = contact_forces[link_id, :3]
        if math.sqrt(fx * fx + fy * fy + fz * fz) > threshold:
            count += 1.0
    return count


@configclass(kw_only=True)
class UndesiredContactsRewardCfg(RewardTermCfg):
    threshold: float

    def __call__(self, ctx) -> RewardTerm:
        del ctx
        return RewardTerm(undesired_contacts_reward, np.float32(self.threshold))
