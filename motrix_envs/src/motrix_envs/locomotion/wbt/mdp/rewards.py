# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""WBT command-backed reward terms."""

from dataclasses import dataclass
from typing import cast

import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, RewardTerm, RewardTermCfg
from motrix_env_core.math.quaternion import rotation_distance
from motrix_env_core.mdp.action import JointPositionActionState


def global_ref_position_reward(ctx: ManagerContext, sigma: float) -> torch.Tensor:
    motion = ctx.commands["motion"]
    steps = motion.steps[:, 0]
    error = motion.clip.reference_body_pos_w[steps] - ctx.sim["tracked_body_pos"][:, motion.reference_index]
    return torch.exp(-torch.sum(error * error, dim=-1) / (sigma * sigma))


@configclass(kw_only=True)
class GlobalRefPositionRewardCfg(RewardTermCfg):
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(global_ref_position_reward, float(self.sigma))


def global_ref_orientation_reward(ctx: ManagerContext, sigma: float) -> torch.Tensor:
    motion = ctx.commands["motion"]
    steps = motion.steps[:, 0]
    target = motion.clip.reference_body_quat_w[steps]
    quaternion = ctx.sim["tracked_body_quat"][:, motion.reference_index]
    distance = rotation_distance(target, quaternion)
    return torch.exp(-(distance * distance) / (sigma * sigma))


@configclass(kw_only=True)
class GlobalRefOrientationRewardCfg(RewardTermCfg):
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(global_ref_orientation_reward, float(self.sigma))


def relative_body_position_reward(ctx: ManagerContext, sigma: float) -> torch.Tensor:
    motion = ctx.commands["motion"]
    diff = motion.target_body_position_relative - ctx.sim["tracked_body_pos"]
    error_sq = torch.sum(diff * diff, dim=(-2, -1)) / diff.shape[-2]
    return torch.exp(-error_sq / (sigma * sigma))


@configclass(kw_only=True)
class RelativeBodyPositionRewardCfg(RewardTermCfg):
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(relative_body_position_reward, float(self.sigma))


def ee_body_pos_reward(ctx: ManagerContext, body_indices: tuple[int, ...], sigma: float) -> torch.Tensor:
    motion = ctx.commands["motion"]
    position = ctx.sim["tracked_body_pos"]
    error = motion.target_body_position_relative[:, body_indices, :] - position[:, body_indices, :]
    error_sq = torch.sum(error * error, dim=(-2, -1)) / len(body_indices)
    return torch.exp(-error_sq / (sigma * sigma))


@configclass(kw_only=True)
class EeBodyPosRewardCfg(RewardTermCfg):
    """Full-3D end-effector position tracking (ankles, wrists).

    zh_CN: 末端 body（踝、腕）的三维位置专项跟踪奖励。

    Foot placement error lives mostly in the horizontal plane, and each foot
    is 1/14 of the all-body relative-position mean — too diluted to shape
    landing accuracy. This term tracks the full 3D end-effector error
    directly, including the height component.
    """

    body_names: tuple[str, ...] = ()
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        if not self.body_names:
            raise ValueError(
                "EeBodyPosRewardCfg requires a non-empty body_names: the kernel "
                "divides the squared tracking error by the number of bodies."
            )
        tracked_body_names = ctx.cfg.commands.motion.tracked_body_names
        body_indices = tuple(tracked_body_names.index(name) for name in self.body_names)
        return RewardTerm(ee_body_pos_reward, body_indices, float(self.sigma))


def relative_body_orientation_reward(ctx: ManagerContext, sigma: float) -> torch.Tensor:
    motion = ctx.commands["motion"]
    quaternion = ctx.sim["tracked_body_quat"]
    distance = rotation_distance(motion.target_body_orientation_relative, quaternion)
    error_sq = torch.sum(distance * distance, dim=-1) / quaternion.shape[-2]
    return torch.exp(-error_sq / (sigma * sigma))


@configclass(kw_only=True)
class RelativeBodyOrientationRewardCfg(RewardTermCfg):
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(relative_body_orientation_reward, float(self.sigma))


def global_body_linear_velocity_reward(ctx: ManagerContext, sigma: float) -> torch.Tensor:
    motion = ctx.commands["motion"]
    steps = motion.steps[:, 0]
    velocity = ctx.sim["tracked_body_linear_velocity"]
    error = motion.clip.tracked_bodies_lin_vel_w[steps] - velocity
    error_sq = torch.sum(error * error, dim=(-2, -1)) / velocity.shape[-2]
    return torch.exp(-error_sq / (sigma * sigma))


@configclass(kw_only=True)
class GlobalBodyLinearVelocityRewardCfg(RewardTermCfg):
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(global_body_linear_velocity_reward, float(self.sigma))


def global_body_angular_velocity_reward(ctx: ManagerContext, sigma: float) -> torch.Tensor:
    motion = ctx.commands["motion"]
    steps = motion.steps[:, 0]
    velocity = ctx.sim["tracked_body_angular_velocity"]
    error = motion.clip.tracked_bodies_ang_vel_w[steps] - velocity
    error_sq = torch.sum(error * error, dim=(-2, -1)) / velocity.shape[-2]
    return torch.exp(-error_sq / (sigma * sigma))


@configclass(kw_only=True)
class GlobalBodyAngularVelocityRewardCfg(RewardTermCfg):
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(global_body_angular_velocity_reward, float(self.sigma))


@dataclass(frozen=True, slots=True)
class DofLimitParams:
    midpoint: torch.Tensor
    half_range: torch.Tensor
    soft_limit: float
    cap: float


def dof_limit_reward(ctx: ManagerContext, params: DofLimitParams) -> torch.Tensor:
    dof_position = ctx.sim["robot_dof_pos"]
    violation = torch.abs(dof_position - params.midpoint) - params.half_range * params.soft_limit
    total = torch.sum(torch.where(violation > 0.0, violation, 0.0), dim=-1)
    return torch.where(total > params.cap, params.cap, total)


@configclass(kw_only=True)
class DofLimitRewardCfg(RewardTermCfg):
    soft_limit: float
    cap: float

    def __call__(self, ctx) -> RewardTerm:
        action = cast(JointPositionActionState, ctx.action_terms["joint_position"].state)
        params = DofLimitParams(
            midpoint=(action.joint_lower + action.joint_upper) * 0.5,
            half_range=(action.joint_upper - action.joint_lower) * 0.5,
            soft_limit=float(self.soft_limit),
            cap=float(self.cap),
        )
        return RewardTerm(dof_limit_reward, params)


def undesired_contacts_reward(ctx: ManagerContext, threshold: float) -> torch.Tensor:
    contact_forces = ctx.sim["undesired_contact_forces"]
    forces = contact_forces[..., :3]
    norm = torch.sqrt(torch.sum(forces * forces, dim=-1))
    return torch.sum((norm > threshold).to(contact_forces.dtype), dim=-1)


@configclass(kw_only=True)
class UndesiredContactsRewardCfg(RewardTermCfg):
    threshold: float

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(undesired_contacts_reward, float(self.threshold))
