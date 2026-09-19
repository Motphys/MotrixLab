# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""WBT command-backed reward terms."""

import math
from typing import cast

import numpy as np
from numba import literally

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
def ee_body_pos_z_reward(ctx: ManagerContext, body_indices: tuple[int, ...], sigma: np.float32) -> float:
    tracked_body_pos = ctx.sim["tracked_body_pos"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    error_sq = 0.0
    for body_id in body_indices:
        diff = motion.target_body_position_relative[body_id, 2] - tracked_body_pos[body_id, 2]
        error_sq += diff * diff
    return math.exp(-(error_sq / len(body_indices)) / (sigma * sigma))


@configclass(kw_only=True)
class EeBodyPosZRewardCfg(RewardTermCfg):
    """Extra height-tracking reward on end-effector bodies (ankles, wrists).

    zh_CN: 末端 body（踝、腕）的高度专项跟踪奖励。

    Mirrors UniLab's ``motion_ee_body_pos_z``: during a flip the extremity
    heights carry the launch/rotation signal that the all-body mean dilutes,
    so this term tracks their z error separately.
    """

    body_names: tuple[str, ...] = ()
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        tracked_body_names = ctx.cfg.commands.motion.tracked_body_names
        body_indices = tuple(tracked_body_names.index(name) for name in self.body_names)
        return RewardTerm(ee_body_pos_z_reward, body_indices, np.float32(self.sigma))


@dispatch
def ee_body_pos_reward(ctx: ManagerContext, body_indices: tuple[int, ...], sigma: np.float32) -> float:
    tracked_body_pos = ctx.sim["tracked_body_pos"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    error_sq = 0.0
    for body_id in body_indices:
        diff = motion.target_body_position_relative[body_id] - tracked_body_pos[body_id]
        error_sq += float(np.dot(diff, diff))
    return math.exp(-(error_sq / len(body_indices)) / (sigma * sigma))


@configclass(kw_only=True)
class EeBodyPosRewardCfg(RewardTermCfg):
    """Full-3D end-effector position tracking (ankles, wrists).

    zh_CN: 末端 body（踝、腕）的三维位置专项跟踪奖励。

    ``motion_ee_body_pos_z`` only constrains height; foot placement error
    lives mostly in the horizontal plane, and each foot is 1/14 of the
    all-body relative-position mean — too diluted to shape landing accuracy.
    This term tracks the full 3D end-effector error directly.
    """

    body_names: tuple[str, ...] = ()
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        tracked_body_names = ctx.cfg.commands.motion.tracked_body_names
        body_indices = tuple(tracked_body_names.index(name) for name in self.body_names)
        return RewardTerm(ee_body_pos_reward, body_indices, np.float32(self.sigma))


@dispatch
def flight_tuck_reward(
    ctx: ManagerContext,
    dof_indices: tuple[int, ...],
    ankle_body_ids: tuple[int, ...],
    ground_z: np.float32,
    sigma: np.float32,
) -> float:
    motion: WbtMotionCommand = ctx.commands["motion"]
    step = motion.steps[0]
    ref_z = -1.0
    for i in ankle_body_ids:
        ref_z = max(ref_z, motion.clip.tracked_bodies_pos_w[step, i, 2])
    if ref_z <= ground_z:
        # Grounded frames carry no tuck signal; the all-body terms handle them.
        return 0.0
    dof_pos = ctx.sim["robot_dof_pos"]
    clip_pos = motion.clip.joint_pos[step]
    error_sq = 0.0
    for j in dof_indices:
        diff = clip_pos[j] - dof_pos[j]
        error_sq += diff * diff
    return math.exp(-(error_sq / len(dof_indices)) / (sigma * sigma))


@configclass(kw_only=True)
class FlightTuckRewardCfg(RewardTermCfg):
    """Flight-window tuck reward on flexion joints (knees, hip pitch).

    zh_CN: 腾空段团身奖励（膝、髋屈曲专项跟踪）。

    Active only while the reference is airborne (both reference ankles above
    ``ground_z``): a tighter tuck spins the flip faster, and the all-body mean
    position reward dilutes exactly this signal. Empirically load-bearing: the
    landing breakthrough regressed from length ~326 to ~138 without it.
    Joints are resolved by name against the model body joint order, which the
    clip loader enforces on the clip column order.
    """

    body_names: tuple[str, ...] = ()
    joint_names: tuple[str, ...] = ()
    ground_z: float
    sigma: float

    def __call__(self, ctx) -> RewardTerm:
        tracked_body_names = ctx.cfg.commands.motion.tracked_body_names
        ankle_body_ids = tuple(tracked_body_names.index(name) for name in self.body_names)
        joint_names = ctx.model.bodies["robot"].joint_names
        dof_indices = tuple(joint_names.index(name) for name in self.joint_names)
        return RewardTerm(
            flight_tuck_reward,
            dof_indices,
            ankle_body_ids,
            np.float32(self.ground_z),
            np.float32(self.sigma),
        )


@dispatch
def curriculum_action_rate_reward(ctx: ManagerContext, action_name: str) -> float:
    action_name = literally(action_name)
    action: WbtJointPositionAction = ctx.actions[action_name]
    delta = action.current - action.previous
    rate = float(np.dot(delta, delta))
    motion: WbtMotionCommand = ctx.commands["motion"]
    return rate * motion.action_rate_scale[0]


@configclass(kw_only=True)
class CurriculumActionRateRewardCfg(RewardTermCfg):
    """Action-rate penalty scaled by the episode-length curriculum controller.

    zh_CN: 动作率惩罚，倍率由 episode 长度课程控制器（滞环）驱动。

    The motion command maintains an EMA of mean episode length; once it clears
    ``curriculum_high`` the penalty multiplier ramps up toward
    ``curriculum_max_scale`` (anti-jitter phase), and falls back toward 1.0 if
    it drops below ``curriculum_low`` — a closed loop that only taxes jitter
    once the skill is reliably landing.
    """

    action_name: str = "joint_position"

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(curriculum_action_rate_reward, self.action_name)


@dispatch
def flight_gated_action_rate_reward(ctx: ManagerContext, action_name: str, lead: np.int64) -> float:
    action_name = literally(action_name)
    action: WbtJointPositionAction = ctx.actions[action_name]
    motion: WbtMotionCommand = ctx.commands["motion"]
    step = motion.steps[0]
    # The launch (and its violent joint accelerations) lives in the short
    # window from takeoff through touchdown; taxing it there makes any
    # imperfect attempt strictly worse than not attempting (raw L2 ~30-47
    # against saturated tracking kernels), so the tax is skipped inside the
    # window and collected everywhere else.
    if (motion.flight_start - lead) <= step <= motion.land_check_step:
        return 0.0
    delta = action.current - action.previous
    return float(np.dot(delta, delta))


@configclass(kw_only=True)
class FlightGatedActionRateRewardCfg(RewardTermCfg):
    """Action-rate penalty suspended over the launch/flight/landing window.

    zh_CN: 起跳/飞行/落地窗口内暂停征收的动作率惩罚。

    With a heavy flat weight (e.g. -0.5) a from-scratch policy converges to a
    smooth small-hop local optimum: every imperfect launch attempt pays the
    full tax immediately while its tracking rewards are still saturated at
    zero. Gating the tax to the stance/hold phases keeps the smoothing
    benefit where jitter matters (visually and for balance) without pricing
    out the explosive launch.
    """

    action_name: str = "joint_position"
    lead: int = 5

    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(flight_gated_action_rate_reward, self.action_name, np.int64(self.lead))


@dispatch
def flight_rotation_progress_reward(ctx: ManagerContext, cap: np.float32) -> float:
    motion: WbtMotionCommand = ctx.commands["motion"]
    step = motion.steps[0]
    if not (motion.flight_start <= step <= motion.flight_end):
        return 0.0
    pelvis_ang_vel = ctx.sim["tracked_body_angular_velocity"][0]
    ref_ang_vel = motion.clip.tracked_bodies_ang_vel_w[step, 0]
    # Reward pitch-axis rotation speed in the reference's direction, linear
    # and capped: every radian earned pays immediately (exp kernels cannot do
    # this — they saturate to zero exactly when rotation is still partial).
    direction = 1.0 if ref_ang_vel[1] >= 0.0 else -1.0
    progress = pelvis_ang_vel[1] * direction
    return min(max(progress, 0.0), cap)


@configclass(kw_only=True)
class FlightRotationProgressRewardCfg(RewardTermCfg):
    """Dense, unsaturating rotation-progress reward inside the flight window.

    zh_CN: 飞行窗口内按参考方向的俯仰转速线性计分（不饱和）的稠密奖励。
    """

    cap: float = 15.0

    def __call__(self, ctx) -> RewardTerm:
        del ctx
        return RewardTerm(flight_rotation_progress_reward, np.float32(self.cap))


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
