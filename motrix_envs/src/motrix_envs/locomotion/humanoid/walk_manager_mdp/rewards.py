# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reward terms for the humanoid velocity-tracking task."""

import math

import numpy as np
import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerContext,
    RewardTerm,
    RewardTermCfg,
)
from motrix_env_core.manager.context import BuildContext
from motrix_env_core.manager.dispatch import dispatch
from motrix_env_core.math.quaternion_array import rotate_inverse_components
from motrix_env_core.mdp.terrain import HeightFieldGrid, heightfield_lookup
from motrix_env_core.sim import (
    BatchLinkPositionQuery,
    BatchLinkQuaternionQuery,
    JointPositionQuery,
    LinkAngularVelocityQuery,
    LinkQuaternionQuery,
    SitePositionQuery,
)
from motrix_envs.locomotion.humanoid.walk_manager_mdp.terrain import ground_height_grid


def _expected_foot_height(phi: torch.Tensor, swing_height: float) -> torch.Tensor:
    """Expected foot height from gait phase using the direct env's Bezier profile (float64, ``(N,)``)."""
    x = (phi.to(torch.float64) + math.pi) / (2.0 * math.pi)
    swing = float(swing_height)
    t = torch.where(x <= 0.5, 2.0 * x, 2.0 * x - 1.0)
    y_start = torch.where(x <= 0.5, 0.0, swing)
    y_end = torch.where(x <= 0.5, swing, 0.0)
    return y_start + (y_end - y_start) * (t**3 + 3.0 * (t**2 * (1.0 - t)))


@dispatch
def penalty_ang_vel_xy_reward(ctx: ManagerContext, base_quat: torch.Tensor, base_ang_vel: torch.Tensor) -> torch.Tensor:
    vx, vy, _ = rotate_inverse_components(base_quat, base_ang_vel)
    walk = ctx.commands["walk"]
    return (vx * vx + vy * vy) * walk.penalty_scale[0]


@configclass(kw_only=True)
class PenaltyAngVelXyRewardCfg(RewardTermCfg):
    def __call__(self, ctx) -> RewardTerm:
        link = ctx.model.bodies["robot"].base_link_name
        return RewardTerm(
            penalty_ang_vel_xy_reward,
            LinkQuaternionQuery(link=link),
            LinkAngularVelocityQuery(link=link),
        )


@dispatch
def penalty_orientation_reward(ctx: ManagerContext, base_quat: torch.Tensor) -> torch.Tensor:
    gravity = base_quat.new_zeros(base_quat.shape[0], 3)
    gravity[:, 2] = -1.0
    gx, gy, _ = rotate_inverse_components(base_quat, gravity)
    walk = ctx.commands["walk"]
    return (gx * gx + gy * gy) * walk.penalty_scale[0]


@configclass(kw_only=True)
class PenaltyOrientationRewardCfg(RewardTermCfg):
    def __call__(self, ctx) -> RewardTerm:
        link = ctx.model.bodies["robot"].base_link_name
        return RewardTerm(
            penalty_orientation_reward,
            LinkQuaternionQuery(link=link),
        )


@dispatch
def penalty_action_rate_reward(ctx: ManagerContext) -> torch.Tensor:
    action = ctx.actions["joint_position"]
    delta = action.current() - action.previous()
    walk = ctx.commands["walk"]
    return (delta * delta).sum(dim=1) * walk.penalty_scale[0]


@configclass(kw_only=True)
class PenaltyActionRateRewardCfg(RewardTermCfg):
    def __call__(self, ctx) -> RewardTerm:
        return RewardTerm(penalty_action_rate_reward)


@dispatch
def feet_phase_reward(
    ctx: ManagerContext,
    swing_height: float,
    feet_phase_sigma: float,
    heightfield: HeightFieldGrid,
    sole_l: torch.Tensor,
    sole_r: torch.Tensor,
) -> torch.Tensor:
    walk = ctx.commands["walk"]
    soles = (sole_l, sole_r)
    error = None
    for foot in range(2):
        expected = _expected_foot_height(walk.phase[:, foot], swing_height)
        sole = soles[foot]
        ground_z = heightfield_lookup(heightfield, sole[:, 0], sole[:, 1])
        delta = sole[:, 2].to(torch.float64) - ground_z - expected
        error = delta * delta if error is None else error + delta * delta
    return torch.exp(-error / float(feet_phase_sigma)).to(torch.float32)


@configclass(kw_only=True)
class FeetPhaseRewardCfg(RewardTermCfg):
    """Foot-clearance tracking reward against the gait-phase reference.

    ``sole_l_site``/``sole_r_site`` carry the resolved sole-site names;
    ``ground_geom`` names the floor geom used for flat-ground height lookups.
    """

    sole_l_site: str = ""
    sole_r_site: str = ""
    swing_height: float = 0.09
    feet_phase_sigma: float = 0.008
    ground_geom: str = ""

    def __call__(self, ctx) -> RewardTerm:
        if not self.sole_l_site or not self.sole_r_site or not self.ground_geom:
            raise ValueError("FeetPhaseRewardCfg requires sole_l_site, sole_r_site, and ground_geom.")
        return RewardTerm(
            feet_phase_reward,
            np.float32(self.swing_height),
            np.float32(self.feet_phase_sigma),
            ground_height_grid(ctx, self.ground_geom),
            SitePositionQuery(site=self.sole_l_site),
            SitePositionQuery(site=self.sole_r_site),
        )


@dispatch
def pose_reward(
    ctx: ManagerContext, default_joint_angles: torch.Tensor, pose_weights: torch.Tensor, joint_pos: torch.Tensor
) -> torch.Tensor:
    delta = joint_pos - default_joint_angles
    walk = ctx.commands["walk"]
    return ((delta * delta) * pose_weights).sum(dim=1) * walk.penalty_scale[0]


@configclass(kw_only=True)
class PoseRewardCfg(RewardTermCfg):
    """Weighted deviation from the robot's default key pose.

    ``pose_weights`` must cover every body joint; coverage is validated at
    build time.
    """

    pose_weights: dict[str, float] = {}

    def __call__(self, ctx) -> RewardTerm:
        body = ctx.model.bodies["robot"]
        missing = sorted(set(body.joint_names).difference(self.pose_weights))
        if missing:
            raise KeyError(f"pose_weights must cover all joints; missing={missing}")
        weights = np.asarray([self.pose_weights[name] for name in body.joint_names], dtype=np.float32)
        if np.any(weights < 0.0):
            raise ValueError("pose_weights must be non-negative")
        # The query must use the compiled body joint names (prefix/suffix
        # resolved) so it stays aligned with body.init_joint_pos and weights.
        return RewardTerm(
            pose_reward,
            body.init_joint_pos,
            weights,
            JointPositionQuery(joints=body.joint_names),
        )


@dispatch
def penalty_close_feet_xy_reward(
    ctx: ManagerContext, close_feet_threshold: float, base_quat: torch.Tensor, foot_pos: torch.Tensor
) -> torch.Tensor:
    # Lateral foot separation in the base frame: rotate the world-frame foot
    # delta's XY components (z forced to zero — vertical separation is
    # intentionally ignored) back into the base frame and take its y
    # magnitude. Rotating the world x-axis instead (as a yaw proxy) flips the
    # yaw sign under the inverse rotation and misjudges staggered feet as
    # crossing.
    delta = foot_pos.new_zeros(foot_pos.shape[0], 3)
    delta[:, :2] = foot_pos[:, 0, :2] - foot_pos[:, 1, :2]
    _, lateral, _ = rotate_inverse_components(base_quat, delta)
    walk = ctx.commands["walk"]
    return torch.where(lateral.abs() < close_feet_threshold, walk.penalty_scale[0], torch.zeros_like(lateral))


@configclass(kw_only=True)
class PenaltyCloseFeetXyRewardCfg(RewardTermCfg):
    """Penalize lateral foot separation below ``close_feet_threshold``."""

    close_feet_threshold: float = 0.15

    def __call__(self, ctx) -> RewardTerm:
        body = ctx.model.bodies["robot"]
        return RewardTerm(
            penalty_close_feet_xy_reward,
            np.float32(self.close_feet_threshold),
            LinkQuaternionQuery(link=body.base_link_name),
            BatchLinkPositionQuery(links=ctx.cfg.scene.objs.robot.resolved_foot_link_names),
        )


@dispatch
def penalty_feet_ori_reward(
    ctx: ManagerContext, default_foot_gravity: torch.Tensor, foot_quat: torch.Tensor
) -> torch.Tensor:
    # foot_quat is (N, F, 4); rotate gravity by each foot's inverse quaternion.
    num_feet = foot_quat.shape[1]
    gravity = foot_quat.new_zeros(*foot_quat.shape[:2], 3)
    gravity[..., 2] = -1.0
    gx, gy, gz = rotate_inverse_components(foot_quat.reshape(-1, 4), gravity.reshape(-1, 3))
    reference = torch.as_tensor(np.asarray(default_foot_gravity))
    total = None
    for foot in range(num_feet):
        # Flat layout is lane-major (lane * F + foot): strided slices pick one foot.
        fx = gx[foot::num_feet]
        fy = gy[foot::num_feet]
        fz = gz[foot::num_feet]
        ref = reference[foot]
        cx = fy * ref[2] - fz * ref[1]
        cy = fz * ref[0] - fx * ref[2]
        cz = fx * ref[1] - fy * ref[0]
        norm = torch.sqrt(cx * cx + cy * cy + cz * cz)
        total = norm if total is None else total + norm
    walk = ctx.commands["walk"]
    return total * walk.penalty_scale[0]


@configclass(kw_only=True)
class PenaltyFeetOriRewardCfg(RewardTermCfg):
    def __call__(self, ctx) -> RewardTerm:
        robot = ctx.cfg.scene.objs.robot
        # Default foot gravity at the init key pose, from BodyModel's
        # compile-time init-pose FK snapshot.
        default_foot_gravity = _default_foot_gravity(ctx, robot)
        return RewardTerm(
            penalty_feet_ori_reward,
            default_foot_gravity,
            BatchLinkQuaternionQuery(links=robot.resolved_foot_link_names),
        )


def _default_foot_gravity(ctx: BuildContext, robot) -> np.ndarray:
    """Rotate world gravity into each default foot frame (host-side numpy)."""
    body = ctx.model.bodies["robot"]
    quats = np.stack([body.init_link_quats[body.link_names.index(link)] for link in robot.resolved_foot_link_names])
    gravity = np.broadcast_to(np.array((0.0, 0.0, -1.0)), quats.shape[:-1] + (3,))
    gx, gy, gz = rotate_inverse_components(quats, gravity)
    return np.stack((gx, gy, gz), axis=-1).astype(np.float32)
