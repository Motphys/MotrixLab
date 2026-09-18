# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reward terms for the humanoid velocity-tracking task."""

import math

import numpy as np
from numba import njit

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerContext,
    RewardTerm,
    RewardTermCfg,
)
from motrix_env_core.mdp.terrain import HeightFieldGrid, heightfield_lookup
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.numba.math.quaternion import rotate_inverse_components
from motrix_env_core.sim import (
    BatchLinkPositionQuery,
    BatchLinkQuaternionQuery,
    JointPositionQuery,
    LinkAngularVelocityQuery,
    LinkQuaternionQuery,
    SitePositionQuery,
)
from motrix_envs.locomotion.humanoid.walk_manager_mdp.command import WalkCommand
from motrix_envs.locomotion.humanoid.walk_manager_mdp.terrain import ground_height_grid


@njit(inline="always")
def _expected_foot_height(phi: float, swing_height: float) -> float:
    """Expected foot height from gait phase using the direct env's Bezier profile."""
    x = (phi + math.pi) / (2.0 * math.pi)

    def bezier(y_start, y_end, t):
        return y_start + (y_end - y_start) * (t**3 + 3.0 * (t**2 * (1.0 - t)))

    if x <= 0.5:
        return bezier(0.0, swing_height, 2.0 * x)
    return bezier(swing_height, 0.0, 2.0 * x - 1.0)


@dispatch
def penalty_ang_vel_xy_reward(ctx: ManagerContext, base_quat: np.ndarray, base_ang_vel: np.ndarray) -> float:
    vx, vy, _ = rotate_inverse_components(base_quat, base_ang_vel)
    walk: WalkCommand = ctx.commands["walk"]
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
def penalty_orientation_reward(ctx: ManagerContext, base_quat: np.ndarray) -> float:
    gx, gy, _ = rotate_inverse_components(base_quat, (0.0, 0.0, -1.0))
    walk: WalkCommand = ctx.commands["walk"]
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
def penalty_action_rate_reward(ctx: ManagerContext) -> float:
    action = ctx.actions["joint_position"]
    delta = action.current - action.previous
    walk: WalkCommand = ctx.commands["walk"]
    return float(np.dot(delta, delta)) * walk.penalty_scale[0]


@configclass(kw_only=True)
class PenaltyActionRateRewardCfg(RewardTermCfg):
    def __call__(self, ctx) -> RewardTerm:
        del ctx
        return RewardTerm(penalty_action_rate_reward)


@dispatch
def feet_phase_reward(
    ctx: ManagerContext,
    swing_height: np.float32,
    feet_phase_sigma: np.float32,
    heightfield: HeightFieldGrid,
    sole_l: np.ndarray,
    sole_r: np.ndarray,
) -> float:
    walk: WalkCommand = ctx.commands["walk"]
    error = 0.0
    for foot in range(2):
        expected = _expected_foot_height(walk.phase[foot], swing_height)
        sole = sole_l if foot == 0 else sole_r
        ground_z = heightfield_lookup(heightfield, sole[0], sole[1])
        delta = sole[2] - ground_z - expected
        error += delta * delta
    return math.exp(-error / feet_phase_sigma)


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
    ctx: ManagerContext, default_joint_angles: np.ndarray, pose_weights: np.ndarray, joint_pos: np.ndarray
) -> float:
    delta = joint_pos - default_joint_angles
    walk: WalkCommand = ctx.commands["walk"]
    return float(np.dot(delta * delta, pose_weights)) * walk.penalty_scale[0]


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
    ctx: ManagerContext, close_feet_threshold: np.float32, base_quat: np.ndarray, foot_pos: np.ndarray
) -> float:
    left = foot_pos[0]
    right = foot_pos[1]
    # Lateral foot separation in the base frame: rotate the world-frame foot
    # delta's XY components (z forced to zero — vertical separation is
    # intentionally ignored) back into the base frame and take its y
    # magnitude. Rotating the world x-axis instead (as a yaw proxy) flips the
    # yaw sign under the inverse rotation and misjudges staggered feet as
    # crossing.
    _, lateral, _ = rotate_inverse_components(base_quat, (left[0] - right[0], left[1] - right[1], 0.0))
    distance = abs(lateral)
    walk: WalkCommand = ctx.commands["walk"]
    if distance < close_feet_threshold:
        return 1.0 * walk.penalty_scale[0]
    return 0.0


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
def penalty_feet_ori_reward(ctx: ManagerContext, default_foot_gravity: np.ndarray, foot_quat: np.ndarray) -> float:
    total = 0.0
    for foot in range(2):
        gx, gy, gz = rotate_inverse_components(foot_quat[foot], (0.0, 0.0, -1.0))
        reference = default_foot_gravity[foot]
        # |cross(foot_gravity, reference)|
        cx = gy * reference[2] - gz * reference[1]
        cy = gz * reference[0] - gx * reference[2]
        cz = gx * reference[1] - gy * reference[0]
        total += math.sqrt(cx * cx + cy * cy + cz * cz)
    walk: WalkCommand = ctx.commands["walk"]
    return total * walk.penalty_scale[0]


@configclass(kw_only=True)
class PenaltyFeetOriRewardCfg(RewardTermCfg):
    def __call__(self, ctx) -> RewardTerm:
        robot = ctx.cfg.scene.objs.robot
        body = ctx.model.bodies["robot"]
        # Default foot gravity at the init key pose, from BodyModel's
        # compile-time init-pose FK snapshot.
        gravity_vec = (0.0, 0.0, -1.0)
        default_foot_gravity = np.stack(
            [
                rotate_inverse_components(body.init_link_quats[body.link_names.index(link)], gravity_vec)
                for link in robot.resolved_foot_link_names
            ]
        ).astype(np.float32)
        return RewardTerm(
            penalty_feet_ori_reward,
            default_foot_gravity,
            BatchLinkQuaternionQuery(links=robot.resolved_foot_link_names),
        )
