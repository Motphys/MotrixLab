# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""WBT command-backed observation terms."""

from dataclasses import dataclass

import numpy as np
import torch

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager import (
    ManagerContext,
    ObservationTermCfg,
    ObsTerm,
)
from motrix_env_core.math import quaternion_array
from motrix_env_core.math.quaternion_array import relative_orientation, rotate_inverse
from motrix_env_core.mdp.observations import (
    UniformNoiseCfg,
    add_uniform_noise,
)
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand


def motion_joint_obs(ctx: ManagerContext, out: torch.Tensor) -> None:
    motion = ctx.commands["motion"]
    steps = motion.steps[:, 0]
    joint_count = motion.clip.joint_pos.shape[1]
    out[:, :joint_count] = motion.clip.joint_pos[steps]
    out[:, joint_count:] = motion.clip.joint_vel[steps]


@configclass(kw_only=True)
class MotionJointObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        motion = ctx.command_terms["motion"]
        if not isinstance(motion, WbtMotionCommand):
            raise TypeError(f"WBT motion command must be WbtMotionCommand, got {type(motion).__name__}.")
        return ObsTerm(2 * motion.clip.joint_pos.shape[1], motion_joint_obs)


def motion_reference_position_obs(ctx: ManagerContext, out: torch.Tensor) -> None:
    motion = ctx.commands["motion"]
    reference_index = motion.reference_index
    reference_pos = ctx.sim["tracked_body_pos"][:, reference_index]
    reference_quat = ctx.sim["tracked_body_quat"][:, reference_index]
    local = rotate_inverse(
        reference_quat,
        motion.clip.reference_body_pos_w[motion.steps[:, 0]] - reference_pos,
    )
    out.copy_(local)


@configclass(kw_only=True)
class MotionReferencePositionObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(3, motion_reference_position_obs)


def motion_reference_orientation_obs(
    ctx: ManagerContext,
    out: torch.Tensor,
    noise_amplitude: float,
) -> None:
    motion = ctx.commands["motion"]
    reference_quat = ctx.sim["tracked_body_quat"][:, motion.reference_index]
    local = relative_orientation(reference_quat, motion.clip.reference_body_quat_w[motion.steps[:, 0]])
    out.copy_(local)
    add_uniform_noise(ctx, out, noise_amplitude)


@configclass(kw_only=True)
class MotionReferenceOrientationObsCfg(ObservationTermCfg):
    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(6, motion_reference_orientation_obs, np.float32(self.noise.amplitude))


def robot_body_position_in_reference_frame_obs(ctx: ManagerContext, out: torch.Tensor) -> None:
    motion = ctx.commands["motion"]
    reference_index = motion.reference_index
    reference_pos = ctx.sim["tracked_body_pos"][:, reference_index]
    local = rotate_inverse(
        ctx.sim["tracked_body_quat"][:, reference_index][:, None, :],
        ctx.sim["tracked_body_pos"] - reference_pos[:, None, :],
    )
    out.copy_(local.reshape(local.shape[0], -1))


@configclass(kw_only=True)
class RobotBodyPositionInReferenceFrameObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        num_tracked = len(ctx.cfg.commands.motion.tracked_body_names)
        return ObsTerm(3 * num_tracked, robot_body_position_in_reference_frame_obs)


def robot_body_orientation_obs(ctx: ManagerContext, out: torch.Tensor) -> None:
    # Bug-compatible with the historical per-lane oracle: body zero writes its
    # 6D matrix over the shared out[:4] inverse-quaternion scratch, and every
    # later body reads those four matrix entries as its reference quaternion.
    # Do not silently repair this alias; migrate the observation contract
    # explicitly.
    tracked_body_quat = ctx.sim["tracked_body_quat"]
    reference_quat = tracked_body_quat[:, ctx.commands["motion"].reference_index]
    first = quaternion_array.matrix6d(
        quaternion_array.mul(quaternion_array.inverse(reference_quat), tracked_body_quat[:, 0, :])
    )
    rest = quaternion_array.matrix6d(quaternion_array.mul(first[:, None, :4], tracked_body_quat[:, 1:, :]))
    out.copy_(torch.cat((first[:, None, :], rest), dim=1).reshape(tracked_body_quat.shape[0], -1))


@configclass(kw_only=True)
class RobotBodyOrientationObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        num_tracked = len(ctx.cfg.commands.motion.tracked_body_names)
        return ObsTerm(6 * num_tracked, robot_body_orientation_obs)


@dataclass(frozen=True, slots=True)
class RelativePositionParams:
    reference: torch.Tensor
    noise_amplitude: float


def dof_pos_rel_obs(ctx: ManagerContext, out: torch.Tensor, params: RelativePositionParams) -> None:
    out.copy_(ctx.sim["robot_dof_pos"] - params.reference)
    add_uniform_noise(ctx, out, params.noise_amplitude)


@configclass(kw_only=True)
class DofPosRelObsCfg(ObservationTermCfg):
    """Joint positions relative to the scene robot's default key pose.

    Reads the WBT-owned ``robot_dof_pos`` query, which is declared in motion
    joint order; the reference pose is reordered to the same joint order from
    the robot's ``RobotCfg.key_pose``.
    """

    reference_key_pose: str = "default"
    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx) -> ObsTerm:
        # The motion command's __call__() already validates that
        # ``robot_dof_pos`` is a JointPositionQuery in motion joint order.
        queried_joints = ctx.model.bodies["robot"].joint_names
        robot = ctx.cfg.scene.objs.robot
        if not isinstance(robot, RobotCfg):
            raise TypeError(f"WBT scene robot must be RobotCfg, got {type(robot).__name__}.")
        robot_name = robot.resolved_base_link_name
        try:
            key_pose = np.asarray(robot.key_pose.poses[self.reference_key_pose], dtype=np.float32)
        except KeyError as error:
            raise ValueError(
                f"WBT key pose {self.reference_key_pose!r} is not defined on RobotCfg {robot_name!r}; "
                f"available poses: {sorted(robot.key_pose.poses)}."
            ) from error
        resolved_names = (robot.resolve_name(name) for name in robot.key_pose.joint_names)
        positions = dict(zip(resolved_names, key_pose, strict=True))
        missing = sorted(set(queried_joints).difference(positions))
        if missing:
            raise ValueError(f"RobotCfg {robot_name!r} key pose is missing queried joints: {missing}.")
        reference = np.asarray([positions[name] for name in queried_joints], dtype=np.float32)
        params = RelativePositionParams(torch.from_numpy(reference), float(self.noise.amplitude))
        return ObsTerm(reference.shape[0], dof_pos_rel_obs, params)


def dof_vel_obs(ctx: ManagerContext, out: torch.Tensor, noise_amplitude: float) -> None:
    out.copy_(ctx.sim["robot_dof_vel"])
    add_uniform_noise(ctx, out, noise_amplitude)


@configclass(kw_only=True)
class DofVelObsCfg(ObservationTermCfg):
    """Joint velocities from the WBT-owned ``robot_dof_vel`` query."""

    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx) -> ObsTerm:
        num_joints = len(ctx.model.bodies["robot"].joint_names)
        return ObsTerm(num_joints, dof_vel_obs, np.float32(self.noise.amplitude))
