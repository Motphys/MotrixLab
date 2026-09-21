# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""WBT command-backed observation terms."""

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import RobotCfg
from motrix_env_core.manager import (
    ManagerContext,
    ObservationTermCfg,
    ObsTerm,
    kernel_data,
    njit,
)
from motrix_env_core.manager.math.quaternion import inverse as quat_inverse
from motrix_env_core.manager.math.quaternion import mul as quat_mul
from motrix_env_core.manager.math.quaternion import rotate_inverse, to_matrix_first_two_columns
from motrix_env_core.mdp.noise import add_uniform_noise
from motrix_env_core.mdp.observations import (
    UniformNoiseCfg,
)
from motrix_env_core.numba.kernel_data import SharedArray
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand


@njit(inline="always")
def _write_relative_orientation_6d(
    reference_quat_w: np.ndarray,
    target_quat_w: np.ndarray,
    out: np.ndarray,
) -> None:
    relative_quat = out[:4]
    quat_inverse(reference_quat_w, relative_quat)
    quat_mul(relative_quat, target_quat_w, relative_quat)
    to_matrix_first_two_columns(relative_quat, out[:6])


@dispatch
def motion_joint_obs(ctx: ManagerContext, out: np.ndarray) -> None:
    motion: WbtMotionCommand = ctx.commands["motion"]
    joint_count = motion.clip.joint_pos.shape[1]
    out[:joint_count] = motion.clip.joint_pos[motion.steps[0]]
    out[joint_count:] = motion.clip.joint_vel[motion.steps[0]]


@configclass(kw_only=True)
class MotionJointObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        motion = ctx.command_terms["motion"]
        if not isinstance(motion, WbtMotionCommand):
            raise TypeError(f"WBT motion command must be WbtMotionCommand, got {type(motion).__name__}.")
        return ObsTerm(2 * motion.clip.joint_pos.shape[1], motion_joint_obs)


@dispatch
def motion_reference_position_obs(ctx: ManagerContext, out: np.ndarray) -> None:
    tracked_body_pos = ctx.sim["tracked_body_pos"]
    tracked_body_quat = ctx.sim["tracked_body_quat"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    reference_pos = tracked_body_pos[motion.reference_index]
    reference_quat = tracked_body_quat[motion.reference_index]
    target_ref_pos = motion.clip.reference_body_pos_w[motion.steps[0]]
    rotate_inverse(
        reference_quat,
        (
            target_ref_pos[0] - reference_pos[0],
            target_ref_pos[1] - reference_pos[1],
            target_ref_pos[2] - reference_pos[2],
        ),
        out,
    )


@configclass(kw_only=True)
class MotionReferencePositionObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        del ctx
        return ObsTerm(3, motion_reference_position_obs)


@dispatch
def motion_reference_orientation_obs(ctx: ManagerContext, out: np.ndarray, noise_amplitude: np.float32) -> None:
    tracked_body_quat = ctx.sim["tracked_body_quat"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    rand = ctx.rand
    reference_quat = tracked_body_quat[motion.reference_index]
    _write_relative_orientation_6d(reference_quat, motion.clip.reference_body_quat_w[motion.steps[0]], out)
    add_uniform_noise(out, noise_amplitude, rand.state)


@configclass(kw_only=True)
class MotionReferenceOrientationObsCfg(ObservationTermCfg):
    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx) -> ObsTerm:
        del ctx
        return ObsTerm(6, motion_reference_orientation_obs, np.float32(self.noise.amplitude))


@dispatch
def robot_body_position_in_reference_frame_obs(ctx: ManagerContext, out: np.ndarray) -> None:
    tracked_body_pos = ctx.sim["tracked_body_pos"]
    tracked_body_quat = ctx.sim["tracked_body_quat"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    reference_pos = tracked_body_pos[motion.reference_index]
    reference_quat = tracked_body_quat[motion.reference_index]
    for body_id in range(tracked_body_pos.shape[0]):
        offset = body_id * 3
        rotate_inverse(
            reference_quat,
            (
                tracked_body_pos[body_id, 0] - reference_pos[0],
                tracked_body_pos[body_id, 1] - reference_pos[1],
                tracked_body_pos[body_id, 2] - reference_pos[2],
            ),
            out[offset : offset + 3],
        )


@configclass(kw_only=True)
class RobotBodyPositionInReferenceFrameObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        num_tracked = len(ctx.cfg.commands.motion.tracked_body_names)
        return ObsTerm(3 * num_tracked, robot_body_position_in_reference_frame_obs)


@dispatch
def robot_body_orientation_obs(ctx: ManagerContext, out: np.ndarray) -> None:
    tracked_body_quat = ctx.sim["tracked_body_quat"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    quat_inverse(tracked_body_quat[motion.reference_index], out[:4])
    for body_id in range(tracked_body_quat.shape[0]):
        offset = body_id * 6
        relative_quat = out[offset : offset + 4]
        relative_quat[:] = out[:4]
        quat_mul(relative_quat, tracked_body_quat[body_id], relative_quat)
        to_matrix_first_two_columns(relative_quat, out[offset : offset + 6])


@configclass(kw_only=True)
class RobotBodyOrientationObsCfg(ObservationTermCfg):
    def __call__(self, ctx) -> ObsTerm:
        num_tracked = len(ctx.cfg.commands.motion.tracked_body_names)
        return ObsTerm(6 * num_tracked, robot_body_orientation_obs)


@kernel_data
class RelativePositionParams:
    reference: SharedArray
    noise_amplitude: np.float32


@dispatch
def dof_pos_rel_obs(ctx: ManagerContext, out: np.ndarray, params: RelativePositionParams) -> None:
    dof_pos = ctx.sim["robot_dof_pos"]
    rand = ctx.rand
    np.subtract(dof_pos, params.reference, out)
    add_uniform_noise(out, params.noise_amplitude, rand.state)


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
        params = RelativePositionParams(reference, np.float32(self.noise.amplitude))
        return ObsTerm(reference.shape[0], dof_pos_rel_obs, params)


@dispatch
def dof_vel_obs(ctx: ManagerContext, out: np.ndarray, noise_amplitude: np.float32) -> None:
    dof_vel = ctx.sim["robot_dof_vel"]
    rand = ctx.rand
    out[:] = dof_vel
    add_uniform_noise(out, noise_amplitude, rand.state)


@configclass(kw_only=True)
class DofVelObsCfg(ObservationTermCfg):
    """Joint velocities from the WBT-owned ``robot_dof_vel`` query."""

    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx) -> ObsTerm:
        num_joints = len(ctx.model.bodies["robot"].joint_names)
        return ObsTerm(num_joints, dof_vel_obs, np.float32(self.noise.amplitude))
