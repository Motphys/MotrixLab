# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""WBT termination terms and raw diagnostic metrics."""

import math

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, TerminationTerm, TerminationTermCfg
from motrix_env_core.manager.math.quaternion import rotation_distance
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_envs.locomotion.wbt.mdp.action import WbtJointPositionAction
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand


@configclass(kw_only=True)
class _WbtTerminationCfg(TerminationTermCfg):
    threshold: float


@dispatch
def bad_ref_z_termination(ctx: ManagerContext, threshold: np.float32) -> bool:
    tracked_body_pos = ctx.sim["tracked_body_pos"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    error = abs(motion.clip.reference_body_pos_w[motion.steps[0], 2] - tracked_body_pos[motion.reference_index, 2])
    ctx.metrics["ref_z_abs_err"][0] = error
    return error > threshold


@dispatch
def bad_ref_z_phased_termination(
    ctx: ManagerContext,
    threshold_out: np.float32,
    threshold_in: np.float32,
) -> bool:
    """Phase-dependent anchor-z termination for clips with an aerial window.

    Outside the flight window the loose ``threshold_out`` lets an early policy
    survive the violent crouch/run-up phases; inside it the tight
    ``threshold_in`` forces the pelvis to actually leave the ground (a
    threshold tight enough that no grounded pose — including tiptoe — can
    satisfy the reference apex).
    """
    tracked_body_pos = ctx.sim["tracked_body_pos"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    step = motion.steps[0]
    error = abs(motion.clip.reference_body_pos_w[step, 2] - tracked_body_pos[motion.reference_index, 2])
    ctx.metrics["ref_z_abs_err"][0] = error
    if motion.flight_start <= step <= motion.flight_end:
        return error > threshold_in
    return error > threshold_out


@configclass(kw_only=True)
class BadRefZPhasedTerminationCfg(_WbtTerminationCfg):
    threshold: float = 0.5  # outside the flight window
    threshold_in: float = 0.25  # inside the flight window

    def __call__(self, ctx) -> TerminationTerm:
        return TerminationTerm(
            bad_ref_z_phased_termination,
            np.float32(self.threshold),
            np.float32(self.threshold_in),
            metric_names=("ref_z_abs_err",),
        )


@dispatch
def bad_ref_position_phased_termination(
    ctx: ManagerContext,
    threshold_out: np.float32,
    threshold_in: np.float32,
) -> bool:
    """Phased 3D anchor-position termination for clips with an aerial window.

    Grounded phases are fully controllable, so planar drift there is a genuine
    failure (tight ``threshold_out``); mid-flight the robot is ballistic and
    cannot correct position, so only catastrophic divergence dies
    (loose ``threshold_in``).
    """
    tracked_body_pos = ctx.sim["tracked_body_pos"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    step = motion.steps[0]
    error = motion.clip.reference_body_pos_w[step] - tracked_body_pos[motion.reference_index]
    distance = math.sqrt(float(np.dot(error, error)))
    ctx.metrics["ref_pos_abs_err"][0] = distance
    if motion.flight_start <= step <= motion.flight_end:
        return distance > threshold_in
    return distance > threshold_out


@configclass(kw_only=True)
class BadRefPositionPhasedTerminationCfg(_WbtTerminationCfg):
    threshold: float = 0.8  # outside the flight window
    threshold_in: float = 1.0  # inside the flight window

    def __call__(self, ctx) -> TerminationTerm:
        return TerminationTerm(
            bad_ref_position_phased_termination,
            np.float32(self.threshold),
            np.float32(self.threshold_in),
            metric_names=("ref_pos_abs_err",),
        )


@configclass(kw_only=True)
class BadRefZTerminationCfg(_WbtTerminationCfg):
    def __call__(self, ctx) -> TerminationTerm:
        del ctx
        return TerminationTerm(
            bad_ref_z_termination,
            np.float32(self.threshold),
            metric_names=("ref_z_abs_err",),
        )


@dispatch
def bad_ref_orientation_termination(ctx: ManagerContext, threshold: np.float32) -> bool:
    tracked_body_quat = ctx.sim["tracked_body_quat"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    target = motion.clip.reference_body_quat_w[motion.steps[0]]
    motion_gravity_z = 2.0 * (target[0] * target[0] + target[1] * target[1]) - 1.0
    robot_quat = tracked_body_quat[motion.reference_index]
    robot_gravity_z = 2.0 * (robot_quat[0] * robot_quat[0] + robot_quat[1] * robot_quat[1]) - 1.0
    error = abs(motion_gravity_z - robot_gravity_z)
    ctx.metrics["ref_ori_abs_err"][0] = error
    return error > threshold


@configclass(kw_only=True)
class BadRefOrientationTerminationCfg(_WbtTerminationCfg):
    def __call__(self, ctx) -> TerminationTerm:
        del ctx
        return TerminationTerm(
            bad_ref_orientation_termination,
            np.float32(self.threshold),
            metric_names=("ref_ori_abs_err",),
        )


@dispatch
def bad_ref_position_termination(ctx: ManagerContext, threshold: np.float32) -> bool:
    """Holosoma-aligned anchor tracking termination: full 3D world-frame error.

    Unlike the z-only variant this also catches planar drift, which the exp
    reward kernels cannot penalize once the anchor leaves their bandwidth.
    """
    tracked_body_pos = ctx.sim["tracked_body_pos"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    error = motion.clip.reference_body_pos_w[motion.steps[0]] - tracked_body_pos[motion.reference_index]
    distance = math.sqrt(float(np.dot(error, error)))
    ctx.metrics["ref_pos_abs_err"][0] = distance
    return distance > threshold


@configclass(kw_only=True)
class BadRefPositionTerminationCfg(_WbtTerminationCfg):
    def __call__(self, ctx) -> TerminationTerm:
        del ctx
        return TerminationTerm(
            bad_ref_position_termination,
            np.float32(self.threshold),
            metric_names=("ref_pos_abs_err",),
        )


@dispatch
def bad_ref_full_orientation_termination(ctx: ManagerContext, threshold: np.float32) -> bool:
    """Holosoma-aligned anchor orientation termination: full quaternion error.

    The legacy variant only compares the gravity-z component (roll/pitch);
    holosoma's ``bad_ref_ori`` measures the complete orientation error.
    """
    tracked_body_quat = ctx.sim["tracked_body_quat"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    error = rotation_distance(
        motion.clip.reference_body_quat_w[motion.steps[0]],
        tracked_body_quat[motion.reference_index],
    )
    ctx.metrics["ref_full_ori_err"][0] = error
    return error > threshold


@configclass(kw_only=True)
class BadRefFullOrientationTerminationCfg(_WbtTerminationCfg):
    def __call__(self, ctx) -> TerminationTerm:
        del ctx
        return TerminationTerm(
            bad_ref_full_orientation_termination,
            np.float32(self.threshold),
            metric_names=("ref_full_ori_err",),
        )


@dispatch
def bad_motion_body_position_termination(
    ctx: ManagerContext,
    body_indices: tuple[int, ...],
    threshold: np.float32,
) -> bool:
    """Holosoma-aligned end-effector tracking termination.

    Terminates when any configured end effector drifts further than
    ``threshold`` from its reference target (the yaw/height-anchored body
    target the relative rewards track), so sloppy limb motion dies early and
    the adaptive sampler re-samples those frames instead of polluting the
    buffer until timeout.
    """
    tracked_body_pos = ctx.sim["tracked_body_pos"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    count = len(body_indices)
    if count == 0:
        ctx.metrics["ee_body_pos_err"][0] = 0.0
        return False
    error_sq_sum = 0.0
    for index in range(count):
        body_id = body_indices[index]
        diff = motion.target_body_position_relative[body_id] - tracked_body_pos[body_id]
        error_sq_sum += float(np.dot(diff, diff))
    mean_err = math.sqrt(error_sq_sum / count)
    ctx.metrics["ee_body_pos_err"][0] = mean_err
    return mean_err > threshold


@configclass(kw_only=True)
class BadMotionBodyPositionTerminationCfg(_WbtTerminationCfg):
    body_names: tuple[str, ...] = ()

    def __call__(self, ctx) -> TerminationTerm:
        tracked_body_names = ctx.cfg.commands.motion.tracked_body_names
        body_indices = tuple(tracked_body_names.index(name) for name in self.body_names)
        return TerminationTerm(
            bad_motion_body_position_termination,
            body_indices,
            np.float32(self.threshold),
            metric_names=("ee_body_pos_err",),
        )


@dispatch
def bad_body_z_termination(
    ctx: ManagerContext,
    body_indices: tuple[int, ...],
    threshold: np.float32,
) -> bool:
    tracked_body_pos = ctx.sim["tracked_body_pos"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    error = 0.0
    for body_id in body_indices:
        error = max(
            error,
            abs(motion.target_body_position_relative[body_id, 2] - tracked_body_pos[body_id, 2]),
        )
    ctx.metrics["body_z_max_err"][0] = error
    return error > threshold


@configclass(kw_only=True)
class BadBodyZTerminationCfg(_WbtTerminationCfg):
    body_names: tuple[str, ...] = ()

    def __call__(self, ctx) -> TerminationTerm:
        tracked_body_names = ctx.cfg.commands.motion.tracked_body_names
        body_indices = tuple(tracked_body_names.index(name) for name in self.body_names)
        return TerminationTerm(
            bad_body_z_termination,
            body_indices,
            np.float32(self.threshold),
            metric_names=("body_z_max_err",),
        )


@dispatch
def bad_dof_position_termination(ctx: ManagerContext, threshold: np.float32) -> bool:
    dof_pos = ctx.sim["robot_dof_pos"]
    action: WbtJointPositionAction = ctx.actions["joint_position"]
    error = 0.0
    finite = True
    for joint_id in range(dof_pos.shape[0]):
        position = dof_pos[joint_id]
        if math.isfinite(position):
            violation = max(action.joint_lower[joint_id] - position, 0.0)
            violation += max(position - action.joint_upper[joint_id], 0.0)
            error = max(error, violation)
        else:
            finite = False
            error = math.inf
    ctx.metrics["dof_limit_violation_max"][0] = error
    return (not finite) or error > threshold


@configclass(kw_only=True)
class BadDofPositionTerminationCfg(_WbtTerminationCfg):
    def __call__(self, ctx) -> TerminationTerm:
        del ctx
        return TerminationTerm(
            bad_dof_position_termination,
            np.float32(self.threshold),
            metric_names=("dof_limit_violation_max",),
        )


@dispatch
def bad_dof_velocity_termination(ctx: ManagerContext, threshold: np.float32) -> bool:
    dof_vel = ctx.sim["robot_dof_vel"]
    error = 0.0
    finite = True
    for joint_id in range(dof_vel.shape[0]):
        velocity = dof_vel[joint_id]
        if math.isfinite(velocity):
            error = max(error, abs(velocity))
        else:
            finite = False
            error = math.inf
    ctx.metrics["dof_vel_abs_max"][0] = error
    return (not finite) or error > threshold


@configclass(kw_only=True)
class BadDofVelocityTerminationCfg(_WbtTerminationCfg):
    def __call__(self, ctx) -> TerminationTerm:
        del ctx
        return TerminationTerm(
            bad_dof_velocity_termination,
            np.float32(self.threshold),
            metric_names=("dof_vel_abs_max",),
        )
