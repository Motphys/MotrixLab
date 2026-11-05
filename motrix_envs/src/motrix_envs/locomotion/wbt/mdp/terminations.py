# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""WBT termination terms and raw diagnostic metrics."""

import numpy as np
import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, TerminationTerm, TerminationTermCfg


@configclass(kw_only=True)
class _WbtTerminationCfg(TerminationTermCfg):
    threshold: float


def bad_ref_z_termination(ctx: ManagerContext, threshold: float) -> torch.Tensor:
    motion = ctx.commands["motion"]
    steps = motion.steps[:, 0]
    error = torch.abs(
        motion.clip.reference_body_pos_w[steps][..., 2] - ctx.sim["tracked_body_pos"][:, motion.reference_index][..., 2]
    )
    ctx.metrics["ref_z_abs_err"][:, 0] = error
    return error > threshold


@configclass(kw_only=True)
class BadRefZTerminationCfg(_WbtTerminationCfg):
    def __call__(self, ctx) -> TerminationTerm:
        return TerminationTerm(
            bad_ref_z_termination,
            np.float32(self.threshold),
            metric_names=("ref_z_abs_err",),
        )


def bad_ref_orientation_termination(ctx: ManagerContext, threshold: float) -> torch.Tensor:
    motion = ctx.commands["motion"]
    steps = motion.steps[:, 0]
    target = motion.clip.reference_body_quat_w[steps]
    actual = ctx.sim["tracked_body_quat"][:, motion.reference_index]
    error = torch.abs(2 * (target[..., 0] ** 2 + target[..., 1] ** 2 - actual[..., 0] ** 2 - actual[..., 1] ** 2))
    ctx.metrics["ref_ori_abs_err"][:, 0] = error
    return error > threshold


@configclass(kw_only=True)
class BadRefOrientationTerminationCfg(_WbtTerminationCfg):
    def __call__(self, ctx) -> TerminationTerm:
        return TerminationTerm(
            bad_ref_orientation_termination,
            np.float32(self.threshold),
            metric_names=("ref_ori_abs_err",),
        )


def bad_motion_body_position_termination(
    ctx: ManagerContext,
    body_indices: tuple[int, ...],
    threshold: float,
) -> torch.Tensor:
    """Holosoma-aligned tracked-body tracking termination.

    Terminates when the mean position error of the configured tracked bodies
    (any subset of ``tracked_body_names``, not just end effectors) against
    their reference targets (the yaw/height-anchored body targets the
    relative rewards track) exceeds ``threshold``.
    """
    motion = ctx.commands["motion"]
    if not body_indices:
        error = torch.zeros_like(motion.steps[:, 0], dtype=torch.float32)
        ctx.metrics["motion_body_pos_err"][:, 0] = error
        return torch.zeros_like(error, dtype=torch.bool)
    diff = motion.target_body_position_relative[:, body_indices, :] - ctx.sim["tracked_body_pos"][:, body_indices, :]
    error_sq_sum = torch.sum(diff * diff, dim=(-2, -1))
    mean_err = torch.sqrt(error_sq_sum / len(body_indices))
    ctx.metrics["motion_body_pos_err"][:, 0] = mean_err
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
            metric_names=("motion_body_pos_err",),
        )


def bad_body_z_termination(
    ctx: ManagerContext,
    body_indices: tuple[int, ...],
    threshold: float,
) -> torch.Tensor:
    motion = ctx.commands["motion"]
    target = motion.target_body_position_relative[:, body_indices, :]
    actual = ctx.sim["tracked_body_pos"][:, body_indices, :]
    if target.shape[-2] == 0:
        error = torch.zeros_like(torch.sum(target, dim=(-2, -1)))
    else:
        error = torch.max(torch.abs(target[..., 2] - actual[..., 2]), dim=-1).values
    ctx.metrics["body_z_max_err"][:, 0] = error
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


def bad_dof_position_termination(ctx: ManagerContext, threshold: float) -> torch.Tensor:
    action = ctx.actions["joint_position"]
    position = ctx.sim["robot_dof_pos"]
    zero = torch.zeros_like(position)
    violation = torch.maximum(action.joint_lower - position, zero) + torch.maximum(position - action.joint_upper, zero)
    finite = torch.all(torch.isfinite(position), dim=-1)
    error = torch.max(violation, dim=-1).values
    error = torch.where(finite, error, torch.full_like(error, float("inf")))
    ctx.metrics["dof_limit_violation_max"][:, 0] = error
    return error > threshold


@configclass(kw_only=True)
class BadDofPositionTerminationCfg(_WbtTerminationCfg):
    def __call__(self, ctx) -> TerminationTerm:
        return TerminationTerm(
            bad_dof_position_termination,
            np.float32(self.threshold),
            metric_names=("dof_limit_violation_max",),
        )
