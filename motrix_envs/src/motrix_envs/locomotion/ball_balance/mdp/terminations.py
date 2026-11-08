# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Termination terms for the manager-based ball-balance task."""

import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, TerminationTerm, TerminationTermCfg
from motrix_env_core.math.quaternion import rotate_inverse as rotate_inverse_batch


@configclass(kw_only=True)
class _BallBalanceTerminationCfg(TerminationTermCfg):
    threshold: float


def bad_base_z_termination(ctx: ManagerContext, threshold: float) -> torch.Tensor:
    error = ctx.sim["robot_base_pos"][:, 2]
    ctx.metrics["base_z"][:, 0] = error
    return error < threshold


@configclass(kw_only=True)
class BadBaseZTerminationCfg(_BallBalanceTerminationCfg):
    """Terminate when the base falls below its balanced on-ball height."""

    def __call__(self, ctx) -> TerminationTerm:
        return TerminationTerm(
            bad_base_z_termination,
            float(self.threshold),
            metric_names=("base_z",),
        )


def bad_orientation_termination(ctx: ManagerContext, threshold: float) -> torch.Tensor:
    base_quat = ctx.sim["robot_base_quat"]
    gravity = base_quat.new_zeros(base_quat.shape[0], 3)
    gravity[:, 2] = -1.0
    gravity = rotate_inverse_batch(base_quat, gravity)
    tilt_sq = gravity[:, 0] * gravity[:, 0] + gravity[:, 1] * gravity[:, 1]
    ctx.metrics["base_tilt"][:, 0] = tilt_sq
    return tilt_sq > threshold * threshold


@configclass(kw_only=True)
class BadOrientationTerminationCfg(_BallBalanceTerminationCfg):
    """Terminate when the base tilts farther than ``threshold`` from upright."""

    def __call__(self, ctx) -> TerminationTerm:
        return TerminationTerm(
            bad_orientation_termination,
            float(self.threshold),
            metric_names=("base_tilt",),
        )


def ball_escaped_termination(ctx: ManagerContext, threshold: float) -> torch.Tensor:
    feet_pos = ctx.sim["feet_pos"]
    ball_pos = ctx.sim["ball_pos"]
    mid_x = 0.5 * (feet_pos[:, 0, 0] + feet_pos[:, 1, 0])
    mid_y = 0.5 * (feet_pos[:, 0, 1] + feet_pos[:, 1, 1])
    error_x = ball_pos[:, 0] - mid_x
    error_y = ball_pos[:, 1] - mid_y
    distance_sq = error_x * error_x + error_y * error_y
    ctx.metrics["ball_feet_distance"][:, 0] = distance_sq
    return distance_sq > threshold * threshold


@configclass(kw_only=True)
class BallEscapedTerminationCfg(_BallBalanceTerminationCfg):
    """Terminate when the ball rolls out from under the feet."""

    def __call__(self, ctx) -> TerminationTerm:
        return TerminationTerm(
            ball_escaped_termination,
            float(self.threshold),
            metric_names=("ball_feet_distance",),
        )
