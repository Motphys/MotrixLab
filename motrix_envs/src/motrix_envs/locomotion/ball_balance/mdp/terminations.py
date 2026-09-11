# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Termination terms for the manager-based ball-balance task."""

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import ManagerContext, ManagerEnv, TerminationTerm, TerminationTermCfg
from motrix_env_core.manager.math.quaternion import rotate_inverse
from motrix_env_core.numba.manager.dispatch import dispatch


@configclass(kw_only=True)
class _BallBalanceTerminationCfg(TerminationTermCfg):
    threshold: float


@dispatch
def bad_base_z_termination(ctx: ManagerContext, threshold: np.float32) -> bool:
    base_pos = ctx.sim["robot_base_pos"]
    error = base_pos[2]
    ctx.metrics["base_z"][0] = error
    return error < threshold


@configclass(kw_only=True)
class BadBaseZTerminationCfg(_BallBalanceTerminationCfg):
    """Terminate when the base falls below its balanced on-ball height."""

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        return TerminationTerm(
            bad_base_z_termination,
            np.float32(self.threshold),
            metric_names=("base_z",),
        )


@dispatch
def bad_orientation_termination(ctx: ManagerContext, threshold: np.float32) -> bool:
    base_quat = ctx.sim["robot_base_quat"]
    gravity = np.empty(3, dtype=np.float32)
    rotate_inverse(base_quat, (0.0, 0.0, -1.0), gravity)
    tilt_sq = gravity[0] * gravity[0] + gravity[1] * gravity[1]
    ctx.metrics["base_tilt"][0] = tilt_sq
    return tilt_sq > threshold * threshold


@configclass(kw_only=True)
class BadOrientationTerminationCfg(_BallBalanceTerminationCfg):
    """Terminate when the base tilts farther than ``threshold`` from upright."""

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        return TerminationTerm(
            bad_orientation_termination,
            np.float32(self.threshold),
            metric_names=("base_tilt",),
        )


@dispatch
def ball_escaped_termination(ctx: ManagerContext, threshold: np.float32) -> bool:
    feet_pos = ctx.sim["feet_pos"]
    ball_pos = ctx.sim["ball_pos"]
    mid_x = 0.5 * (feet_pos[0, 0] + feet_pos[1, 0])
    mid_y = 0.5 * (feet_pos[0, 1] + feet_pos[1, 1])
    error_x = ball_pos[0] - mid_x
    error_y = ball_pos[1] - mid_y
    distance_sq = error_x * error_x + error_y * error_y
    ctx.metrics["ball_feet_distance"][0] = distance_sq
    return distance_sq > threshold * threshold


@configclass(kw_only=True)
class BallEscapedTerminationCfg(_BallBalanceTerminationCfg):
    """Terminate when the ball rolls out from under the feet."""

    def __call__(self, env: ManagerEnv) -> TerminationTerm:
        del env
        return TerminationTerm(
            ball_escaped_termination,
            np.float32(self.threshold),
            metric_names=("ball_feet_distance",),
        )
