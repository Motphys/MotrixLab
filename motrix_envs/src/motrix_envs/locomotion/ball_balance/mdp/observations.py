# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Observation terms for the manager-based ball-balance task."""

import numpy as np
import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerContext,
    ObservationTermCfg,
    ObsTerm,
)
from motrix_env_core.math.quaternion_array import rotate_inverse as rotate_inverse_batch
from motrix_env_core.mdp.observations import UniformNoiseCfg


def ball_relative_position_obs(ctx: ManagerContext, out: torch.Tensor, noise_amplitude: float) -> None:
    relative = ctx.sim["ball_pos"] - ctx.sim["robot_base_pos"]
    out.copy_(rotate_inverse_batch(ctx.sim["robot_base_quat"], relative))
    if noise_amplitude != 0.0:
        out += ctx.rand.next_uniform(out.shape[1]) * noise_amplitude


@configclass(kw_only=True)
class BallRelativePositionObsCfg(ObservationTermCfg):
    """Ball position relative to the robot base, in the base frame."""

    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(3, ball_relative_position_obs, np.float32(self.noise.amplitude))


def ball_relative_velocity_obs(ctx: ManagerContext, out: torch.Tensor, noise_amplitude: float) -> None:
    out.copy_(rotate_inverse_batch(ctx.sim["robot_base_quat"], ctx.sim["ball_lin_vel"]))
    if noise_amplitude != 0.0:
        out += ctx.rand.next_uniform(out.shape[1]) * noise_amplitude


@configclass(kw_only=True)
class BallRelativeVelocityObsCfg(ObservationTermCfg):
    """Ball linear velocity in the robot base frame."""

    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(3, ball_relative_velocity_obs, np.float32(self.noise.amplitude))


def ball_position_obs(ctx: ManagerContext, out: torch.Tensor) -> None:
    out.copy_(ctx.sim["ball_pos"])


@configclass(kw_only=True)
class BallPositionObsCfg(ObservationTermCfg):
    """Ball position in the world frame (value observations only)."""

    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(3, ball_position_obs)


def ball_velocity_obs(ctx: ManagerContext, out: torch.Tensor) -> None:
    out.copy_(ctx.sim["ball_lin_vel"])


@configclass(kw_only=True)
class BallVelocityObsCfg(ObservationTermCfg):
    """Ball linear velocity in the world frame (value observations only)."""

    def __call__(self, ctx) -> ObsTerm:
        return ObsTerm(3, ball_velocity_obs)
