# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Observation terms for the manager-based ball-balance task."""

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerContext,
    ManagerEnv,
    ObservationTermCfg,
    ObsTerm,
)
from motrix_env_core.manager.math.quaternion import rotate_inverse
from motrix_env_core.mdp.noise import add_uniform_noise
from motrix_env_core.mdp.observations import UniformNoiseCfg
from motrix_env_core.numba.manager.dispatch import dispatch


@dispatch
def projected_gravity_obs(ctx: ManagerContext, out: np.ndarray, noise_amplitude: np.float32) -> None:
    base_quat = ctx.sim["robot_base_quat"]
    rotate_inverse(base_quat, (0.0, 0.0, -1.0), out)
    add_uniform_noise(out, noise_amplitude, ctx.rand.state)


@configclass(kw_only=True)
class ProjectedGravityObsCfg(ObservationTermCfg):
    """Gravity direction expressed in the robot base frame."""

    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        del env
        return ObsTerm(3, projected_gravity_obs, np.float32(self.noise.amplitude))


@dispatch
def ball_relative_position_obs(ctx: ManagerContext, out: np.ndarray, noise_amplitude: np.float32) -> None:
    ball_pos = ctx.sim["ball_pos"]
    base_pos = ctx.sim["robot_base_pos"]
    base_quat = ctx.sim["robot_base_quat"]
    rotate_inverse(
        base_quat,
        (ball_pos[0] - base_pos[0], ball_pos[1] - base_pos[1], ball_pos[2] - base_pos[2]),
        out,
    )
    add_uniform_noise(out, noise_amplitude, ctx.rand.state)


@configclass(kw_only=True)
class BallRelativePositionObsCfg(ObservationTermCfg):
    """Ball position relative to the robot base, in the base frame."""

    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        del env
        return ObsTerm(3, ball_relative_position_obs, np.float32(self.noise.amplitude))


@dispatch
def ball_relative_velocity_obs(ctx: ManagerContext, out: np.ndarray, noise_amplitude: np.float32) -> None:
    ball_lin_vel = ctx.sim["ball_lin_vel"]
    base_quat = ctx.sim["robot_base_quat"]
    rotate_inverse(base_quat, (ball_lin_vel[0], ball_lin_vel[1], ball_lin_vel[2]), out)
    add_uniform_noise(out, noise_amplitude, ctx.rand.state)


@configclass(kw_only=True)
class BallRelativeVelocityObsCfg(ObservationTermCfg):
    """Ball linear velocity in the robot base frame."""

    noise: UniformNoiseCfg = UniformNoiseCfg()

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        del env
        return ObsTerm(3, ball_relative_velocity_obs, np.float32(self.noise.amplitude))


@dispatch
def ball_position_obs(ctx: ManagerContext, out: np.ndarray) -> None:
    out[:] = ctx.sim["ball_pos"]


@configclass(kw_only=True)
class BallPositionObsCfg(ObservationTermCfg):
    """Ball position in the world frame (value observations only)."""

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        del env
        return ObsTerm(3, ball_position_obs)


@dispatch
def ball_velocity_obs(ctx: ManagerContext, out: np.ndarray) -> None:
    out[:] = ctx.sim["ball_lin_vel"]


@configclass(kw_only=True)
class BallVelocityObsCfg(ObservationTermCfg):
    """Ball linear velocity in the world frame (value observations only)."""

    def __call__(self, env: ManagerEnv) -> ObsTerm:
        del env
        return ObsTerm(3, ball_velocity_obs)
