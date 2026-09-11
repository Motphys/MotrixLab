# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Simulator reset terms for the ball-balance task."""

from __future__ import annotations

import math

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerContext,
    ManagerEnv,
    ResetTerm,
    ResetTermCfg,
)
from motrix_env_core.numba.kernel_data import Map, SharedArray, kernel_data
from motrix_env_core.numba.manager.dispatch import dispatch
from motrix_env_core.numba.math import quaternion as numba_quaternion
from motrix_env_core.sim import (
    BodyAngularVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    JointPositionWrite,
    JointVelocityWrite,
)


@dispatch
def _reset_body_pos(
    ctx: ManagerContext,
    sim_writes: Map[np.ndarray],
    spawn: tuple[np.float32, np.float32, np.float32],
    noise_scale: tuple[np.float32, np.float32, np.float32],
) -> None:
    position = sim_writes["position"]
    for index in range(3):
        position[0, index] = spawn[index] + ctx.rand.next_uniform() * noise_scale[index]


@configclass(kw_only=True)
class BodyPosResetCfg(ResetTermCfg):
    """Reset the floating base to a fixed spawn pose with uniform noise."""

    spawn: tuple[float, float, float]
    noise: tuple[float, float, float] = (0.02, 0.02, 0.005)
    noise_scale: float = 1.0

    def __call__(self, env: ManagerEnv) -> ResetTerm:
        body = env.cfg.scene.objs.robot.resolved_base_link_name
        return ResetTerm(
            _reset_body_pos,
            tuple(np.asarray(self.spawn, dtype=np.float32)),
            tuple(np.asarray(self.noise, dtype=np.float32) * np.float32(self.noise_scale)),
            writes={"position": BodyPositionWrite((body,))},
        )


@dispatch
def _reset_body_rot(
    ctx: ManagerContext,
    sim_writes: Map[np.ndarray],
    noise_scale: tuple[np.float32, np.float32, np.float32],
) -> None:
    rotation = sim_writes["rotation"]
    noisy_quat = np.empty(4, dtype=np.float32)
    numba_quaternion.from_euler(
        ctx.rand.next_uniform() * noise_scale[0],
        ctx.rand.next_uniform() * noise_scale[1],
        ctx.rand.next_uniform() * noise_scale[2],
        noisy_quat,
    )
    rotation[0] = noisy_quat
    norm = math.sqrt(
        rotation[0, 0] * rotation[0, 0]
        + rotation[0, 1] * rotation[0, 1]
        + rotation[0, 2] * rotation[0, 2]
        + rotation[0, 3] * rotation[0, 3]
    )
    rotation[0] /= norm


@configclass(kw_only=True)
class BodyRotResetCfg(ResetTermCfg):
    """Reset the floating base to upright with small Euler noise."""

    noise: tuple[float, float, float] = (0.05, 0.05, 0.05)
    noise_scale: float = 1.0

    def __call__(self, env: ManagerEnv) -> ResetTerm:
        body = env.cfg.scene.objs.robot.resolved_base_link_name
        amplitude = np.asarray(self.noise, dtype=np.float32) * np.float32(self.noise_scale)
        return ResetTerm(
            _reset_body_rot,
            tuple(amplitude),
            writes={"rotation": BodyRotationWrite((body,))},
        )


@dispatch
def _reset_body_lin_vel(
    ctx: ManagerContext,
    sim_writes: Map[np.ndarray],
    noise_scale: tuple[np.float32, np.float32, np.float32],
) -> None:
    linear_velocity = sim_writes["linear_velocity"]
    for index in range(3):
        linear_velocity[0, index] = ctx.rand.next_uniform() * noise_scale[index]


@configclass(kw_only=True)
class BodyLinVelResetCfg(ResetTermCfg):
    """Reset the floating base velocity to zero with uniform noise."""

    noise: tuple[float, float, float] = (0.1, 0.1, 0.05)
    noise_scale: float = 1.0

    def __call__(self, env: ManagerEnv) -> ResetTerm:
        body = env.cfg.scene.objs.robot.resolved_base_link_name
        amplitude = np.asarray(self.noise, dtype=np.float32) * np.float32(self.noise_scale)
        return ResetTerm(
            _reset_body_lin_vel,
            tuple(amplitude),
            writes={"linear_velocity": BodyLinearVelocityWrite((body,))},
        )


@dispatch
def _reset_body_rot_vel(
    ctx: ManagerContext,
    sim_writes: Map[np.ndarray],
    noise_scale: tuple[np.float32, np.float32, np.float32],
) -> None:
    angular_velocity = sim_writes["angular_velocity"]
    for index in range(3):
        angular_velocity[0, index] = ctx.rand.next_uniform() * noise_scale[index]


@configclass(kw_only=True)
class BodyRotVelResetCfg(ResetTermCfg):
    """Reset the floating base angular velocity to zero with uniform noise."""

    noise: tuple[float, float, float] = (0.2, 0.2, 0.2)
    noise_scale: float = 1.0

    def __call__(self, env: ManagerEnv) -> ResetTerm:
        body = env.cfg.scene.objs.robot.resolved_base_link_name
        amplitude = np.asarray(self.noise, dtype=np.float32) * np.float32(self.noise_scale)
        return ResetTerm(
            _reset_body_rot_vel,
            tuple(amplitude),
            writes={"angular_velocity": BodyAngularVelocityWrite((body,))},
        )


@kernel_data
class DofResetParams:
    default_pos: SharedArray
    joint_lower: SharedArray
    joint_upper: SharedArray
    noise_scale: np.float32


@dispatch
def _reset_body_dof_pos(ctx: ManagerContext, sim_writes: Map[np.ndarray], params: DofResetParams) -> None:
    position = sim_writes["position"]
    velocity = sim_writes["velocity"]
    velocity[:] = 0.0
    rng = ctx.rand
    for index in range(position.shape[0]):
        value = params.default_pos[index] + rng.next_uniform() * params.noise_scale
        position[index] = min(max(value, params.joint_lower[index]), params.joint_upper[index])


@configclass(kw_only=True)
class BodyDofPosResetCfg(ResetTermCfg):
    """Reset articulated DOFs to the default key pose and zero velocity."""

    noise: float = 0.05
    noise_scale: float = 1.0

    def __call__(self, env: ManagerEnv) -> ResetTerm:
        robot = env.cfg.scene.objs.robot
        joints = tuple(robot.resolve_name(name) for name in robot.key_pose.joint_names)
        if "default" not in robot.key_pose.poses:
            raise ValueError("ball-balance robot must define key pose 'default'")
        default_pos = np.asarray(robot.key_pose.poses["default"], dtype=np.float32)
        joint_lower, joint_upper = env.model.others["robot_joint_position_limits"]
        expected_joint_shape = (len(joints),)
        if (
            default_pos.shape != expected_joint_shape
            or joint_lower.shape != expected_joint_shape
            or joint_upper.shape != expected_joint_shape
        ):
            raise ValueError(
                "ball-balance robot joint state must match robot_dof_pos: "
                f"default={default_pos.shape}, lower={joint_lower.shape}, "
                f"upper={joint_upper.shape}, dof_pos={expected_joint_shape}."
            )
        return ResetTerm(
            _reset_body_dof_pos,
            DofResetParams(
                default_pos=default_pos,
                joint_lower=joint_lower,
                joint_upper=joint_upper,
                noise_scale=np.float32(self.noise * self.noise_scale),
            ),
            writes={"position": JointPositionWrite(joints), "velocity": JointVelocityWrite(joints)},
        )


@dispatch
def _reset_ball(
    ctx: ManagerContext,
    sim_writes: Map[np.ndarray],
    spawn: tuple[np.float32, np.float32, np.float32],
    noise_scale: tuple[np.float32, np.float32, np.float32],
    velocity_noise_scale: tuple[np.float32, np.float32, np.float32],
) -> None:
    position = sim_writes["position"]
    linear_velocity = sim_writes["linear_velocity"]
    angular_velocity = sim_writes["angular_velocity"]
    for index in range(3):
        position[0, index] = spawn[index] + ctx.rand.next_uniform() * noise_scale[index]
        linear_velocity[0, index] = ctx.rand.next_uniform() * velocity_noise_scale[index]
        angular_velocity[0, index] = 0.0


@configclass(kw_only=True)
class BallResetCfg(ResetTermCfg):
    """Reset the balance ball to rest at its spawn point with uniform noise."""

    ball_link_name: str
    spawn: tuple[float, float, float]
    noise: tuple[float, float, float] = (0.01, 0.01, 0.0)
    velocity_noise: tuple[float, float, float] = (0.05, 0.05, 0.0)
    noise_scale: float = 1.0

    def __call__(self, env: ManagerEnv) -> ResetTerm:
        return ResetTerm(
            _reset_ball,
            tuple(np.asarray(self.spawn, dtype=np.float32)),
            tuple(np.asarray(self.noise, dtype=np.float32) * np.float32(self.noise_scale)),
            tuple(np.asarray(self.velocity_noise, dtype=np.float32) * np.float32(self.noise_scale)),
            writes={
                "position": BodyPositionWrite((self.ball_link_name,)),
                "linear_velocity": BodyLinearVelocityWrite((self.ball_link_name,)),
                "angular_velocity": BodyAngularVelocityWrite((self.ball_link_name,)),
            },
        )
