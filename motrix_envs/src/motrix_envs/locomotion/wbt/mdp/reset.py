# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Simulator reset terms for whole-body tracking."""

from __future__ import annotations

import math

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerContext,
    ResetTerm,
    ResetTermCfg,
)
from motrix_env_core.numba.kernel_data import Map
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
from motrix_envs.locomotion.wbt.mdp.action import WbtJointPositionAction
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommand


@dispatch
def _reset_body_pos(
    ctx: ManagerContext,
    sim_writes: Map[np.ndarray],
    noise_scale: tuple[np.float32, np.float32, np.float32],
) -> None:
    position = sim_writes["position"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    position[0] = motion.clip.root_body_pos_w[motion.steps[0]]
    for index in range(3):
        position[0, index] += ctx.rand.next_uniform() * noise_scale[index]


@configclass(kw_only=True)
class BodyPosResetCfg(ResetTermCfg):
    """Reset floating-body world position from the WBT motion."""

    noise: tuple[float, float, float] = (0.05, 0.05, 0.01)
    noise_scale: float = 1.0

    def __call__(self, ctx) -> ResetTerm:
        body = ctx.cfg.scene.objs.robot.resolved_base_link_name
        amplitude = np.asarray(self.noise, dtype=np.float32) * np.float32(self.noise_scale)
        return ResetTerm(
            _reset_body_pos,
            tuple(amplitude),
            writes={"position": BodyPositionWrite((body,))},
        )


@dispatch
def _reset_body_rot(
    ctx: ManagerContext,
    sim_writes: Map[np.ndarray],
    noise_scale: tuple[np.float32, np.float32, np.float32],
) -> None:
    rotation = sim_writes["rotation"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    rotation[0] = motion.clip.root_body_quat_w[motion.steps[0]]
    noisy_quat = np.empty((4,), dtype=np.float32)
    numba_quaternion.from_euler(
        ctx.rand.next_uniform() * noise_scale[0],
        ctx.rand.next_uniform() * noise_scale[1],
        ctx.rand.next_uniform() * noise_scale[2],
        noisy_quat,
    )
    base_quat = np.empty((4,), dtype=np.float32)
    base_quat[:] = rotation[0]
    numba_quaternion.mul(noisy_quat, base_quat, rotation[0])
    norm = math.sqrt(float(np.dot(rotation[0], rotation[0])))
    rotation[0] /= norm


@configclass(kw_only=True)
class BodyRotResetCfg(ResetTermCfg):
    """Reset floating-body world rotation from the WBT motion."""

    noise: tuple[float, float, float] = (0.1, 0.1, 0.2)
    noise_scale: float = 1.0

    def __call__(self, ctx) -> ResetTerm:
        body = ctx.cfg.scene.objs.robot.resolved_base_link_name
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
    motion: WbtMotionCommand = ctx.commands["motion"]
    linear_velocity[0] = motion.clip.root_body_lin_vel_w[motion.steps[0]]
    for index in range(3):
        linear_velocity[0, index] += ctx.rand.next_uniform() * noise_scale[index]


@configclass(kw_only=True)
class BodyLinVelResetCfg(ResetTermCfg):
    """Reset floating-body world linear velocity from the WBT motion."""

    noise: tuple[float, float, float] = (0.5, 0.5, 0.2)
    noise_scale: float = 1.0

    def __call__(self, ctx) -> ResetTerm:
        body = ctx.cfg.scene.objs.robot.resolved_base_link_name
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
    motion: WbtMotionCommand = ctx.commands["motion"]
    angular_velocity[0] = motion.clip.root_body_ang_vel_w[motion.steps[0]]
    for index in range(3):
        angular_velocity[0, index] += ctx.rand.next_uniform() * noise_scale[index]


@configclass(kw_only=True)
class BodyRotVelResetCfg(ResetTermCfg):
    """Reset floating-body world angular velocity from the WBT motion."""

    noise: tuple[float, float, float] = (0.52, 0.52, 0.78)
    noise_scale: float = 1.0

    def __call__(self, ctx) -> ResetTerm:
        body = ctx.cfg.scene.objs.robot.resolved_base_link_name
        amplitude = np.asarray(self.noise, dtype=np.float32) * np.float32(self.noise_scale)
        return ResetTerm(
            _reset_body_rot_vel,
            tuple(amplitude),
            writes={"angular_velocity": BodyAngularVelocityWrite((body,))},
        )


@dispatch
def _reset_body_dof_pos(ctx: ManagerContext, sim_writes: Map[np.ndarray], noise_scale: np.float32) -> None:
    position = sim_writes["position"]
    velocity = sim_writes["velocity"]
    motion: WbtMotionCommand = ctx.commands["motion"]
    action: WbtJointPositionAction = ctx.actions["joint_position"]
    step = motion.steps[0]
    position[:] = motion.clip.joint_pos[step]
    velocity[:] = motion.clip.joint_vel[step]
    rng = ctx.rand
    for index in range(position.shape[0]):
        value = position[index] + rng.next_uniform() * noise_scale
        position[index] = min(max(value, action.joint_lower[index]), action.joint_upper[index])


@configclass(kw_only=True)
class BodyDofPosResetCfg(ResetTermCfg):
    """Reset articulated DOF position and velocity from the WBT motion."""

    noise: float = 0.1
    noise_scale: float = 1.0

    def __call__(self, ctx) -> ResetTerm:
        joints = ctx.cfg.commands.motion.joint_names
        return ResetTerm(
            _reset_body_dof_pos,
            np.float32(self.noise * self.noise_scale),
            writes={"position": JointPositionWrite(joints), "velocity": JointVelocityWrite(joints)},
        )
