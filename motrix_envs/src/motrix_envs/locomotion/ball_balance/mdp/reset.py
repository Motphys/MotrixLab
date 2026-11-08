# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Simulator reset terms for the ball-balance task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerContext,
    ResetTerm,
    ResetTermCfg,
)
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.math.quaternion import from_euler
from motrix_env_core.sim import (
    BodyAngularVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    JointPositionWrite,
    JointVelocityWrite,
)


def _reset_body_pos(
    ctx: ManagerContext, sim_writes: Map[torch.Tensor], spawn: torch.Tensor, noise_amplitude: torch.Tensor
) -> None:
    noise = ctx.rand.next_uniform(3)[ctx.reset_ids]
    sim_writes["position"][:, 0].copy_(spawn + noise * noise_amplitude)


@configclass(kw_only=True)
class BodyPosResetCfg(ResetTermCfg):
    """Reset the floating base to a fixed spawn pose with uniform noise."""

    spawn: tuple[float, float, float]
    noise: tuple[float, float, float] = (0.02, 0.02, 0.005)
    noise_scale: float = 1.0

    def __call__(self, ctx) -> ResetTerm:
        body = ctx.cfg.scene.objs.robot.resolved_base_link_name
        return ResetTerm(
            _reset_body_pos,
            torch.tensor(self.spawn, dtype=torch.float32),
            torch.tensor(self.noise, dtype=torch.float32) * self.noise_scale,
            writes={"position": BodyPositionWrite((body,))},
        )


def _reset_body_rot(ctx: ManagerContext, sim_writes: Map[torch.Tensor], noise_amplitude: torch.Tensor) -> None:
    angles = ctx.rand.next_uniform(3)[ctx.reset_ids] * noise_amplitude
    quat = from_euler(angles)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    sim_writes["rotation"][:, 0].copy_(quat)


@configclass(kw_only=True)
class BodyRotResetCfg(ResetTermCfg):
    """Reset the floating base to upright with small Euler noise."""

    noise: tuple[float, float, float] = (0.05, 0.05, 0.05)
    noise_scale: float = 1.0

    def __call__(self, ctx) -> ResetTerm:
        body = ctx.cfg.scene.objs.robot.resolved_base_link_name
        amplitude = torch.tensor(self.noise, dtype=torch.float32) * self.noise_scale
        return ResetTerm(
            _reset_body_rot,
            amplitude,
            writes={"rotation": BodyRotationWrite((body,))},
        )


def _reset_body_lin_vel(ctx: ManagerContext, sim_writes: Map[torch.Tensor], noise_amplitude: torch.Tensor) -> None:
    noise = ctx.rand.next_uniform(3)[ctx.reset_ids]
    sim_writes["linear_velocity"][:, 0].copy_(noise * noise_amplitude)


@configclass(kw_only=True)
class BodyLinVelResetCfg(ResetTermCfg):
    """Reset the floating base velocity to zero with uniform noise."""

    noise: tuple[float, float, float] = (0.1, 0.1, 0.05)
    noise_scale: float = 1.0

    def __call__(self, ctx) -> ResetTerm:
        body = ctx.cfg.scene.objs.robot.resolved_base_link_name
        amplitude = torch.tensor(self.noise, dtype=torch.float32) * self.noise_scale
        return ResetTerm(
            _reset_body_lin_vel,
            amplitude,
            writes={"linear_velocity": BodyLinearVelocityWrite((body,))},
        )


def _reset_body_rot_vel(ctx: ManagerContext, sim_writes: Map[torch.Tensor], noise_amplitude: torch.Tensor) -> None:
    noise = ctx.rand.next_uniform(3)[ctx.reset_ids]
    sim_writes["angular_velocity"][:, 0].copy_(noise * noise_amplitude)


@configclass(kw_only=True)
class BodyRotVelResetCfg(ResetTermCfg):
    """Reset the floating base angular velocity to zero with uniform noise."""

    noise: tuple[float, float, float] = (0.2, 0.2, 0.2)
    noise_scale: float = 1.0

    def __call__(self, ctx) -> ResetTerm:
        body = ctx.cfg.scene.objs.robot.resolved_base_link_name
        amplitude = torch.tensor(self.noise, dtype=torch.float32) * self.noise_scale
        return ResetTerm(
            _reset_body_rot_vel,
            amplitude,
            writes={"angular_velocity": BodyAngularVelocityWrite((body,))},
        )


@dataclass(frozen=True, slots=True)
class DofResetParams:
    default_pos: torch.Tensor
    joint_lower: torch.Tensor
    joint_upper: torch.Tensor
    noise_scale: float


def _reset_body_dof_pos(ctx: ManagerContext, sim_writes: Map[torch.Tensor], params: DofResetParams) -> None:
    noise = ctx.rand.next_uniform(params.default_pos.shape[0])[ctx.reset_ids]
    value = params.default_pos + noise * params.noise_scale
    sim_writes["position"].copy_(value.clamp(min=params.joint_lower, max=params.joint_upper))
    sim_writes["velocity"].zero_()


@configclass(kw_only=True)
class BodyDofPosResetCfg(ResetTermCfg):
    """Reset articulated DOFs to the default key pose and zero velocity."""

    noise: float = 0.05
    noise_scale: float = 1.0

    def __call__(self, ctx) -> ResetTerm:
        robot = ctx.cfg.scene.objs.robot
        joints = tuple(robot.resolve_name(name) for name in robot.key_pose.joint_names)
        if "default" not in robot.key_pose.poses:
            raise ValueError("ball-balance robot must define key pose 'default'")
        default_pos = np.asarray(robot.key_pose.poses["default"], dtype=np.float32)
        joint_lower, joint_upper = ctx.model.others["robot_joint_position_limits"]
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
                default_pos=torch.as_tensor(default_pos),
                joint_lower=torch.as_tensor(joint_lower),
                joint_upper=torch.as_tensor(joint_upper),
                noise_scale=float(self.noise * self.noise_scale),
            ),
            writes={"position": JointPositionWrite(joints), "velocity": JointVelocityWrite(joints)},
        )


def _reset_ball(
    ctx: ManagerContext,
    sim_writes: Map[torch.Tensor],
    spawn: torch.Tensor,
    noise_amplitude: torch.Tensor,
    velocity_noise_amplitude: torch.Tensor,
) -> None:
    # Position and velocity draws interleave per axis, so one 6-wide draw is
    # split into even (position) and odd (velocity) slots.
    samples = ctx.rand.next_uniform(6)[ctx.reset_ids]
    sim_writes["position"][:, 0].copy_(spawn + samples[:, 0::2] * noise_amplitude)
    sim_writes["linear_velocity"][:, 0].copy_(samples[:, 1::2] * velocity_noise_amplitude)
    sim_writes["angular_velocity"].zero_()


@configclass(kw_only=True)
class BallResetCfg(ResetTermCfg):
    """Reset the balance ball to rest at its spawn point with uniform noise."""

    ball_link_name: str
    spawn: tuple[float, float, float]
    noise: tuple[float, float, float] = (0.01, 0.01, 0.0)
    velocity_noise: tuple[float, float, float] = (0.05, 0.05, 0.0)
    noise_scale: float = 1.0

    def __call__(self, ctx) -> ResetTerm:
        return ResetTerm(
            _reset_ball,
            torch.tensor(self.spawn, dtype=torch.float32),
            torch.tensor(self.noise, dtype=torch.float32) * self.noise_scale,
            torch.tensor(self.velocity_noise, dtype=torch.float32) * self.noise_scale,
            writes={
                "position": BodyPositionWrite((self.ball_link_name,)),
                "linear_velocity": BodyLinearVelocityWrite((self.ball_link_name,)),
                "angular_velocity": BodyAngularVelocityWrite((self.ball_link_name,)),
            },
        )
