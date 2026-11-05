# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Simulator reset terms for whole-body tracking."""

from __future__ import annotations

import numpy as np
import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerContext,
    ResetTerm,
    ResetTermCfg,
)
from motrix_env_core.manager.dispatch import dispatch
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.math.quaternion_array import mul
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
    sim_writes: Map[torch.Tensor],
    noise_amplitude: torch.Tensor,
) -> None:
    motion = ctx.commands["motion"]
    position = sim_writes["position"]
    position[:, 0, :] = motion.clip.root_body_pos_w[motion.steps[:, 0]]
    # next_uniform() returns [-1, 1), so this is centered uniform noise in
    # [-scale, +scale) per axis.
    position[:, 0, :] += ctx.rand.next_uniform(3) * noise_amplitude


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
            torch.from_numpy(amplitude),
            writes={"position": BodyPositionWrite((body,))},
        )


@dispatch
def _reset_body_rot(
    ctx: ManagerContext,
    sim_writes: Map[torch.Tensor],
    noise_amplitude: torch.Tensor,
) -> None:
    motion = ctx.commands["motion"]
    rotation = sim_writes["rotation"]
    base_quat = motion.clip.root_body_quat_w[motion.steps[:, 0]]
    samples = ctx.rand.next_uniform(3) * noise_amplitude
    # Euler roll/pitch/yaw noise left-multiplied onto the frame quaternion,
    # renormalized. Trig runs in float64 and rounds to float32 on store; the
    # squared-norm reduction accumulates sequentially in float32 and the
    # square root goes through float64 before rounding back — torch's direct
    # float32 sqrt is 1 ulp off IEEE on CPU for values just below 1. The
    # final divide stays float32.
    roll, pitch, yaw = (samples.double() * 0.5).unbind(dim=-1)
    cr, cp, cy = torch.cos(roll), torch.cos(pitch), torch.cos(yaw)
    sr, sp, sy = torch.sin(roll), torch.sin(pitch), torch.sin(yaw)
    noise_quat = torch.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dim=-1,
    ).to(torch.float32)
    noisy = mul(noise_quat, base_quat)
    squared = noisy * noisy
    norm_squared = squared[:, 0:1]
    for index in range(1, squared.shape[1]):
        norm_squared = norm_squared + squared[:, index : index + 1]
    norm = torch.sqrt(norm_squared.double()).to(torch.float32)
    rotation[:, 0, :] = noisy / norm


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
            torch.from_numpy(amplitude),
            writes={"rotation": BodyRotationWrite((body,))},
        )


@dispatch
def _reset_body_lin_vel(
    ctx: ManagerContext,
    sim_writes: Map[torch.Tensor],
    noise_amplitude: torch.Tensor,
) -> None:
    motion = ctx.commands["motion"]
    linear_velocity = sim_writes["linear_velocity"]
    linear_velocity[:, 0, :] = motion.clip.root_body_lin_vel_w[motion.steps[:, 0]]
    linear_velocity[:, 0, :] += ctx.rand.next_uniform(3) * noise_amplitude


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
            torch.from_numpy(amplitude),
            writes={"linear_velocity": BodyLinearVelocityWrite((body,))},
        )


@dispatch
def _reset_body_rot_vel(
    ctx: ManagerContext,
    sim_writes: Map[torch.Tensor],
    noise_amplitude: torch.Tensor,
) -> None:
    motion = ctx.commands["motion"]
    angular_velocity = sim_writes["angular_velocity"]
    angular_velocity[:, 0, :] = motion.clip.root_body_ang_vel_w[motion.steps[:, 0]]
    angular_velocity[:, 0, :] += ctx.rand.next_uniform(3) * noise_amplitude


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
            torch.from_numpy(amplitude),
            writes={"angular_velocity": BodyAngularVelocityWrite((body,))},
        )


@dispatch
def _reset_body_dof_pos(ctx: ManagerContext, sim_writes: Map[torch.Tensor], noise_scale: float) -> None:
    motion = ctx.commands["motion"]
    steps = motion.steps[:, 0]
    position = sim_writes["position"]
    velocity = sim_writes["velocity"]
    position[:] = motion.clip.joint_pos[steps]
    velocity[:] = motion.clip.joint_vel[steps]
    action = ctx.actions["joint_position"]
    position += ctx.rand.next_uniform(position.shape[1]) * noise_scale
    # clamp has no out= form Dynamo accepts on the arena-backed
    # (non-contiguous) write view; copy_ back instead.
    position.copy_(torch.clamp(position, min=action.joint_lower, max=action.joint_upper))


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
