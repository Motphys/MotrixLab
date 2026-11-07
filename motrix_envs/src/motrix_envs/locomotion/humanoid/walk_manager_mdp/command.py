# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Velocity-command and gait-phase command term."""

import math
from dataclasses import dataclass

import numpy as np
import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    CommandCfg,
    CommandTerm,
    ManagerContext,
    ManagerEnv,
    metric,
)
from motrix_env_core.manager.commands import ResetContext
from motrix_env_core.manager.dispatch import dispatch


def _phase(
    command: torch.Tensor,
    phase_out: torch.Tensor,
    sin_cos_out: torch.Tensor,
    phase_offset: torch.Tensor,
    steps: torch.Tensor,
    phase_step: torch.Tensor,
) -> None:
    """Refresh both lanes' phase clocks over whole batches, pinning standing commands to ``pi``.

    The clock arithmetic runs in float64 (the kernel's ``math.pi``
    promotion) and is stored back as float32; standing commands pin both
    lanes to ``pi`` per lane.
    """
    tau = 2.0 * math.pi
    steps = steps.to(torch.float64)[:, None]
    phase_step = phase_step.to(torch.float64)[:, None]
    raw = (steps * phase_step + phase_offset.to(torch.float64) + math.pi) % tau - math.pi
    speed = torch.sqrt(command[:, 0] * command[:, 0] + command[:, 1] * command[:, 1])
    stand = (speed.to(torch.float64) < 0.01) & (command[:, 2].abs().to(torch.float64) < 0.01)
    phase = raw.to(torch.float32)
    phase = torch.where(stand[:, None], torch.full_like(phase, math.pi), phase)
    sin = torch.sin(phase.to(torch.float64)).to(torch.float32)
    cos = torch.cos(phase.to(torch.float64)).to(torch.float32)
    sin_cos_out[:, 0] = sin[:, 0]
    sin_cos_out[:, 1] = sin[:, 1]
    sin_cos_out[:, 2] = cos[:, 0]
    sin_cos_out[:, 3] = cos[:, 1]
    phase_out.copy_(phase)


def _uniform_range(u: torch.Tensor, low, high) -> torch.Tensor:
    """Float32 ``low + (high - low) * 0.5 * (u + 1)`` matching the kernel convention."""
    return low + (high - low) * 0.5 * (u + 1.0)


def _resample_command(
    ctx: ManagerContext,
    commands: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    stand_prob,
    resample: torch.Tensor,
) -> None:
    """Resample velocity commands on ``resample`` lanes, preserving other lanes' RNG rows.

    The draw counts and per-lane order are fixed: one 3-draw uniform batch
    then one stand-probability draw; lanes that do not resample keep their
    pre-draw RNG state.
    """
    if not bool(resample.any()):
        return
    previous_state = ctx.rand.state.clone()
    u = ctx.rand.next_uniform(3)
    sampled = _uniform_range(u, low, high)
    stand_u = ctx.rand.next_uniform(1)[:, 0]
    stand = (stand_u.to(torch.float64) + 1.0) * 0.5 < float(stand_prob)
    sampled = torch.where(stand[:, None], torch.zeros_like(sampled), sampled)
    keep_state = ~resample[:, None]
    ctx.rand.state.copy_(torch.where(keep_state, previous_state, ctx.rand.state))
    commands.copy_(torch.where(resample[:, None], sampled, commands))


@dataclass(frozen=True, slots=True)
class WalkCommand(CommandTerm):
    """Per-environment velocity command and gait-phase clock.

    Mirrors the direct env's ``commands`` / ``phase`` episode state:
    commands resample every ``resample_steps`` transitions, the phase advances
    by ``phase_step`` per step from a per-env offset, and standing commands pin
    the phase to ``pi``.

    Attributes:
        vel_limit_low: Lower velocity-command bounds ``(vx, vy, wz)``.
        vel_limit_high: Upper velocity-command bounds ``(vx, vy, wz)``.
        stand_prob: Probability of resampling a zero (standing) command.
        resample_steps: Command resampling interval in control steps.
        gait_period_base: Nominal gait period in seconds.
        gait_period_width: Per-episode uniform gait-period jitter half-width
            in seconds; 0 freezes the period at ``gait_period_base``.
        curriculum_enabled: Whether the penalty-scale curriculum is active.
        penalty_scale: Current penalty-curriculum scale (host EMA in
            ``reset``; read by the penalty reward kernels).
        avg_ep_len: Curriculum EMA of the average episode length.
        level_down_threshold: Average episode length below which the penalty
            scale decreases.
        level_up_threshold: Average episode length above which the penalty
            scale increases.
        degree: Multiplicative curriculum step per update.
        min_scale: Curriculum scale lower clamp.
        max_scale: Curriculum scale upper clamp.
        phase_offset: Per-lane phase offsets, ``(2,)``; the right lane is
            offset by ``pi`` and both are re-randomized at reset.
        sin_cos: Published ``sin``/``cos`` of both lanes' phase,
            ``(sin_l, sin_r, cos_l, cos_r)``; consumed by the gait-phase
            observation.
        phase: Both lanes' phase clocks, ``(2,)`` in ``[-pi, pi]``.
        phase_step: Per-lane phase increment per control step,
            ``2*pi*ctrl_dt/gait_period``, resampled per episode when
            ``gait_period_width`` is positive.
        steps: Control steps since the last command resample (published as
            the ``command_steps`` metric).
    """

    vel_limit_low: torch.Tensor
    vel_limit_high: torch.Tensor
    stand_prob: float
    resample_steps: float
    gait_period_base: float
    gait_period_width: float
    curriculum_enabled: bool
    penalty_scale: torch.Tensor
    avg_ep_len: torch.Tensor
    level_down_threshold: float
    level_up_threshold: float
    degree: float
    min_scale: float
    max_scale: float

    # Per-environment state. ``phase`` and ``steps`` are published as metrics;
    # the kernel lowering hands each lane a writable row view.
    # Contract goal vector (inherited CommandTerm.command row view).
    # command: torch.Tensor
    phase_offset: torch.Tensor
    sin_cos: torch.Tensor
    phase: torch.Tensor
    phase_step: torch.Tensor
    steps: torch.Tensor = metric(name="command_steps", dtype=torch.float32)

    @staticmethod
    @dispatch
    def update(receiver, ctx: ManagerContext) -> None:
        _phase(
            receiver.command,
            receiver.phase,
            receiver.sin_cos,
            receiver.phase_offset,
            receiver.steps[:, 0],
            receiver.phase_step[:, 0],
        )

    @staticmethod
    @dispatch
    def advance(receiver, ctx: ManagerContext) -> None:
        steps = receiver.steps
        steps[:, 0] += 1.0
        resample = torch.remainder(steps[:, 0], float(receiver.resample_steps)) == 0.0
        _resample_command(
            ctx,
            receiver.command,
            receiver.vel_limit_low,
            receiver.vel_limit_high,
            receiver.stand_prob,
            resample,
        )

    @staticmethod
    @dispatch
    def reset_env(receiver, ctx: ManagerContext) -> None:
        receiver.steps[:, 0] = 0.0
        width = float(receiver.gait_period_width)
        if width > 0.0:
            u = ctx.rand.next_uniform(1)[:, 0]
            jitter = _uniform_range(u, -width, width)
            period = float(receiver.gait_period_base) + jitter.to(torch.float64)
            receiver.phase_step[:, 0] = (2.0 * math.pi * ctx.dt / period.clamp(min=0.1)).to(torch.float32)
        first = _uniform_range(ctx.rand.next_uniform(1)[:, 0], -math.pi, math.pi)
        offsets = torch.empty_like(receiver.phase_offset)
        offsets[:, 0] = first
        offsets[:, 1] = ((first.to(torch.float64) + 2.0 * math.pi) % (2.0 * math.pi) - math.pi).to(torch.float32)
        receiver.phase_offset.copy_(offsets)
        _resample_command(
            ctx,
            receiver.command,
            receiver.vel_limit_low,
            receiver.vel_limit_high,
            receiver.stand_prob,
            torch.ones_like(first, dtype=torch.bool),
        )
        _phase(
            receiver.command,
            receiver.phase,
            receiver.sin_cos,
            receiver.phase_offset,
            receiver.steps[:, 0],
            receiver.phase_step[:, 0],
        )

    def reset(self, ctx: ResetContext) -> None:
        """Update the penalty-scale curriculum from this round's episode ends.

        ``ctx.env_ids`` is exactly the set of done lanes (terminated or
        truncated), so the EMA sample matches the direct env's
        ``done = terminated | truncated``.
        """
        penalty_scale = self.penalty_scale.numpy()
        avg_ep_len = self.avg_ep_len.numpy()
        ctx.metrics["penalty_scale"] = float(penalty_scale[0])
        if not self.curriculum_enabled or ctx.env_ids.numel() == 0:
            return
        ep_len = self.steps[ctx.env_ids, 0].to(torch.float64) + 1.0
        avg_ep_len[0] = 0.99 * avg_ep_len[0] + 0.01 * float(ep_len.mean())
        if avg_ep_len[0] < self.level_down_threshold:
            penalty_scale[0] *= 1.0 - self.degree
        elif avg_ep_len[0] > self.level_up_threshold:
            penalty_scale[0] *= 1.0 + self.degree
        penalty_scale[0] = min(max(float(penalty_scale[0]), float(self.min_scale)), float(self.max_scale))


@configclass(kw_only=True)
class WalkCommandCfg(CommandCfg):
    """Velocity-command resampling parameters (mirrors the direct ``commands`` group)."""

    resampling_time: float = 10.0
    ctrl_dt: float = 0.02
    gait_period: float = 1.0
    # Per-episode uniform gait-period jitter: period ~ U(g-w, g+w). 0 disables.
    gait_period_randomization_width: float = 0.0
    stand_prob: float = 0.2
    curriculum_enabled: bool = True
    initial_scale: float = 0.5
    min_scale: float = 0.5
    max_scale: float = 1.0
    level_down_threshold: float = 150.0
    level_up_threshold: float = 750.0
    degree: float = 0.001
    vel_limit: list[list[float]] = (
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
    )

    def __call__(self, env: ManagerEnv) -> WalkCommand:
        num_envs = env.num_envs
        return WalkCommand(
            vel_limit_low=torch.tensor(self.vel_limit[0], dtype=torch.float32),
            vel_limit_high=torch.tensor(self.vel_limit[1], dtype=torch.float32),
            stand_prob=float(self.stand_prob),
            resample_steps=float(max(int(round(self.resampling_time / self.ctrl_dt)), 1)),
            phase_step=torch.from_numpy(
                np.full(
                    (num_envs, 1),
                    2.0 * math.pi * self.ctrl_dt / self.gait_period,
                    dtype=np.float32,
                )
            ),
            gait_period_base=float(self.gait_period),
            gait_period_width=float(self.gait_period_randomization_width),
            curriculum_enabled=self.curriculum_enabled,
            penalty_scale=torch.from_numpy(
                np.full(
                    (1,),
                    self.initial_scale if self.curriculum_enabled else 1.0,
                    dtype=np.float32,
                )
            ),
            avg_ep_len=torch.zeros(1, dtype=torch.float32),
            level_down_threshold=float(self.level_down_threshold),
            level_up_threshold=float(self.level_up_threshold),
            degree=float(self.degree),
            min_scale=float(self.min_scale),
            max_scale=float(self.max_scale),
            command=torch.zeros((num_envs, 3), dtype=torch.float32),
            phase_offset=torch.zeros((num_envs, 2), dtype=torch.float32),
            sin_cos=torch.zeros((num_envs, 4), dtype=torch.float32),
            phase=torch.zeros((num_envs, 2), dtype=torch.float32),
            steps=torch.zeros((num_envs, 1), dtype=torch.float32),
        )
