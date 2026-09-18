# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Velocity-command and gait-phase command term."""

import math

import numpy as np
from numba import njit

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    CommandCfg,
    CommandTerm,
    ManagerContext,
    ManagerEnv,
    SharedArray,
    kernel_data,
    metric,
)
from motrix_env_core.numba.manager.commands import ResetContext
from motrix_env_core.numba.manager.dispatch import dispatch


@njit(inline="always")
def _lane_phase(cmd, phase_out, sin_cos_out, phase_offset, steps, phase_dt) -> None:
    """Refresh both lanes' phase clocks, pinning standing commands to ``pi``."""
    tau = 2.0 * math.pi
    for i in range(2):
        phase_out[i] = (steps * phase_dt + phase_offset[i] + math.pi) % tau - math.pi
    speed = math.sqrt(cmd[0] * cmd[0] + cmd[1] * cmd[1])
    if speed < 0.01 and abs(cmd[2]) < 0.01:
        # Standing commands pin both lanes, matching the direct env's
        # ``phase[stand] = pi``: the gait clock must freeze for both feet.
        phase_out[:] = math.pi
    for i in range(2):
        sin_cos_out[i] = math.sin(phase_out[i])
        sin_cos_out[2 + i] = math.cos(phase_out[i])


@njit(inline="always")
def _lane_resample_phase_offset(ctx, phase_offset) -> None:
    first = ctx.rand.uniform_range(np.float32(-math.pi), np.float32(math.pi))
    phase_offset[0] = first
    phase_offset[1] = (first + 2.0 * math.pi) % (2.0 * math.pi) - math.pi


@njit(inline="always")
def _lane_resample_command(ctx, commands, low, high, stand_prob) -> None:
    rand = ctx.rand
    for index in range(3):
        commands[index] = rand.uniform_range(low[index], high[index])
    if (rand.next_uniform() + 1.0) * 0.5 < stand_prob:
        commands[:] = 0.0


@kernel_data
class WalkCommand(CommandTerm):
    """Per-environment velocity command and gait-phase clock.

    Mirrors the direct env's ``info["commands"]`` / ``info["phase"]`` state:
    commands resample every ``resample_steps`` transitions, the phase advances
    by ``phase_dt`` per step from a per-env offset, and standing commands pin
    the phase to ``pi``.
    """

    vel_limit_low: SharedArray
    vel_limit_high: SharedArray
    stand_prob: np.float32
    resample_steps: np.float32
    phase_dt: np.float32
    # Curriculum state (host EMA in reset(ctx); kernel/host read the scale).
    curriculum_enabled: bool
    penalty_scale: SharedArray
    avg_ep_len: SharedArray
    level_down_threshold: np.float32
    level_up_threshold: np.float32
    degree: np.float32
    min_scale: np.float32
    max_scale: np.float32

    # Per-environment state. ``phase`` and ``steps`` are published as metrics;
    # the kernel lowering hands each lane a writable row view.
    # Contract goal vector (inherited CommandTerm.command row view).
    # command: np.ndarray
    phase_offset: np.ndarray
    sin_cos: np.ndarray
    phase: np.ndarray
    steps: np.ndarray = metric(name="command_steps", dtype=np.float32)

    @dispatch
    def update(self, ctx: ManagerContext) -> None:
        _lane_phase(self.command, self.phase, self.sin_cos, self.phase_offset, self.steps[0], self.phase_dt)

    @dispatch
    def advance(self, ctx: ManagerContext) -> None:
        self.steps[0] += 1.0
        if self.steps[0] % self.resample_steps == 0.0:
            _lane_resample_command(ctx, self.command, self.vel_limit_low, self.vel_limit_high, self.stand_prob)

    @dispatch
    def reset_env(self, ctx: ManagerContext) -> None:
        self.steps[0] = 0.0
        _lane_resample_phase_offset(ctx, self.phase_offset)
        _lane_resample_command(ctx, self.command, self.vel_limit_low, self.vel_limit_high, self.stand_prob)
        _lane_phase(self.command, self.phase, self.sin_cos, self.phase_offset, self.steps[0], self.phase_dt)

    def reset(self, ctx: ResetContext) -> None:
        """Update the penalty-scale curriculum from this round's episode ends.

        ``ctx.env_ids`` is exactly the set of done lanes (terminated or
        truncated), so the EMA sample matches the direct env's
        ``done = terminated | truncated``.
        """
        ctx.metrics["penalty_scale"] = float(self.penalty_scale[0])
        if not self.curriculum_enabled or ctx.env_ids.size == 0:
            return
        ep_len = self.steps[ctx.env_ids, 0].astype(np.float64) + 1.0
        self.avg_ep_len[0] = np.float32(0.99 * self.avg_ep_len[0] + 0.01 * float(ep_len.mean()))
        if self.avg_ep_len[0] < self.level_down_threshold:
            self.penalty_scale[0] *= np.float32(1.0 - self.degree)
        elif self.avg_ep_len[0] > self.level_up_threshold:
            self.penalty_scale[0] *= np.float32(1.0 + self.degree)
        self.penalty_scale[0] = np.float32(
            min(max(float(self.penalty_scale[0]), float(self.min_scale)), float(self.max_scale))
        )


@configclass(kw_only=True)
class WalkCommandCfg(CommandCfg):
    """Velocity-command resampling parameters (mirrors the direct ``commands`` group)."""

    resampling_time: float = 10.0
    ctrl_dt: float = 0.02
    gait_period: float = 1.0
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
            vel_limit_low=np.asarray(self.vel_limit[0], dtype=np.float32),
            vel_limit_high=np.asarray(self.vel_limit[1], dtype=np.float32),
            stand_prob=np.float32(self.stand_prob),
            resample_steps=np.float32(max(int(round(self.resampling_time / self.ctrl_dt)), 1)),
            phase_dt=np.float32(2.0 * math.pi * self.ctrl_dt / self.gait_period),
            curriculum_enabled=self.curriculum_enabled,
            penalty_scale=np.full(
                (1,),
                self.initial_scale if self.curriculum_enabled else 1.0,
                dtype=np.float32,
            ),
            avg_ep_len=np.zeros((1,), dtype=np.float32),
            level_down_threshold=np.float32(self.level_down_threshold),
            level_up_threshold=np.float32(self.level_up_threshold),
            degree=np.float32(self.degree),
            min_scale=np.float32(self.min_scale),
            max_scale=np.float32(self.max_scale),
            command=np.zeros((num_envs, 3), dtype=np.float32),
            phase_offset=np.zeros((num_envs, 2), dtype=np.float32),
            sin_cos=np.zeros((num_envs, 4), dtype=np.float32),
            phase=np.zeros((num_envs, 2), dtype=np.float32),
            steps=np.zeros((num_envs, 1), dtype=np.float32),
        )
