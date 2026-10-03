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
def _lane_phase(
    cmd: np.ndarray,
    phase_out: np.ndarray,
    sin_cos_out: np.ndarray,
    phase_offset: np.ndarray,
    steps: float,
    phase_step: float,
) -> None:
    """Refresh both lanes' phase clocks, pinning standing commands to ``pi``."""
    tau = 2.0 * math.pi
    for i in range(2):
        phase_out[i] = (steps * phase_step + phase_offset[i] + math.pi) % tau - math.pi
    speed = math.sqrt(cmd[0] * cmd[0] + cmd[1] * cmd[1])
    if speed < 0.01 and abs(cmd[2]) < 0.01:
        # Standing commands pin both lanes, matching the direct env's
        # ``phase[stand] = pi``: the gait clock must freeze for both feet.
        phase_out[:] = math.pi
    for i in range(2):
        sin_cos_out[i] = math.sin(phase_out[i])
        sin_cos_out[2 + i] = math.cos(phase_out[i])


@njit(inline="always")
def _lane_resample_phase_offset(ctx: ManagerContext, phase_offset: np.ndarray) -> None:
    first = ctx.rand.uniform_range(np.float32(-math.pi), np.float32(math.pi))
    phase_offset[0] = first
    phase_offset[1] = (first + 2.0 * math.pi) % (2.0 * math.pi) - math.pi


@njit(inline="always")
def _lane_resample_command(
    ctx: ManagerContext, commands: np.ndarray, low: np.ndarray, high: np.ndarray, stand_prob: np.float32
) -> None:
    rand = ctx.rand
    for index in range(3):
        commands[index] = rand.uniform_range(low[index], high[index])
    if (rand.next_uniform() + 1.0) * 0.5 < stand_prob:
        commands[:] = 0.0


@kernel_data
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

    vel_limit_low: SharedArray
    vel_limit_high: SharedArray
    stand_prob: np.float32
    resample_steps: np.float32
    gait_period_base: np.float32
    gait_period_width: np.float32
    curriculum_enabled: bool
    penalty_scale: SharedArray
    avg_ep_len: SharedArray
    level_down_threshold: np.float32
    level_up_threshold: np.float32
    degree: np.float32
    min_scale: np.float32
    max_scale: np.float32
    # Initial-state randomization curriculum: shared mix fraction read by the
    # walk reset kernel; widened on the host from ``init_state_curriculum_start``
    # toward 1.0 over ``init_state_curriculum_steps`` control steps.
    init_state_mix: SharedArray
    init_state_step_count: SharedArray
    init_state_curriculum_steps: np.float32
    init_state_curriculum_start: np.float32
    # Terrain difficulty curriculum: every tile origin on the grid (shared,
    # row-major rows x cols) plus per-lane row/column bindings. The host
    # moves a lane's row up after a long episode and down after an early
    # fall; the reset kernel resolves the spawn origin from the binding.
    terrain_curriculum_enabled: bool
    terrain_origin_grid: SharedArray
    terrain_levels: np.ndarray
    terrain_cols: np.ndarray
    terrain_cols_count: np.int64
    terrain_rows_count: np.int64
    terrain_move_up_steps: np.float32

    # Per-environment state. ``phase`` and ``steps`` are published as metrics;
    # the kernel lowering hands each lane a writable row view.
    # Contract goal vector (inherited CommandTerm.command row view).
    # command: np.ndarray
    phase_offset: np.ndarray
    sin_cos: np.ndarray
    phase: np.ndarray
    phase_step: np.ndarray
    steps: np.ndarray = metric(name="command_steps", dtype=np.float32)
    episode_steps: np.ndarray = metric(name="episode_steps", dtype=np.float32)

    @dispatch
    def update(self, ctx: ManagerContext) -> None:
        _lane_phase(
            self.command,
            self.phase,
            self.sin_cos,
            self.phase_offset,
            self.steps[0],
            self.phase_step[0],
        )

    @dispatch
    def advance(self, ctx: ManagerContext) -> None:
        self.steps[0] += 1.0
        self.episode_steps[0] += 1.0
        if self.steps[0] % self.resample_steps == 0.0:
            _lane_resample_command(ctx, self.command, self.vel_limit_low, self.vel_limit_high, self.stand_prob)

    @dispatch
    def reset_env(self, ctx: ManagerContext) -> None:
        self.steps[0] = 0.0
        if self.gait_period_width > np.float32(0.0):
            period = float(self.gait_period_base) + ctx.rand.uniform_range(
                -self.gait_period_width, self.gait_period_width
            )
            self.phase_step[0] = np.float32(2.0 * math.pi * ctx.dt / max(period, 0.1))
        _lane_resample_phase_offset(ctx, self.phase_offset)
        _lane_resample_command(ctx, self.command, self.vel_limit_low, self.vel_limit_high, self.stand_prob)
        _lane_phase(
            self.command,
            self.phase,
            self.sin_cos,
            self.phase_offset,
            self.steps[0],
            self.phase_step[0],
        )

    def reset(self, ctx: ResetContext) -> None:
        """Update both curricula from this round's episode ends.

        ``ctx.env_ids`` is exactly the set of done lanes (terminated or
        truncated), so the EMA sample matches the direct env's
        ``done = terminated | truncated``. The terrain curriculum moves a
        lane one row up when its episode survived near the timeout and one
        row down on an early fall, then clears the lane's episode counter.
        """
        ctx.metrics["penalty_scale"] = float(self.penalty_scale[0])
        if ctx.env_ids.size == 0:
            return
        if self.terrain_curriculum_enabled:
            for env_id in ctx.env_ids:
                if self.episode_steps[env_id, 0] >= self.terrain_move_up_steps:
                    self.terrain_levels[env_id, 0] = min(
                        int(self.terrain_levels[env_id, 0]) + 1, int(self.terrain_rows_count) - 1
                    )
                else:
                    self.terrain_levels[env_id, 0] = max(int(self.terrain_levels[env_id, 0]) - 1, 0)
                self.episode_steps[env_id, 0] = 0.0
            ctx.metrics["terrain_level"] = float(np.mean(self.terrain_levels))
        if not self.curriculum_enabled:
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

    def on_transition(self) -> None:
        """Widen the initial-state randomization curriculum by one step."""
        if self.init_state_mix[0] >= np.float32(1.0):
            return
        self.init_state_step_count[0] += np.float32(1.0)
        progress = min(float(self.init_state_step_count[0]) / float(self.init_state_curriculum_steps), 1.0)
        self.init_state_mix[0] = np.float32(
            float(self.init_state_curriculum_start) + (1.0 - float(self.init_state_curriculum_start)) * progress
        )


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
    init_state_curriculum_steps: int = 0
    init_state_curriculum_start: float = 1.0
    # Terrain difficulty curriculum (requires the walk reset term's tile
    # spawn): lanes start at terrain_start_ratio of the tile rows and move
    # up after surviving terrain_move_up_ratio of the episode, down on falls.
    terrain_curriculum: bool = False
    terrain_start_ratio: float = 0.2
    terrain_move_up_ratio: float = 0.9
    vel_limit: list[list[float]] = (
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
    )

    def __call__(self, env: ManagerEnv) -> WalkCommand:
        num_envs = env.num_envs
        terrain = _build_terrain_curriculum(env, self) if self.terrain_curriculum else None
        return WalkCommand(
            vel_limit_low=np.asarray(self.vel_limit[0], dtype=np.float32),
            vel_limit_high=np.asarray(self.vel_limit[1], dtype=np.float32),
            stand_prob=np.float32(self.stand_prob),
            resample_steps=np.float32(max(int(round(self.resampling_time / self.ctrl_dt)), 1)),
            phase_step=np.full(
                (num_envs, 1),
                np.float32(2.0 * math.pi * self.ctrl_dt / self.gait_period),
                dtype=np.float32,
            ),
            gait_period_base=np.float32(self.gait_period),
            gait_period_width=np.float32(self.gait_period_randomization_width),
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
            init_state_mix=np.full(
                (1,),
                self.init_state_curriculum_start if self.init_state_curriculum_steps > 0 else 1.0,
                dtype=np.float32,
            ),
            init_state_step_count=np.zeros((1,), dtype=np.float32),
            init_state_curriculum_steps=np.float32(max(self.init_state_curriculum_steps, 1)),
            init_state_curriculum_start=np.float32(self.init_state_curriculum_start),
            terrain_curriculum_enabled=self.terrain_curriculum,
            terrain_origin_grid=(
                terrain["origin_grid"] if terrain else np.zeros((1, 3), dtype=np.float32)
            ).reshape(-1, 3),
            terrain_levels=(
                terrain["levels"] if terrain else np.zeros((num_envs, 1), dtype=np.int64)
            ),
            terrain_cols=terrain["cols"] if terrain else np.zeros((num_envs, 1), dtype=np.int64),
            terrain_cols_count=np.int64(terrain["cols_count"] if terrain else 1),
            terrain_rows_count=np.int64(terrain["rows_count"] if terrain else 1),
            terrain_move_up_steps=np.float32(
                terrain["move_up_steps"] if terrain else np.float32(np.finfo(np.float32).max)
            ),
            command=np.zeros((num_envs, 3), dtype=np.float32),
            phase_offset=np.zeros((num_envs, 2), dtype=np.float32),
            sin_cos=np.zeros((num_envs, 4), dtype=np.float32),
            phase=np.zeros((num_envs, 2), dtype=np.float32),
            steps=np.zeros((num_envs, 1), dtype=np.float32),
            episode_steps=np.zeros((num_envs, 1), dtype=np.float32),
        )


def _build_terrain_curriculum(env: ManagerEnv, cfg: WalkCommandCfg) -> dict:
    """Build the tile-origin grid and per-lane bindings for tile-spawn cfgs."""
    from motrix_env_core.config.scene import HFieldTerrainCfg  # noqa: TC001
    from motrix_envs.locomotion.humanoid.walk_manager_mdp.reset import _tile_origin_grid
    from motrix_envs.locomotion.humanoid.walk_manager_mdp.terrain import ground_height_grid

    reset_cfg = env.cfg.sim_reset.humanoid_state
    if not reset_cfg.tile_spawn or reset_cfg.spawn_tiles is None:
        raise ValueError("WalkCommandCfg.terrain_curriculum requires tile_spawn with spawn_tiles.")
    floor_obj = getattr(env.cfg.scene.objs, "floor", None)
    if not isinstance(floor_obj, HFieldTerrainCfg):
        raise ValueError("WalkCommandCfg.terrain_curriculum requires a height-field ground geom.")
    grid = ground_height_grid(env, env.cfg.ground_heightfield_geom)
    tiles = reset_cfg.spawn_tiles
    rows, cols = tiles
    origin_grid = _tile_origin_grid(grid, tiles, reset_cfg.spawn_border).reshape(rows * cols, 3)
    levels = np.full((env.num_envs, 1), int(cfg.terrain_start_ratio * rows), dtype=np.int64)
    cols_of_env = (np.arange(env.num_envs) * cols // env.num_envs).astype(np.int64).reshape(-1, 1)
    # ctrl_dt on this cfg is wired to the env's control period in the task cfg.
    episode_steps = 20.0 / max(cfg.ctrl_dt, 1e-6)
    return {
        "origin_grid": np.ascontiguousarray(origin_grid),
        "levels": levels,
        "cols": cols_of_env,
        "cols_count": cols,
        "rows_count": rows,
        "move_up_steps": np.float32(cfg.terrain_move_up_ratio * episode_steps),
    }
