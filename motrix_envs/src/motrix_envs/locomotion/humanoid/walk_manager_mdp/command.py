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
from motrix_env_core.numba.math.quaternion import rotate_vector
from motrix_envs.locomotion.humanoid.walk_manager_mdp.terrain_sampling import TerrainSamplingState


@njit(inline="always")
def _terrain_step_progress(
    command: np.ndarray,
    previous_xy: np.ndarray,
    previous_quat: np.ndarray,
    position: np.ndarray,
    dt: float,
    min_speed: float,
) -> tuple[float, float]:
    """Commanded distance and signed, capped progress along this step's command.

    Commands are body-frame velocities; displacement is world-frame. Using the
    orientation at the start of the transition also handles turning trajectories.
    Capping each step prevents overspeed from paying for subsequent stalled steps.
    """
    speed = math.sqrt(command[0] ** 2 + command[1] ** 2)
    if speed <= min_speed:
        return 0.0, 0.0
    direction = rotate_vector(previous_quat, (command[0], command[1], 0.0))
    planar_speed = math.sqrt(direction[0] ** 2 + direction[1] ** 2)
    if planar_speed <= 1e-8:
        return 0.0, 0.0
    expected = planar_speed * dt
    progress = (
        (position[0] - previous_xy[0]) * direction[0] + (position[1] - previous_xy[1]) * direction[1]
    ) / planar_speed
    return expected, min(max(progress, -expected), expected)


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
def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@njit(inline="always")
def _lane_resample_command(
    ctx: ManagerContext,
    commands: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    stand_prob: np.float32,
    heading_prob: np.float32,
    heading_target: np.ndarray,
    is_heading: np.ndarray,
    heading_low: float,
    heading_high: float,
) -> None:
    rand = ctx.rand
    for index in range(3):
        commands[index] = rand.uniform_range(low[index], high[index])
    draw = (rand.next_uniform() + 1.0) * 0.5
    is_heading[0] = draw >= stand_prob and draw < stand_prob + heading_prob
    if draw < stand_prob:
        commands[:] = 0.0
        heading_target[0] = 0.0
    elif is_heading[0]:
        heading_target[0] = rand.uniform_range(heading_low, heading_high)
    else:
        heading_target[0] = 0.0


@njit(inline="always")
def _apply_heading_command(
    commands: np.ndarray,
    heading_target: float,
    is_heading: bool,
    base_quat: np.ndarray,
    stiffness: float,
    yaw_low: float,
    yaw_high: float,
) -> None:
    if not is_heading:
        return
    x, y, z, w = base_quat
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    error = _wrap_to_pi(heading_target - yaw)
    commands[2] = min(max(stiffness * error, yaw_low), yaw_high)


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
    heading_prob: np.float32
    heading_control_stiffness: np.float32
    resample_steps: np.float32
    resample_steps_high: np.float32
    resample_steps_left: np.ndarray
    heading_target: np.ndarray
    is_heading: np.ndarray
    heading_command: bool
    heading_low: np.float32
    heading_high: np.float32
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
    # promotes half-tile traversal and demotes poor command-aligned progress;
    # the reset kernel resolves the spawn origin from the binding.
    terrain_curriculum_enabled: bool
    terrain_origin_grid: SharedArray
    terrain_levels: np.ndarray
    terrain_cols: np.ndarray
    terrain_cols_count: np.int64
    terrain_rows_count: np.int64
    terrain_max_level: np.int64
    terrain_tile_length: np.float32
    terrain_move_down_ratio: np.float32
    terrain_progress_promotion: bool
    terrain_require_survival_for_promotion: bool
    terrain_move_up_ratio: np.float32
    terrain_min_exploration_ratio: np.float32
    terrain_max_origin_radius: np.ndarray
    terrain_failure_demote_ratio: np.float32
    terrain_min_level: np.int64
    terrain_min_command_speed: np.float32
    terrain_commanded_distance: np.ndarray
    terrain_progress: np.ndarray
    terrain_terminal_xy: np.ndarray
    terrain_previous_xy: np.ndarray
    terrain_previous_quat: np.ndarray
    terrain_spawn_xy: np.ndarray
    terrain_balanced_columns: bool
    terrain_column_row_counter: SharedArray
    terrain_sampling_enabled: bool
    terrain_sampling_fraction: np.float32
    terrain_sampling_max_multiplier: np.float32
    terrain_sampling_ema_alpha: np.float32
    terrain_sampling_min_count: np.int64
    terrain_sampling_seed: np.int64
    terrain_sampling_failure_ema: SharedArray
    terrain_sampling_evidence_counts: SharedArray
    terrain_sampling_counter: SharedArray
    terrain_sampling_level_pool: SharedArray

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
        if self.heading_command:
            _apply_heading_command(
                self.command,
                self.heading_target[0],
                self.is_heading[0] != 0,
                ctx.sim["terrain_curriculum_base_quat"],
                float(self.heading_control_stiffness),
                float(self.vel_limit_low[2]),
                float(self.vel_limit_high[2]),
            )
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
        if self.terrain_curriculum_enabled:
            position = ctx.sim["terrain_curriculum_base_pos"]
            expected, progress = _terrain_step_progress(
                self.command,
                self.terrain_previous_xy,
                self.terrain_previous_quat,
                position,
                float(ctx.dt),
                float(self.terrain_min_command_speed),
            )
            self.terrain_commanded_distance[0] += np.float32(expected)
            self.terrain_progress[0] += np.float32(progress)
            self.terrain_terminal_xy[:] = position[:2]
            self.terrain_previous_xy[:] = position[:2]
            self.terrain_previous_quat[:] = ctx.sim["terrain_curriculum_base_quat"]
            origin_index = self.terrain_levels[0] * self.terrain_cols_count + self.terrain_cols[0]
            origin = self.terrain_origin_grid[origin_index]
            radius = math.sqrt((position[0] - origin[0]) ** 2 + (position[1] - origin[1]) ** 2)
            self.terrain_max_origin_radius[0] = max(self.terrain_max_origin_radius[0], radius)
        self.resample_steps_left[0] -= 1
        if self.resample_steps_left[0] <= 0:
            interval = self.resample_steps
            if self.resample_steps_high > self.resample_steps:
                interval = ctx.rand.uniform_range(self.resample_steps, self.resample_steps_high)
            self.resample_steps_left[0] = max(int(round(interval)), 1)
            _lane_resample_command(
                ctx,
                self.command,
                self.vel_limit_low,
                self.vel_limit_high,
                self.stand_prob,
                self.heading_prob if self.heading_command else np.float32(0.0),
                self.heading_target,
                self.is_heading,
                np.float32(self.heading_low),
                np.float32(self.heading_high),
            )
            if self.heading_command:
                _apply_heading_command(
                    self.command,
                    self.heading_target[0],
                    self.is_heading[0] != 0,
                    ctx.sim["terrain_curriculum_base_quat"],
                    float(self.heading_control_stiffness),
                    float(self.vel_limit_low[2]),
                    float(self.vel_limit_high[2]),
                )
            _lane_phase(
                self.command,
                self.phase,
                self.sin_cos,
                self.phase_offset,
                self.steps[0],
                self.phase_step[0],
            )

    @dispatch
    def reset_env(self, ctx: ManagerContext) -> None:
        self.steps[0] = 0.0
        self.episode_steps[0] = 0.0
        interval = self.resample_steps
        if self.resample_steps_high > self.resample_steps:
            interval = ctx.rand.uniform_range(self.resample_steps, self.resample_steps_high)
        self.resample_steps_left[0] = max(int(round(interval)), 1)
        if self.terrain_curriculum_enabled:
            self.terrain_commanded_distance[0] = 0.0
            self.terrain_progress[0] = 0.0
        if self.gait_period_width > np.float32(0.0):
            period = float(self.gait_period_base) + ctx.rand.uniform_range(
                -self.gait_period_width, self.gait_period_width
            )
            self.phase_step[0] = np.float32(2.0 * math.pi * ctx.dt / max(period, 0.1))
        _lane_resample_phase_offset(ctx, self.phase_offset)
        _lane_resample_command(
            ctx,
            self.command,
            self.vel_limit_low,
            self.vel_limit_high,
            self.stand_prob,
            self.heading_prob if self.heading_command else np.float32(0.0),
            self.heading_target,
            self.is_heading,
            np.float32(self.heading_low),
            np.float32(self.heading_high),
        )
        if self.heading_command:
            _apply_heading_command(
                self.command,
                self.heading_target[0],
                self.is_heading[0] != 0,
                ctx.sim["terrain_curriculum_base_quat"],
                float(self.heading_control_stiffness),
                float(self.vel_limit_low[2]),
                float(self.vel_limit_high[2]),
            )
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
        ``done = terminated | truncated``. The terrain curriculum uses the
        terminal planar displacement from the actual spawn: half-tile traversal
        promotes. Optional progress promotion also credits survived, well-tracked
        turns/reversals that leave the spawn platform. Early failures and poor
        timeout progress demote. Up/down tracking thresholds provide hysteresis;
        zero-demand episodes cannot earn progress promotion.
        """
        ctx.metrics["penalty_scale"] = float(self.penalty_scale[0])
        if ctx.env_ids.size == 0:
            return
        if self.terrain_curriculum_enabled:
            ids = ctx.env_ids
            active = self.episode_steps[ids, 0] > 0.0
            levels = self.terrain_levels[ids, 0].copy()
            if self.terrain_sampling_enabled:
                sampling = TerrainSamplingState(
                    int(self.terrain_cols_count),
                    int(self.terrain_rows_count),
                    adaptive_fraction=float(self.terrain_sampling_fraction),
                    max_multiplier=float(self.terrain_sampling_max_multiplier),
                    ema_alpha=float(self.terrain_sampling_ema_alpha),
                    min_count=int(self.terrain_sampling_min_count),
                )
                sampling.failure_ema = self.terrain_sampling_failure_ema
                sampling.evidence_counts = self.terrain_sampling_evidence_counts
                sampling.update(self.terrain_cols[ids], levels, ctx.terminated[ids], active)
                self.terrain_sampling_failure_ema[:] = sampling.failure_ema
                self.terrain_sampling_evidence_counts[:] = sampling.evidence_counts
            delta = self.terrain_terminal_xy[ids] - self.terrain_spawn_xy[ids]
            distance = np.linalg.norm(delta, axis=1)
            failed = ctx.terminated[ids]
            promoted = active & (distance > 0.5 * float(self.terrain_tile_length))
            if self.terrain_require_survival_for_promotion:
                promoted &= ~failed
            commanded = self.terrain_commanded_distance[ids, 0]
            progress = self.terrain_progress[ids, 0]
            if self.terrain_progress_promotion:
                # Reversals and well-tracked turns should not erase traversal.
                # Require a complete surviving episode, enough signed progress,
                # good tracking and departure from the central spawn platform.
                progress_success = (
                    ~failed
                    & (commanded > 0.0)
                    & (progress > 0.5 * float(self.terrain_tile_length))
                    & (progress >= float(self.terrain_move_up_ratio) * commanded)
                    & (
                        self.terrain_max_origin_radius[ids, 0]
                        >= float(self.terrain_min_exploration_ratio) * float(self.terrain_tile_length)
                    )
                )
                promoted |= active & progress_success
            poor_progress = (commanded > 0.0) & (progress < float(self.terrain_move_down_ratio) * commanded)
            # A fall is only a severe curriculum failure when it happens near
            # the spawn.  A late fall after meaningful traversal should not
            # erase the evidence that this level was nearly mastered. Timeout
            # episodes use the command-aligned progress test instead.
            early_failure = failed & (
                distance < float(self.terrain_failure_demote_ratio) * float(self.terrain_tile_length)
            )
            demoted = active & ~promoted & (early_failure | (~failed & poor_progress))
            self.terrain_levels[ids, 0] = np.clip(
                levels + promoted.astype(np.int64) - demoted.astype(np.int64),
                int(self.terrain_min_level),
                int(self.terrain_max_level),
            )
            if self.terrain_balanced_columns:
                active_ids = ids[active]
                graded_levels = self.terrain_levels[active_ids, 0]
                # Alternate assignments independently per graded row, carrying
                # parity across partial resets. Never sample a destination row.
                for row in np.unique(graded_levels):
                    row_ids = active_ids[graded_levels == row]
                    counter = self.terrain_column_row_counter[row]
                    self.terrain_cols[row_ids, 0] = (counter + np.arange(row_ids.size)) % 2
                    self.terrain_column_row_counter[row] += row_ids.size
            completed = max(int(np.count_nonzero(active)), 1)
            ctx.metrics["terrain_promote_rate"] = float(np.count_nonzero(promoted) / completed)
            ctx.metrics["terrain_demote_rate"] = float(np.count_nonzero(demoted) / completed)
            self.episode_steps[ids, 0] = 0.0
            self.terrain_commanded_distance[ids, 0] = 0.0
            self.terrain_progress[ids, 0] = 0.0
            self.terrain_max_origin_radius[ids, 0] = 0.0
            if self.terrain_sampling_enabled:
                active_ids = ids[active]
                # Share the current frontier within each column, not per lane.
                # Snapshot every binding after grading and before reassignment;
                # temporarily empty columns retain their last nonempty histogram.
                num_cols = int(self.terrain_cols_count)
                num_levels = int(self.terrain_rows_count)
                pool = np.bincount(
                    self.terrain_cols[:, 0] * num_levels + self.terrain_levels[:, 0],
                    minlength=num_cols * num_levels,
                ).reshape(num_cols, num_levels)
                occupied = pool.sum(axis=1) > 0
                self.terrain_sampling_level_pool[occupied] = pool[occupied]
                if active_ids.size:
                    rng = np.random.default_rng(
                        np.random.SeedSequence([int(self.terrain_sampling_seed), int(self.terrain_sampling_counter[0])])
                    )
                    old_cols = self.terrain_cols[active_ids, 0].copy()
                    new_cols = sampling.resample(active_ids, rng)
                    self.terrain_sampling_counter[0] += 1
                    self.terrain_cols[active_ids, 0] = new_cols
                    changed_mask = new_cols != old_cols
                    changed_ids = active_ids[changed_mask]
                    # Partial fix: same-column lanes retain their newly graded
                    # level; only cross-column moves inherit the population pool.
                    # Cross-column grade erasure remains: this C x L pool is
                    # fragile, so mixed-task sampling stays disabled until a
                    # proper persistent reservoir replaces population snapshots.
                    if changed_ids.size:
                        selected_pool = self.terrain_sampling_level_pool[new_cols[changed_mask]]
                        cdf = np.cumsum(selected_pool, axis=1)
                        draws = rng.random(changed_ids.size) * cdf[:, -1]
                        self.terrain_levels[changed_ids, 0] = np.sum(draws[:, None] >= cdf, axis=1)
                probabilities = sampling.probabilities()
                counts = np.bincount(self.terrain_cols[:, 0], minlength=num_cols)
                fractions = counts / len(self.terrain_cols)
                totals = np.bincount(self.terrain_cols[:, 0], weights=self.terrain_levels[:, 0], minlength=num_cols)
                pool_means = (
                    self.terrain_sampling_level_pool @ np.arange(num_levels)
                ) / self.terrain_sampling_level_pool.sum(axis=1)
                column_means = np.divide(totals, counts, out=pool_means, where=counts > 0)
                for col in range(num_cols):
                    ctx.metrics[f"terrain_sampling_probability_col_{col}"] = float(probabilities[col])
                    ctx.metrics[f"terrain_sampling_fraction_col_{col}"] = float(fractions[col])
                    ctx.metrics[f"terrain_level_col_{col}"] = float(column_means[col])
                # Fixed base column weights, independent of adaptive occupancy.
                ctx.metrics["terrain_level"] = float(sampling.base_probabilities @ column_means)
                ctx.metrics["terrain_level_sampling_weighted"] = float(np.mean(self.terrain_levels))
            else:
                for col in range(int(self.terrain_cols_count)):
                    mask = self.terrain_cols[:, 0] == col
                    if np.any(mask):
                        ctx.metrics[f"terrain_level_col_{col}"] = float(np.mean(self.terrain_levels[mask, 0]))
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
    resampling_time_range: tuple[float, float] | None = None
    ctrl_dt: float = 0.02
    gait_period: float = 1.0
    # Per-episode uniform gait-period jitter: period ~ U(g-w, g+w). 0 disables.
    gait_period_randomization_width: float = 0.0
    stand_prob: float = 0.2
    heading_command: bool = False
    heading_range: tuple[float, float] = (-math.pi, math.pi)
    heading_control_stiffness: float = 0.5
    heading_prob: float = 0.3
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
    # spawn): lanes start at terrain_start_ratio of the tile rows, promote
    # after surviving half-tile traversal, and demote after failure or when
    # command-aligned progress lags the command-integrated distance.
    terrain_curriculum: bool = False
    terrain_start_ratio: float = 0.2
    terrain_tile_length: float = 8.0
    terrain_move_down_ratio: float = 0.5
    # Optional turn/reversal-friendly promotion; retain the traversal route.
    terrain_progress_promotion: bool = False
    terrain_require_survival_for_promotion: bool = False
    terrain_move_up_ratio: float = 0.7
    # Training-only balanced assignments after grading, preserving each row.
    terrain_balanced_columns: bool = False
    terrain_sampling_enabled: bool = False
    terrain_sampling_fraction: float = 0.25
    terrain_sampling_max_multiplier: float = 2.0
    terrain_sampling_ema_alpha: float = 0.05
    terrain_sampling_min_count: int = 20
    terrain_sampling_seed: int = 1
    terrain_min_exploration_ratio: float = 0.25
    terrain_failure_demote_ratio: float = 0.25
    terrain_min_level: int = 1
    terrain_max_level: int | None = None
    terrain_min_command_speed: float = 0.05
    vel_limit: list[list[float]] = (
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.terrain_move_down_ratio < self.terrain_move_up_ratio <= 1.0:
            raise ValueError("terrain tracking thresholds must satisfy 0 <= down < up <= 1")
        if self.terrain_max_level is not None and self.terrain_max_level < self.terrain_min_level:
            raise ValueError("terrain_max_level must be at least terrain_min_level")
        if self.terrain_balanced_columns and self.terrain_sampling_enabled:
            raise ValueError("terrain_balanced_columns requires terrain_sampling_enabled=False")
        if self.terrain_max_level is not None and self.terrain_sampling_enabled:
            raise ValueError("terrain_max_level requires terrain_sampling_enabled=False")
        if not 0.0 < self.terrain_min_exploration_ratio <= 0.5:
            raise ValueError("terrain_min_exploration_ratio must lie in (0, 0.5]")
        if self.resampling_time <= 0.0:
            raise ValueError("resampling_time must be positive")
        if self.resampling_time_range is not None:
            lo, hi = self.resampling_time_range
            if not 0.0 < lo <= hi:
                raise ValueError("resampling_time_range must be positive and ordered")
        if not 0.0 <= self.stand_prob <= 1.0:
            raise ValueError("stand_prob must lie in [0, 1]")
        if not 0.0 <= self.heading_prob <= 1.0:
            raise ValueError("heading_prob must lie in [0, 1]")
        if self.heading_command and self.stand_prob + self.heading_prob > 1.0:
            raise ValueError("stand_prob + heading_prob must not exceed 1")
        if self.heading_range[0] > self.heading_range[1]:
            raise ValueError("heading_range must be ordered")
        if self.heading_control_stiffness < 0.0:
            raise ValueError("heading_control_stiffness must be non-negative")

    def __call__(self, env: ManagerEnv) -> WalkCommand:
        num_envs = env.num_envs
        terrain = _build_terrain_curriculum(env, self) if self.terrain_curriculum else None
        if self.terrain_sampling_enabled:
            if terrain is None:
                raise ValueError("adaptive terrain sampling requires terrain_curriculum")
            TerrainSamplingState(
                terrain["cols_count"],
                terrain["rows_count"],
                adaptive_fraction=self.terrain_sampling_fraction,
                max_multiplier=self.terrain_sampling_max_multiplier,
                ema_alpha=self.terrain_sampling_ema_alpha,
                min_count=self.terrain_sampling_min_count,
            )
        if self.terrain_balanced_columns:
            if terrain is None:
                raise ValueError("terrain_balanced_columns requires terrain_curriculum")
            if terrain["cols_count"] != 2:
                raise ValueError("terrain_balanced_columns requires exactly 2 terrain columns")
            if self.terrain_sampling_enabled:
                raise ValueError("terrain_balanced_columns requires terrain_sampling_enabled=False")
        num_cols = terrain["cols_count"] if terrain else 1
        num_levels = terrain["rows_count"] if terrain else 1
        initial_cols = terrain["cols"] if terrain else np.zeros((num_envs, 1), dtype=np.int64)
        initial_levels = terrain["levels"] if terrain else np.zeros((num_envs, 1), dtype=np.int64)
        level_pool = np.bincount(
            (initial_cols * num_levels + initial_levels).reshape(-1), minlength=num_cols * num_levels
        ).reshape(num_cols, num_levels)
        # Small environments may not bind every column at startup. Bootstrap
        # only those columns at the configured starting difficulty.
        level_pool[level_pool.sum(axis=1) == 0, initial_levels[0, 0]] = 1
        lo, hi = self.resampling_time_range or (self.resampling_time, self.resampling_time)
        return WalkCommand(
            vel_limit_low=np.asarray(self.vel_limit[0], dtype=np.float32),
            vel_limit_high=np.asarray(self.vel_limit[1], dtype=np.float32),
            stand_prob=np.float32(self.stand_prob),
            heading_prob=np.float32(self.heading_prob),
            heading_control_stiffness=np.float32(self.heading_control_stiffness),
            heading_command=self.heading_command,
            heading_target=np.zeros((num_envs, 1), dtype=np.float32),
            is_heading=np.zeros((num_envs, 1), dtype=np.bool_),
            heading_low=np.float32(self.heading_range[0]),
            heading_high=np.float32(self.heading_range[1]),
            resample_steps=np.float32(max(int(round(lo / self.ctrl_dt)), 1)),
            resample_steps_high=np.float32(max(int(round(hi / self.ctrl_dt)), 1)),
            resample_steps_left=np.zeros((num_envs, 1), dtype=np.int64),
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
            terrain_origin_grid=(terrain["origin_grid"] if terrain else np.zeros((1, 3), dtype=np.float32)).reshape(
                -1, 3
            ),
            terrain_levels=(terrain["levels"] if terrain else np.zeros((num_envs, 1), dtype=np.int64)),
            terrain_cols=terrain["cols"] if terrain else np.zeros((num_envs, 1), dtype=np.int64),
            terrain_cols_count=np.int64(terrain["cols_count"] if terrain else 1),
            terrain_rows_count=np.int64(terrain["rows_count"] if terrain else 1),
            terrain_max_level=np.int64(terrain["max_level"] if terrain else 0),
            terrain_tile_length=np.float32(self.terrain_tile_length),
            terrain_move_down_ratio=np.float32(self.terrain_move_down_ratio),
            terrain_progress_promotion=self.terrain_progress_promotion,
            terrain_require_survival_for_promotion=self.terrain_require_survival_for_promotion,
            terrain_move_up_ratio=np.float32(self.terrain_move_up_ratio),
            terrain_min_exploration_ratio=np.float32(self.terrain_min_exploration_ratio),
            terrain_max_origin_radius=np.zeros((num_envs, 1), dtype=np.float32),
            terrain_failure_demote_ratio=np.float32(self.terrain_failure_demote_ratio),
            terrain_min_level=np.int64(self.terrain_min_level),
            terrain_min_command_speed=np.float32(self.terrain_min_command_speed),
            terrain_commanded_distance=np.zeros((num_envs, 1), dtype=np.float32),
            terrain_progress=np.zeros((num_envs, 1), dtype=np.float32),
            terrain_terminal_xy=np.zeros((num_envs, 2), dtype=np.float32),
            terrain_previous_xy=np.zeros((num_envs, 2), dtype=np.float32),
            terrain_previous_quat=np.zeros((num_envs, 4), dtype=np.float32),
            terrain_spawn_xy=np.zeros((num_envs, 2), dtype=np.float32),
            terrain_balanced_columns=self.terrain_balanced_columns,
            terrain_column_row_counter=np.zeros((num_levels,), dtype=np.int64),
            terrain_sampling_enabled=self.terrain_sampling_enabled,
            terrain_sampling_fraction=np.float32(self.terrain_sampling_fraction),
            terrain_sampling_max_multiplier=np.float32(self.terrain_sampling_max_multiplier),
            terrain_sampling_ema_alpha=np.float32(self.terrain_sampling_ema_alpha),
            terrain_sampling_min_count=np.int64(self.terrain_sampling_min_count),
            terrain_sampling_seed=np.int64(self.terrain_sampling_seed),
            terrain_sampling_failure_ema=np.full(
                (terrain["cols_count"] if terrain else 1, terrain["rows_count"] if terrain else 1),
                0.5,
                dtype=np.float64,
            ),
            terrain_sampling_evidence_counts=np.zeros(
                (terrain["cols_count"] if terrain else 1, terrain["rows_count"] if terrain else 1), dtype=np.int64
            ),
            terrain_sampling_counter=np.zeros((1,), dtype=np.int64),
            terrain_sampling_level_pool=level_pool,
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
    if not 0 <= cfg.terrain_min_level < rows:
        raise ValueError("terrain_min_level must be a valid terrain row")
    origin_grid = _tile_origin_grid(grid, tiles, reset_cfg.spawn_border).reshape(rows * cols, 3)
    max_level = rows - 1 if cfg.terrain_max_level is None else cfg.terrain_max_level
    if not 0 <= cfg.terrain_min_level <= max_level < rows:
        raise ValueError("terrain levels must satisfy 0 <= terrain_min_level <= terrain_max_level < rows")
    start_level = np.clip(int(cfg.terrain_start_ratio * rows), cfg.terrain_min_level, max_level)
    levels = np.full((env.num_envs, 1), start_level, dtype=np.int64)
    cols_of_env = (np.arange(env.num_envs) * cols // env.num_envs).astype(np.int64).reshape(-1, 1)
    return {
        "origin_grid": np.ascontiguousarray(origin_grid),
        "levels": levels,
        "cols": cols_of_env,
        "cols_count": cols,
        "rows_count": rows,
        "max_level": max_level,
    }
