# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for terrain curriculum progress and reset semantics."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from motrix_env_core.manager import ManagerEnv
from motrix_envs.locomotion.humanoid.g1_stairs import make_g129dof_walk_mixed_cfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.command import WalkCommand, WalkCommandCfg, _terrain_step_progress


def test_balanced_columns_preserve_grades_and_independent_row_parity():
    command, ctx = _terrain_command_for_reset([False, False, True, False, False, False])
    command = replace(command, terrain_balanced_columns=True, terrain_cols_count=np.int64(2))
    command.terrain_levels[:, 0] = [1, 1, 2, 2, 1, 2]
    command.terrain_cols[:] = 1
    command.terrain_terminal_xy[:] = 0
    command.terrain_terminal_xy[0, 0] = 5  # promotion to row 2
    command.episode_steps[4, 0] = 0  # startup reset: no assignment
    ctx.env_ids = np.array([0, 1, 2, 3, 4])  # lane 5 is still running
    command.reset(ctx)
    np.testing.assert_array_equal(command.terrain_levels[:, 0], [2, 1, 1, 2, 1, 2])
    np.testing.assert_array_equal(command.terrain_cols[:, 0], [0, 0, 1, 1, 1, 1])
    np.testing.assert_array_equal(command.terrain_column_row_counter, [0, 2, 2])
    assert command.episode_steps[5, 0] == 1

    # Odd partial batches carry parity separately for each destination row.
    command.episode_steps[[1, 3], 0] = 1
    command.terrain_commanded_distance[:] = 0
    ctx.env_ids = np.array([3, 1])
    command.reset(ctx)
    np.testing.assert_array_equal(command.terrain_cols[[1, 3], 0], [0, 0])
    np.testing.assert_array_equal(command.terrain_column_row_counter, [0, 3, 3])
    command.episode_steps[1, 0] = 1
    ctx.env_ids = np.array([1])
    command.reset(ctx)
    assert command.terrain_cols[1, 0] == 1
    np.testing.assert_array_equal(command.terrain_column_row_counter, [0, 4, 3])
    command.reset(ctx)  # no completed transition since previous reset
    command.reset(SimpleNamespace(env_ids=np.array([], dtype=np.int64), terminated=ctx.terminated, metrics={}))
    np.testing.assert_array_equal(command.terrain_column_row_counter, [0, 4, 3])


def test_disabled_balancing_leaves_columns_and_counters_unchanged():
    command, ctx = _terrain_command_for_reset([False, True])
    command.terrain_cols[:, 0] = [1, 0]
    command.terrain_column_row_counter[:] = [2, 3, 4]
    command.reset(ctx)
    np.testing.assert_array_equal(command.terrain_cols[:, 0], [1, 0])
    np.testing.assert_array_equal(command.terrain_column_row_counter, [2, 3, 4])


@pytest.mark.parametrize("columns", [1, 3, 4])
def test_balanced_columns_require_two_columns(monkeypatch, columns):
    from motrix_envs.locomotion.humanoid.walk_manager_mdp import command as command_module

    monkeypatch.setattr(command_module, "_build_terrain_curriculum", lambda env, cfg: {"cols_count": columns})
    cfg = WalkCommandCfg(terrain_curriculum=True, terrain_balanced_columns=True)
    with pytest.raises(ValueError, match="exactly 2 terrain columns"):
        cfg(SimpleNamespace(num_envs=2))


def test_balanced_columns_require_curriculum_and_exclude_adaptive_sampling():
    with pytest.raises(ValueError, match="requires terrain_curriculum"):
        WalkCommandCfg(terrain_balanced_columns=True)(SimpleNamespace(num_envs=2))
    with pytest.raises(ValueError, match="terrain_sampling_enabled=False"):
        WalkCommandCfg(terrain_balanced_columns=True, terrain_sampling_enabled=True)


def test_balanced_columns_for_play_does_not_mutate_training_cfg():
    cfg = make_g129dof_walk_mixed_cfg()
    assert cfg.for_play() is cfg  # default retains the existing mode-agnostic path
    cfg.commands.walk.terrain_balanced_columns = True
    play = cfg.for_play()
    assert not play.commands.walk.terrain_balanced_columns
    assert cfg.commands.walk.terrain_balanced_columns
    assert play.commands.walk.terrain_curriculum == cfg.commands.walk.terrain_curriculum


def test_command_aligned_progress_handles_body_frame_reversal_and_caps():
    quat_identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    expected, progress = _terrain_step_progress(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        quat_identity,
        np.array([0.02, 0.0, 0.0], dtype=np.float32),
        0.02,
        0.05,
    )
    assert expected == pytest.approx(0.02)
    assert progress == pytest.approx(0.02)

    # A reversal/circle step is scored by signed aligned progress, not raw speed.
    _, reverse_progress = _terrain_step_progress(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.02, 0.0], dtype=np.float32),
        quat_identity,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        0.02,
        0.05,
    )
    assert reverse_progress == pytest.approx(-0.02)
    # Reversing the command as well must count as good progress, even though
    # the two-step net displacement returns to the spawn position.
    _, followed_reversal = _terrain_step_progress(
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.02, 0.0], dtype=np.float32),
        quat_identity,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        0.02,
        0.05,
    )
    assert followed_reversal == pytest.approx(0.02)

    _, capped = _terrain_step_progress(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        quat_identity,
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        0.02,
        0.05,
    )
    assert capped == pytest.approx(0.02)

    # Body command [1, 0] under +90-degree yaw points along world +Y.
    quarter_turn = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
    _, yaw_progress = _terrain_step_progress(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
        quarter_turn,
        np.array([0.0, 0.02, 0.0], dtype=np.float32),
        0.02,
        0.05,
    )
    assert yaw_progress == pytest.approx(0.02)


def _terrain_command_for_reset(terminated):
    n = len(terminated)
    terminal_xy = np.zeros((n, 2), dtype=np.float32)
    terminal_xy[0, 0] = 5
    return WalkCommand(
        vel_limit_low=np.zeros(3, dtype=np.float32),
        vel_limit_high=np.ones(3, dtype=np.float32),
        stand_prob=np.float32(0),
        heading_prob=np.float32(0),
        heading_control_stiffness=np.float32(0.5),
        heading_command=False,
        heading_target=np.zeros((n, 1), dtype=np.float32),
        is_heading=np.zeros((n, 1), dtype=np.bool_),
        heading_low=np.float32(-np.pi),
        heading_high=np.float32(np.pi),
        resample_steps=np.float32(10),
        resample_steps_high=np.float32(10),
        resample_steps_left=np.zeros((n, 1), dtype=np.int64),
        phase_step=np.ones((n, 1), dtype=np.float32),
        gait_period_base=np.float32(1),
        gait_period_width=np.float32(0),
        curriculum_enabled=False,
        penalty_scale=np.ones(1, dtype=np.float32),
        avg_ep_len=np.zeros(1, dtype=np.float32),
        level_down_threshold=np.float32(0),
        level_up_threshold=np.float32(1e9),
        degree=np.float32(0),
        min_scale=np.float32(0),
        max_scale=np.float32(1),
        init_state_mix=np.ones(1, dtype=np.float32),
        init_state_step_count=np.zeros(1, dtype=np.float32),
        init_state_curriculum_steps=np.float32(1),
        init_state_curriculum_start=np.float32(1),
        terrain_curriculum_enabled=True,
        terrain_origin_grid=np.zeros((2, 3), dtype=np.float32),
        terrain_levels=np.ones((n, 1), dtype=np.int64),
        terrain_cols=np.zeros((n, 1), dtype=np.int64),
        terrain_cols_count=np.int64(1),
        terrain_rows_count=np.int64(3),
        terrain_max_level=np.int64(2),
        terrain_tile_length=np.float32(8),
        terrain_move_down_ratio=np.float32(0.5),
        terrain_progress_promotion=True,
        terrain_require_survival_for_promotion=False,
        terrain_balanced_columns=False,
        terrain_column_row_counter=np.zeros(3, dtype=np.int64),
        terrain_sampling_enabled=False,
        terrain_sampling_fraction=np.float32(0.25),
        terrain_sampling_max_multiplier=np.float32(2),
        terrain_sampling_ema_alpha=np.float32(0.05),
        terrain_sampling_min_count=np.int64(20),
        terrain_sampling_seed=np.int64(1),
        terrain_sampling_failure_ema=np.full((1, 3), 0.5),
        terrain_sampling_evidence_counts=np.zeros((1, 3), dtype=np.int64),
        terrain_sampling_counter=np.zeros(1, dtype=np.int64),
        terrain_sampling_level_pool=np.array([[0, n, 0]], dtype=np.int64),
        terrain_move_up_ratio=np.float32(0.7),
        terrain_min_exploration_ratio=np.float32(0.25),
        terrain_max_origin_radius=np.zeros((n, 1), dtype=np.float32),
        terrain_failure_demote_ratio=np.float32(0.25),
        terrain_min_level=np.int64(1),
        terrain_min_command_speed=np.float32(0.05),
        terrain_commanded_distance=np.ones((n, 1), dtype=np.float32),
        terrain_progress=np.ones((n, 1), dtype=np.float32),
        terrain_terminal_xy=terminal_xy,
        terrain_previous_xy=np.zeros((n, 2), dtype=np.float32),
        terrain_previous_quat=np.zeros((n, 4), dtype=np.float32),
        terrain_spawn_xy=np.zeros((n, 2), dtype=np.float32),
        command=np.zeros((n, 3), dtype=np.float32),
        phase_offset=np.zeros((n, 2), dtype=np.float32),
        sin_cos=np.zeros((n, 4), dtype=np.float32),
        phase=np.zeros((n, 2), dtype=np.float32),
        steps=np.zeros((n, 1), dtype=np.float32),
        episode_steps=np.ones((n, 1), dtype=np.float32),
    ), SimpleNamespace(env_ids=np.arange(n), terminated=np.asarray(terminated), metrics={})


def test_reset_promotes_survivor_and_demotes_failure_without_loop():
    command, ctx = _terrain_command_for_reset([False, True])
    command.terrain_levels[:, 0] = [1, 2]
    command.reset(ctx)
    np.testing.assert_array_equal(command.terrain_levels[:, 0], [2, 1])
    np.testing.assert_array_equal(command.episode_steps[:, 0], 0)


def test_late_failure_after_traversal_is_not_demoted():
    command, ctx = _terrain_command_for_reset([True, False])
    command.terrain_terminal_xy[:] = [[3.0, 0.0], [0.0, 0.0]]
    command.reset(ctx)
    np.testing.assert_array_equal(command.terrain_levels[:, 0], [1, 1])


def test_reset_startup_zero_steps_is_neutral():
    command, ctx = _terrain_command_for_reset([True, True])
    command.episode_steps[:] = 0
    command.reset(ctx)
    np.testing.assert_array_equal(command.terrain_levels[:, 0], [1, 1])


def test_reset_subset_does_not_touch_other_lanes():
    command, _ = _terrain_command_for_reset([False, False])
    ctx = SimpleNamespace(env_ids=np.array([0]), terminated=np.array([False]), metrics={})
    command.reset(ctx)
    assert command.terrain_levels[:, 0].tolist() == [2, 1]
    assert command.episode_steps[1, 0] == 1
    assert command.terrain_commanded_distance[1, 0] == 1


def test_timeout_with_standing_command_is_neutral():
    command, ctx = _terrain_command_for_reset([False, False])
    command.terrain_terminal_xy[:] = 0
    command.terrain_commanded_distance[:] = 0
    command.terrain_progress[:] = 0
    command.reset(ctx)
    np.testing.assert_array_equal(command.terrain_levels[:, 0], [1, 1])


def test_timeout_with_poor_progress_demotes():
    command, ctx = _terrain_command_for_reset([False, False])
    command.terrain_levels[:] = 2
    command.terrain_terminal_xy[:] = 0
    command.terrain_commanded_distance[:] = 1
    command.terrain_progress[:] = 0.1
    command.reset(ctx)
    np.testing.assert_array_equal(command.terrain_levels[:, 0], [1, 1])


def test_progress_promotion_requires_tracking_and_spatial_coverage():
    command, ctx = _terrain_command_for_reset([False, False])
    command.terrain_terminal_xy[:] = 0.0
    command.terrain_commanded_distance[:, 0] = [6.0, 6.0]
    command.terrain_progress[:, 0] = [5.0, 5.0]
    command.terrain_max_origin_radius[:, 0] = [2.1, 0.5]
    command.reset(ctx)
    # The first lane meets both absolute progress and 70% tracking plus 2m
    # coverage; the second has good tracking but remains on the spawn platform.
    np.testing.assert_array_equal(command.terrain_levels[:, 0], [2, 1])
    assert ctx.metrics["terrain_promote_rate"] == pytest.approx(0.5)
    assert ctx.metrics["terrain_demote_rate"] == pytest.approx(0.0)


def test_progress_promotion_rejects_low_tracking_ratio():
    command, ctx = _terrain_command_for_reset([False, False])
    command.terrain_terminal_xy[:] = 0.0
    command.terrain_commanded_distance[:, 0] = [10.0, 0.0]
    command.terrain_progress[:, 0] = [6.0, 0.0]
    command.terrain_max_origin_radius[:, 0] = 3.0
    command.reset(ctx)
    np.testing.assert_array_equal(command.terrain_levels[:, 0], [1, 1])


def test_progress_promotion_rejects_stationary_and_failed_episodes():
    command, ctx = _terrain_command_for_reset([False, True])
    command.terrain_terminal_xy[:] = 0.0
    command.terrain_commanded_distance[:, 0] = [8.0, 8.0]
    command.terrain_progress[:, 0] = [8.0, 8.0]
    command.terrain_max_origin_radius[:, 0] = [0.0, 3.0]
    command.reset(ctx)
    # Standing never accumulates demand in the real kernel, while a failed lane
    # cannot use the optional survived-progress route.
    np.testing.assert_array_equal(command.terrain_levels[:, 0], [1, 1])


def test_survival_gate_rejects_failed_half_tile_traversal():
    command, ctx = _terrain_command_for_reset([True, False])
    command = replace(command, terrain_require_survival_for_promotion=True)
    command.terrain_terminal_xy[:] = [[5.0, 0.0], [5.0, 0.0]]
    command.reset(ctx)
    np.testing.assert_array_equal(command.terrain_levels[:, 0], [1, 2])


def _adaptive_command(cols=(0, 1), levels=(1, 2), seed=1):
    command, _ = _terrain_command_for_reset([False, False])
    command = replace(
        command,
        terrain_sampling_enabled=True,
        terrain_cols_count=np.int64(2),
        terrain_sampling_seed=np.int64(seed),
        terrain_sampling_failure_ema=np.full((2, 3), 0.5),
        terrain_sampling_evidence_counts=np.zeros((2, 3), dtype=np.int64),
        terrain_sampling_level_pool=np.array([[0, 1, 0], [0, 0, 1]], dtype=np.int64),
    )
    command.terrain_cols[:, 0] = cols
    command.terrain_levels[:, 0] = levels
    command.terrain_terminal_xy[:] = 0
    command.terrain_commanded_distance[:] = 0
    return command


def test_sampler_transfers_current_column_mastery_and_preserves_partial_reset():
    command = _adaptive_command()
    ctx = SimpleNamespace(env_ids=np.array([0]), terminated=np.array([False, False]), metrics={})
    command.reset(ctx)
    # Seed1 chooses column1: a new lane inherits that column's mastered level,
    # not an untrained per-lane history. The running donor lane is untouched.
    np.testing.assert_array_equal(command.terrain_cols[:, 0], [1, 1])
    np.testing.assert_array_equal(command.terrain_levels[:, 0], [2, 2])
    np.testing.assert_array_equal(command.terrain_sampling_level_pool, [[0, 1, 0], [0, 0, 1]])
    assert command.episode_steps[1, 0] == 1
    assert command.terrain_sampling_evidence_counts[0, 1] == 1
    assert ctx.metrics["terrain_level_col_0"] == pytest.approx(1)
    assert ctx.metrics["terrain_level_col_1"] == pytest.approx(2)
    assert ctx.metrics["terrain_level"] == pytest.approx(1.5)
    assert ctx.metrics["terrain_level_sampling_weighted"] == pytest.approx(2)


def test_sampler_does_not_transfer_mastery_across_columns():
    command = _adaptive_command(levels=(2, 1))
    ctx = SimpleNamespace(env_ids=np.array([0]), terminated=np.array([False, False]), metrics={})
    command.reset(ctx)
    assert command.terrain_cols[0, 0] == 1
    assert command.terrain_levels[0, 0] == 1
    np.testing.assert_array_equal(command.terrain_sampling_level_pool, [[0, 0, 1], [0, 1, 0]])


def test_sampler_snapshots_postgrade_frontier_and_records_old_failure_cell():
    command = _adaptive_command(levels=(1, 1), seed=2)
    command.terrain_terminal_xy[0] = [5, 0]
    ctx = SimpleNamespace(env_ids=np.array([0]), terminated=np.array([True, False]), metrics={})
    command.reset(ctx)
    # Legacy survival flag=False keeps the net-traversal promotion route.
    assert command.terrain_cols[0, 0] == 0
    assert command.terrain_levels[0, 0] == 2
    np.testing.assert_array_equal(command.terrain_sampling_level_pool[0], [0, 0, 1])
    assert command.terrain_sampling_evidence_counts[0, 1] == 1
    assert command.terrain_sampling_evidence_counts[0, 2] == 0
    assert command.terrain_sampling_failure_ema[0, 1] > 0.5


def test_sampler_empty_column_retains_latest_histogram():
    command = _adaptive_command()
    ctx = SimpleNamespace(env_ids=np.array([0]), terminated=np.array([False, False]), metrics={})
    command.reset(ctx)  # column0 becomes empty, retaining its level1 histogram
    command.episode_steps[0, 0] = 1
    command = replace(command, terrain_sampling_seed=np.int64(2))
    command.terrain_sampling_counter[0] = 0
    command.reset(ctx)  # seed2 chooses the now-empty column0
    assert command.terrain_cols[0, 0] == 0
    assert command.terrain_levels[0, 0] == 1
    np.testing.assert_array_equal(command.terrain_sampling_level_pool[0], [0, 1, 0])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_sampler_seeded_column_draws_and_vectorized_empirical_levels(seed):
    from motrix_envs.locomotion.humanoid.walk_manager_mdp.terrain_sampling import TerrainSamplingState

    command = _adaptive_command(cols=(0, 0), seed=seed)
    ctx = SimpleNamespace(env_ids=np.arange(2), terminated=np.array([False, False]), metrics={})
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    expected_cols = TerrainSamplingState(2, 3).resample(ctx.env_ids, rng)
    changed = expected_cols != command.terrain_cols[:, 0]
    pool = np.array([[0, 1, 1], [0, 0, 1]])
    cdf = np.cumsum(pool[expected_cols[changed]], axis=1)
    draws = rng.random(np.count_nonzero(changed)) * cdf[:, -1]
    expected_levels = command.terrain_levels[:, 0].copy()
    expected_levels[changed] = np.sum(draws[:, None] >= cdf, axis=1)
    twin = _adaptive_command(cols=(0, 0), seed=seed)
    command.reset(ctx)
    twin.reset(SimpleNamespace(env_ids=np.arange(2), terminated=ctx.terminated, metrics={}))
    np.testing.assert_array_equal(command.terrain_cols[:, 0], expected_cols)
    np.testing.assert_array_equal(command.terrain_levels[:, 0], expected_levels)
    np.testing.assert_array_equal(twin.terrain_cols, command.terrain_cols)
    np.testing.assert_array_equal(twin.terrain_levels, command.terrain_levels)
    # Empty columns use their latest histogram mean, not occupied-column levels.
    column_means = [
        expected_levels[expected_cols == col].mean()
        if np.any(expected_cols == col)
        else pool[col] @ np.arange(3) / pool[col].sum()
        for col in range(2)
    ]
    for col, mean in enumerate(column_means):
        assert ctx.metrics[f"terrain_level_col_{col}"] == pytest.approx(mean)
    assert ctx.metrics["terrain_level"] == pytest.approx(np.mean(column_means))
    assert ctx.metrics["terrain_level_sampling_weighted"] == pytest.approx(expected_levels.mean())


def test_explicit_terrain_bound_rejects_unbounded_adaptive_resampling():
    with pytest.raises(ValueError, match="terrain_sampling_enabled=False"):
        WalkCommandCfg(terrain_max_level=2, terrain_sampling_enabled=True)


def test_curriculum_equal_bounds_hold_level_for_promotions_and_failures():
    command, ctx = _terrain_command_for_reset([False, True])
    command = replace(command, terrain_min_level=np.int64(1), terrain_max_level=np.int64(1))
    for _ in range(3):
        command.episode_steps[:] = 1
        command.terrain_terminal_xy[:] = [[5, 0], [0, 0]]
        command.reset(ctx)
        np.testing.assert_array_equal(command.terrain_levels[:, 0], [1, 1])


def test_curriculum_upper_bound_prevents_success_promoting_above_range():
    command, ctx = _terrain_command_for_reset([False])
    command = replace(command, terrain_max_level=np.int64(1))
    command.reset(ctx)
    assert command.terrain_levels[0, 0] == 1
    assert ctx.metrics["terrain_promote_rate"] == 1.0


def test_sampler_single_env_same_column_retains_repeated_promotions():
    command, ctx = _terrain_command_for_reset([False])
    command = replace(
        command,
        terrain_sampling_enabled=True,
        terrain_rows_count=np.int64(5),
        terrain_max_level=np.int64(4),
        terrain_sampling_failure_ema=np.full((1, 5), 0.5),
        terrain_sampling_evidence_counts=np.zeros((1, 5), dtype=np.int64),
        terrain_sampling_level_pool=np.array([[0, 0, 1, 0, 0]], dtype=np.int64),
    )
    command.terrain_levels[:] = 2
    for expected_level in (3, 4):
        command.episode_steps[:] = 1
        command.reset(ctx)
        assert command.terrain_cols[0, 0] == 0
        assert command.terrain_levels[0, 0] == expected_level
        assert ctx.metrics["terrain_level"] == pytest.approx(expected_level)


def test_sampler_single_env_same_column_retains_demotion():
    command, ctx = _terrain_command_for_reset([True])
    command = replace(command, terrain_sampling_enabled=True)
    command.terrain_levels[:] = 2
    command.terrain_terminal_xy[:] = 0
    command.reset(ctx)
    assert command.terrain_cols[0, 0] == 0
    assert command.terrain_levels[0, 0] == 1
    assert ctx.metrics["terrain_demote_rate"] == pytest.approx(1)


@pytest.mark.parametrize(
    "initial_levels, terminated, expected_levels", [((1, 1), [False, False], [2, 1]), ((1, 2), [False, True], [2, 1])]
)
def test_sampler_same_column_population_preserves_grades_and_mean(
    monkeypatch, initial_levels, terminated, expected_levels
):
    from motrix_envs.locomotion.humanoid.walk_manager_mdp.terrain_sampling import TerrainSamplingState

    command = _adaptive_command(cols=(0, 0), levels=initial_levels, seed=2)
    command.terrain_terminal_xy[0] = [5, 0]
    ctx = SimpleNamespace(env_ids=np.arange(2), terminated=np.asarray(terminated), metrics={})
    monkeypatch.setattr(
        TerrainSamplingState, "resample", lambda self, env_ids, rng: np.zeros(env_ids.size, dtype=np.int64)
    )
    previous_mean = command.terrain_levels.mean()
    command.reset(ctx)
    np.testing.assert_array_equal(command.terrain_cols[:, 0], [0, 0])
    np.testing.assert_array_equal(command.terrain_levels[:, 0], expected_levels)
    expected_mean = np.mean(expected_levels)
    assert ctx.metrics["terrain_level_col_0"] == pytest.approx(expected_mean)
    assert ctx.metrics["terrain_level_sampling_weighted"] == pytest.approx(expected_mean)
    assert command.terrain_levels.mean() - previous_mean == pytest.approx(0.5 if not terminated[1] else 0)


def test_sampler_initial_reset_does_not_draw_or_update_evidence():
    command, ctx = _terrain_command_for_reset([False, False])
    command = replace(command, terrain_sampling_enabled=True)
    command.episode_steps[:] = 0
    pool = command.terrain_sampling_level_pool.copy()
    cols = command.terrain_cols.copy()
    command.reset(ctx)
    command.reset(ctx)
    assert command.terrain_sampling_counter[0] == 0
    assert command.terrain_sampling_evidence_counts.sum() == 0
    np.testing.assert_array_equal(command.terrain_levels[:, 0], [1, 1])
    np.testing.assert_array_equal(command.terrain_cols, cols)
    np.testing.assert_array_equal(command.terrain_sampling_level_pool, pool)
    command.reset(SimpleNamespace(env_ids=np.array([], dtype=np.int64), terminated=ctx.terminated, metrics={}))
    assert command.terrain_sampling_counter[0] == 0
    np.testing.assert_array_equal(command.terrain_sampling_level_pool, pool)


@pytest.mark.parametrize("sampling", [False, True])
def test_compiled_mixed_terrain_environment_smoke(sampling):
    cfg = make_g129dof_walk_mixed_cfg()
    cfg.commands.walk.terrain_sampling_enabled = sampling
    env = ManagerEnv(cfg, num_envs=2)
    try:
        state = env.init_state()
        assert state.obs.policy.shape[0] == 2
        walk = env.command_terms["walk"]
        np.testing.assert_array_equal(
            walk.terrain_levels[:, 0], int(env.cfg.commands.walk.terrain_start_ratio * walk.terrain_rows_count)
        )
        assert walk.terrain_sampling_level_pool.shape == (walk.terrain_cols_count, walk.terrain_rows_count)
        assert np.all(walk.terrain_sampling_level_pool.sum(axis=1) > 0)
        occupied_counts = np.bincount(walk.terrain_cols[:, 0], minlength=int(walk.terrain_cols_count))
        occupied = occupied_counts > 0
        np.testing.assert_array_equal(walk.terrain_sampling_level_pool.sum(axis=1)[occupied], occupied_counts[occupied])
        np.testing.assert_allclose(walk.terrain_spawn_xy, walk.terrain_terminal_xy)
        np.testing.assert_allclose(walk.terrain_spawn_xy, env.sim_data["terrain_curriculum_base_pos"][:, :2])
        origins = walk.terrain_origin_grid[
            walk.terrain_levels[:, 0] * walk.terrain_cols_count + walk.terrain_cols[:, 0]
        ]
        np.testing.assert_allclose(
            walk.terrain_max_origin_radius[:, 0],
            np.linalg.norm(walk.terrain_spawn_xy - origins[:, :2], axis=1),
            atol=1e-6,
        )
        radius = walk.terrain_max_origin_radius.copy()
        env.step(np.zeros((2, env.num_actuators), dtype=np.float32))
        assert np.all(walk.terrain_max_origin_radius >= radius)
        env.reset(np.arange(2, dtype=np.int64))
    finally:
        del env
