# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Slope-only terrain isolation and registered manager reset contracts."""

import numpy as np

from motrix_env_core import registry
from motrix_env_core.config.scene import PyramidSlopeTerrainGeneratorCfg
from motrix_envs.locomotion.humanoid.g1_slope import make_g129dof_walk_slope_cfg
from motrix_envs.locomotion.humanoid.g1_stairs import make_g129dof_walk_mixed_cfg


def test_slope_terrain_preserves_mixed_slope_generators():
    slope = make_g129dof_walk_slope_cfg()
    mixed = make_g129dof_walk_mixed_cfg()
    slope_cells = [region.generator for region in slope.scene.assets.terrain.generator.regions]
    reference = [
        region.generator
        for region in mixed.scene.assets.terrain.generator.regions
        if isinstance(region.generator, PyramidSlopeTerrainGeneratorCfg)
    ]
    assert slope_cells == reference

    # Config equality alone misses rectangular-map physical-axis distortion.
    # Compare every generated slope tile against its mixed-map counterpart.
    def tile_patches(asset):
        heights = asset.generator.generate(asset.size, asset.shape)
        patches = []
        for region in asset.generator.regions:
            if isinstance(region.generator, PyramidSlopeTerrainGeneratorCfg):
                start = np.rint((np.asarray(region.center) - np.asarray(region.size) / 2) * asset.shape).astype(int)
                end = np.rint((np.asarray(region.center) + np.asarray(region.size) / 2) * asset.shape).astype(int)
                patches.append(heights[start[0] : end[0], start[1] : end[1]])
        return patches

    for actual, expected in zip(tile_patches(slope.scene.assets.terrain), tile_patches(mixed.scene.assets.terrain)):
        np.testing.assert_allclose(actual, expected, atol=1e-7)
    rows, cols = slope.sim_reset.humanoid_state.spawn_tiles
    assert rows * cols == len(slope_cells)
    assert slope.observations.policy.base_lin_vel is None
    assert not slope.commands.walk.terrain_sampling_enabled


def test_balanced_slope_reset_compiles_and_spawns_on_new_column():
    cfg = make_g129dof_walk_slope_cfg()
    cfg.commands.walk.terrain_balanced_columns = True
    cfg.sim_reset.humanoid_state.tile_xy_offset_range = 0.0
    env = registry.resolve("g1-walk-slope", env_cfg=cfg).make(num_envs=4, seed=17)
    try:
        state = env.init_state()
        env.compile()
        state.terminated[:] = False
        walk = env.command_terms["walk"]
        np.testing.assert_array_equal(walk.terrain_column_row_counter, 0)
        # Grading must be neutral here; verify the binding reaches compiled
        # reset writes, not just the host command's arrays.
        levels = walk.terrain_levels.copy()
        walk.episode_steps[:] = 1
        walk.terrain_commanded_distance[:] = 0
        walk.terrain_terminal_xy[:] = walk.terrain_spawn_xy
        env.reset(np.arange(4, dtype=np.int64))
        np.testing.assert_array_equal(walk.terrain_levels, levels)
        np.testing.assert_array_equal(walk.terrain_cols[:, 0], [0, 1, 0, 1])
        origins = walk.terrain_origin_grid[levels[:, 0] * 2 + walk.terrain_cols[:, 0]]
        np.testing.assert_allclose(walk.terrain_spawn_xy, origins[:, :2])
        np.testing.assert_allclose(env.sim_data["terrain_curriculum_base_pos"][:, :2], origins[:, :2])
        assert walk.terrain_column_row_counter[levels[0, 0]] == 4
        # A subsequent odd partial reset keeps row parity and other spawns.
        before = walk.terrain_spawn_xy.copy()
        for lane, expected_col in [(0, 0), (2, 1)]:
            walk.episode_steps[lane, 0] = 1
            env.reset(np.array([lane], dtype=np.int64))
            assert walk.terrain_cols[lane, 0] == expected_col
            index = walk.terrain_levels[lane, 0] * 2 + expected_col
            np.testing.assert_allclose(walk.terrain_spawn_xy[lane], walk.terrain_origin_grid[index, :2])
        np.testing.assert_allclose(walk.terrain_spawn_xy[[1, 3]], before[[1, 3]])
    finally:
        del env


def test_registered_slope_environment_compiles_and_resets_both_columns():
    cfg = make_g129dof_walk_slope_cfg()
    env = registry.resolve("g1-walk-slope", env_cfg=cfg).make(num_envs=4, mode="play", seed=17)
    try:
        state = env.init_state()
        env.compile()
        walk = env.command_terms["walk"]
        assert set(walk.terrain_cols[:, 0]) == {0, 1}
        initial_level = np.clip(
            int(cfg.commands.walk.terrain_start_ratio * walk.terrain_rows_count),
            cfg.commands.walk.terrain_min_level,
            walk.terrain_max_level,
        )
        np.testing.assert_array_equal(walk.terrain_levels, initial_level)
        for _ in range(3):
            state = env.step(np.zeros((4, env.num_actuators), dtype=np.float32))
            assert np.all(np.isfinite(state.obs.policy))
        env.reset(np.arange(4, dtype=np.int64))
        np.testing.assert_allclose(walk.terrain_spawn_xy, env.sim_data["terrain_curriculum_base_pos"][:, :2])
        assert set(walk.terrain_cols[:, 0]) == {0, 1}
        assert np.all(walk.terrain_levels >= cfg.commands.walk.terrain_min_level)
        assert np.all(walk.terrain_levels <= walk.terrain_max_level)
    finally:
        del env
