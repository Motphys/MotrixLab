# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.terrain import HeightFieldGrid
from motrix_envs.locomotion.humanoid.g1_stairs import make_g129dof_walk_mixed_cfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp import reset
from motrix_envs.locomotion.humanoid.walk_manager_mdp.reset import (
    _spawn_ground_height,
    _tile_spawn_ground_height,
)


def _grid(values):
    return HeightFieldGrid(
        heights=np.asarray(values, dtype=np.float32),
        origin=np.zeros(2, dtype=np.float32),
        spacing=np.ones(2, dtype=np.float32),
        z0=np.zeros(1, dtype=np.float32),
        constant=np.float32(0),
        enabled=True,
    )


def _lookup(grid, x, y):
    if not grid.enabled:
        return float(grid.constant)
    fx = min(max((x - grid.origin[0]) / grid.spacing[0], 0.0), grid.heights.shape[1] - 1e-6)
    fy = min(max((y - grid.origin[1]) / grid.spacing[1], 0.0), grid.heights.shape[0] - 1e-6)
    col, row = int(fx), int(fy)
    tx, ty = fx - col, fy - row
    top = grid.heights[row, col] * (1 - tx) + grid.heights[row, col + 1] * tx
    bottom = grid.heights[row + 1, col] * (1 - tx) + grid.heights[row + 1, col + 1] * tx
    return float(top * (1 - ty) + bottom * ty + grid.z0[0])


def test_tile_clearance_disabled_preserves_legacy_center_patch(monkeypatch):
    monkeypatch.setattr(reset, "heightfield_lookup", _lookup)
    grid = _grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert _tile_spawn_ground_height.py_func(grid, 1.0, 1.0, 0.0) == 0.0
    assert _spawn_ground_height.py_func(grid, 1.0, 1.0) == 0.0


def test_tile_clearance_uses_footprint_highest_point(monkeypatch):
    monkeypatch.setattr(reset, "heightfield_lookup", _lookup)
    grid = _grid([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
    assert _tile_spawn_ground_height.py_func(grid, 1.0, 1.0, 0.2) > _tile_spawn_ground_height.py_func(
        grid, 1.0, 1.0, 0.0
    )


def test_tile_clearance_tracks_bilinear_slope(monkeypatch):
    monkeypatch.setattr(reset, "heightfield_lookup", _lookup)
    grid = _grid([[0, 0, 0], [0, 1, 1], [0, 2, 2]])
    center = _tile_spawn_ground_height.py_func(grid, 1.0, 1.0, 0.0)
    edge = _tile_spawn_ground_height.py_func(grid, 1.0, 1.0, 0.2)
    assert edge >= center


def test_tile_clearance_compiles_and_lifts_inverted_slope_spawn():
    cfg = make_g129dof_walk_mixed_cfg()
    cfg.commands.walk.terrain_sampling_enabled = False
    cfg.sim_reset.humanoid_state.tile_ground_height_radius = 0.2
    env = ManagerEnv(cfg, num_envs=4, seed=71)
    try:
        walk = env.command_terms["walk"]
        walk.terrain_levels[:] = 6
        walk.terrain_cols[:] = 7
        state = env.init_state()
        params = env.sim_reset_terms["humanoid_state"].args[0]
        positions = env.sim_data["terrain_curriculum_base_pos"]
        for position in positions:
            x, y = position[:2]
            ground = max(
                _lookup(params.heightfield, px, py)
                for px, py in ((x, y), (x - 0.2, y), (x + 0.2, y), (x, y - 0.2), (x, y + 0.2))
            )
            assert position[2] >= params.init_pose[2] + ground - 1e-5
        assert np.all(np.isfinite(state.obs.policy))
    finally:
        del env
