# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Numerical contracts for terrain-relative foot orientation."""

import math
from dataclasses import replace

import numpy as np

from motrix_env_core.manager import ManagerEnv
from motrix_env_core.numba.math.quaternion import from_euler
from motrix_envs.locomotion.humanoid.g1_stairs import make_g129dof_walk_mixed_cfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.rewards import (
    PenaltyFeetOriRewardCfg,
    _terrain_relative_foot_ori_error_from_samples,
)


def _error(slope_x, slope_y, quat):
    distance = np.float32(0.15)
    return _terrain_relative_foot_ori_error_from_samples(
        quat,
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        np.float32(-slope_x * distance),
        np.float32(slope_x * distance),
        np.float32(-slope_y * distance),
        np.float32(slope_y * distance),
        distance,
    )


def test_terrain_relative_orientation_flat_and_disabled():
    identity = from_euler(0.0, 0.0, 0.0)
    assert _error(0.0, 0.0, identity) == 0.0
    assert PenaltyFeetOriRewardCfg(weight=-1.0).terrain_relative is False
    with np.testing.assert_raises(ValueError):
        PenaltyFeetOriRewardCfg(weight=-1.0, terrain_relative=True, normal_sample_distance=0.0)


def test_terrain_relative_orientation_planar_slope():
    slope = 0.5
    matching = from_euler(0.0, -math.atan(slope), 0.0)
    wrong = from_euler(0.0, 0.0, 0.0)

    assert _error(slope, 0.0, matching) < 1e-6
    assert _error(slope, 0.0, wrong) > 0.1


def test_enabled_terrain_relative_reward_compiles_and_steps():
    cfg = make_g129dof_walk_mixed_cfg()
    cfg.rewards.penalty_feet_ori = replace(
        cfg.rewards.penalty_feet_ori,
        terrain_relative=True,
        weight=-1.0,
        ground_geom="floor",
    )
    env = ManagerEnv(cfg, num_envs=2)
    try:
        state = env.init_state()
        assert np.all(np.isfinite(state.obs.policy))
        next_state = env.step(np.zeros((2, env.num_actuators), dtype=np.float32))
        assert np.all(np.isfinite(next_state.obs.policy))
    finally:
        del env
