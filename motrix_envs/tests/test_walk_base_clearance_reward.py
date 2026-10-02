# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from motrix_env_core.mdp.terrain import HeightFieldGrid, heightfield_lookup
from motrix_envs.locomotion.humanoid.walk_manager_mdp import rewards
from motrix_envs.locomotion.humanoid.walk_manager_mdp.rewards import (
    PenaltyBaseClearanceRewardCfg,
    _squared_hinge,
    penalty_base_clearance_reward,
)


def _grid(values, *, origin=(0.0, 0.0), spacing=(1.0, 1.0), z0=0.0):
    return HeightFieldGrid(
        heights=np.asarray(values, dtype=np.float32),
        origin=np.asarray(origin, dtype=np.float32),
        spacing=np.asarray(spacing, dtype=np.float32),
        z0=np.asarray([z0], dtype=np.float32),
        constant=np.float32(0.0),
        enabled=True,
    )


def _ctx(scale):
    return SimpleNamespace(commands={"walk": SimpleNamespace(penalty_scale=np.asarray([scale], dtype=np.float32))})


def test_base_clearance_reward_matches_squared_hinge_on_ramp(monkeypatch):
    # The 1 m x 1 m grid ramps in x; at x=.25 the ground is .25 m.
    monkeypatch.setattr(rewards, "heightfield_lookup", heightfield_lookup.py_func)
    grid = _grid([[0.0, 1.0], [0.0, 1.0]])
    reward = penalty_base_clearance_reward(
        _ctx(9.0), np.float32(0.5), grid, np.asarray([0.25, 0.4, 0.55], dtype=np.float32), "walk"
    )
    expected = (0.5 - (0.55 - 0.25)) ** 2 * 9.0
    assert reward == pytest.approx(expected)


def test_base_clearance_reward_zero_at_or_above_minimum(monkeypatch):
    monkeypatch.setattr(rewards, "heightfield_lookup", heightfield_lookup.py_func)
    grid = _grid([[0.0, 0.0], [0.0, 0.0]])
    for base_z in (0.5, 0.8):
        reward = penalty_base_clearance_reward(
            _ctx(1.0), np.float32(0.5), grid, np.asarray([0.2, 0.3, base_z], dtype=np.float32), "walk"
        )
        assert reward == pytest.approx(0.0)


def test_clearance_reward_uses_grid_world_coordinates_and_datum(monkeypatch):
    monkeypatch.setattr(rewards, "heightfield_lookup", heightfield_lookup.py_func)
    grid = _grid([[0.0, 1.0], [2.0, 3.0]], origin=(-4.0, 8.0), spacing=(2.0, 4.0), z0=3.0)
    # Halfway in both axes: ground=3+1.5; clearance=.3 regardless of datum.
    value = penalty_base_clearance_reward(
        _ctx(0.5), np.float32(0.5), grid, np.asarray([-3.0, 10.0, 4.8], dtype=np.float32), "walk"
    )
    assert value == pytest.approx(0.2**2 * 0.5, abs=1e-7)


def test_clearance_reward_flat_ground_uses_constant(monkeypatch):
    monkeypatch.setattr(rewards, "heightfield_lookup", heightfield_lookup.py_func)
    grid = HeightFieldGrid(
        heights=np.zeros((1, 1), dtype=np.float32),
        origin=np.zeros(2, dtype=np.float32),
        spacing=np.ones(2, dtype=np.float32),
        z0=np.zeros(1, dtype=np.float32),
        constant=np.float32(2.0),
        enabled=False,
    )
    value = penalty_base_clearance_reward(
        _ctx(1.0), np.float32(0.6), grid, np.asarray([30.0, -10.0, 2.4], dtype=np.float32), "walk"
    )
    assert value == pytest.approx(0.2**2, abs=1e-7)


def test_squared_hinge_compiles_and_has_expected_boundary():
    assert _squared_hinge.py_func(-1.0) == 0.0
    assert _squared_hinge.py_func(0.0) == 0.0
    assert _squared_hinge.py_func(0.3) == pytest.approx(0.09)
    assert _squared_hinge(0.3) == pytest.approx(0.09)


@pytest.mark.parametrize("minimum_height", [0.0, -0.1, np.inf, np.nan])
def test_base_clearance_cfg_rejects_invalid_minimum_height(minimum_height):
    with pytest.raises(ValueError, match="minimum_height"):
        PenaltyBaseClearanceRewardCfg(minimum_height=minimum_height, weight=-100.0)
