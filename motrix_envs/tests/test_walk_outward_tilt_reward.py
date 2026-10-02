# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

from motrix_envs.locomotion.humanoid.walk_manager_mdp.rewards import (
    PenaltyOutwardTiltRewardCfg,
    _outward_tilt_penalty,
    penalty_outward_tilt_reward,
)


def test_outward_tilt_penalty_deadzone_and_recovery():
    threshold = 1.0 - np.cos(np.deg2rad(15.0))
    assert _outward_tilt_penalty.py_func(threshold, 0.2, 0.0, -0.98, 0.0, 1.0) == 0.0
    assert _outward_tilt_penalty.py_func(threshold, 0.2, 0.0, -0.5, 0.0, 1.0) > 0.0
    assert _outward_tilt_penalty.py_func(threshold, 0.2, 0.0, -0.5, 0.0, -1.0) == 0.0


@pytest.mark.parametrize("yaw", [0.0, 0.7, -1.9])
def test_outward_tilt_dispatch_transforms_world_omega_and_preserves_quaternion_sign(yaw):
    angle = 0.4
    # Rz(yaw) Ry(angle), XYZW. World angular velocity Rz(yaw) [0, rate, 0].
    q = np.array(
        [
            -np.sin(yaw / 2) * np.sin(angle / 2),
            np.cos(yaw / 2) * np.sin(angle / 2),
            np.sin(yaw / 2) * np.cos(angle / 2),
            np.cos(yaw / 2) * np.cos(angle / 2),
        ]
    )
    ctx = SimpleNamespace(commands={"walk": SimpleNamespace(penalty_scale=np.array([0.7]))})
    threshold = 1 - np.cos(0.2)
    omega = np.array([-np.sin(yaw), np.cos(yaw), 0.0]) * 0.6
    expected = ((1 - np.cos(angle)) - threshold) * np.sin(angle) * 0.6 * 0.7
    for sign in (1, -1):
        assert penalty_outward_tilt_reward(ctx, threshold, sign * q, omega) == pytest.approx(expected)
        assert penalty_outward_tilt_reward(ctx, threshold, sign * q, -omega) == pytest.approx(0.0)
    assert penalty_outward_tilt_reward(ctx, threshold, q, np.array([0.0, 0.0, 1.0])) == pytest.approx(0.0, abs=1e-15)


def test_outward_tilt_penalty_compiles_and_is_finite():
    value = _outward_tilt_penalty(0.1, 0.2, 0.1, -0.8, 0.3, 0.4)
    assert np.isfinite(value)


@pytest.mark.parametrize("angle", [0.0, -0.1, np.pi / 2, np.inf, np.nan])
def test_outward_tilt_cfg_rejects_invalid_deadzone(angle):
    with pytest.raises(ValueError, match="deadzone_angle"):
        PenaltyOutwardTiltRewardCfg(deadzone_angle=angle, weight=-1.0)


def test_outward_tilt_cfg_uses_quaternion_and_angular_velocity_queries():
    angle = 0.31
    cfg = PenaltyOutwardTiltRewardCfg(deadzone_angle=angle, weight=-1.0)
    body = SimpleNamespace(base_link_name="base")
    term = cfg(SimpleNamespace(model=SimpleNamespace(bodies={"robot": body})))
    assert term.args[0] == pytest.approx(1.0 - np.cos(angle), abs=1e-7)
    assert term.args[1].link == "base"
    assert term.args[2].link == "base"
