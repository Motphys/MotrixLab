# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Heading numerics and fused command/reset lifecycle behavior."""

import math

import numpy as np
import pytest

from motrix_env_core.manager import ManagerEnv
from motrix_env_core.numba.math.quaternion import from_euler
from motrix_envs.locomotion.humanoid.g1_stairs import make_g129dof_walk_mixed_cfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.command import _apply_heading_command


@pytest.mark.parametrize(
    ("yaw", "target", "stiffness", "low", "high", "expected"),
    [
        (0.0, 0.4, 0.5, -1.0, 1.0, 0.2),
        (math.pi / 2, 0.0, 1.0, -2.0, 2.0, -math.pi / 2),
        (-math.pi / 2, 0.0, 1.0, -2.0, 2.0, math.pi / 2),
        (math.pi - 0.1, -math.pi + 0.1, 1.0, -1.0, 1.0, 0.2),
        (-math.pi + 0.1, math.pi - 0.1, 1.0, -1.0, 1.0, -0.2),
        (0.0, 2.0, 2.0, -0.3, 0.7, 0.7),
        (0.0, -2.0, 2.0, -0.3, 0.7, -0.3),
    ],
)
def test_heading_shortest_turn_gain_and_asymmetric_clamp(yaw, target, stiffness, low, high, expected):
    command = np.array([0.6, -0.2, 0.9], dtype=np.float32)
    # Nonzero roll and pitch distinguish projected-forward yaw from quaternion z alone.
    quaternion = from_euler(0.3, -0.2, yaw)
    _apply_heading_command(command, target, True, quaternion, stiffness, low, high)
    np.testing.assert_array_equal(command[:2], np.array([0.6, -0.2], dtype=np.float32))
    assert command[2] == pytest.approx(expected, abs=2e-6)


def test_positive_pi_tie_wraps_to_negative_pi():
    command = np.zeros(3, dtype=np.float32)
    quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    _apply_heading_command(command, math.pi, True, quaternion, 1.0, -4.0, 4.0)
    assert command[2] == pytest.approx(-math.pi)


def test_nonheading_preserves_sampled_command():
    command = np.array([0.6, -0.2, 0.9], dtype=np.float32)
    before = command.copy()
    _apply_heading_command(command, 1.2, False, from_euler(0.0, 0.0, 0.5), 1.0, -0.1, 0.1)
    np.testing.assert_array_equal(command, before)


def _assert_heading_matches_spawn(env):
    walk = env.command_terms["walk"]
    quaternion = env.sim_data["terrain_curriculum_base_quat"]
    x, y, z, w = quaternion.T
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    error = (walk.heading_target[:, 0] - yaw + math.pi) % (2.0 * math.pi) - math.pi
    expected = np.clip(
        float(walk.heading_control_stiffness) * error,
        walk.vel_limit_low[2],
        walk.vel_limit_high[2],
    )
    heading = walk.is_heading[:, 0].astype(bool)
    np.testing.assert_allclose(walk.command[heading, 2], expected[heading], atol=2e-6)
    standing = np.all(walk.command == 0.0, axis=1)
    assert not np.any(heading & standing)
    np.testing.assert_allclose(walk.phase[standing], math.pi, atol=2e-6)
    assert np.ptp(yaw) > 1.0


def test_seeded_fused_heading_reset_modes_partial_reset_and_countdown():
    cfg = make_g129dof_walk_mixed_cfg()
    # These are test inputs, not assertions about tunable task defaults.
    cfg.commands.walk.heading_command = True
    cfg.commands.walk.heading_prob = 0.3
    cfg.commands.walk.stand_prob = 0.1
    cfg.commands.walk.resampling_time_range = (0.06, 0.14)
    cfg.commands.walk.curriculum_enabled = False
    cfg.sim_reset.humanoid_state.spawn_yaw_range = math.pi
    env = ManagerEnv(cfg, num_envs=128, seed=9182)
    try:
        env.init_state()
        walk = env.command_terms["walk"]
        assert walk.command.shape == (128, 3)
        assert walk.resample_steps_left.shape == (128, 1)
        backing = walk.command
        counts = np.zeros(3, dtype=np.int64)
        for _ in range(8):
            _assert_heading_matches_spawn(env)
            heading = walk.is_heading[:, 0].astype(bool)
            standing = np.all(walk.command == 0.0, axis=1)
            # One categorical draw: 30% heading, 10% standing, 60% raw yaw.
            # This deliberately differs from mjlab's independent flag sampling.
            counts += [heading.sum(), standing.sum(), (~heading & ~standing).sum()]
            assert np.all((walk.resample_steps_left >= 3) & (walk.resample_steps_left <= 7))
            env.reset(np.arange(128, dtype=np.int64))
        np.testing.assert_allclose(counts / counts.sum(), [0.3, 0.1, 0.6], atol=0.05)
        assert np.unique(walk.resample_steps_left).size > 1

        selected = np.array([2, 7, 19], dtype=np.int64)
        untouched = np.ones(128, dtype=bool)
        untouched[selected] = False
        arrays = (walk.command, walk.heading_target, walk.is_heading, walk.resample_steps_left, walk.steps)
        before = [array.copy() for array in arrays]
        env.reset(selected)
        for array, previous in zip(arrays, before, strict=True):
            np.testing.assert_array_equal(array[untouched], previous[untouched])
        _assert_heading_matches_spawn(env)
        assert walk.command is backing

        # Forced expiry exercises derived heading commands in next-action observations.
        walk.resample_steps_left[:] = 1
        env.step(np.zeros((128, env.num_actuators), dtype=np.float32))
        _assert_heading_matches_spawn(env)
        assert np.all((walk.resample_steps_left >= 3) & (walk.resample_steps_left <= 7))
        assert walk.command is backing
    finally:
        del env
