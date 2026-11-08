# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for humanoid-walk reset-time dynamics randomization."""

from contextlib import nullcontext
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.action import JointPositionActionCfg, JointPositionActionState, JointPositionActionTerm
from motrix_envs.locomotion.humanoid.g1 import make_g129dof_walk_flat_cfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.randomization import WalkRandomizationCfg


def test_walk_randomization_cfg_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="kp_scale_range"):
        WalkRandomizationCfg(kp_scale_range=(1.1, 0.9))
    with pytest.raises(ValueError, match="sliding_friction_range"):
        WalkRandomizationCfg(sliding_friction_range=(0.0, 1.25))
    with pytest.raises(ValueError, match="base_com_offset_noise"):
        WalkRandomizationCfg(base_com_offset_noise=(0.05, -0.05, 0.05))
    with pytest.raises(ValueError, match="base_mass_offset_range"):
        WalkRandomizationCfg(base_mass_offset_range=(3.0, -1.0))


def _tiny_delay_manager(num_envs: int = 2, num_actuators: int = 3, lo: int = 0, hi: int = 2):
    action_state = JointPositionActionState(
        action_queue=torch.from_numpy(np.zeros((num_envs, max(hi + 1, 2), num_actuators), dtype=np.float32)),
        default_angles=torch.from_numpy(np.zeros(num_actuators, dtype=np.float32)),
        joint_lower=torch.from_numpy(np.zeros(num_actuators, dtype=np.float32)),
        joint_upper=torch.from_numpy(np.ones(num_actuators, dtype=np.float32)),
        action_scales=torch.from_numpy(np.ones(num_actuators, dtype=np.float32)),
        delay_steps=torch.from_numpy(np.full(num_envs, hi, dtype=np.int64)),
        action_ptr=torch.from_numpy(np.zeros(1, dtype=np.int64)),
        delay_lo=lo,
        delay_hi=hi,
    )
    term = JointPositionActionTerm(
        gym.spaces.Box(-np.inf, np.inf, shape=(num_actuators,), dtype=np.float32),
        action_state,
    )
    # Exercise real manager history/lifecycle code, replacing only simulator I/O
    # and unrelated command resets. No history algorithm is duplicated here.
    env = ManagerEnv.__new__(ManagerEnv)
    env._num_envs = num_envs
    env._action_space = term.action_space
    env._action_terms = {"joint_position": term}
    env._action_slices = {"joint_position": slice(0, num_actuators)}
    env._action_actuators = {"joint_position": ()}
    controls = torch.empty((num_envs, num_actuators), dtype=torch.float32)
    env._action_writes = SimpleNamespace(buffer=lambda name: controls, execute=lambda: None)
    env._command_terms = {}
    env._state = SimpleNamespace(terminated=torch.zeros(num_envs, dtype=torch.bool))
    env._compiled_manager_program = SimpleNamespace(
        context=SimpleNamespace(
            reset_episode=torch.zeros((num_envs, 1), dtype=torch.bool),
            terminated=env._state.terminated,
        )
    )
    env.perf = SimpleNamespace(scope=lambda name: nullcontext())
    env._reset_sim_rows = lambda env_ids: None
    return env, term.state, controls


def test_action_delay_applies_delayed_action():
    env, action_state, controls = _tiny_delay_manager(lo=1, hi=2)
    action_state.delay_steps[:] = 2
    actions = torch.full(controls.shape, 1.0, dtype=torch.float32)
    env.apply_action(actions, env._state)
    # Only the raw action exists so far; unwritten history remains zero.
    assert bool((controls == 0.0).all())
    assert bool((action_state.action_queue[:, int(action_state.action_ptr[0])] == 1.0).all())
    for value in (2.0, 3.0, 4.0):
        env.apply_action(torch.full_like(actions, value), env._state)
    # Two-step delay: the applied action is the one from two steps ago.
    assert bool((controls == 2.0).all())


def test_action_delay_disabled_passes_actions_through_and_keeps_history():
    env, action_state, controls = _tiny_delay_manager(lo=0, hi=0)
    actions = torch.full(controls.shape, 0.5, dtype=torch.float32)
    env.apply_action(actions, env._state)
    assert bool((controls == 0.5).all())
    ptr = int(action_state.action_ptr[0])
    assert bool((action_state.action_queue[:, ptr] == 0.5).all())
    assert bool((action_state.action_queue[:, (ptr - 1) % action_state.action_queue.shape[1]] == 0.0).all())


def test_action_queue_wraps_and_selects_delay_per_lane():
    env, action_state, controls = _tiny_delay_manager(num_envs=2, lo=0, hi=2)
    action_state.delay_steps[:] = torch.tensor([0, 2])
    width = action_state.action_queue.shape[1]
    for value in range(1, width + 2):
        actions = torch.full(controls.shape, value, dtype=torch.float32)
        env.apply_action(actions, env._state)
        assert int(action_state.action_ptr[0]) == value % width
        assert controls[0, 0] == value
        expected_lane_1 = 0.0 if value <= 2 else value - 2
        assert controls[1, 0] == expected_lane_1


def test_action_reset_clears_only_selected_lane_history_and_keeps_pointer():
    env, action_state, controls = _tiny_delay_manager(num_envs=2, lo=0, hi=2)
    for value in (1.0, 2.0, 3.0):
        env.apply_action(torch.full(controls.shape, value, dtype=torch.float32), env._state)
    ptr_before_reset = action_state.action_ptr.clone()
    other_lane_history = action_state.action_queue[1].clone()
    env.reset(torch.tensor([0], dtype=torch.long))
    assert bool((action_state.action_queue[0] == 0.0).all())
    torch.testing.assert_close(action_state.action_queue[1], other_lane_history, rtol=0, atol=0)
    torch.testing.assert_close(action_state.action_ptr, ptr_before_reset, rtol=0, atol=0)


def test_action_delay_reset_resamples_within_range():
    env, action_state, _ = _tiny_delay_manager(lo=0, hi=2)
    env.reset(torch.arange(action_state.action_queue.shape[0], dtype=torch.long))
    assert bool(((action_state.delay_steps >= 0) & (action_state.delay_steps <= 2)).all())


def test_joint_position_term_methods_manage_history_and_specific_state():
    env, action_state, controls = _tiny_delay_manager(lo=0, hi=2)
    term = env._action_terms["joint_position"]
    for value in (1.0, 2.0, 3.0):
        env.apply_action(torch.full(controls.shape, value, dtype=torch.float32), env._state)
    pointer = int(action_state.action_ptr[0])
    term.process(torch.full(controls.shape, 4.0, dtype=torch.float32))
    assert int(action_state.action_ptr[0]) == (pointer + 1) % action_state.action_queue.shape[1]
    torch.testing.assert_close(action_state.current(), torch.full_like(action_state.current(), 4.0))
    term.reset(torch.tensor([0], dtype=torch.long))
    assert bool((action_state.action_queue[0] == 0.0).all())
    assert bool((action_state.action_queue[1, 0] == 3.0).all())


def test_flat_cfg_randomization_settings_are_wired():
    cfg = make_g129dof_walk_flat_cfg()
    randomization = cfg.sim_reset.humanoid_state.randomization
    assert randomization.enabled
    assert randomization.kp_scale_range[0] < randomization.kp_scale_range[1]
    assert randomization.sliding_friction_range is not None
    assert randomization.sliding_friction_range[0] < randomization.sliding_friction_range[1]
    assert cfg.actions.joint_position.action_delay_steps[0] < cfg.actions.joint_position.action_delay_steps[1]
    assert cfg.commands.walk.gait_period_randomization_width > 0.0
    for key in ("randomize_actuator_kd", "randomize_friction_default", "randomize_link_masses", "randomize_base_com"):
        assert key in cfg.queries.model


def test_disabled_action_delay_has_zero_cfg_default():
    assert JointPositionActionCfg().action_delay_steps == (0, 0)


def test_walk_env_randomized_reset_samples_within_ranges():
    cfg = make_g129dof_walk_flat_cfg()
    env = ManagerEnv(cfg, num_envs=2)
    try:
        env.step(torch.zeros((2, env.num_actuators), dtype=torch.float32))
        env.reset(torch.arange(2, dtype=torch.long))
        walk = env.command_terms["walk"]
        # Phase increments invert the configured gait-period interval.
        command_cfg = cfg.commands.walk
        min_period = max(command_cfg.gait_period - command_cfg.gait_period_randomization_width, 0.1)
        max_period = max(command_cfg.gait_period + command_cfg.gait_period_randomization_width, 0.1)
        min_phase_step = 2.0 * np.pi * cfg.ctrl_dt / max_period
        max_phase_step = 2.0 * np.pi * cfg.ctrl_dt / min_period
        assert bool((walk.phase_step.numpy() >= min_phase_step - 1e-6).all())
        assert bool((walk.phase_step.numpy() <= max_phase_step + 1e-6).all())
        action_state = env.action_terms["joint_position"].state
        delay_lo, delay_hi = cfg.actions.joint_position.action_delay_steps
        assert bool(((action_state.delay_steps >= delay_lo) & (action_state.delay_steps <= delay_hi)).all())
    finally:
        del env
