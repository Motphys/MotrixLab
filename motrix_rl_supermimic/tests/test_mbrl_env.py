# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle and delegation contracts of MbrlEnvBase on a toy task.

The toy is a damped single joint tracking a per-frame target: canonical
state ``(theta, theta_dot)``, velocity-controlled through a fake backend.
It proves the provider builds and tests without any task package or robot
constants.
"""

import gymnasium.spaces as gym_spaces
import pytest
import torch
from motrix_rl_supermimic.toy import ToyEnv, ToyTaskMDP
from motrix_rl_supermimic.toy import register as register_toy

from motrix_rl_supermimic import StateSpec

register_toy()


@pytest.fixture()
def env() -> ToyEnv:
    torch.manual_seed(0)
    env = ToyEnv(num_envs=3)
    env.init_state()
    return env


def test_spaces_follow_spec(env: ToyEnv):
    assert env.observation_space.shape == (4,)
    assert env.action_space.shape == (1,)
    assert isinstance(env.action_space, gym_spaces.Box)


def test_step_delegates_reward_and_advances_frame(env: ToyEnv):
    action = torch.full((3, 1), 0.1)
    state = env.step(action)

    assert env.frame.tolist() == [1, 1, 1]  # advanced by the toy refresh hook
    # Same inputs the base consumed: post-physics state, advanced frame,
    # current action, zero action history on the first step.
    expected = env.task_mdp.reward(env.state(), env.frame, action, torch.zeros(3, 1))
    torch.testing.assert_close(state.reward, expected)
    torch.testing.assert_close(env.prev_action, action)
    assert state.obs.policy.shape == (3, 4)
    assert not bool(state.terminated.any())


def test_observation_uses_rotated_action_history(env: ToyEnv):
    env.step(torch.full((3, 1), 0.3))
    state = env.step(torch.full((3, 1), -0.2))

    manual = env.task_mdp.observation(env.state(), env.prev_action, env.context, env.frame)
    torch.testing.assert_close(state.obs.policy, manual, rtol=1e-3, atol=1e-2)


def test_first_step_prev_action_is_zero(env: ToyEnv):
    action = torch.full((3, 1), 0.5)
    state = env.step(action)
    expected = env.task_mdp.reward(env.state(), env.frame, action, torch.zeros(3, 1))
    torch.testing.assert_close(state.reward, expected)


def test_termination_and_autoreset_clear_action_history(env: ToyEnv):
    env.sim.theta[:] = 2.5  # over the hard-termination threshold
    state = env.step(torch.zeros(3, 1))
    assert bool(state.terminated.all())

    env.sim.theta[:] = 2.5
    state = env.step(torch.full((3, 1), 0.7))
    assert bool(state.terminated.all())
    # auto-reset already ran: action history rows cleared, frame restarted
    assert env.prev_action.abs().sum().item() == 0.0
    assert env.frame.tolist() == [0, 0, 0]
    assert env.sim.reset_rows  # backend received the reset writes


def test_soft_termination_is_available_but_not_env_termination(env: ToyEnv):
    env.sim.theta[:] = 1.6
    s = env.state()
    soft = env.task_mdp.soft_termination(s, env.frame)
    assert bool((soft > 0.5).all())  # deep in the death regime
    assert not bool(env._termination(s).any())  # hard threshold still not hit


def test_state_spec_groups_partition_and_validate():
    spec = ToyTaskMDP.spec
    assert [w for w, _ in spec.groups()] == [3, 4, 3, 3, 1, 1]
    with pytest.raises(ValueError, match="state_groups"):
        StateSpec(
            state_dim=3,
            action_dim=1,
            context_dim=0,
            observation_dim=1,
            critic_observation_dim=1,
            command_dim=1,
            state_groups=((1, 1.0),),
        )
