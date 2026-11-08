# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from motrix_env_core.array.env import ArrayEnvState, EnvObs
from motrix_env_core.direct.env import DirectEnv

_NUM_ENVS = 2


_OBS_DIM = 3


_ACT_DIM = 2


def _make_env():
    policy_space = gym.spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([-2.0, -1.0], dtype=np.float32),
        high=np.array([2.0, 3.0], dtype=np.float32),
    )
    env = Mock(spec=DirectEnv)
    state = ArrayEnvState(
        obs=EnvObs(torch.zeros((_NUM_ENVS, _OBS_DIM), dtype=torch.float32)),
        reward=torch.ones(_NUM_ENVS, dtype=torch.float32),
        terminated=torch.zeros(_NUM_ENVS, dtype=torch.bool),
        truncated=torch.zeros(_NUM_ENVS, dtype=torch.bool),
        episode_steps=torch.zeros(_NUM_ENVS, dtype=torch.int64),
    )
    env.cfg = SimpleNamespace(max_episode_steps=100)
    env.num_envs = _NUM_ENVS
    env.action_space = action_space
    env.policy_observation_space = policy_space
    env.value_observation_space = policy_space
    env.has_value_observation = False
    env.state = state
    env.init_state.return_value = state
    env.step.return_value = state
    return env


def _wrap_env(monkeypatch, env, renderer):
    pytest.importorskip("rsl_rl")
    wrapper_module = importlib.import_module("motrix_rl_rslrl.torch.wrap")
    monkeypatch.setattr(wrapper_module, "create_renderer", Mock(return_value=renderer))
    trainer_module = importlib.import_module("motrix_rl_rslrl.torch")
    return trainer_module.wrap_env(env, torch.device("cpu"), render=object())


def test_rl_trainer_supports_sim_backend(monkeypatch) -> None:
    env = _make_env()
    renderer = Mock()
    wrapped = _wrap_env(monkeypatch, env, renderer)

    obs, _ = wrapped.reset()
    next_obs, rewards, dones, extras = wrapped.step(torch.zeros((_NUM_ENVS, _ACT_DIM)))
    assert all(tensor.device.type == "cpu" for tensor in (obs["policy"], next_obs["policy"], rewards, dones))
    assert extras["time_outs"].device.type == "cpu"

    actions = env.step.call_args.args[0]
    assert isinstance(actions, torch.Tensor)
    assert actions.device.type == "cpu"

    wrapped.render("ignored", ignored=True)
    wrapped.close()
    renderer.render.assert_called_once_with()
    renderer.close.assert_called_once_with()
