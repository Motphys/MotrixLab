# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from motrix_env_core.array.env import ArrayEnvState, NpObs
from motrix_env_core.direct.env import DirectEnv
from motrix_env_motrixsim.torch_env import TorchEnv, TorchEnvState, TorchObs

_TRAINERS = ["skrl.torch", "skrl.jax"]

_SIM_BACKENDS = ["np", "torch"]


_NUM_ENVS = 2


_OBS_DIM = 3


_ACT_DIM = 2


def _make_env(sim_backend: str):
    policy_space = gym.spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([-2.0, -1.0], dtype=np.float32),
        high=np.array([2.0, 3.0], dtype=np.float32),
    )
    if sim_backend == "np":
        env = Mock(spec=DirectEnv)
        state = ArrayEnvState(
            obs=NpObs(np.zeros((_NUM_ENVS, _OBS_DIM), dtype=np.float32)),
            reward=np.ones(_NUM_ENVS, dtype=np.float32),
            terminated=np.zeros(_NUM_ENVS, dtype=bool),
            truncated=np.zeros(_NUM_ENVS, dtype=bool),
            episode_steps=np.zeros(_NUM_ENVS, dtype=np.uint64),
        )
    else:
        env = Mock(spec=TorchEnv)
        env.device = (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        )
        state = TorchEnvState(
            data=None,
            obs=TorchObs(torch.zeros((_NUM_ENVS, _OBS_DIM), device=env.device)),
            reward=torch.ones(_NUM_ENVS, device=env.device),
            terminated=torch.zeros(_NUM_ENVS, dtype=torch.bool, device=env.device),
            truncated=torch.zeros(_NUM_ENVS, dtype=torch.bool, device=env.device),
            episode_steps=torch.zeros(_NUM_ENVS, dtype=torch.int64, device=env.device),
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


def _wrap_env(monkeypatch, trainer: str, sim_backend: str, env, renderer):
    pytest.importorskip("skrl")
    if trainer == "skrl.jax":
        pytest.importorskip("jax")
    package = {"skrl.torch": "motrix_rl_skrl.torch", "skrl.jax": "motrix_rl_skrl.jax"}[trainer]
    wrapper_module = importlib.import_module(f"{package}.wrap_{sim_backend}")
    monkeypatch.setattr(wrapper_module, "create_renderer", Mock(return_value=renderer))
    trainer_module = importlib.import_module(package)
    return trainer_module.wrap_env(env, render=object())


@pytest.mark.parametrize("trainer", _TRAINERS)
@pytest.mark.parametrize("sim_backend", _SIM_BACKENDS)
def test_rl_trainer_supports_sim_backend(monkeypatch, trainer: str, sim_backend: str) -> None:
    env = _make_env(sim_backend)
    renderer = Mock()
    wrapped = _wrap_env(monkeypatch, trainer, sim_backend, env, renderer)

    if trainer == "skrl.torch":
        obs, _ = wrapped.reset()
        result = wrapped.step(torch.zeros((_NUM_ENVS, _ACT_DIM), device=wrapped.device))
        assert all(tensor.device == wrapped.device for tensor in (obs, *result[:4]))
    elif trainer == "skrl.jax":
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        obs, _ = wrapped.reset()
        result = wrapped.step(jnp.zeros((_NUM_ENVS, _ACT_DIM)))
        assert all(isinstance(array, jax.Array) for array in (obs, *result[:4]))
        assert all(array.device == wrapped.device for array in (obs, *result[:4]))
    actions = env.step.call_args.args[0]
    assert isinstance(actions, np.ndarray if sim_backend == "np" else torch.Tensor)
    if sim_backend == "torch":
        assert actions.device == env.device

    wrapped.render("ignored", ignored=True)
    wrapped.close()
    renderer.render.assert_called_once_with()
    renderer.close.assert_called_once_with()
