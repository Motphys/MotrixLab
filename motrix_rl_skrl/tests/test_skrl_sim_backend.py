# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from motrix_env_core.array.env import ArrayEnvState, EnvObs
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.renderer import RecordConfig, RenderConfig

_TRAINERS = ["skrl.torch", "skrl.jax"]


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


def _wrap_env(monkeypatch, trainer: str, env, renderer):
    pytest.importorskip("skrl")
    if trainer == "skrl.jax":
        pytest.importorskip("jax")
    package = {"skrl.torch": "motrix_rl_skrl.torch", "skrl.jax": "motrix_rl_skrl.jax"}[trainer]
    wrapper_module = importlib.import_module(f"{package}.wrap")
    create_renderer = Mock(return_value=renderer)
    monkeypatch.setattr(wrapper_module, "create_renderer", create_renderer)
    trainer_module = importlib.import_module(package)
    recording = RenderConfig(headless=True, record=RecordConfig(path=Path("video.mp4"), fps=30, num_frames=60))
    wrapped = trainer_module.wrap_env(env, render=recording)
    create_renderer.assert_called_once_with(env, recording)
    return wrapped


@pytest.mark.parametrize("trainer", _TRAINERS)
def test_rl_trainer_supports_sim_backend(monkeypatch, trainer: str) -> None:
    env = _make_env()
    renderer = Mock()
    wrapped = _wrap_env(monkeypatch, trainer, env, renderer)

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
    assert isinstance(actions, torch.Tensor)
    assert actions.device.type == "cpu"

    wrapped.render("ignored", ignored=True)
    wrapped.close()
    renderer.render.assert_called_once_with()
    renderer.close.assert_called_once_with()


@pytest.mark.parametrize("trainer", _TRAINERS)
def test_provider_passes_render_init_kwargs_to_registry(monkeypatch, trainer):
    pytest.importorskip("skrl")
    backend = trainer.split(".")[1]
    if backend == "jax":
        pytest.importorskip("jax")
    module = importlib.import_module(f"motrix_rl_skrl.{backend}.train.ppo")
    recording = RenderConfig(
        headless=True,
        record=RecordConfig(width=640, height=480, path=Path("video.mp4"), fps=30, num_frames=60),
    )
    instance = SimpleNamespace(
        _env_name="cartpole", _sim="test-sim", _context=SimpleNamespace(seed=7), _render=recording
    )
    make = Mock()
    monkeypatch.setattr(module.env_registry, "make", make)

    result = module.Trainer._make_env(instance, 2, mode="play")

    make.assert_called_once_with(
        "cartpole",
        num_envs=2,
        mode="play",
        sim="test-sim",
        seed=7,
        headless=True,
        record=recording.record,
    )
    assert result is make.return_value
