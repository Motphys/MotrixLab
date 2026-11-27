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
from motrix_env_core.observations import ObservationGroupSpec, ObservationSpec
from motrix_env_core.renderer import RecordConfig, RenderConfig

_TRAINERS = ["skrl.torch", "skrl.jax"]


_NUM_ENVS = 2


_OBS_DIM = 3


_ACT_DIM = 2


def _make_env(history_length: int = 1, value_dim: int | None = None) -> Mock:
    policy_space = gym.spaces.Box(-np.inf, np.inf, (_OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([-2.0, -1.0], dtype=np.float32),
        high=np.array([2.0, 3.0], dtype=np.float32),
    )
    groups = {"policy": ObservationGroupSpec(policy_space, history_length)}
    observations = {"policy": torch.zeros((_NUM_ENVS, _OBS_DIM * history_length), dtype=torch.float32)}
    if value_dim is not None:
        groups["value"] = ObservationGroupSpec(
            gym.spaces.Box(-np.inf, np.inf, (value_dim,), dtype=np.float32), history_length
        )
        observations["value"] = torch.zeros((_NUM_ENVS, value_dim * history_length), dtype=torch.float32)
    observation_spec = ObservationSpec(groups)
    env = Mock(spec=DirectEnv)
    env.device = torch.device("cpu")
    state = ArrayEnvState(
        obs=EnvObs(observations),
        reward=torch.ones(_NUM_ENVS, dtype=torch.float32),
        terminated=torch.zeros(_NUM_ENVS, dtype=torch.bool),
        truncated=torch.zeros(_NUM_ENVS, dtype=torch.bool),
        episode_steps=torch.zeros(_NUM_ENVS, dtype=torch.int64),
    )
    env.cfg = SimpleNamespace(max_episode_steps=100)
    env.num_envs = _NUM_ENVS
    env.action_space = action_space
    env.observation_spec = observation_spec
    env.state = state
    env.initialize.return_value = state
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
@pytest.mark.parametrize(("history_length", "value_dim"), [(1, None), (2, 5)])
def test_rl_trainer_supports_sim_backend(monkeypatch, trainer: str, history_length: int, value_dim: int | None) -> None:
    env = _make_env(history_length, value_dim)
    renderer = Mock()
    wrapped = _wrap_env(monkeypatch, trainer, env, renderer)
    assert wrapped.observation_space.shape == (_OBS_DIM * history_length,)
    assert wrapped.state_space.shape == ((value_dim or _OBS_DIM) * history_length,)
    assert wrapped.state().shape == (_NUM_ENVS, (value_dim or _OBS_DIM) * history_length)

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
        _env_name="cartpole",
        _sim="test-sim",
        _context=SimpleNamespace(run=None, sim_device=None, seed=7),
        _render=recording,
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
