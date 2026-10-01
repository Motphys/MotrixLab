# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from motrix_rl_rslrl.torch.train import ppo
from motrix_rl_rslrl.torch.train.ppo import Trainer, add_runtime_config

from motrix_rl.config import CheckpointConfig, LoggingConfig


def test_add_runtime_config():
    cfg = {"max_iterations": 10}

    result = add_runtime_config(
        cfg,
        LoggingConfig(backend="wandb", interval=100),
        CheckpointConfig(interval=4),
    )

    assert result is cfg
    assert result["logger"] == "wandb"
    assert result["save_interval"] == 4


def test_add_runtime_config_disables_periodic_checkpoints():
    cfg = {"max_iterations": 10}

    result = add_runtime_config(
        cfg,
        LoggingConfig(backend="tensorboard", interval=100),
        CheckpointConfig(interval=0),
    )

    assert result["logger"] == "tensorboard"
    assert result["save_interval"] == 11


def test_add_runtime_config_rejects_unknown_logger():
    with pytest.raises(ValueError, match="logging backend"):
        add_runtime_config(
            {"max_iterations": 10},
            LoggingConfig(backend="unknown", interval=100),
            CheckpointConfig(interval=0),
        )


@pytest.mark.parametrize(
    ("entrypoint", "args", "num_envs", "mode"),
    (
        ("train", (), 2, "train"),
        ("play", ("policy.pt",), 3, "play"),
    ),
)
def test_trainer_sets_seed_before_creating_environment(monkeypatch, entrypoint, args, num_envs, mode):
    class StopAfterMakeEnv(Exception):
        pass

    calls = []
    trainer = object.__new__(Trainer)
    trainer._context = SimpleNamespace(seed=17, num_envs=2, play_num_envs=3)
    trainer._env_name = "demo"
    trainer._sim = None
    trainer._render = None
    trainer._set_seed = lambda seed: calls.append(("seed", seed))

    def make_env(env_name, **kwargs):
        calls.append(("make_env", env_name, kwargs["num_envs"], kwargs["mode"]))
        raise StopAfterMakeEnv

    monkeypatch.setattr(ppo.env_registry, "make", make_env)

    with pytest.raises(StopAfterMakeEnv):
        getattr(trainer, entrypoint)(*args)

    assert calls == [("seed", 17), ("make_env", "demo", num_envs, mode)]


@pytest.mark.parametrize("seed", (17, None))
def test_trainer_seed_covers_python_numpy_and_torch(monkeypatch, seed):
    calls = []
    trainer = object.__new__(Trainer)
    monkeypatch.setattr(ppo.random, "seed", lambda value: calls.append(("python", value)))
    monkeypatch.setattr(ppo.np.random, "seed", lambda value: calls.append(("numpy", value)))
    monkeypatch.setattr(ppo.torch, "manual_seed", lambda value: calls.append(("torch", value)))
    monkeypatch.setattr(ppo.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ppo.torch.cuda, "manual_seed_all", lambda value: calls.append(("cuda", value)))

    trainer._set_seed(seed)

    expected = [] if seed is None else [("python", seed), ("numpy", seed), ("torch", seed), ("cuda", seed)]
    assert calls == expected
