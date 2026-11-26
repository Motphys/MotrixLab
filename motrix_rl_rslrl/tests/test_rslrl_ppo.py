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

    expected = set() if seed is None else {("python", seed), ("numpy", seed), ("torch", seed), ("cuda", seed)}
    assert set(calls) == expected


@pytest.mark.parametrize(
    ("resume_from", "loaded_iteration", "expected_learning_iterations"),
    (
        (None, None, 3000),
        ("checkpoint.pt", 2000, 1000),
        ("checkpoint.pt", 3000, 0),
    ),
)
def test_trainer_resumes_rslrl_training_state(
    monkeypatch, tmp_path, resume_from, loaded_iteration, expected_learning_iterations
):
    trainer = object.__new__(Trainer)
    trainer._context = SimpleNamespace(
        seed=17,
        num_envs=2,
        checkpoint_format="pt",
        run_dir=tmp_path,
        resume_from=resume_from,
    )
    trainer._env_name = "demo"
    trainer._sim = None
    trainer._render = None
    trainer._set_seed = lambda seed: None
    trainer._create_rslrl_config = lambda: {"max_iterations": 3000}

    calls = []

    class FakeRunner:
        current_learning_iteration = 0 if loaded_iteration is None else loaded_iteration

        def load(self, path):
            calls.append(("load", path))
            self.current_learning_iteration = loaded_iteration

        def learn(self, num_learning_iterations):
            calls.append(("learn", num_learning_iterations))

        def save(self, path):
            calls.append(("save", path))

    fake_runner = FakeRunner()
    monkeypatch.setattr(ppo, "resolve_default_device", lambda: "cpu")
    monkeypatch.setattr(ppo.env_registry, "make", lambda *args, **kwargs: object())
    monkeypatch.setattr(ppo, "wrap_env", lambda env, device, render: object())
    monkeypatch.setattr(ppo, "OnPolicyRunner", lambda *args, **kwargs: fake_runner)
    monkeypatch.setattr(
        ppo.checkpoints, "final_checkpoint_path", lambda *args, **kwargs: tmp_path / "checkpoints" / "latest.pt"
    )
    monkeypatch.setattr(ppo.checkpoints, "record_checkpoint_artifact", lambda *args, **kwargs: None)

    trainer.train()

    expected = [("learn", expected_learning_iterations), ("save", str(tmp_path / "checkpoints" / "latest.pt"))]
    if resume_from is not None:
        expected.insert(0, ("load", resume_from))
    assert calls == expected
