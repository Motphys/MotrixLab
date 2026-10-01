# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from motrix_rl_skrl.ppo import SkrlPpoTrainerBase, add_runtime_config, ppo_memory_size

from motrix_rl.config import CheckpointConfig, LoggingConfig


class Scheduler:
    pass


class Scaler:
    pass


@dataclass
class DummyEnv:
    observation_space: str = "obs-space"
    state_space: str = "state-space"
    device: str = "cpu"


def test_add_runtime_config_for_training():
    cfg = {
        "learning_rate_scheduler": "KLAdaptiveLR",
        "rewards_shaper_scale": 2.0,
        "mixed_precision": True,
    }

    result = add_runtime_config(
        cfg,
        DummyEnv(),
        scheduler_cls=Scheduler,
        scaler_cls=Scaler,
        run_dir="runs/cartpole/skrl/torch/ppo/26-07-24_12-00-00-000000",
        logging=LoggingConfig(backend="tensorboard", interval=17),
        checkpoint=CheckpointConfig(interval=23),
        remove_mixed_precision=True,
    )

    assert result is cfg
    assert result["learning_rate_scheduler"] is Scheduler
    assert "mixed_precision" not in result
    assert result["rewards_shaper"](3.0, 0, 0) == 6.0
    assert result["observation_preprocessor"] is Scaler
    assert result["observation_preprocessor_kwargs"] == {"size": "obs-space", "device": "cpu"}
    assert result["state_preprocessor"] is Scaler
    assert result["state_preprocessor_kwargs"] == {"size": "state-space", "device": "cpu"}
    assert result["value_preprocessor"] is Scaler
    assert result["value_preprocessor_kwargs"] == {"size": 1, "device": "cpu"}
    assert result["experiment"]["write_interval"] == 17
    assert result["experiment"]["checkpoint_interval"] == 23
    assert result["experiment"]["directory"] == "runs/cartpole/skrl/torch/ppo"
    assert result["experiment"]["experiment_name"] == "26-07-24_12-00-00-000000"


def test_add_runtime_config_for_play_disables_logging():
    cfg = {
        "learning_rate_scheduler": None,
        "rewards_shaper_scale": 1.0,
    }

    result = add_runtime_config(
        cfg,
        DummyEnv(),
        scheduler_cls=Scheduler,
        scaler_cls=Scaler,
        logging=LoggingConfig(backend="tensorboard", interval=100),
        checkpoint=CheckpointConfig(interval=0),
    )

    assert result["learning_rate_scheduler"] is None
    assert result["rewards_shaper"] is None
    assert "rewards_shaper_scale" not in result
    assert result["experiment"]["write_interval"] == 0
    assert result["experiment"]["checkpoint_interval"] == 0
    assert result["experiment"]["directory"] == ""
    assert result["experiment"]["experiment_name"] == ""


def test_ppo_memory_size_uses_rollouts_for_auto_size():
    assert ppo_memory_size(-1, {"rollouts": 32}) == 32
    assert ppo_memory_size(128, {"rollouts": 32}) == 128


@pytest.mark.parametrize(
    ("entrypoint", "args", "num_envs", "mode"),
    (
        ("train", (), 2, "train"),
        ("play", ("policy.pt",), 3, "play"),
    ),
)
def test_trainer_sets_seed_before_creating_environment(entrypoint, args, num_envs, mode):
    class StopAfterMakeEnv(Exception):
        pass

    calls = []
    trainer = object.__new__(SkrlPpoTrainerBase)
    trainer._rlcfg = object()
    trainer._context = SimpleNamespace(seed=17, num_envs=2, play_num_envs=3)
    trainer._set_seed = lambda seed: calls.append(("seed", seed))

    def make_env(*, num_envs, mode):
        calls.append(("make_env", num_envs, mode))
        raise StopAfterMakeEnv

    trainer._make_env = make_env

    with pytest.raises(StopAfterMakeEnv):
        getattr(trainer, entrypoint)(*args)

    assert calls == [("seed", 17), ("make_env", num_envs, mode)]
