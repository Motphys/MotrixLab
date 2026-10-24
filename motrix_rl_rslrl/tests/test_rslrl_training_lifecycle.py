# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from motrix_rl_rslrl.torch.train import ppo


@pytest.mark.parametrize("failure_stage", ["wrap", "runner", "writer", "update", "success"])
def test_training_owns_writer_and_wrapper_lifecycle(monkeypatch, failure_stage):
    trainer = object.__new__(ppo.Trainer)
    trainer._context = SimpleNamespace(seed=None, num_envs=2, run_dir="unused")
    trainer._env_name = "toy"
    trainer._sim = trainer._render = None
    trainer._create_rslrl_config = lambda: {}
    env = Mock()
    wrapped = Mock()
    runner = SimpleNamespace(is_distributed=False, gpu_world_size=1, gpu_global_rank=0)
    log = Mock()
    error = RuntimeError(failure_stage)

    def fail_at(stage, value):
        if failure_stage == stage:
            raise error
        return value

    monkeypatch.setattr(ppo.env_registry, "make", lambda *args, **kwargs: env)
    monkeypatch.setattr(ppo, "wrap_env", lambda *args, **kwargs: fail_at("wrap", wrapped))
    monkeypatch.setattr(ppo, "OnPolicyRunner", lambda *args, **kwargs: fail_at("runner", runner))
    monkeypatch.setattr(ppo, "RslrlTrainingLogger", lambda *args, **kwargs: fail_at("writer", log))
    monkeypatch.setattr(ppo, "run_sync", lambda *args: fail_at("update", None))
    if failure_stage == "success":
        trainer.train()
    else:
        with pytest.raises(RuntimeError) as caught:
            trainer.train()
        assert caught.value is error
    assert wrapped.close.call_count == (failure_stage != "wrap")
    assert log.close.call_count == (failure_stage in ("update", "success"))


def test_cleanup_failure_does_not_replace_training_interrupt(monkeypatch):
    trainer = object.__new__(ppo.Trainer)
    trainer._context = SimpleNamespace(seed=None, num_envs=2, run_dir="unused")
    trainer._env_name = "toy"
    trainer._sim = trainer._render = None
    trainer._create_rslrl_config = lambda: {}
    wrapped, log = Mock(), Mock()
    wrapped.close.side_effect = RuntimeError("close")
    log.close.side_effect = RuntimeError("writer close")
    runner = SimpleNamespace(is_distributed=False, gpu_world_size=1, gpu_global_rank=0)
    error = KeyboardInterrupt()
    monkeypatch.setattr(ppo.env_registry, "make", Mock())
    monkeypatch.setattr(ppo, "wrap_env", lambda *args, **kw: wrapped)
    monkeypatch.setattr(ppo, "OnPolicyRunner", lambda *args, **kw: runner)
    monkeypatch.setattr(ppo, "RslrlTrainingLogger", lambda *args, **kw: log)
    monkeypatch.setattr(ppo, "run_sync", Mock(side_effect=error))
    with pytest.raises(KeyboardInterrupt) as caught:
        trainer.train()
    assert caught.value is error
    wrapped.close.assert_called_once()
    log.close.assert_called_once()
