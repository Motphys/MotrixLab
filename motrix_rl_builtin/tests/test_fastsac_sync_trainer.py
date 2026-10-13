# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.utils.tensorboard

from motrix_rl.config import CheckpointConfig, LoggingConfig
from motrix_rl.contracts import TrainerContext


def test_sync_train_closes_tensorboard_writer_when_training_fails(tmp_path, monkeypatch):
    from motrix_rl_builtin.fastsac.sync.train import Trainer

    import motrix_envs  # noqa: F401 registers environments used by the trainer

    writer = Mock()
    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", Mock(return_value=writer))
    monkeypatch.setattr(Trainer, "_make_env", Mock())
    monkeypatch.setattr(Trainer, "_make_agent", Mock(return_value=SimpleNamespace(device="cpu", global_step=0)))

    def fail_after_logging(self, agent, env, training_log, **kwargs):
        training_log.scalars({"train/qf_loss": 1.0}, 0, env_steps=1)  # opens TensorBoard
        raise RuntimeError("training failed")

    monkeypatch.setattr(Trainer, "_run_loop", fail_after_logging)
    context = TrainerContext(
        run=None,
        env_name="cartpole",
        run_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        sim=None,
        checkpoint_format="pt",
        num_envs=1,
        play_num_envs=1,
        seed=None,
        rl_cfg=SimpleNamespace(trainer=SimpleNamespace(num_learning_iterations=1)),
        logging=LoggingConfig(backend="tensorboard", interval=1),
        checkpoint=CheckpointConfig(interval=0),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        Trainer(context=context).train()

    writer.close.assert_called_once_with()
