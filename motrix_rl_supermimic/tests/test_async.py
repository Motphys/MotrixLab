# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Async topology smoke: spawned CPU collector feeding the learner ring."""

from pathlib import Path

import pytest
import torch
from motrix_rl_supermimic.config import AgentCfg, CollectorCfg, SuperMimicCfg, TrainerCfg
from motrix_rl_supermimic.toy import register as register_toy

pytestmark = pytest.mark.integration


def test_async_trainer_runs(tmp_path):
    pytest.importorskip("motrix_rl")
    register_toy()
    from motrix_rl_supermimic.async_impl import AsyncTrainer

    from motrix_rl.contracts import TrainerContext
    from motrix_rl.metrics import MetricsLogger

    cfg = SuperMimicCfg(
        asynchronous=True,
        agent=AgentCfg(hidden=32, hist=4),
        trainer=TrainerCfg(
            iterations=20,
            buffer_steps=64,
            utd=0.5,
            horizon=4,
            goal_frames=2,
            wm_window=2,
            wm_batches=1,
            wm_batch=64,
            pi_batch=64,
            warmup_steps=4,
            wm_only_iters=2,
            critic_epochs=1,
            critic_minibatches=2,
            lr=3e-3,
            log_every=5,
            ckpt_every=20,
        ),
        collector=CollectorCfg(block_steps=2, queue_blocks=2, snapshot_every=5, worker_timeout=60.0,
                               policy_device="cpu", preregister="motrix_rl_supermimic.toy"),
    )
    context = TrainerContext(
        run=None,
        env_name="supermimic-toy",
        run_dir=Path(tmp_path),
        checkpoint_dir=Path(tmp_path) / "ckpt",
        sim=None,
        checkpoint_format="pt",
        num_envs=4,
        play_num_envs=1,
        seed=0,
        rl_cfg=cfg,
        logging=None,
        checkpoint=None,
        metrics=MetricsLogger.disabled(),
    )
    trainer = AsyncTrainer(context=context)
    trainer.train()

    ckpt = Path(tmp_path) / "ckpt"
    assert (ckpt / "last.pt").exists()
    snapshot = torch.load(ckpt / "last.pt", map_location="cpu", weights_only=False)
    assert snapshot["iteration"] == 20
