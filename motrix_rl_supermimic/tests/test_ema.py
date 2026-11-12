# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Policy EMA contracts: averaging, checkpoint layout, and resume."""

import torch
from motrix_rl_supermimic.config import AgentCfg, SuperMimicCfg, TrainerCfg
from motrix_rl_supermimic.learner import Learner
from motrix_rl_supermimic.toy import register as register_toy
from test_mbrl_env import ToyEnv


def test_ema_averages_and_checkpoints(tmp_path):
    register_toy()
    torch.manual_seed(0)
    env = ToyEnv(num_envs=8)
    env.init_state()
    cfg = SuperMimicCfg(
        agent=AgentCfg(hidden=32, hist=4, pi_ema=0.995),
        trainer=TrainerCfg(
            iterations=6,
            buffer_steps=64,
            utd=0.5,
            horizon=4,
            goal_frames=2,
            wm_window=2,
            wm_batches=1,
            wm_batch=64,
            pi_batch=64,
            warmup_steps=8,
            wm_only_iters=2,
            critic_epochs=1,
            critic_minibatches=2,
            lr=3e-3,
            log_every=2,
            ckpt_every=6,
        ),
    )
    learner = Learner(cfg, env, "cpu", tmp_path)
    assert learner.pi_avg is not None

    before = torch.cat([p.flatten() for p in learner.pi.parameters()])
    learner.collect(4)
    for _ in range(4):
        learner.wm_update()
    learner.policy_update()  # moves pi and updates the average

    after = torch.cat([p.flatten() for p in learner.pi.parameters()])
    avg = torch.cat([p.flatten() for p in learner.pi_avg.module.parameters()])
    assert not torch.allclose(before, after)
    # The average moves toward the live policy but is strictly between the two.
    moved = (avg - before).abs().max()
    live_moved = (after - before).abs().max()
    assert 0 < float(moved) <= float(live_moved) + 1e-6

    learner.train()  # reaches ckpt_every and saves
    ck = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert set(ck) >= {"pi", "pi_live", "pi_avg", "pi_avg_n_averaged"}
    live = torch.cat([p.flatten() for p in ck["pi_live"].values() if p.is_floating_point()])
    avg_ck = torch.cat([v.flatten() for k, v in ck["pi_avg"].items() if v.is_floating_point()])
    assert not torch.allclose(live[: avg_ck.numel()], avg_ck[: live.numel()])
    # 'pi' stores the live weights (the task-compatible layout); the average
    # lives under 'pi_avg' with its own counter.
    pi_ck = torch.cat([v.flatten() for v in ck["pi"].values() if v.is_floating_point()])
    torch.testing.assert_close(pi_ck, live)

    # Resume restores both the live policy and the EMA continuation state.
    fresh = Learner(cfg, env, "cpu", tmp_path / "fresh")
    fresh.load(str(tmp_path / "last.pt"))
    assert fresh.pi_avg is not None
    assert int(fresh.pi_avg.n_averaged) > 0
    for a, b in zip(learner.pi.parameters(), fresh.pi.parameters()):
        torch.testing.assert_close(a, b)


def test_ema_off_by_default(tmp_path):
    register_toy()
    env = ToyEnv(num_envs=4)
    env.init_state()
    cfg = SuperMimicCfg(
        agent=AgentCfg(hidden=16, hist=2),
        trainer=TrainerCfg(
            iterations=2,
            buffer_steps=32,
            utd=0.5,
            horizon=2,
            goal_frames=2,
            wm_window=2,
            wm_batches=1,
            wm_batch=16,
            pi_batch=16,
            warmup_steps=4,
            wm_only_iters=1,
            critic_epochs=1,
            critic_minibatches=1,
            log_every=1,
            ckpt_every=2,
        ),
    )
    learner = Learner(cfg, env, "cpu", tmp_path)
    assert learner.pi_avg is None
    learner.train()
    ck = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert "pi_avg" not in ck
