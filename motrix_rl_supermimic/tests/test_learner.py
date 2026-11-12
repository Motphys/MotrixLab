# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""End-to-end learner smoke on the toy task: collect → WM fit → imagination BPTT."""

import torch
from motrix_rl_supermimic.config import AgentCfg, SuperMimicCfg, TrainerCfg
from motrix_rl_supermimic.learner import Learner
from motrix_rl_supermimic.task_mdp import StateSpec
from test_mbrl_env import ToyEnv, ToyTaskMDP  # noqa: F401  (reuse fake backend + toy task)


def test_toy_training_smoke(tmp_path):
    torch.manual_seed(0)
    env = ToyEnv(num_envs=8)
    env.init_state()
    cfg = SuperMimicCfg(
        agent=AgentCfg(hidden=32, hist=4),
        trainer=TrainerCfg(
            iterations=30,
            buffer_steps=128,
            utd=0.5,
            horizon=4,
            goal_frames=2,
            wm_window=2,
            wm_batches=2,
            wm_batch=64,
            pi_batch=64,
            warmup_steps=8,
            wm_only_iters=2,
            critic_epochs=1,
            critic_minibatches=2,
            lr=3e-3,
            log_every=10,
            ckpt_every=15,
        ),
    )
    rows = []
    learner = Learner(cfg, env, "cpu", tmp_path, metrics=lambda it, m: rows.append(m))
    # Phase 1: WM only — loss must drop across the first training iterations.
    learner.collect(4)
    wm_losses = []
    for _ in range(60):
        learner.collect(1)
        wm_losses.append(learner.wm_update())
    assert wm_losses[-1] < wm_losses[0]

    # Phase 2: full loop including imagination BPTT and the critic.
    learner.train()
    assert learner.iteration == cfg.trainer.iterations
    assert rows and all(torch.isfinite(torch.tensor(r["wm_loss"])) for r in rows)
    assert (tmp_path / "last.pt").exists()

    # Phase 3: resume path round-trips module weights.
    fresh = Learner(cfg, env, "cpu", tmp_path / "fresh")
    fresh.load(str(tmp_path / "last.pt"))
    assert fresh.iteration == learner.iteration
    for key in ("wm", "pi", "critic"):
        for a, b in zip(getattr(learner, key).parameters(), getattr(fresh, key).parameters()):
            torch.testing.assert_close(a, b)


def test_canonical_layout_rejects_noncanonical_groups():
    bad = StateSpec(
        state_dim=15,
        action_dim=1,
        context_dim=0,
        observation_dim=1,
        critic_observation_dim=1,
        command_dim=1,
        state_groups=((15, 1.0),),  # not the canonical (3,4,3,3,nJ,nJ) partition
    )
    import pytest

    from motrix_rl_supermimic import state as st

    with pytest.raises(ValueError, match="Canonical state layout"):
        st.canonical_offsets(bad)
