# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Test task composition and transition feeder for the async learner."""

import torch
from motrix_rl_supermimic.mbrl_env import MbrlEnv, assemble_task
from motrix_rl_supermimic.toy import ToyEnvCfg, register
from motrix_rl_supermimic.trainer import SuperMimicTrainer


def make_toy_task(device: str = "cpu", *, cfg: ToyEnvCfg | None = None):
    cfg = cfg or ToyEnvCfg()
    return assemble_task(cfg.for_imagined(), device)


def make_toy(num_envs: int = 1, cfg: ToyEnvCfg | None = None) -> MbrlEnv:
    register()
    cfg = cfg or ToyEnvCfg()
    env = MbrlEnv(cfg, num_envs=num_envs)
    return env


def ingest_toy_steps(learner, env: MbrlEnv, count: int) -> None:
    """Step Manager physics and feed owned term-recorded transitions."""
    for _ in range(count):
        with torch.no_grad():
            action = torch.randn(env.num_envs, env.action_space.shape[0]).clamp(-0.5, 0.5)
            result = env.step(action)
            transition = env.snapshot_transition(result)
            views = {key: value.unsqueeze(0) for key, value in transition.items()}
        SuperMimicTrainer._ingest(learner, 1, views)
