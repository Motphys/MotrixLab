# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Test task composition and transition feeder for the async learner."""

import torch
from _supermimic_toy import ToyEnvCfg, register
from motrix_rl_supermimic.env import SuperMimicEnv
from motrix_rl_supermimic.imagined_task import ImaginedTask
from motrix_rl_supermimic.transitions import Transitions
from motrix_rl_supermimic.world_context import WorldContextTerm, WorldContextTermCfg

from motrix_env_core.config import configclass
from motrix_env_core.manager.context import BuildContext, ManagerContext


def capture_reference_context(ctx: ManagerContext, out: torch.Tensor) -> None:
    out.copy_(ctx.commands["reference"].context)


@configclass(kw_only=True)
class ReferenceContextCfg(WorldContextTermCfg):
    size: int = 1
    center: tuple[float, ...] = (0.0,)
    scale: tuple[float, ...] = (1.0,)

    def __call__(self, ctx: BuildContext) -> WorldContextTerm:
        return WorldContextTerm(self.size, capture_reference_context, self.center, self.scale)


def make_toy_task(device: str = "cpu", *, cfg: ToyEnvCfg | None = None) -> ImaginedTask:
    cfg = cfg or ToyEnvCfg()
    return ImaginedTask(cfg.for_imagined(), device)


def make_toy(num_envs: int = 1, cfg: ToyEnvCfg | None = None) -> SuperMimicEnv:
    register()
    cfg = cfg or ToyEnvCfg()
    env = SuperMimicEnv(cfg, num_envs=num_envs)
    return env


def ingest_toy_steps(learner, env: SuperMimicEnv, count: int) -> None:
    """Step Manager physics and feed owned term-recorded transitions."""
    for _ in range(count):
        with torch.no_grad():
            action = torch.randn(env.num_envs, env.action_space.shape[0]).clamp(-0.5, 0.5)
            result = env.step(action)
            transition = env.snapshot_transition(result)
            views = {key: value.unsqueeze(0) for key, value in transition.items()}
        learner.ingest(Transitions.from_dict(views))
