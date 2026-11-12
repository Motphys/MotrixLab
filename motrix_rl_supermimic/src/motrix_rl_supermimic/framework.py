# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SuperMimic provider registration: task-agnostic world-model MBRL agent."""

from __future__ import annotations

from motrix_rl.contracts import AgentProvider, RlFramework, TrainerBase, TrainerContext
from motrix_rl_supermimic.config import SuperMimicCfg


class SuperMimicProvider(AgentProvider[SuperMimicCfg]):
    config_type = SuperMimicCfg

    @property
    def train_backend(self) -> str:
        return "torch"

    @property
    def agent_name(self) -> str:
        return "supermimic"

    @property
    def checkpoint_format(self) -> str | None:
        return "pt"

    def create_trainer(self, context: TrainerContext[SuperMimicCfg]) -> TrainerBase:
        if context.rl_cfg.asynchronous:
            from motrix_rl_supermimic.async_impl.train import AsyncTrainer

            return AsyncTrainer(context=context)
        from motrix_rl_supermimic.sync import Trainer

        return Trainer(context=context)

    def total_env_steps(self, config: SuperMimicCfg, num_envs: int) -> int:
        return config.trainer.iterations * num_envs


class SuperMimicFramework(RlFramework):
    def __init__(self) -> None:
        super().__init__((SuperMimicProvider(),))

    @property
    def name(self) -> str:
        return "supermimic"
