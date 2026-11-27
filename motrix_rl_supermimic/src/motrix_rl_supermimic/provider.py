# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SuperMimic provider: task-agnostic world-model MBRL agent."""

from __future__ import annotations

from motrix_rl.contracts import AgentProvider, TrainerBase, TrainerContext
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
        from motrix_rl_supermimic.trainer import SuperMimicTrainer

        return SuperMimicTrainer(context=context)

    def total_env_steps(self, config: SuperMimicCfg, num_envs: int) -> int:
        return config.trainer.iterations * num_envs
