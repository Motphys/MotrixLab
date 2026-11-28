# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""APPO provider assembled into the Motrix framework by the builtin plugin."""

from motrix_rl.contracts import AgentProvider, TrainerBase, TrainerContext
from motrix_rl_appo.config import AppoCfg


class AppoProvider(AgentProvider[AppoCfg]):
    config_type = AppoCfg

    @property
    def train_backend(self) -> str:
        return "torch"

    @property
    def agent_name(self) -> str:
        return "appo"

    @property
    def checkpoint_format(self) -> str:
        return "pt"

    def total_env_steps(self, config: AppoCfg, num_envs: int) -> int:
        return config.max_iterations * config.rollout_steps * num_envs

    def create_trainer(self, context: TrainerContext[AppoCfg]) -> TrainerBase:
        from motrix_rl_appo.trainer import AppoTrainer

        return AppoTrainer(context=context)
