# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl import frameworks
from motrix_rl.deploy.api import OnnxPolicyExporter
from motrix_rl_builtin.fastsac.config import FastSacCfg
from motrix_rl_interface import AgentProvider, ExecutionMode, RlFramework, TrainerBase, TrainerContext


class FastSacProvider(AgentProvider[FastSacCfg]):
    config_type = FastSacCfg

    @property
    def train_backend(self) -> str:
        return "torch"

    @property
    def agent_name(self) -> str:
        return "fastsac"

    @property
    def checkpoint_format(self) -> str | None:
        return "pt"

    @property
    def execution_modes(self) -> tuple[ExecutionMode, ...]:
        return (ExecutionMode.INLINE_RUNTIME, ExecutionMode.PROCESS_RUNTIME)

    def create_trainer(self, context: TrainerContext[FastSacCfg]) -> TrainerBase:
        if context.rl_cfg.asynchronous:
            from motrix_rl_builtin.fastsac.async_impl.train import Trainer
        else:
            from motrix_rl_builtin.fastsac.sync.train import Trainer

        return Trainer(context=context)

    def create_policy_exporter(self) -> OnnxPolicyExporter:
        from motrix_rl_builtin.fastsac.export import FastSacOnnxExporter

        return FastSacOnnxExporter()


class MotrixFramework(RlFramework):
    def __init__(self) -> None:
        super().__init__((FastSacProvider(),))

    @property
    def name(self) -> str:
        return "motrix"


def register_framework() -> None:
    frameworks.register_framework(MotrixFramework())
