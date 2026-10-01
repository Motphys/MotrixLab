# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl import frameworks
from motrix_rl.contracts import AgentProvider, RlFramework, TrainerBase, TrainerContext
from motrix_rl.deploy.api import OnnxPolicyExporter
from motrix_rl_rslrl.cfg import RslrlCfg


class RslrlPpoProvider(AgentProvider[RslrlCfg]):
    config_type = RslrlCfg

    @property
    def train_backend(self) -> str:
        return "torch"

    @property
    def agent_name(self) -> str:
        return "ppo"

    @property
    def checkpoint_format(self) -> str | None:
        return "pt"

    def create_trainer(self, context: TrainerContext[RslrlCfg]) -> TrainerBase:
        from motrix_rl_rslrl.torch.train.ppo import Trainer

        return Trainer(context=context)

    def create_policy_exporter(self) -> OnnxPolicyExporter:
        from motrix_rl_rslrl.export import RslrlPpoOnnxExporter

        return RslrlPpoOnnxExporter()


class RslrlFramework(RlFramework):
    def __init__(self) -> None:
        super().__init__((RslrlPpoProvider(),))

    @property
    def name(self) -> str:
        return "rslrl"


def register_framework() -> None:
    frameworks.register_framework(RslrlFramework())
