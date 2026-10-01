# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl import frameworks
from motrix_rl.contracts import AgentProvider, RlFramework, TrainerBase, TrainerContext
from motrix_rl.deploy.api import OnnxPolicyExporter
from motrix_rl_skrl.config import SkrlCfg


class _SkrlPpoProvider(AgentProvider[SkrlCfg]):
    config_type = SkrlCfg

    @property
    def agent_name(self) -> str:
        return "ppo"


class SkrlPpoJaxProvider(_SkrlPpoProvider):
    @property
    def train_backend(self) -> str:
        return "jax"

    @property
    def checkpoint_format(self) -> str | None:
        return "pickle"

    def create_trainer(self, context: TrainerContext[SkrlCfg]) -> TrainerBase:
        from motrix_rl_skrl.jax.train.ppo import Trainer

        return Trainer(context=context)


class SkrlPpoTorchProvider(_SkrlPpoProvider):
    @property
    def train_backend(self) -> str:
        return "torch"

    @property
    def checkpoint_format(self) -> str | None:
        return "pt"

    def create_trainer(self, context: TrainerContext[SkrlCfg]) -> TrainerBase:
        from motrix_rl_skrl.torch.train.ppo import Trainer

        return Trainer(context=context)

    def create_policy_exporter(self) -> OnnxPolicyExporter:
        from motrix_rl_skrl.torch.export import SkrlPpoTorchOnnxExporter

        return SkrlPpoTorchOnnxExporter()


class SkrlFramework(RlFramework):
    def __init__(self) -> None:
        super().__init__(
            (
                SkrlPpoJaxProvider(),
                SkrlPpoTorchProvider(),
            )
        )

    @property
    def name(self) -> str:
        return "skrl"


def register_framework() -> None:
    frameworks.register_framework(SkrlFramework())
