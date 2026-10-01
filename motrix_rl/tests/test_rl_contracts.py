# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from motrix_rl.contracts import AgentProvider, RlFramework, TrainerContext


@dataclass
class Config:
    value: int = 1


class Provider(AgentProvider[Config]):
    config_type = Config

    @property
    def train_backend(self):
        return "test"

    @property
    def agent_name(self):
        return "unit"

    @property
    def checkpoint_format(self):
        return "bin"

    def create_trainer(self, context: TrainerContext[Config]):
        raise NotImplementedError


class Framework(RlFramework):
    @property
    def name(self):
        return "unit"


def test_provider_and_framework_contracts():
    framework = Framework((Provider(),))

    assert framework.supported_agents() == ("unit",)
    assert framework.supported_train_backends("unit") == ("test",)
    assert framework.get_agent_provider("unit", "test").checkpoint_format == "bin"
    assert Provider().validate_config(Config(3)).value == 3
