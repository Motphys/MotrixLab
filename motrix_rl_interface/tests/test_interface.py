from dataclasses import dataclass

from motrix_rl_interface import AgentProvider, ExecutionMode, RlFramework, TrainerBase, TrainerContext


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

    @property
    def execution_modes(self):
        return (ExecutionMode.NATIVE,)

    def create_trainer(self, context: TrainerContext[Config]):
        raise NotImplementedError


class Framework(RlFramework):
    @property
    def name(self):
        return "unit"


def test_interface_provider_and_framework_contracts():
    framework = Framework((Provider(),))

    assert framework.supported_agents() == ("unit",)
    assert framework.supported_train_backends("unit") == ("test",)
    assert framework.get_agent_provider("unit", "test").checkpoint_format == "bin"
    assert Provider().validate_config(Config(3)).value == 3


def test_interface_has_no_algorithm_dependency():
    assert TrainerBase.__module__ == "motrix_rl_interface.contracts"
