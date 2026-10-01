# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from motrix_env_core import registry as env_registry
from motrix_rl import frameworks, runner, runs, utils
from motrix_rl.config import CheckpointConfig, LoggingConfig, TaskMeta, TrainConfig
from motrix_rl.frameworks import RlFramework


@dataclass
class DemoCfg:
    value: int = 1


def test_runner_train_creates_run_and_returns_result(monkeypatch, tmp_path):
    monkeypatch.setattr(frameworks, "_frameworks", {})
    monkeypatch.setattr("motrix_rl.plugins.load_plugins", lambda: None)
    monkeypatch.setattr(env_registry, "contains", lambda env_name: env_name == "demo-env")
    monkeypatch.setattr(utils, "get_device_supports", lambda: utils.DeviceSupports(torch=True))

    provider = _RunnerProvider()
    frameworks.register_framework(_RunnerFramework("external", (provider,)))

    result = runner.train(
        runner.TrainRequest(
            config=TrainConfig(
                task=TaskMeta(env="demo-env", rllib="external", algo="demo", train_backend="custom"),
                num_envs=2,
                play_num_envs=3,
                seed=7,
                algo=DemoCfg(),
                logging=LoggingConfig(backend="tensorboard", interval=20),
                checkpoint=CheckpointConfig(interval=50),
            ),
            runs_root=tmp_path,
        )
    )

    assert provider.trainer is not None
    assert provider.trainer.train_called
    assert result.run.metadata.env_name == "demo-env"
    assert result.run.metadata.rllib == "external"
    assert result.run.metadata.train_backend == "custom"
    assert result.run.metadata.algo == "demo"
    assert result.run.metadata.checkpoint_format == "pt"
    assert result.run.metadata.seed == 7
    assert result.run.run_dir.parent == tmp_path / "demo-env" / "external" / "custom" / "demo"
    assert provider.trainer.context.rl_cfg == DemoCfg()
    assert provider.trainer.context.num_envs == 2
    assert provider.trainer.context.play_num_envs == 3
    assert provider.trainer.context.seed == 7
    assert provider.trainer.context.logging == LoggingConfig(backend="tensorboard", interval=20)
    assert provider.trainer.context.checkpoint == CheckpointConfig(interval=50)
    assert runs.task_config_path(result.run.run_dir).is_file()

    handle = runner.create_run_handle(
        result.run,
        cfg_override={"value": 2},
        play_num_envs=6,
    )
    assert handle.trainer.context.num_envs == 2
    assert handle.trainer.context.play_num_envs == 6
    assert handle.trainer.context.rl_cfg == DemoCfg(value=2)
    assert handle.trainer.context.logging == LoggingConfig(backend="tensorboard", interval=20)
    assert handle.trainer.context.checkpoint == CheckpointConfig(interval=50)


class _RunnerTrainer(frameworks.TrainerBase):
    def __init__(self, context: frameworks.TrainerContext[DemoCfg]) -> None:
        self.context = context
        self.train_called = False

    def train(self) -> None:
        self.train_called = True

    def play(self, policy: str) -> None:
        raise NotImplementedError


class _RunnerFramework(RlFramework):
    def __init__(self, name: str, providers) -> None:
        self._name = name
        super().__init__(providers)

    @property
    def name(self) -> str:
        return self._name


@dataclass
class _RunnerProvider(frameworks.AgentProvider[DemoCfg]):
    config_type = DemoCfg

    trainer: _RunnerTrainer | None = None

    @property
    def train_backend(self) -> str:
        return "custom"

    @property
    def agent_name(self) -> str:
        return "demo"

    @property
    def checkpoint_format(self) -> str | None:
        return "pt"

    def create_trainer(
        self,
        context: frameworks.TrainerContext[DemoCfg],
    ) -> frameworks.TrainerBase:
        self.trainer = _RunnerTrainer(context)
        return self.trainer
