# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest

from motrix_env_core.renderer import RenderConfig
from motrix_rl import frameworks, runs
from motrix_rl.config import CheckpointConfig, LoggingConfig
from motrix_rl.frameworks import RlFramework
from motrix_rl.method import RlMethod, parse_method
from motrix_rl.result import TrainResult


@dataclass
class _UnitCfg:
    value: int = 1


@dataclass
class _OtherUnitCfg:
    value: int = 2


def _create_context(run, *, rl_cfg, **kwargs):
    return frameworks.create_trainer_context(
        run,
        num_envs=4,
        play_num_envs=2,
        seed=7,
        rl_cfg=rl_cfg,
        logging=LoggingConfig(backend="tensorboard", interval=100),
        checkpoint=CheckpointConfig(interval=0),
        **kwargs,
    )


def test_parse_method_prefers_rllib_algo_form():
    assert parse_method("skrl.ppo") == RlMethod(rllib="skrl", algo="ppo")
    assert parse_method("rslrl.ppo") == RlMethod(rllib="rslrl", algo="ppo")
    assert parse_method("motrix.fastsac") == RlMethod(rllib="motrix", algo="fastsac")


def test_parse_method_supports_deprecated_rllib_shorthand():
    assert parse_method("ppo", deprecated_rllib="skrl") == RlMethod(rllib="skrl", algo="ppo")

    with pytest.raises(ValueError, match="conflicts"):
        parse_method("skrl.ppo", deprecated_rllib="rslrl")

    with pytest.raises(ValueError, match="rllib.algo"):
        parse_method("ppo")


def test_frameworks_register_provider(monkeypatch):
    monkeypatch.setattr(frameworks, "_frameworks", {})
    monkeypatch.setattr("motrix_rl.plugins.load_plugins", lambda: None)

    provider = _UnitProvider("skrl", "torch", "unit")
    _register_unit_provider(provider)

    assert frameworks.agent_exists("skrl", "unit", "torch")
    assert frameworks.get_agent_provider("skrl", "unit", "torch") is provider


def test_frameworks_expose_provider_queries(monkeypatch):
    monkeypatch.setattr(frameworks, "_frameworks", {})
    monkeypatch.setattr("motrix_rl.plugins.load_plugins", lambda: None)

    provider = _UnitProvider("skrl", "torch", "unit")
    _register_unit_provider(provider)

    assert frameworks.supported_agents("skrl") == ("unit",)
    assert frameworks.get_config_type("skrl", "unit") is _UnitCfg
    assert frameworks.supported_train_backends("skrl", "unit") == ("torch",)
    assert frameworks.iter_agent_providers(framework_name="skrl", train_backend="torch") == (provider,)


def test_framework_rejects_backend_config_type_mismatch():
    with pytest.raises(ValueError, match="inconsistent config types"):
        _UnitFramework(
            "mixed",
            (
                _UnitProvider("mixed", "torch", "unit"),
                _OtherUnitProvider("mixed", "jax", "unit"),
            ),
        )


def test_frameworks_create_standard_trainer(monkeypatch, tmp_path):
    monkeypatch.setattr(frameworks, "_frameworks", {})
    monkeypatch.setattr("motrix_rl.plugins.load_plugins", lambda: None)
    provider = _UnitProvider("skrl", "torch", "unit")
    _register_unit_provider(provider)
    run = runs.create_run_context(
        env_name="demo-env",
        rllib="skrl",
        train_backend="torch",
        algo="unit",
        seed=None,
        checkpoint_format="pt",
        runs_root=tmp_path,
    )
    trainer = frameworks.create_trainer(
        _create_context(
            run,
            rl_cfg=_UnitCfg(value=4),
            render=RenderConfig(),
            resume_from="checkpoint.pt",
        ),
    )
    assert isinstance(trainer, frameworks.TrainerHandle)
    assert isinstance(trainer.trainer, _UnitTrainer)
    assert trainer.trainer.args[0].env_name == "demo-env"
    assert trainer.trainer.args[0].run_dir == run.run_dir
    assert trainer.trainer.args[0].checkpoint_dir == run.checkpoint_dir
    assert trainer.trainer.args[0].checkpoint_format == "pt"
    assert isinstance(trainer.trainer.args[0].render, RenderConfig)
    assert trainer.trainer.args[0].num_envs == 4
    assert trainer.trainer.args[0].play_num_envs == 2
    assert trainer.trainer.args[0].seed == 7
    assert trainer.trainer.args[0].rl_cfg == _UnitCfg(value=4)
    assert trainer.trainer.args[0].resume_from == "checkpoint.pt"


def test_frameworks_pass_trainer_context(monkeypatch, tmp_path):
    monkeypatch.setattr(frameworks, "_frameworks", {})
    monkeypatch.setattr("motrix_rl.plugins.load_plugins", lambda: None)
    provider = _UnitProvider("skrl", "torch", "unit")
    _register_unit_provider(provider)

    run = runs.create_run_context(
        env_name="demo-env",
        rllib="skrl",
        train_backend="torch",
        algo="unit",
        seed=None,
        checkpoint_format="pt",
        runs_root=tmp_path,
    )
    trainer = frameworks.create_trainer(_create_context(run, rl_cfg=_UnitCfg()))

    assert trainer.trainer.context.env_name == "demo-env"
    assert trainer.trainer.context.run_dir == run.run_dir
    assert trainer.trainer.context.checkpoint_dir == run.checkpoint_dir
    assert trainer.trainer.context.checkpoint_format == "pt"
    assert trainer.trainer.context.render is None
    assert trainer.trainer.context.rl_cfg == _UnitCfg()
    assert trainer.trainer.context.resume_from is None


@pytest.mark.parametrize(
    ("logging", "checkpoint", "message"),
    [
        (LoggingConfig(backend="", interval=100), CheckpointConfig(interval=0), "logging.backend"),
        (LoggingConfig(backend="tensorboard", interval=0), CheckpointConfig(interval=0), "logging.interval"),
        (LoggingConfig(backend="tensorboard", interval=100), CheckpointConfig(interval=-1), "checkpoint.interval"),
    ],
)
def test_create_trainer_context_validates_runtime_config(tmp_path, logging, checkpoint, message):
    run = runs.create_run_context(
        env_name="demo-env",
        rllib="skrl",
        train_backend="torch",
        algo="unit",
        seed=None,
        checkpoint_format="pt",
        runs_root=tmp_path,
    )

    with pytest.raises(ValueError, match=message):
        frameworks.create_trainer_context(
            run,
            num_envs=4,
            play_num_envs=2,
            seed=7,
            rl_cfg=_UnitCfg(),
            logging=logging,
            checkpoint=checkpoint,
        )


def test_trainer_handle_returns_framework_train_result(monkeypatch, tmp_path):
    monkeypatch.setattr(frameworks, "_frameworks", {})
    monkeypatch.setattr("motrix_rl.plugins.load_plugins", lambda: None)
    provider = _UnitProvider("skrl", "torch", "unit")
    _register_unit_provider(provider)

    run = runs.create_run_context(
        env_name="demo-env",
        rllib="skrl",
        train_backend="torch",
        algo="unit",
        seed=None,
        checkpoint_format="pt",
        runs_root=tmp_path,
    )
    trainer = frameworks.create_trainer(_create_context(run, rl_cfg=_UnitCfg()))
    result = trainer.train()

    assert trainer.trainer.train_called
    assert result == TrainResult(run=run)


def test_frameworks_reject_missing_run_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(frameworks, "_frameworks", {})
    monkeypatch.setattr("motrix_rl.plugins.load_plugins", lambda: None)
    provider = _UnitProvider("skrl", "torch", "unit", fail_on_create=True)
    _register_unit_provider(provider)

    run = runs.create_run_context(
        env_name="demo-env",
        rllib="skrl",
        train_backend="jax",
        algo="unit",
        seed=None,
        checkpoint_format="pickle",
        runs_root=tmp_path,
    )

    with pytest.raises(ValueError, match="No agent provider found"):
        frameworks.create_trainer(_create_context(run, rl_cfg=_UnitCfg()))


def test_frameworks_reject_provider_config_type_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(frameworks, "_frameworks", {})
    monkeypatch.setattr("motrix_rl.plugins.load_plugins", lambda: None)
    _register_unit_provider(_UnitProvider("skrl", "torch", "unit"))
    run = runs.create_run_context(
        env_name="demo-env",
        rllib="skrl",
        train_backend="torch",
        algo="unit",
        seed=None,
        checkpoint_format="pt",
        runs_root=tmp_path,
    )

    with pytest.raises(TypeError, match="expects _UnitCfg"):
        frameworks.create_trainer(_create_context(run, rl_cfg={"value": 1}))


class _UnitTrainer(frameworks.TrainerBase):
    def __init__(
        self,
        *,
        context: frameworks.TrainerContext[_UnitCfg],
    ) -> None:
        self.args = (context,)
        self.context = context
        self.train_called = False

    def train(self):
        self.train_called = True
        return "ignored backend result"

    def play(self, policy: str) -> None:
        raise NotImplementedError


def _register_unit_provider(provider: "_UnitProvider") -> None:
    frameworks.register_framework(_UnitFramework(provider.rllib, (provider,)))


class _UnitFramework(RlFramework):
    def __init__(self, name: str, providers) -> None:
        self._name = name
        super().__init__(providers)

    @property
    def name(self) -> str:
        return self._name


@dataclass(frozen=True)
class _UnitProvider(frameworks.AgentProvider[_UnitCfg]):
    config_type = _UnitCfg

    rllib: str
    _train_backend: str
    _agent_name: str
    fail_on_create: bool = False

    @property
    def train_backend(self) -> str:
        return self._train_backend

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def checkpoint_format(self) -> str | None:
        return "pt"

    def create_trainer(
        self,
        context: frameworks.TrainerContext[_UnitCfg],
    ) -> frameworks.TrainerBase:
        if self.fail_on_create:
            raise AssertionError("trainer should not be constructed")
        return _UnitTrainer(context=context)


class _OtherUnitProvider(_UnitProvider):
    config_type = _OtherUnitCfg
