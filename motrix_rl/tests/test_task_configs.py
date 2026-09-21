# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Self-consistency net for the YAML RL configs (issue #151).

Every registered task (discovered by scanning configs/task/**) composes into a
fully-populated, typed base-schema instance. Because the base schemas carry no
field defaults, an incomplete base value file surfaces here as a
``MissingMandatoryValue`` at ``to_object`` time.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.object_type import ObjectType
from hydra.errors import MissingConfigException
from omegaconf import MISSING, OmegaConf

from motrix_rl import frameworks, runner, runs
from motrix_rl.config import TaskConfig
from motrix_rl.rslrl.cfg import RslrlRunnerCfg

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
RSLRL_BASE_CONFIG = CONFIG_DIR / "algo_base" / "rslrl.ppo.yaml"


def _discover_task_specs():
    """Discover Hydra task options; compose tests below validate their contents."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        loader = GlobalHydra.instance().config_loader()
        env_names = loader.get_group_options("task", results_filter=ObjectType.GROUP)
        options = [
            (env_name, option) for env_name in env_names for option in loader.get_group_options(f"task/{env_name}")
        ]
    GlobalHydra.instance().clear()

    specs = []
    for env_name, option in options:
        parts = option.split(".")
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid task option '{env_name}/{option}'")
        rllib, algo = parts[:2]
        backend = parts[2] if len(parts) == 3 else None
        specs.append((env_name, rllib, algo, backend))
    return sorted(specs, key=lambda spec: (spec[0], spec[1], spec[2], spec[3] or ""))


ALL_SPECS = _discover_task_specs()


def _spec_id(spec) -> str:
    env, rllib, algo, backend = spec
    return f"{env}/{rllib}.{algo}" + (f".{backend}" if backend else "")


def _compose_task_option(
    env: str,
    rllib: str,
    algo: str,
    backend: str | None = None,
    config_dir=CONFIG_DIR,
    overrides: list[str] | None = None,
):
    option = f"{env}/{rllib}.{algo}" + (f".{backend}" if backend else "")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=[f"task={option}", *(overrides or [])])
    result = OmegaConf.to_object(cfg)
    if not isinstance(result, TaskConfig):
        raise TypeError(f"Expected TaskConfig, got {type(result).__name__}")
    return result


def test_specs_discovered():
    assert ALL_SPECS, "no task configs discovered under configs/task/**"


def test_rslrl_base_config_matches_runner_schema():
    config_fields = set(yaml.safe_load(RSLRL_BASE_CONFIG.read_text())) - {"defaults"}
    schema_fields = {item.name for item in fields(RslrlRunnerCfg)}

    assert config_fields == schema_fields


@pytest.mark.parametrize("spec", ALL_SPECS, ids=[_spec_id(s) for s in ALL_SPECS])
def test_task_composes_to_typed_config(spec, tmp_path):
    env, rllib, algo, backend = spec
    task_cfg = _compose_task_option(env, rllib, algo, backend)
    cfg = task_cfg.algo
    assert isinstance(cfg, frameworks.get_config_type(rllib, algo))

    assert task_cfg.task.env == env
    assert task_cfg.task.rllib == rllib
    assert task_cfg.task.algo == algo
    assert task_cfg.task.train_backend == backend
    assert task_cfg.num_envs > 0
    assert task_cfg.play_num_envs > 0

    runs.write_task_config(tmp_path, task_cfg)
    restored = runs.read_task_config(tmp_path, frameworks.get_config_type(rllib, algo))
    assert restored.task == task_cfg.task
    assert restored.num_envs == task_cfg.num_envs
    assert restored.play_num_envs == task_cfg.play_num_envs
    assert restored.seed == task_cfg.seed
    assert restored.logging == task_cfg.logging
    assert restored.checkpoint == task_cfg.checkpoint
    assert restored.algo == task_cfg.algo


@pytest.mark.parametrize(
    ("env", "rllib", "algo"),
    [
        ("dex-evt-walk-flat", "motrix", "fastsac"),
        ("dex-evt-walk-rough", "motrix", "fastsac"),
        ("g1-walk-flat", "motrix", "fastsac"),
        ("go2-walk-flat", "motrix", "fastsac"),
        ("k1-walk-flat", "motrix", "fastsac"),
        ("k1-walk-rough", "motrix", "fastsac"),
        ("go2-walk-flat", "rslrl", "ppo"),
    ],
)
def test_task_checkpoint_policy_composes_at_root(env, rllib, algo):
    task_cfg = _compose_task_option(env, rllib, algo)

    # The checkpoint block must compose to the root TaskConfig (runs.py reads it
    # there) rather than stay MISSING or nest under ``algo``; ``_compose_task_option``
    # raises MissingMandatoryValue otherwise. The exact interval is operational
    # config, not a contract, so it is not pinned (only the runtime validity >= 0).
    assert task_cfg.checkpoint.interval >= 0


def test_motrix_fastsac_asynchronous_switches_trainer(tmp_path):
    import motrix_envs  # noqa: F401 registers built-in environments
    from motrix_rl.fastsac.async_impl.train import Trainer as AsyncTrainer
    from motrix_rl.fastsac.config import FastSacAsyncOptionsCfg
    from motrix_rl.fastsac.sync.train import Trainer as SyncTrainer

    async_cfg = _compose_task_option(
        "g1-walk-flat",
        "motrix",
        "fastsac",
        overrides=["algo.trainer.async_options.utd_mode=learner_bound"],
    )
    sync_cfg = _compose_task_option(
        "g1-walk-flat",
        "motrix",
        "fastsac",
        overrides=["algo.asynchronous=false"],
    )

    assert async_cfg.algo.asynchronous is True
    assert sync_cfg.algo.asynchronous is False
    assert isinstance(async_cfg.algo.trainer.async_options, FastSacAsyncOptionsCfg)
    assert async_cfg.algo.trainer.async_options.utd_mode == "learner_bound"
    assert async_cfg.algo.trainer.async_options.collector_inference_device == "cuda"
    assert async_cfg.algo.trainer.async_options.collector_compile is True
    assert async_cfg.algo.trainer.async_options.collector_amp is True

    for index, (task_cfg, trainer_type) in enumerate(((sync_cfg, SyncTrainer), (async_cfg, AsyncTrainer))):
        run = runs.create_run_context(
            env_name=task_cfg.task.env,
            rllib=task_cfg.task.rllib,
            train_backend="torch",
            algo=task_cfg.task.algo,
            seed=task_cfg.seed,
            checkpoint_format="pt",
            runs_root=tmp_path / str(index),
        )
        runs.write_task_config(run.run_dir, task_cfg)

        assert isinstance(runner.create_run_handle(run).trainer, trainer_type)


def test_external_algo_schema_and_task_config_can_be_registered(tmp_path):
    @dataclass
    class ExternalCfg:
        value: int = MISSING

    class ExternalProvider(frameworks.AgentProvider[ExternalCfg]):
        config_type = ExternalCfg

        @property
        def train_backend(self) -> str:
            return "custom"

        @property
        def agent_name(self) -> str:
            return "demo"

        @property
        def checkpoint_format(self) -> str | None:
            return None

        def create_trainer(self, context):
            raise NotImplementedError

    class ExternalFramework(frameworks.RlFramework):
        @property
        def name(self) -> str:
            return "external_test"

    frameworks.register_framework(ExternalFramework((ExternalProvider(),)))

    (tmp_path / "algo_base").mkdir()
    (tmp_path / "task" / "external-env").mkdir(parents=True)
    (tmp_path / "algo_base" / "external_test.demo.yaml").write_text(
        "defaults:\n  - _external_test_demo_schema\n  - _self_\nvalue: 1\n"
    )
    (tmp_path / "task" / "external-env" / "external_test.demo.yaml").write_text(
        "# @package _global_\n"
        "defaults:\n  - /algo_base@algo: external_test.demo\n  - _self_\n"
        "task:\n  env: external-env\n  rllib: external_test\n  algo: demo\n  train_backend: null\n"
        "num_envs: 8\nplay_num_envs: 2\nseed: 3\nalgo:\n  value: 7\n"
    )
    (tmp_path / "train.yaml").write_text(
        "defaults:\n"
        "  - train_schema\n"
        "  - _self_\n"
        "  - task: external-env/external_test.demo\n"
        "logging:\n  backend: tensorboard\n  interval: 100\n"
        "checkpoint:\n  interval: 0\n"
    )

    cfg = _compose_task_option("external-env", "external_test", "demo", config_dir=tmp_path).algo

    assert isinstance(cfg, ExternalCfg)
    assert cfg.value == 7

    (tmp_path / "task" / "external-env" / "external_test.demo.custom.yaml").write_text(
        "# @package _global_\ndefaults:\n  - missing-parent\n  - _self_\n"
    )
    with pytest.raises(MissingConfigException, match="missing-parent"):
        _compose_task_option("external-env", "external_test", "demo", "custom", config_dir=tmp_path)
