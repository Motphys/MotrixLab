# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest
from motrix_rl_builtin.fastsac.config import FastSacCfg
from motrix_rl_builtin.fastsac.provider import FastSacProvider
from motrix_rl_supermimic.config import SuperMimicCfg
from motrix_rl_supermimic.provider import SuperMimicProvider

from motrix_rl import frameworks


@pytest.mark.integration
def test_builtin_plugin_registers_algorithms():
    assert frameworks.supported_agents("motrix") == ("fastsac", "supermimic")
    assert frameworks.get_config_type("motrix", "fastsac") is FastSacCfg
    assert frameworks.supported_train_backends("motrix", "fastsac") == ("torch",)
    assert isinstance(frameworks.get_agent_provider("motrix", "fastsac", "torch"), FastSacProvider)
    assert frameworks.get_config_type("motrix", "supermimic") is SuperMimicCfg
    assert frameworks.supported_train_backends("motrix", "supermimic") == ("torch",)
    assert isinstance(frameworks.get_agent_provider("motrix", "supermimic", "torch"), SuperMimicProvider)


@pytest.mark.integration
def test_supermimic_defaults_compose_to_provider_config():
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from motrix_rl.plugins import load_plugins

    load_plugins()
    with initialize_config_module(version_base=None, config_module="motrix_rl.configs"):
        cfg = compose(config_name=None, overrides=["+algo_base@algo=motrix.supermimic"])

    assert isinstance(OmegaConf.to_object(cfg.algo), SuperMimicCfg)


@pytest.mark.parametrize(("iterations", "num_envs", "expected"), [(1, 1, 1), (100, 64, 6400), (0, 8, 0)])
def test_fastsac_total_env_steps_counts_one_batch_per_iteration(iterations, num_envs, expected):
    cfg = FastSacCfg()
    cfg.trainer.num_learning_iterations = iterations

    assert FastSacProvider().total_env_steps(cfg, num_envs) == expected
