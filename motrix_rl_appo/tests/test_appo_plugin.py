# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore
from motrix_rl_appo.config import AppoCfg
from motrix_rl_appo.provider import AppoProvider
from omegaconf import OmegaConf

from motrix_rl import frameworks


def test_installed_plugin_discovery_and_structured_defaults() -> None:
    provider = frameworks.get_agent_provider("motrix", "appo", "torch")
    assert isinstance(provider, AppoProvider)
    node = ConfigStore.instance().load("algo_base/motrix.appo.yaml").node
    config = OmegaConf.to_object(node)
    assert isinstance(config, AppoCfg)
    assert provider.total_env_steps(config, 7) == config.max_iterations * config.rollout_steps * 7
