# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Framework registration contracts: plugin install, agent lookup, algo defaults."""


def test_plugin_registers_framework_and_schema():
    from motrix_rl_supermimic.plugin import register

    from motrix_rl import frameworks

    register()

    assert frameworks.exists("supermimic")
    assert frameworks.supported_agents("supermimic") == ("supermimic",)
    provider = frameworks.get_agent_provider("supermimic", "supermimic", "torch")
    assert provider is not None and provider.agent_name == "supermimic"

    from hydra.core.config_store import ConfigStore

    node = ConfigStore.instance().repo["algo_base"]["supermimic.supermimic.yaml"].node
    assert node["trainer"]["horizon"] == 32


def test_total_env_steps():
    from motrix_rl_supermimic.config import SuperMimicCfg, TrainerCfg
    from motrix_rl_supermimic.framework import SuperMimicProvider

    cfg = SuperMimicCfg(trainer=TrainerCfg(iterations=100))
    assert SuperMimicProvider().total_env_steps(cfg, 8) == 800
