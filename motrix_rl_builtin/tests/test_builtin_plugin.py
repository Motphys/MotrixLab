# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl_builtin.fastsac.config import FastSacCfg
from motrix_rl_builtin.fastsac.framework import FastSacProvider

from motrix_rl import frameworks


def test_builtin_plugin_registers_fastsac():
    assert frameworks.supported_agents("motrix") == ("fastsac",)
    assert frameworks.get_config_type("motrix", "fastsac") is FastSacCfg
    assert frameworks.supported_train_backends("motrix", "fastsac") == ("torch",)
    assert isinstance(frameworks.get_agent_provider("motrix", "fastsac", "torch"), FastSacProvider)
