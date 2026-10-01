# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_rl_rslrl.framework import RslrlPpoProvider

from motrix_rl import frameworks


def test_rslrl_plugin_registers_ppo():
    assert isinstance(frameworks.get_agent_provider("rslrl", "ppo", "torch"), RslrlPpoProvider)
