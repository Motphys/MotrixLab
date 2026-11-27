# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore
from motrix_rl_supermimic.config import SuperMimicCfg

from motrix_rl import frameworks
from motrix_rl_builtin.framework import MotrixFramework


def register() -> None:
    frameworks.register_framework(MotrixFramework())
    # The implementation package owns the defaults; builtin owns their public name.
    ConfigStore.instance().store(group="algo_base", name="motrix.supermimic", node=SuperMimicCfg())
