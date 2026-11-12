# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from hydra.core.config_store import ConfigStore

from motrix_rl import frameworks
from motrix_rl_supermimic.config import SuperMimicCfg
from motrix_rl_supermimic.framework import SuperMimicFramework


_registering = False


def register() -> None:
    # Idempotent: load_plugins() may have already registered this framework
    # through the installed entry point (any registry query does the same,
    # so guard against re-entry instead of querying the registry here).
    global _registering
    if _registering:
        return
    _registering = True
    try:
        frameworks.register_framework(SuperMimicFramework())
    except ValueError:
        # Already registered through the installed entry point (load_plugins)
        # or an earlier call; both install the same schema.
        return
    finally:
        _registering = False
    # Task recipes reference ``/algo_base@algo: supermimic.supermimic``; the
    # typed schema node is installed by framework registration, this node
    # carries the overridable default values.
    ConfigStore.instance().store(group="algo_base", name="supermimic.supermimic", node=SuperMimicCfg())
