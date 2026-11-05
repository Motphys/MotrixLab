# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import pytest

from motrix_env_core.base import EnvCfg  # noqa: E402
from motrix_env_core.config.scene import SceneCfg  # noqa: E402
from motrix_env_core.manager import ManagerEnv  # noqa: E402


def test_manager_env_requires_manager_config() -> None:
    with pytest.raises(TypeError, match="must inherit ManagerBasedEnvCfg"):
        ManagerEnv(EnvCfg(scene=SceneCfg()))
