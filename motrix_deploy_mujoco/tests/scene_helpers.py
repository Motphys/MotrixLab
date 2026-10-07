# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Complete scenes assembled locally by adapter tests."""

from motrix_robots.unitree import UnitreeGo2Robot

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, SceneCfg, SceneObjsCfg


@configclass
class AdapterSceneObjsCfg(SceneObjsCfg):
    floor: FlatTerrainCfg | None = None


def build_scene() -> SceneCfg:
    return SceneCfg(
        objs=AdapterSceneObjsCfg(
            robot=UnitreeGo2Robot(translation=(0.0, 0.0, -0.114)),
            floor=FlatTerrainCfg(friction=(0.6, 0.005, 0.0001)),
        )
    )
