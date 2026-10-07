# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""A robot-free flat-ground deployment world."""

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, SceneCfg, SceneObjsCfg


@configclass
class FlatWorldObjsCfg(SceneObjsCfg):
    floor: FlatTerrainCfg = FlatTerrainCfg(friction=(0.6, 0.005, 0.0001))


def build_scene() -> SceneCfg:
    """Build fresh flat ground with an empty robot slot."""
    return SceneCfg(objs=FlatWorldObjsCfg())
