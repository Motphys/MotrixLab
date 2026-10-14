# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Deterministic procedural rough ground, independent of robots and RL tasks."""

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    HFieldTerrainCfg,
    NoiseTerrainGeneratorCfg,
    ProceduralHFieldAssetCfg,
    SceneAssetsCfg,
    SceneCfg,
    SceneObjsCfg,
    SystemCameraCfg,
)


@configclass
class RoughWorldAssetsCfg(SceneAssetsCfg):
    # Match the built-in quadruped rough-walk terrain without importing its
    # robot-scoped scene or training task configuration.
    terrain: ProceduralHFieldAssetCfg = ProceduralHFieldAssetCfg(
        generator=NoiseTerrainGeneratorCfg(seed=0, height_scale=0.1, flip_y=True),
        size=(64.0, 64.0),
        shape=(320, 320),
    )


@configclass
class RoughWorldObjsCfg(SceneObjsCfg):
    floor: HFieldTerrainCfg = HFieldTerrainCfg(hfield="terrain", friction=(0.6, 0.005, 0.0001))


def build_scene() -> SceneCfg:
    """Build fresh seeded rough ground with an empty robot slot."""
    # Front view matching the quadruped training scenes (e.g. quadruped/cfg.py).
    return SceneCfg(
        assets=RoughWorldAssetsCfg(),
        objs=RoughWorldObjsCfg(),
        system_camera=SystemCameraCfg(distance=10.0, elevation=-25.0, azimuth=90.0),
    )
