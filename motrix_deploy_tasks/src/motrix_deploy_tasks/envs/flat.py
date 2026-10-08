# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""A robot-free flat-ground deployment world."""

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, SceneCfg, SceneObjsCfg, SystemCameraCfg
from motrix_env_core.config.scene.base import SceneVisualCfg
from motrix_env_core.config.scene.light import LightCfg


@configclass
class FlatWorldObjsCfg(SceneObjsCfg):
    floor: FlatTerrainCfg = FlatTerrainCfg(friction=(0.6, 0.005, 0.0001))
    sun: LightCfg = LightCfg(color=(0.7, 0.7, 0.7), illuminance=10_000.0)


def build_scene() -> SceneCfg:
    """Build fresh flat ground with an empty robot slot."""
    return SceneCfg(
        objs=FlatWorldObjsCfg(),
        # Front view matching the G1 WBT training scene (g1/common.py).
        system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
        visual=SceneVisualCfg(
            ambient_light_color=(0.3, 0.3, 0.3),
            ambient_light_brightness=1_000.0,
            head_light_color=(0.6, 0.6, 0.6),
            head_light_luminous_power=1_000.0,
            tone_mapping="none",
        ),
    )
