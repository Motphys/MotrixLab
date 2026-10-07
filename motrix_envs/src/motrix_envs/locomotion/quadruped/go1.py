# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Go1 flat- and rough-terrain walk configuration and environment registration."""

from motrix_robots.unitree import UnitreeGo1Robot

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, HFieldTerrainCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.quadruped.cfg import (
    ControlConfig,
    QuadrupedSceneCfg,
    QuadrupedWalkEnvCfg,
    QuadrupedWalkTerrainSceneAssetsCfg,
    RewardConfig,
    RewardScales,
)
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask


@registry.envcfg("go1-walk-flat")
@configclass
class Go1WalkDirectEnvCfg(QuadrupedWalkEnvCfg):
    """Track walking commands with Unitree Go1 on flat ground.

    zh_CN: 控制 Unitree Go1 在平地上跟踪行走指令。
    """

    render_spacing: float = 0.0
    spawn_xy_range: float = 4.0
    control_config: ControlConfig = ControlConfig(action_scale=0.1)
    reward_config: RewardConfig = RewardConfig(
        scales=RewardScales(
            similar_to_default=-0.03,
            swing_contact=-1.0,
        ),
        # Tighten yaw-rate tracking (default 0.25) so the gait does not yaw/curve
        # while still satisfying the body-frame velocity-tracking reward.
        tracking_ang_vel_sigma=0.05,
        base_height_target=0.275,
    )
    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(
        objs=StandardSceneObjsCfg(
            floor=FlatTerrainCfg(
                material="mat_ground",
                friction=(0.6, 0.005, 0.0001),
            ),
            robot=UnitreeGo1Robot(),
        ),
    )


@registry.envcfg("go1-walk-rough")
@configclass
class Go1WalkRoughDirectEnvCfg(Go1WalkDirectEnvCfg):
    """Track walking commands with Unitree Go1 on a procedural rough height field.

    zh_CN: 控制 Unitree Go1 在程序化粗糙高度场上跟踪行走指令。
    """

    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(
        assets=QuadrupedWalkTerrainSceneAssetsCfg(),
        objs=StandardSceneObjsCfg(
            floor=HFieldTerrainCfg(
                hfield="terrain",
                material="mat_ground",
                friction=(0.6, 0.005, 0.0001),
            ),
            robot=UnitreeGo1Robot(),
        ),
    )


registry.env("go1-walk-flat")(QuadrupedWalkTask)
registry.env("go1-walk-rough")(QuadrupedWalkTask)
