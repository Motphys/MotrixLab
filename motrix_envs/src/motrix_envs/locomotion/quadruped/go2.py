# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Go2 flat- and rough-terrain walk configuration and environment registration."""

from dataclasses import replace

import numpy as np
from motrix_robots.unitree import UnitreeGo2Robot

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, HFieldTerrainCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.quadruped.cfg import (
    Commands,
    QuadrupedSceneCfg,
    QuadrupedWalkEnvCfg,
    QuadrupedWalkRandomizationCfg,
    QuadrupedWalkTerrainSceneAssetsCfg,
    RewardConfig,
    RewardScales,
    VelocityCommandCfg,
)
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask


@registry.envcfg("go2-walk-flat")
@configclass
class Go2WalkDirectEnvCfg(QuadrupedWalkEnvCfg):
    """Track walking commands with Unitree Go2 on flat ground.

    zh_CN: 控制 Unitree Go2 在平地上跟踪行走指令。
    """

    render_spacing: float = 0.0
    spawn_xy_range: float = 4.0
    commands: Commands = Commands(
        velocity=VelocityCommandCfg(
            lower=np.array([-1.0, -0.5, -0.5], dtype=np.float32),
            upper=np.array([1.0, 0.5, 0.5], dtype=np.float32),
            standing_probability=0.1,
            resampling_seconds_range=(5.0, 10.0),
        )
    )
    randomization: QuadrupedWalkRandomizationCfg = QuadrupedWalkRandomizationCfg(
        enabled=True,
        joint_pos_noise=0.03,
        joint_vel_noise=0.1,
        base_lin_vel_noise=(0.1, 0.1, 0.0),
        base_ang_vel_noise=(0.1, 0.1, 0.1),
        action_delay_steps=(0, 1),
        kp_scale_range=(0.9, 1.1),
        damping_scale_range=(0.9, 1.1),
        sliding_friction_range=(0.4, 0.9),
        base_mass_scale_range=(0.9, 1.1),
        base_com_offset_noise=(0.01, 0.01, 0.005),
    )
    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(
        objs=StandardSceneObjsCfg(
            floor=FlatTerrainCfg(
                material="mat_ground",
                friction=(0.6, 0.005, 0.0001),
            ),
            robot=UnitreeGo2Robot(),
        ),
    )

    reward_config: RewardConfig = RewardConfig(
        scales=RewardScales(
            similar_to_default=-1,
        ),
        tracking_ang_vel_sigma=0.05,
        target_foot_height=0.05,
        base_height_target=0.35,
    )

    def for_play(self) -> "Go2WalkDirectEnvCfg":
        """Return a deterministic Go2 evaluation config."""

        cfg = super().for_play()
        return replace(cfg, noise_config=replace(cfg.noise_config, level=0.0))


@registry.envcfg("go2-walk-rough")
@configclass
class Go2WalkRoughDirectEnvCfg(Go2WalkDirectEnvCfg):
    """Track walking commands with Unitree Go2 on a procedural rough height field.

    zh_CN: 控制 Unitree Go2 在程序化粗糙高度场上跟踪行走指令。
    """

    commands: Commands = Commands(
        velocity=VelocityCommandCfg(
            lower=np.array([-0.5, -0.4, -1.0], dtype=np.float32),
            upper=np.array([1.0, 0.4, 1.0], dtype=np.float32),
            standing_probability=0.1,
        )
    )
    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(
        assets=QuadrupedWalkTerrainSceneAssetsCfg(),
        objs=StandardSceneObjsCfg(
            floor=HFieldTerrainCfg(
                hfield="terrain",
                material="mat_ground",
                friction=(0.6, 0.005, 0.0001),
            ),
            robot=UnitreeGo2Robot(),
        ),
    )
    reward_config: RewardConfig = RewardConfig(
        target_foot_height=0.1,
    )


registry.env("go2-walk-flat")(QuadrupedWalkTask)
registry.env("go2-walk-rough")(QuadrupedWalkTask)
