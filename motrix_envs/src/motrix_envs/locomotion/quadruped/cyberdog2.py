# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Cyberdog2 flat- and rough-terrain walk configuration and environment registration."""

from dataclasses import replace

import numpy as np
from motrix_robots.cyberdog2 import Cyberdog2Robot

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.quadruped.cfg import (
    Commands,
    QuadrupedSceneCfg,
    QuadrupedWalkEnvCfg,
    QuadrupedWalkRandomizationCfg,
    RewardConfig,
    RewardScales,
    Sensor,
    VelocityCommandCfg,
)
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask


@registry.envcfg("cyberdog2-walk-flat")
@configclass
class Cyberdog2WalkCfg(QuadrupedWalkEnvCfg):
    """Track walking commands with Cyberdog2 on flat ground.

    zh_CN: 控制 Cyberdog2 在平地上跟踪行走指令。
    """

    render_spacing: float = 0.0
    spawn_xy_range: float = 4.0
    commands: Commands = Commands(
        velocity=VelocityCommandCfg(
            lower=np.array([1.0, -0, -0], dtype=np.float32),
            upper=np.array([1.0, 0, 0], dtype=np.float32),
            standing_probability=0.1,
            resampling_seconds_range=(5.0, 10.0),
        )
    )

    # CyberDog2 keeps its own IMU sensor names; remap them to the task's
    # source-independent sensor roles instead of renaming the model sensors.
    sensor: Sensor = Sensor(
        gyro="imu_ang_vel",
        local_linvel="imu_lin_vel",
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
            robot=Cyberdog2Robot(),
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

    def for_play(self) -> "Cyberdog2WalkCfg":
        """Return a deterministic Cyberdog2 evaluation config."""

        cfg = super().for_play()
        return replace(cfg, noise_config=replace(cfg.noise_config, level=0.0))


registry.env("cyberdog2-walk-flat")(QuadrupedWalkTask)
