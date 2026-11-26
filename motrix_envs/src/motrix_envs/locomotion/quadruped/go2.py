# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Go2 flat-ground v2 walk configuration and environment registration."""

from dataclasses import replace

import numpy as np
from motrix_robots.unitree import UnitreeGo2Robot

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.quadruped.cfg import (
    Commands,
    NoiseConfig,
    QuadrupedSceneCfg,
    QuadrupedWalkEnvCfg,
    QuadrupedWalkRandomizationCfg,
    RewardConfig,
    RewardScales,
    VelocityCommandCfg,
)
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask


@registry.envcfg("go2-walk-flat-v2")
@configclass
class Go2WalkFlatV2DirectEnvCfg(QuadrupedWalkEnvCfg):
    """Track walking commands with a fast-converging yaw-focused v2 recipe.

    zh_CN: 使用快速收敛且聚焦 yaw 能力的 v2 配方控制 Unitree Go2 行走。
    """

    render_spacing: float = 0.0
    spawn_xy_range: float = 4.0
    max_episode_seconds: float = 20.0
    noise_config: NoiseConfig = NoiseConfig(scale_gravity=0.05)
    commands: Commands = Commands(
        velocity=VelocityCommandCfg(
            lower=np.array([-1.0, -0.5, -1.0], dtype=np.float32),
            upper=np.array([1.0, 0.5, 1.0], dtype=np.float32),
            standing_probability=0.50,
            hard_case_probability=0.20,
            hard_cases=(
                (0.0, 0.4, 0.0),
                (0.0, -0.4, 0.0),
                (0.0, 0.5, 0.0),
                (0.0, -0.5, 0.0),
                (0.8, 0.4, 0.0),
                (0.8, -0.4, 0.0),
                (-0.8, 0.4, 0.0),
                (-0.8, -0.4, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 0.5, 0.0),
                (1.0, -0.5, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
                (-1.0, 0.5, 0.0),
                (-1.0, -0.5, 0.0),
                (0.8, 0.4, 0.8),
                (0.8, 0.4, -0.8),
                (0.8, -0.4, 0.8),
                (0.8, -0.4, -0.8),
                (-0.8, 0.4, 0.8),
                (-0.8, 0.4, -0.8),
                (-0.8, -0.4, 0.8),
                (-0.8, -0.4, -0.8),
                (0.0, 0.0, 0.5),
                (0.0, 0.0, -0.5),
                (0.0, 0.0, 0.8),
                (0.0, 0.0, -0.8),
            ),
            # Reach the terminal hard-case probability earlier than v1 so
            # the 2000-iteration recipe still sees enough difficult commands.
            hard_case_curriculum=((0, 0.0), (200, 0.0), (450, 0.08), (700, 0.20), (900, 0.20), (2000, 0.20)),
            hard_case_curriculum_steps_per_iteration=24,
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
        sliding_friction_range=(0.6, 1.4),
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
            # v2 raises the yaw-tracking signal to retain rotation quality
            # under the smaller v2 training sample budget.
            tracking_lin_vel=4.5,
            feet_spacing=0.5,
            tracking_ang_vel=1.6,
            action_rate=-0.12,
            dof_acc=-2e-5,
            ang_vel_xy=-0.5,
            # Candidate A full run: soften linear tracking from the V203
            # recipe to reduce lateral/combined command overtracking.
            flat_orientation=-4.0,
            swing_feet_z=6.0,
            swing_contact=-1.5,
            touchdown_velocity=-0.8,
            similar_to_default=-0.1,
            hip_position=-1.0,
            base_com_position=-0.5,
            base_height=-300.0,
        ),
        excluded_rewards=(),
        tracking_lin_vel_sigma=0.35,
        tracking_lin_vel_forward_sigma=0.16,
        tracking_lin_vel_lateral_sigma=0.25,
        swing_contact_forward_scale=1.5,
        tracking_ang_vel_sigma=0.05,
        target_foot_height=0.09,
        swing_fraction=0.45,
        swing_feet_z_sinusoidal=True,
        swing_feet_height_sigma=0.05,
        foot_height_reference_offset=0.04232,
        swing_rise_fraction=0.35,
        swing_plateau_fraction=0.30,
        touchdown_height_threshold=0.03,
        touchdown_speed_sigma=0.35,
        touchdown_speed_clip=3.0,
        feet_spacing_tolerance=0.03,
        feet_spacing_sigma=0.03,
        hip_position_tolerance=0.08,
        hip_position_sigma=0.08,
        base_height_target=0.305,
    )

    def for_play(self) -> "Go2WalkFlatV2DirectEnvCfg":
        """Return a deterministic Go2 evaluation config."""

        cfg = super().for_play()
        return replace(cfg, noise_config=replace(cfg.noise_config, level=0.0))


registry.env("go2-walk-flat-v2")(QuadrupedWalkTask)
