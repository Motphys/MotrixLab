# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Pollen Robotics Microduck humanoid velocity-tracking presets and registration."""

from dataclasses import replace

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import HFieldTerrainCfg, SystemCameraCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.humanoid import cfg as humanoid_cfg
from motrix_envs.locomotion.humanoid.walk_np import HumanoidVelocityTrackingEnv
from motrix_envs.robot import Microduck


def _make_microduck_robot() -> Microduck:
    return Microduck()


# Pinned Microduck termination collision inventory, owned by the Microduck walk task.
_MICRODUCK_TERMINATION_GEOMS = ("trunk_collision",)


@registry.envcfg("microduck-walk-flat")
def make_microduck_walk_flat_cfg() -> humanoid_cfg.HumanoidVelocityTrackingEnvCfg:
    """Track walking commands with Microduck on flat ground.

    zh_CN: 控制 Microduck 小型双足机器人在平地上跟踪行走指令。
    """

    return humanoid_cfg.HumanoidVelocityTrackingEnvCfg(
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            # Microduck is a ~25 cm robot; frame the lead env much closer than
            # the full-size humanoid default camera (grid center at z=0.75).
            system_camera=SystemCameraCfg(
                lookat=(0.0, 0.0, 0.12),
                distance=0.35,
                elevation=-20.0,
                azimuth=180.0,
            ),
            objs=StandardSceneObjsCfg(robot=_make_microduck_robot()),
        ),
        control_config=humanoid_cfg.ControlCfg(action_scale=0.5),
        commands=humanoid_cfg.CommandsCfg(
            vel_limit=[
                [-1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0],
            ],
        ),
        gait=humanoid_cfg.GaitCfg(
            period=0.5,
            swing_height=0.04,
            feet_phase_sigma=0.002,
        ),
        reward_config=humanoid_cfg.RewardCfg(
            scales=humanoid_cfg.RewardScales(
                tracking_lin_vel=10.0,
                tracking_ang_vel=3.0,
                penalty_action_rate=-0.5,
                feet_phase=8.0,
            ),
            tracking_sigma=0.15,
            close_feet_threshold=0.05,
            pose_weights={
                "left_hip_yaw": 5.0,
                "left_hip_roll": 1.0,
                "left_hip_pitch": 0.01,
                "left_knee": 0.01,
                "left_ankle": 5.0,
                "neck_pitch": 50.0,
                "head_pitch": 50.0,
                "head_yaw": 50.0,
                "head_roll": 50.0,
                "right_hip_yaw": 5.0,
                "right_hip_roll": 1.0,
                "right_hip_pitch": 0.01,
                "right_knee": 0.01,
                "right_ankle": 5.0,
            },
        ),
        asset=humanoid_cfg.AssetCfg(
            foot_height_site_names=("left_foot", "right_foot"),
            ground_geom_name="floor",
            terminate_contact_geom_names=_MICRODUCK_TERMINATION_GEOMS,
        ),
        sim=SimCfg(dt=0.005, solver_iterations=6, solver_tolerance=1e-4),
        spawn_xy_range=0.0,
    )


@registry.envcfg("microduck-walk-rough")
def make_microduck_walk_rough_cfg() -> humanoid_cfg.HumanoidVelocityTrackingEnvCfg:
    """Track walking commands with Microduck over uneven terrain.

    zh_CN: 控制 Microduck 小型双足机器人在起伏地形上跟踪行走指令。
    """

    return replace(
        make_microduck_walk_flat_cfg(),
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            assets=humanoid_cfg.TerrainSceneAssetsCfg(),
            system_camera=SystemCameraCfg(
                lookat=(0.0, 0.0, 0.12),
                distance=0.35,
                elevation=-20.0,
                azimuth=180.0,
            ),
            objs=StandardSceneObjsCfg(
                floor=HFieldTerrainCfg(
                    hfield="terrain",
                    material="mat_ground",
                ),
                robot=_make_microduck_robot(),
            ),
        ),
        spawn_xy_range=4.0,
        render_spacing=0.0,
    )


registry.env("microduck-walk-flat")(HumanoidVelocityTrackingEnv)
registry.env("microduck-walk-rough")(HumanoidVelocityTrackingEnv)
