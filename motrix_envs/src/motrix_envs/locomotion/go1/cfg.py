# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from motrix_robots.unitree import UNITREE_GO1_ASSET_DIR, UnitreeGo1Robot

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import MjcfFileCfg, SceneCfg, SceneObjsCfg
from motrix_env_core.direct.env import DirectEnvCfg


@configclass
class NoiseConfig:
    level: float = 1.0
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@configclass
class ControlConfig:
    stiffness = 80  # [N*m/rad]
    damping = 1  # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale = 0.05


@configclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0, 0.42]

    # the default angles for all joints. key = joint name, value = target angle [rad]
    default_joint_angles = {
        "FL_hip": 0.0,  # [rad]
        "RL_hip": 0.0,  # [rad]
        "FR_hip": -0.0,  # [rad]
        "RR_hip": -0.0,  # [rad]
        "FL_thigh": 0.9,  # [rad]
        "RL_thigh": 0.9,  # [rad]
        "FR_thigh": 0.9,  # [rad]
        "RR_thigh": 0.9,  # [rad]
        "FL_calf": -1.8,  # [rad]
        "RL_calf": -1.8,  # [rad]
        "FR_calf": -1.8,  # [rad]
        "RR_calf": -1.8,  # [rad]
    }


@configclass
class Commands:
    vel_limit = [
        [-1.0, -1.0, -1.0],  # min: vel_x [m/s], vel_y [m/s], ang_vel [rad/s]
        [2.0, 1.0, 1.0],  # max
    ]


@configclass
class Normalization:
    lin_vel = 2
    ang_vel = 0.25
    dof_pos = 1
    dof_vel = 0.05


@configclass
class Asset:
    body_name = "trunk"
    foot_name = "foot"
    ground_name = "floor"
    penalize_contacts_on = ["thigh", "calf"]
    terminate_after_contacts_on = ["trunk"]


@configclass
class Sensor:
    local_linvel = "local_linvel"
    gyro = "gyro"
    feet = ["FR", "FL", "RR", "RL"]


# -- docs-tag-start: go1-reward-config --
@configclass
class RewardConfig:
    scales: dict[str, float] = {
        "termination": -0.0,
        "tracking_lin_vel": 1.0,
        "tracking_ang_vel": 0.5,
        "lin_vel_z": -2.0,
        "ang_vel_xy": -0.05,
        "orientation": -0.0,
        "torques": -0.00001,
        "dof_vel": -0.0,
        "dof_acc": -2.5e-7,
        "base_height": -0.0,
        "feet_air_time": 1.0,
        "collision": -1.0 * 0,
        "action_rate": -0.001,
        "stand_still": -0.0,
        "hip_pos": -1,
        "calf_pos": -0.3 * 0,
    }

    tracking_sigma: float = 0.25
    max_foot_height: float = 0.1


# -- docs-tag-end: go1-reward-config --


@configclass
class Go1TerrainWalkDirectEnvCfg(DirectEnvCfg):
    max_episode_seconds: float = 20.0
    noise_config: NoiseConfig = NoiseConfig()
    control_config: ControlConfig = ControlConfig()
    reward_config: RewardConfig = RewardConfig()
    init_state: InitState = InitState()
    commands: Commands = Commands()
    normalization: Normalization = Normalization()
    asset: Asset = Asset()
    sensor: Sensor = Sensor()
    sim: SimCfg = SimCfg(dt=0.01)
    ctrl_dt: float = 0.01


@registry.envcfg("go1-stairs-terrain-walk")
@configclass
class Go1WalkDirectStairsEnvCfg(Go1TerrainWalkDirectEnvCfg):
    """Control Unitree Go1 to walk over stair terrain.

    zh_CN: 控制 Unitree Go1 在台阶地形上行走。
    """

    render_spacing: float = 0.0
    scene: SceneCfg = SceneCfg(
        file=Path(__file__).parent / "xmls" / "scene_stairs_terrain.xml",
        objs=SceneObjsCfg(
            robot=UnitreeGo1Robot(model=MjcfFileCfg(file=UNITREE_GO1_ASSET_DIR / "go1_motor_actuator.xml"))
        ),
    )
    # Pinned to the scene's geom inventory (single hfield "floor"): the legacy
    # env matched ground/foot geoms by substring over exactly these names.
    # Foot contact-force sensors follow Sensor.feet order.

    @configclass
    class Commands:
        vel_limit = [
            [0.5, -0.0, 0.0],  # min: vel_x [m/s], vel_y [m/s], ang_vel [rad/s]
            [1.0, 0.0, 0.0],  # max
        ]

    @configclass
    class RewardConfig:
        scales: dict[str, float] = {
            "termination": -0.0,
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -0.0,
            "torques": -0.00001,
            "dof_vel": -0.0,
            "dof_acc": -2.5e-7,
            "base_height": -0.0,
            "feet_air_time": 1.0,
            "collision": -1.0 * 0,
            "feet_stumble": -0.1,
            "action_rate": -0.001,
            "stand_still": -0.0,
            "hip_pos": -1,
            "calf_pos": -0.3 * 0,
        }

        tracking_sigma: float = 0.25
        max_foot_height: float = 0.1

    commands: Commands = Commands()
    reward_config: RewardConfig = RewardConfig()
