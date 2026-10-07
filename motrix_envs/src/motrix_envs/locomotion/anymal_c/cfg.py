# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_robots.anymal import AnymalC

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg, SceneObjsCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/xmls/scene.xml"


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
    # stiffness[N*m/rad] uses kp parameter from XML, recorded for reference only
    # damping[N*m*s/rad] uses kv parameter from XML, recorded for reference only
    action_scale = 0.06  # action scale


@configclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0, 0.5]  # Z-axis height matches the initial height of base in XML

    # position randomization range [x_min, y_min, x_max, y_max]
    pos_randomization_range = [-10.0, -10.0, 10.0, 10.0]  # randomly distributed over 20m x 20m range on ground

    # the default angles for all joints. key = joint name, value = target angle [rad]
    default_joint_angles = {
        "LF_HAA": 0.0,  # [rad]
        "RF_HAA": 0.0,  # [rad]
        "LH_HAA": 0.0,  # [rad]
        "RH_HAA": 0.0,  # [rad]
        "LF_HFE": 0.4,  # [rad]
        "RF_HFE": 0.4,  # [rad]
        "LH_HFE": -0.4,  # [rad]
        "RH_HFE": -0.4,  # [rad]
        "LF_KFE": -0.8,  # [rad]
        "RF_KFE": -0.8,  # [rad]
        "LH_KFE": 0.8,  # [rad]
        "RH_KFE": 0.8,  # [rad]
    }


@configclass
class Commands:
    # offset range of target position relative to robot initial position
    # [dx_min, dy_min, yaw_min, dx_max, dy_max, yaw_max]
    # dx/dy: offset relative to robot initial position (meters)
    # yaw: target absolute orientation (radians), random horizontal direction
    pose_command_range = [-5.0, -5.0, -3.14, 5.0, 5.0, 3.14]


@configclass
class Normalization:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05


@configclass
class Asset:
    body_name = "base"
    foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
    terminate_after_contacts_on = ["base"]
    ground_name = "ground"


@configclass
class Sensor:
    base_linvel = "base_linvel"
    base_gyro = "base_gyro"


@configclass
class RewardConfig:
    scales: dict[str, float] = {
        "termination": -400.0,
        "position_tracking": 0.5,
        "fine_position_tracking": 0.5,
        "orientation": -0.2,
    }


@registry.envcfg("anymal_c_navigation_flat")
@configclass
class AnymalCEnvCfg(DirectEnvCfg):
    """Navigate ANYmal-C toward a target on flat ground.

    zh_CN: 控制 ANYmal-C 在平地上朝目标位置导航。
    """

    scene: SceneCfg = SceneCfg(file=model_file, objs=SceneObjsCfg(robot=AnymalC()))
    reset_noise_scale: float = 0.01
    max_episode_seconds: float = 7.0
    sim: SimCfg = SimCfg(dt=0.01)
    ctrl_dt: float = 0.01
    reset_yaw_scale: float = 0.1
    max_dof_vel: float = 100.0  # maximum joint velocity threshold, greater tolerance during early training

    noise_config: NoiseConfig = NoiseConfig()
    control_config: ControlConfig = ControlConfig()
    reward_config: RewardConfig = RewardConfig()
    init_state: InitState = InitState()
    commands: Commands = Commands()
    normalization: Normalization = Normalization()
    asset: Asset = Asset()
    sensor: Sensor = Sensor()
