# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/xmls/mjx_scene.xml"


@configclass
class InitState:
    # robot joint names and default positions [rad]
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
        "finger_joint1",
        "finger_joint2",
    ]
    default_joint_pos = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04], np.float32)
    joint_pos_reset_noise_scale = 0.125


@configclass
class ControlConfig:
    # Position control
    # The actuator defined in xml file is <position ..../>
    # From ctrlrange in actuator in xml
    # Using position control and action as offset effectively solves the problem of large joint angle changes
    actuators = ["actuator1", "actuator2", "actuator3", "actuator4", "actuator5", "actuator6", "actuator7", "actuator8"]
    min_pos = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -np.pi / 2, 0]
    max_pos = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, np.pi / 2, 0.04]


@configclass
class Commands:
    target_pos_x = [0.4, 0.6]
    target_pos_y = [-0.25, 0.25]
    target_pos_z = [0.25, 0.5]


@configclass
class Asset:
    ground_name = "table"
    terminate_after_contacts_on = ["left_finger_pad", "left_finger_pad"]


@registry.envcfg("franka-lift-cube")
@configclass
class FrankaLiftCubeEnvCfg(DirectEnvCfg):
    """Control a Franka arm to grasp and lift a cube.

    zh_CN: 控制 Franka 机械臂抓取并抬升立方体。
    """

    render_spacing: float = 2.0
    scene: SceneCfg = SceneCfg(file=model_file)
    max_episode_seconds: float = 2.5
    sim: SimCfg = SimCfg(dt=0.01, max_contacts=128, max_contact_constraints=384)
    move_speed: float = 1.0
    ctrl_dt: float = 0.01
    reset_noise_scale = 0.05

    init_state: InitState = InitState()
    control_config: ControlConfig = ControlConfig()
    command_config: Commands = Commands()
    asset: Asset = Asset()
