# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/peg_insert_scene_rm65.xml"


@configclass
class InitState:
    joint_names = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper_Left_1_Joint",
    ]
    default_joint_pos = np.array([0.09, 0.71, 0.92, -0.18, 1.19, -0.85, 0.0], np.float32)
    joint_pos_reset_noise_scale = 0.005


@configclass
class ControlConfig:
    actuators = ["actuator1", "actuator2", "actuator3", "actuator4", "actuator5", "actuator6", "actuator7"]
    min_pos = [-2.0, -1.5, -1.5, -2.0, -1.5, -1.57, -0.91]
    max_pos = [2.0, 1.5, 1.5, 2.0, 1.5, 1.57, 0.0]


@configclass
class PegInsertConfig:
    peg_length: float = 0.1
    peg_radius: float = 0.015
    socket_depth: float = 0.08
    socket_radius: float = 0.016
    insertion_threshold: float = 0.045
    success_threshold: float = 0.003
    grasp_threshold: float = 0.04
    grasp_height_threshold: float = 0.06
    safe_approach_height: float = 0.08
    grasp_target_height: float = 0.025
    descend_xy_threshold: float = 0.035


@configclass
class RewardConfig:
    approach_weight: float = 10.0
    reach_weight: float = 10.0
    grasp_weight: float = 45.0
    lift_weight: float = 35.0
    insert_weight: float = 140.0
    align_weight: float = 60.0
    success_bonus: float = 1500.0
    action_penalty_weight: float = 0.01
    joint_vel_penalty_weight: float = 0.0005
    precision_weight: float = 32.0
    orientation_weight: float = 8.0
    grasp_success_bonus: float = 350.0
    anti_shake_weight: float = 0.003
    hand_approach_bonus: float = 6.0
    gripper_close_bonus: float = 20.0
    peg_to_socket_approach_weight: float = 48.0
    depth_insert_bonus: float = 44.0


@configclass
class ActionConfig:
    action_scale: float = 0.025
    gripper_closed: float = -0.91
    gripper_open: float = 0.0


@configclass
class PegInitConfig:
    x_range: tuple = (0.50, 0.62)
    y_range: tuple = (-0.10, 0.10)
    z_pos: float = 0.052
    z_noise_scale: float = 0.0015
    min_socket_xy_dist: float = 0.08
    max_socket_xy_dist: float = 0.18
    max_sample_attempts: int = 32


@registry.envcfg("rm65_insert_peg")
@registry.envcfg("peg-insert")
@configclass
class PegInsertEnvCfg(DirectEnvCfg):
    """Control RM65 to grasp, align, and insert a peg into a socket.

    zh_CN: 控制 RM65 抓取插销、对准插座并完成插入。
    """

    render_spacing: float = 2.0
    scene: SceneCfg = SceneCfg(file=model_file)
    max_episode_seconds: float = 3.0
    sim: SimCfg = SimCfg(dt=0.002, max_contacts=128, max_contact_constraints=384)
    move_speed: float = 0.5
    ctrl_dt: float = 0.01
    reset_noise_scale: float = 0.005

    init_state: InitState = InitState()
    control_config: ControlConfig = ControlConfig()
    peg_config: PegInsertConfig = PegInsertConfig()
    reward_config: RewardConfig = RewardConfig()
    action_config: ActionConfig = ActionConfig()
    peg_init_config: PegInitConfig = PegInitConfig()
