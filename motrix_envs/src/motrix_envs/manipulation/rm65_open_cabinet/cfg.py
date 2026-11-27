# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/xmls/scene.xml"


@configclass
class ResetConfig:
    joint_pos_noise_scale: float = 0


@configclass
class ArmControlConfig:
    action_mode: str = "joint_target"
    action_in_degrees: bool = False
    target_action_normalized: bool = True
    max_joint_speed: float = 1.15
    max_joint_acc: float = 10.47
    use_speed_limit: bool = True
    use_acc_limit: bool = True
    target_smoothing_alpha: float = 0.0
    action_delay_steps: int = 6
    actuator_lag_alpha: float = 0.062
    delay_lag_randomization_enabled: bool = True
    action_delay_steps_min: int = 6
    action_delay_steps_max: int = 7
    actuator_lag_alpha_min: float = 0.059
    actuator_lag_alpha_max: float = 0.064
    speed_acc_randomization_enabled: bool = True
    max_joint_speed_min: float = 0.95
    max_joint_speed_max: float = 1.35
    max_joint_acc_min: float = 8.5
    max_joint_acc_max: float = 12.0


@configclass
class GripperControlConfig:
    action_mode: str = "binary"
    use_sigmoid: bool = True
    close_threshold: float = 0.7
    close_on_threshold: float = 0.78
    open_off_threshold: float = 0.62
    min_switch_interval_s: float = 0.25
    max_speed: float = 4.0
    use_speed_limit: bool = True
    actuator_lag_alpha: float = 0.062


@configclass
class RewardConfig:
    dist_std: float = 0.4
    dist_scale: float = 15.0

    gripper_close_dist: float = 0.035
    gripper_close_reward: float = 140.0
    gripper_close_penalty: float = -8.0

    grasp_dist: float = 0.03
    grasp_close_ratio: float = 0.7
    grasp_hold_steps: int = 6

    open_reward_scale: float = 420.0
    open_delta_reward_scale: float = 260.0
    open_reward_strict_dist: float = 0.08

    grasp_hold_reward_scale: float = 8.0

    grasp_hold_open_scale: float = 10.0

    open_bonus_dist_1: float = 0.15
    open_bonus_reward_1: float = 35.0
    open_bonus_dist_2: float = 0.22
    open_bonus_reward_2: float = 70.0

    slip_penalty: float = 20.0
    slip_penalty_open_scale: float = 65.0
    slip_open_dist_thresh: float = 0.002

    action_penalty_rate_late: float = 4e-3

    finger_penalty_weight: float = 6.0
    finger_penalty_dist: float = 0.6
    finger_align_reward: float = 8.0
    finger_align_close_amount_thresh: float = 0.05
    gripper_switch_penalty: float = 0.8
    gripper_switch_penalty_dist: float = 0.10
    quat_reward_scale: float = 12
    quat_reward_dist_thresh: float = 0.5
    wrong_open_dist: float = 0.032
    action_penalty_switch_step: int = 12000
    action_penalty_rate_early: float = 8e-4
    joint_vel_penalty_rate_early: float = 0.0
    joint_vel_penalty_rate_late: float = 5e-3
    truncation_penalty: float = 10.0


@configclass
class TerminationConfig:
    tcp_behind_handle_threshold: float = -0.02
    max_joint_vel: float = 3.93


@configclass
class ObservationNoiseConfig:
    enabled: bool = True
    joint_noise_enabled: bool = True
    handle_pose_noise_enabled: bool = True
    joint_pos_std: float = 2e-5
    joint_vel_std: float = 1e-4
    target_pos_std: float = 0.01
    target_rot_std: float = 0.017
    target_pos_bias_std: float = 0.015
    target_rot_bias_std: float = 0.03
    bias_resample_prob: float = 0.0
    dropout_prob: float = 0.091
    latency_steps: int = 0
    hold_last_on_dropout: bool = True


@registry.envcfg("rm65-open-cabinet")
@configclass
class RM65OpenCabinetEnvCfg(DirectEnvCfg):
    """Control RM65 to grasp a handle and open the cabinet's bottom drawer.

    zh_CN: 控制 RM65 抓住把手并拉开柜体底部抽屉。
    """

    scene: SceneCfg = SceneCfg(file=model_file)
    max_episode_seconds: float = 30.0
    sim: SimCfg = SimCfg(dt=0.005, max_contacts=256, max_contact_constraints=768)
    ctrl_dt: float = 0.025
    render_spacing: float = 2.0
    action_scale = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    action_history_len: int = 9
    reset: ResetConfig = ResetConfig()
    reward: RewardConfig = RewardConfig()
    termination: TerminationConfig = TerminationConfig()
    arm_control: ArmControlConfig = ArmControlConfig()
    gripper_control: GripperControlConfig = GripperControlConfig()
    observation_noise: ObservationNoiseConfig = ObservationNoiseConfig()
