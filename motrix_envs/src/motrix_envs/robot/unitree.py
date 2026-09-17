# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import KeyPoseCfg, MjcfFileCfg
from motrix_envs.robot.humanoid import HumanoidRobotCfg
from motrix_envs.robot.quadruped import QuadrupedLegCfg, QuadrupedLegsCfg, QuadrupedRobotCfg

UNITREE_G1_ASSET_DIR = Path(__file__).parent / "assets" / "g1"
UNITREE_GO1_ASSET_DIR = Path(__file__).parents[1] / "locomotion" / "go1" / "xmls"
UNITREE_GO2_ASSET_DIR = Path(__file__).parent / "assets" / "go2"
_G1_29DOF_MJCF = UNITREE_G1_ASSET_DIR / "g1_29dof.xml"
# Leg gains follow the Unitree unitree_rl_gym G1 PD table (hip 100, knee 150,
# ankle 40 N*m/rad): the stock menagerie-derived gains (28-99) cannot hold a
# standing pose against gravity, which starves whole-body tracking of any
# learning signal. Arms and waist keep the stock gains (not covered by the
# official table).
_G1_29DOF_OFFICIAL_MJCF = UNITREE_G1_ASSET_DIR / "g1_29dof_official.xml"
_GO1_MJCF = UNITREE_GO1_ASSET_DIR / "go1_position_actuator.xml"
_GO2_MJCF = UNITREE_GO2_ASSET_DIR / "go2_mjx.xml"


@configclass(kw_only=True)
class UnitreeG129Dof(HumanoidRobotCfg):
    """Built-in Unitree G1 29-DOF robot configuration."""

    model: MjcfFileCfg = MjcfFileCfg(file=_G1_29DOF_MJCF)
    base_link_name: str = "pelvis"
    left_foot_link_name: str = "left_ankle_roll_link"
    right_foot_link_name: str = "right_ankle_roll_link"
    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=[
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        poses={
            "default": [
                -0.312,
                0.0,
                0.0,
                0.669,
                -0.363,
                0.0,
                -0.312,
                0.0,
                0.0,
                0.669,
                -0.363,
                0.0,
                0.0,
                0.0,
                0.0,
                0.2,
                0.2,
                0.0,
                0.6,
                0.0,
                0.0,
                0.0,
                0.2,
                -0.2,
                0.0,
                0.6,
                0.0,
                0.0,
                0.0,
            ]
        },
    )


@configclass(kw_only=True)
class UnitreeG129DofOfficialGains(UnitreeG129Dof):
    """Unitree G1 29-DOF with official RL leg gains (hip 100, knee 150, ankle 40).

    zh_CN: 腿部增益对齐 Unitree 官方 RL 配置的 G1 29-DOF 模型。
    """

    model: MjcfFileCfg = MjcfFileCfg(file=_G1_29DOF_OFFICIAL_MJCF)


@configclass(kw_only=True)
class UnitreeGo1Robot(QuadrupedRobotCfg):
    """Built-in Unitree Go1 quadruped robot configuration."""

    model: MjcfFileCfg = MjcfFileCfg(file=_GO1_MJCF)
    base_link_name: str = "trunk"
    legs: QuadrupedLegsCfg = QuadrupedLegsCfg(
        front_left=QuadrupedLegCfg(
            contact_geom_name="FL_foot",
        ),
        front_right=QuadrupedLegCfg(
            contact_geom_name="FR_foot",
        ),
        rear_left=QuadrupedLegCfg(
            contact_geom_name="RL_foot",
        ),
        rear_right=QuadrupedLegCfg(
            contact_geom_name="RR_foot",
        ),
    )
    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=[
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ],
        poses={"default": [0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8, 0.0, 0.9, -1.8]},
    )


@configclass(kw_only=True)
class UnitreeGo2Robot(QuadrupedRobotCfg):
    """Built-in Unitree Go2 quadruped robot configuration."""

    model: MjcfFileCfg = MjcfFileCfg(file=_GO2_MJCF)
    base_link_name: str = "base"
    legs: QuadrupedLegsCfg = QuadrupedLegsCfg(
        front_left=QuadrupedLegCfg(
            contact_geom_name="FL_foot",
        ),
        front_right=QuadrupedLegCfg(
            contact_geom_name="FR_foot",
        ),
        rear_left=QuadrupedLegCfg(
            contact_geom_name="RL_foot",
        ),
        rear_right=QuadrupedLegCfg(
            contact_geom_name="RR_foot",
        ),
    )
    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=[
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ],
        poses={"default": [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5]},
    )
