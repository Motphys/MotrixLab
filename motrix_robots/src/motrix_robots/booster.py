# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import KeyPoseCfg, MjcfFileCfg
from motrix_robots.humanoid import HumanoidRobotCfg

BOOSTER_K1_ASSET_DIR = Path(__file__).parent / "assets" / "k1"
_K1_22DOF_MJCF = BOOSTER_K1_ASSET_DIR / "k1_22dof.xml"


@configclass(kw_only=True)
class BoosterK1(HumanoidRobotCfg):
    """Built-in Booster K1 22-DOF humanoid robot configuration."""

    model: MjcfFileCfg = MjcfFileCfg(file=_K1_22DOF_MJCF)
    base_link_name: str = "Trunk"
    left_foot_link_name: str = "left_foot_link"
    right_foot_link_name: str = "right_foot_link"
    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=[
            "AAHead_yaw",
            "Head_pitch",
            "ALeft_Shoulder_Pitch",
            "Left_Shoulder_Roll",
            "Left_Elbow_Pitch",
            "Left_Elbow_Yaw",
            "ARight_Shoulder_Pitch",
            "Right_Shoulder_Roll",
            "Right_Elbow_Pitch",
            "Right_Elbow_Yaw",
            "Left_Hip_Pitch",
            "Left_Hip_Roll",
            "Left_Hip_Yaw",
            "Left_Knee_Pitch",
            "Left_Ankle_Pitch",
            "Left_Ankle_Roll",
            "Right_Hip_Pitch",
            "Right_Hip_Roll",
            "Right_Hip_Yaw",
            "Right_Knee_Pitch",
            "Right_Ankle_Pitch",
            "Right_Ankle_Roll",
        ],
        poses={
            "default": [
                0.0,
                0.0,
                0.068,
                -1.296,
                0.533,
                -0.487,
                0.030,
                1.317,
                0.437,
                0.435,
                -0.30,
                0.0,
                0.0,
                0.60,
                -0.30,
                0.0,
                -0.30,
                0.0,
                0.0,
                0.60,
                -0.30,
                0.0,
            ]
        },
    )
