# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import KeyPoseCfg, MjcfFileCfg
from motrix_envs.robot.humanoid import HumanoidRobotCfg

MICRODUCK_ASSET_DIR = Path(__file__).parent / "assets" / "microduck"
_MICRODUCK_MJCF = MICRODUCK_ASSET_DIR / "microduck.xml"


@configclass(kw_only=True)
class Microduck(HumanoidRobotCfg):
    """Built-in Pollen Robotics Microduck 14-DoF biped robot configuration.

    Ported from https://github.com/pollen-robotics/microduck_rl (Apache-2.0
    code; mesh assets are CC BY-SA-NC, see the asset directory README).
    """

    model: MjcfFileCfg = MjcfFileCfg(file=_MICRODUCK_MJCF)
    base_link_name: str = "trunk_base"
    left_foot_link_name: str = "ankle_left"
    right_foot_link_name: str = "ankle_right"
    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=[
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
            "left_ankle",
            "neck_pitch",
            "head_pitch",
            "head_yaw",
            "head_roll",
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
            "right_ankle",
        ],
        poses={
            # Upstream keyframe "STAND" (trunk at z=0.12).
            "default": [
                0.0,
                -0.08726646259971647,
                -0.457924,
                -0.004940,
                0.452984,
                0.3490658503988659,
                0.3490658503988659,
                0.0,
                0.0,
                0.0,
                0.08726646259971647,
                0.457924,
                0.004940,
                -0.452984,
            ],
            # Upstream keyframe "SIT" (trunk at z=0.07).
            "sit": [
                0.0,
                0.0,
                -0.5236,
                1.0472,
                0.0,
                0.5,
                1.6,
                0.0,
                0.0,
                0.0,
                0.0,
                0.5236,
                -1.0472,
                0.0,
            ],
            # Upstream keyframe "FOLD" (trunk at z=0.07).
            "fold": [
                0.0,
                0.0,
                1.5708,
                1.5708,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.5708,
                -1.5708,
                0.0,
            ],
        },
    )
