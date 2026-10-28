# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import KeyPoseCfg, MjcfFileCfg
from motrix_robots.quadruped import QuadrupedLegCfg, QuadrupedLegsCfg, QuadrupedRobotCfg

CYBERDOG2_ASSET_DIR = Path(__file__).parent / "assets" / "cyberdog2"
CYBERDOG2_MJCF = CYBERDOG2_ASSET_DIR / "cyberdog2.xml"


@configclass(kw_only=True)
class Cyberdog2Robot(QuadrupedRobotCfg):
    """Cyberdog2 quadruped robot configuration"""

    model: MjcfFileCfg = MjcfFileCfg(file=CYBERDOG2_MJCF)
    base_link_name: str = "body"
    # Contact geoms are the *_foot_collision boxes defined in cyberdog2.xml;
    # unlike Go2, CyberDog2 does not name its foot collision geom "FL_foot".
    legs: QuadrupedLegsCfg = QuadrupedLegsCfg(
        front_left=QuadrupedLegCfg(contact_geom_name="FL_foot_collision"),
        front_right=QuadrupedLegCfg(contact_geom_name="FR_foot_collision"),
        rear_left=QuadrupedLegCfg(contact_geom_name="RL_foot_collision"),
        rear_right=QuadrupedLegCfg(contact_geom_name="RR_foot_collision"),
    )
    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=[
            "FR_abad_joint",
            "FL_abad_joint",
            "RR_abad_joint",
            "RL_abad_joint",
            "FR_hip_joint",
            "FL_hip_joint",
            "RR_hip_joint",
            "RL_hip_joint",
            "FR_knee_joint",
            "FL_knee_joint",
            "RR_knee_joint",
            "RL_knee_joint",
        ],
        poses={"default": [-0.10, 0.10, -0.10, 0.10, 0.80, 0.80, 1.00, 1.00, -1.50, -1.50, -1.50, -1.50]},
    )
