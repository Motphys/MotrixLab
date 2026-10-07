# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import KeyPoseCfg, MjcfFileCfg
from motrix_robots.quadruped import QuadrupedLegCfg, QuadrupedLegsCfg, QuadrupedRobotCfg

ANYMAL_C_ASSET_DIR = Path(__file__).parent / "assets" / "anymal_c"
_ANYMAL_C_MJCF = ANYMAL_C_ASSET_DIR / "anymal_c.xml"


@configclass(kw_only=True)
class AnymalC(QuadrupedRobotCfg):
    """Built-in ANYmal-C quadruped robot configuration."""

    model: MjcfFileCfg = MjcfFileCfg(file=_ANYMAL_C_MJCF)
    base_link_name: str = "base"
    legs: QuadrupedLegsCfg = QuadrupedLegsCfg(
        front_left=QuadrupedLegCfg(
            contact_geom_name="LF_FOOT",
        ),
        front_right=QuadrupedLegCfg(
            contact_geom_name="RF_FOOT",
        ),
        rear_left=QuadrupedLegCfg(
            contact_geom_name="LH_FOOT",
        ),
        rear_right=QuadrupedLegCfg(
            contact_geom_name="RH_FOOT",
        ),
    )
    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=[
            "LF_HAA",
            "LF_HFE",
            "LF_KFE",
            "RF_HAA",
            "RF_HFE",
            "RF_KFE",
            "LH_HAA",
            "LH_HFE",
            "LH_KFE",
            "RH_HAA",
            "RH_HFE",
            "RH_KFE",
        ],
        poses={"default": [0.0, 0.4, -0.8, 0.0, 0.4, -0.8, 0.0, -0.4, 0.8, 0.0, -0.4, 0.8]},
    )


__all__ = ["ANYMAL_C_ASSET_DIR", "AnymalC"]
