# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Isolated Go2 legacy-pose ablation for the flat-walk task."""

from motrix_robots.unitree import UnitreeGo2Robot

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, KeyPoseCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.quadruped.cfg import QuadrupedSceneCfg
from motrix_envs.locomotion.quadruped.go2_walk import Go2WalkDirectEnvCfg
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask


LEGACY_GO2_KEY_POSE = KeyPoseCfg(
    joint_names=[
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ],
    poses={"default": [
        0.1, 0.8, -1.5,
        -0.1, 0.8, -1.5,
        0.1, 1.0, -1.5,
        -0.1, 1.0, -1.5,
    ]},
)


@registry.envcfg("go2-walk-flat-d-legacy-pose")
@configclass
class Go2WalkLegacyPoseEnvCfg(Go2WalkDirectEnvCfg):
    """Current improved task with legacy RL Gym default hip angles."""

    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(
        objs=StandardSceneObjsCfg(
            floor=FlatTerrainCfg(
                material="mat_ground",
                friction=(0.6, 0.005, 0.0001),
            ),
            robot=UnitreeGo2Robot(key_pose=LEGACY_GO2_KEY_POSE),
        ),
    )


registry.env("go2-walk-flat-d-legacy-pose")(QuadrupedWalkTask)


__all__ = ["Go2WalkLegacyPoseEnvCfg", "LEGACY_GO2_KEY_POSE"]
