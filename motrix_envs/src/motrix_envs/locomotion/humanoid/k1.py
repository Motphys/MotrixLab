# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Booster K1 humanoid velocity-tracking presets and registration."""

from dataclasses import replace

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import HFieldTerrainCfg
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.rewards import TrackingAngVelZRewardCfg, TrackingLinVelXyRewardCfg
from motrix_env_core.mdp.terminations import CollidingTerminationCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.humanoid import cfg as humanoid_cfg
from motrix_envs.locomotion.humanoid.cfg import (
    HumanoidVelocityTrackingManagerEnvCfg,
    WalkResetCfg,
    WalkRewardsCfg,
    WalkTerminationsCfg,
)
from motrix_envs.locomotion.humanoid.walk_manager_mdp.reset import WalkStateResetCfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.rewards import (
    FeetPhaseRewardCfg,
    PenaltyActionRateRewardCfg,
    PenaltyCloseFeetXyRewardCfg,
    PoseRewardCfg,
)
from motrix_envs.robot import BoosterK1


def _make_k1_robot() -> BoosterK1:
    return BoosterK1(translation=(0.0, 0.0, -0.46))


# Pinned K1 termination collision inventory, owned by the K1 walk task.
_K1_TERMINATION_GEOMS = (
    "trunk_upper_collision",
    "trunk_lower_collision",
    "head_yaw_collision",
    "head_pitch_collision",
    "left_shoulder_collision",
    "left_elbow_collision",
    "left_hand_collision",
    "right_shoulder_collision",
    "right_elbow_collision",
    "right_hand_collision",
    "left_hip_roll_collision",
    "left_hip_yaw_collision",
    "left_shank_upper_collision",
    "left_shank_lower_collision",
    "right_hip_roll_collision",
    "right_hip_yaw_collision",
    "right_shank_upper_collision",
    "right_shank_lower_collision",
)


def _make_k1_rewards() -> WalkRewardsCfg:
    return WalkRewardsCfg(
        tracking_lin_vel=TrackingLinVelXyRewardCfg(command_name="walk", sigma=0.25, weight=2.0),
        tracking_ang_vel=TrackingAngVelZRewardCfg(command_name="walk", sigma=0.25, weight=1.5),
        penalty_action_rate=PenaltyActionRateRewardCfg(weight=-2.0),
        feet_phase=FeetPhaseRewardCfg(sole_l_site="left_foot", sole_r_site="right_foot", weight=5.0),
        penalty_close_feet_xy=PenaltyCloseFeetXyRewardCfg(close_feet_threshold=0.15, weight=-10.0),
        pose=PoseRewardCfg(
            pose_weights={
                "AAHead_yaw": 50.0,
                "Head_pitch": 50.0,
                "ALeft_Shoulder_Pitch": 50.0,
                "Left_Shoulder_Roll": 50.0,
                "Left_Elbow_Pitch": 50.0,
                "Left_Elbow_Yaw": 50.0,
                "ARight_Shoulder_Pitch": 50.0,
                "Right_Shoulder_Roll": 50.0,
                "Right_Elbow_Pitch": 50.0,
                "Right_Elbow_Yaw": 50.0,
                "Left_Hip_Pitch": 0.01,
                "Left_Hip_Roll": 1.0,
                "Left_Hip_Yaw": 5.0,
                "Left_Knee_Pitch": 0.01,
                "Left_Ankle_Pitch": 5.0,
                "Left_Ankle_Roll": 5.0,
                "Right_Hip_Pitch": 0.01,
                "Right_Hip_Roll": 1.0,
                "Right_Hip_Yaw": 5.0,
                "Right_Knee_Pitch": 0.01,
                "Right_Ankle_Pitch": 5.0,
                "Right_Ankle_Roll": 5.0,
            },
            weight=-0.5,
        ),
    )


def _make_k1_terminations() -> WalkTerminationsCfg:
    return WalkTerminationsCfg(
        colliding=CollidingTerminationCfg(
            termination_geoms=_K1_TERMINATION_GEOMS,
            ground_geom="floor",
        )
    )


@registry.envcfg("k1-walk-flat")
def make_k1_walk_flat_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Track walking commands with Booster K1 on flat ground.

    zh_CN: 控制 Booster K1 在平地上跟踪行走指令。
    """
    return HumanoidVelocityTrackingManagerEnvCfg(
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            objs=StandardSceneObjsCfg(robot=_make_k1_robot()),
        ),
        rewards=_make_k1_rewards(),
        terminations=_make_k1_terminations(),
        sim=SimCfg(dt=0.005, solver_iterations=6, solver_tolerance=1e-4),
    )


@registry.envcfg("k1-walk-rough")
def make_k1_walk_rough_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Track walking commands with Booster K1 over uneven terrain.

    zh_CN: 控制 Booster K1 在起伏地形上跟踪行走指令。
    """
    return replace(
        make_k1_walk_flat_cfg(),
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            assets=humanoid_cfg.TerrainSceneAssetsCfg(),
            objs=StandardSceneObjsCfg(
                floor=HFieldTerrainCfg(
                    hfield="terrain",
                    material="mat_ground",
                ),
                robot=_make_k1_robot(),
            ),
        ),
        sim_reset=WalkResetCfg(humanoid_state=WalkStateResetCfg(spawn_xy_range=4.0)),
        render_spacing=0.0,
    )


registry.env("k1-walk-flat")(ManagerEnv)
registry.env("k1-walk-rough")(ManagerEnv)
