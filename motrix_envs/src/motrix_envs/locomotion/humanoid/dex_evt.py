# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Dex-EVT humanoid velocity-tracking presets and registration."""

from dataclasses import replace

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import FlatTerrainCfg, HFieldTerrainCfg
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
from motrix_envs.robot import DexEvt

# Pinned Dex-EVT termination collision inventory, owned by the Dex-EVT walk task.
_DEX_EVT_TERMINATION_GEOMS = (
    "pelvis_collision",
    "hip_pitch_l_link_collision",
    "hip_yaw_l_link_collision",
    "knee_pitch_l_link_collision",
    "hip_pitch_r_link_collision",
    "hip_yaw_r_link_collision",
    "knee_pitch_r_link_collision",
    "waist_pitch_link_collision",
    "shoulder_yaw_l_link_collision",
    "wrist_pitch_l_link_collision",
    "left_tcp_link_collision",
    "shoulder_yaw_r_link_collision",
    "wrist_pitch_r_link_collision",
    "right_tcp_link_collision",
)


def _make_dex_evt_robot() -> DexEvt:
    return DexEvt(translation=(0.0, 0.0, 0.9785))


def _make_dex_evt_rewards() -> WalkRewardsCfg:
    return WalkRewardsCfg(
        tracking_lin_vel=TrackingLinVelXyRewardCfg(command_name="walk", sigma=0.25, weight=2.0),
        tracking_ang_vel=TrackingAngVelZRewardCfg(command_name="walk", sigma=0.25, weight=1.5),
        penalty_action_rate=PenaltyActionRateRewardCfg(weight=-2.0),
        feet_phase=FeetPhaseRewardCfg(
            sole_l_site="left_foot_contact_point",
            sole_r_site="right_foot_contact_point",
            weight=5.0,
        ),
        penalty_close_feet_xy=PenaltyCloseFeetXyRewardCfg(close_feet_threshold=0.15, weight=-10.0),
        pose=PoseRewardCfg(
            pose_weights={
                "hip_pitch_l_joint": 0.01,
                "hip_roll_l_joint": 1.0,
                "hip_yaw_l_joint": 5.0,
                "knee_pitch_l_joint": 0.01,
                "ankle_pitch_l_joint": 5.0,
                "ankle_roll_l_joint": 5.0,
                "hip_pitch_r_joint": 0.01,
                "hip_roll_r_joint": 1.0,
                "hip_yaw_r_joint": 5.0,
                "knee_pitch_r_joint": 0.01,
                "ankle_pitch_r_joint": 5.0,
                "ankle_roll_r_joint": 5.0,
                "waist_yaw_joint": 50.0,
                "waist_roll_joint": 50.0,
                "waist_pitch_joint": 50.0,
                "shoulder_pitch_l_joint": 50.0,
                "shoulder_roll_l_joint": 50.0,
                "shoulder_yaw_l_joint": 50.0,
                "elbow_pitch_l_joint": 50.0,
                "shoulder_pitch_r_joint": 50.0,
                "shoulder_roll_r_joint": 50.0,
                "shoulder_yaw_r_joint": 50.0,
                "elbow_pitch_r_joint": 50.0,
            },
            weight=-0.5,
        ),
    )


@registry.envcfg("dex-evt-walk-flat")
def make_dex_evt_walk_flat_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Control the Dex-EVT humanoid to walk on flat ground.

    zh_CN: 控制 Dex-EVT 人形机器人在平地上行走。
    """
    return HumanoidVelocityTrackingManagerEnvCfg(
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            objs=StandardSceneObjsCfg(
                floor=FlatTerrainCfg(
                    material="mat_ground",
                    friction=(0.8, 0.005, 0.0001),
                ),
                robot=_make_dex_evt_robot(),
            ),
        ),
        rewards=_make_dex_evt_rewards(),
        terminations=WalkTerminationsCfg(
            colliding=CollidingTerminationCfg(
                termination_geoms=_DEX_EVT_TERMINATION_GEOMS,
                ground_geom="floor",
            )
        ),
        sim=SimCfg(dt=0.005, solver_iterations=6, solver_tolerance=0.0001),
    )


@registry.envcfg("dex-evt-walk-rough")
def make_dex_evt_walk_rough_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Control the Dex-EVT humanoid to walk over uneven terrain.

    zh_CN: 控制 Dex-EVT 人形机器人在起伏地形上行走。
    """
    return replace(
        make_dex_evt_walk_flat_cfg(),
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            assets=humanoid_cfg.TerrainSceneAssetsCfg(),
            objs=StandardSceneObjsCfg(
                floor=HFieldTerrainCfg(
                    hfield="terrain",
                    material="mat_ground",
                ),
                robot=_make_dex_evt_robot(),
            ),
        ),
        sim_reset=WalkResetCfg(humanoid_state=WalkStateResetCfg(spawn_xy_range=4.0)),
        render_spacing=0.0,
    )


registry.env("dex-evt-walk-flat")(ManagerEnv)
registry.env("dex-evt-walk-rough")(ManagerEnv)
