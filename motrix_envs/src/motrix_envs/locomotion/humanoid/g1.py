# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree G1 humanoid velocity-tracking presets and registration."""

from dataclasses import replace

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import HFieldTerrainCfg, SystemCameraCfg
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.action import JointPositionActionCfg
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
from motrix_envs.locomotion.humanoid.walk_manager_mdp.command import WalkCommandCfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.randomization import WalkRandomizationCfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.reset import WalkStateResetCfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.rewards import (
    FeetPhaseRewardCfg,
    PenaltyActionRateRewardCfg,
    PenaltyCloseFeetXyRewardCfg,
    PoseRewardCfg,
)
from motrix_envs.robot import UnitreeG129Dof

# Pinned G1 termination collision inventory, owned by the G1 walk task.
_G1_TERMINATION_GEOMS = (
    "pelvis_collision",
    "left_thigh",
    "right_thigh",
    "torso_collision1",
    "torso_collision2",
    "torso_collision3",
    "head_collision",
    "left_shoulder_yaw_collision",
    "right_shoulder_yaw_collision",
)


# Differentiated pose weights carried over from the direct-env G1 preset:
# the torso and arms are pinned hard (50.0) so the policy cannot lean or yaw
# the upper body to compensate, hip_yaw and ankles are held moderately (5.0),
# and hip_pitch/knee stay nearly free (0.01) for swinging legs.
_G1_POSE_WEIGHTS = {
    **{name: 0.01 for name in ("left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint")},
    **{name: 1.0 for name in ("left_hip_roll_joint", "right_hip_roll_joint")},
    **{
        name: 5.0
        for name in (
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
        )
    },
    **{
        name: 50.0
        for name in (
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
        )
    },
}


def _make_g1_rewards(robot: UnitreeG129Dof) -> WalkRewardsCfg:
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
            pose_weights={name: _G1_POSE_WEIGHTS[name] for name in robot.key_pose.joint_names}, weight=-0.5
        ),
    )


@registry.envcfg("g1-walk-flat")
def make_g129dof_walk_flat_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Track walking commands with Unitree G1 on flat ground.

    zh_CN: 控制 Unitree G1 在平地上跟踪行走指令。
    """
    robot = UnitreeG129Dof()
    return HumanoidVelocityTrackingManagerEnvCfg(
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
            objs=StandardSceneObjsCfg(robot=robot),
        ),
        rewards=_make_g1_rewards(robot),
        terminations=WalkTerminationsCfg(
            colliding=CollidingTerminationCfg(
                termination_geoms=_G1_TERMINATION_GEOMS,
                ground_geom="floor",
            )
        ),
        # Holosoma g1_29dof_randomization parity: reset-time kp/kd, friction,
        # mass, and base-com randomization plus per-episode gait-period jitter
        # and a [0, 1]-step control delay. Push disturbance is not included.
        actions=humanoid_cfg.WalkActionsCfg(
            joint_position=JointPositionActionCfg(
                action_scale=0.25,
                action_scales_by_effort_limit_over_p_gain=False,
                action_delay_steps=(0, 1),
            )
        ),
        commands=humanoid_cfg.WalkCommandsCfg(walk=WalkCommandCfg(gait_period_randomization_width=0.2)),
        sim_reset=WalkResetCfg(
            humanoid_state=WalkStateResetCfg(
                randomization=WalkRandomizationCfg(
                    enabled=True,
                    kp_scale_range=(0.9, 1.1),
                    damping_scale_range=(0.9, 1.1),
                    sliding_friction_range=(0.5, 1.25),
                    link_mass_scale_range=(0.9, 1.2),
                    base_mass_offset_range=(-1.0, 3.0),
                    base_com_offset_noise=(0.05, 0.05, 0.05),
                )
            )
        ),
        sim=SimCfg(dt=0.01, solver_iterations=3, solver_tolerance=1e-4),
    )


@registry.envcfg("g1-walk-rough")
def make_g129dof_walk_rough_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Track walking commands with Unitree G1 over uneven terrain.

    zh_CN: 控制 Unitree G1 在起伏地形上跟踪行走指令。
    """
    return replace(
        make_g129dof_walk_flat_cfg(),
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
            assets=humanoid_cfg.TerrainSceneAssetsCfg(),
            objs=StandardSceneObjsCfg(
                floor=HFieldTerrainCfg(hfield="terrain", material="mat_ground"),
                robot=UnitreeG129Dof(),
            ),
        ),
        sim_reset=WalkResetCfg(humanoid_state=WalkStateResetCfg(spawn_xy_range=4.0)),
        render_spacing=0.0,
    )


registry.env("g1-walk-flat")(ManagerEnv)
registry.env("g1-walk-rough")(ManagerEnv)
