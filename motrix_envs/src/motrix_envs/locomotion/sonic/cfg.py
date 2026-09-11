# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Manager configuration shared by the SONIC G1 task variants."""

from __future__ import annotations

from copy import deepcopy

from omegaconf import MISSING

from motrix_env_core.config import configclass
from motrix_env_core.manager import (
    ManagerActionsCfg,
    ManagerBasedEnvCfg,
    ManagerCommandsCfg,
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ManagerResetCfg,
    ManagerRewardsCfg,
    ManagerTerminationsCfg,
    SimQueriesCfg,
)
from motrix_env_core.sim import (
    ActuatorKpQuery,
    BatchLinkAngularVelocityQuery,
    BatchLinkLinearVelocityQuery,
    BatchLinkNetContactForceQuery,
    BatchLinkPositionQuery,
    BatchLinkQuaternionQuery,
    BodyJointPositionLimitsQuery,
    JointPositionQuery,
    JointVelocityQuery,
)
from motrix_envs.locomotion.sonic import mdp
from motrix_envs.locomotion.wbt.mdp.reset import (
    BodyDofPosResetCfg,
    BodyLinVelResetCfg,
    BodyPosResetCfg,
    BodyRotResetCfg,
    BodyRotVelResetCfg,
)


@configclass
class SonicActionsCfg(ManagerActionsCfg):
    joint_position: mdp.SonicJointPositionActionCfg = mdp.SonicJointPositionActionCfg()


@configclass
class SonicCommandsCfg(ManagerCommandsCfg):
    motion: mdp.SonicMotionCommandCfg = mdp.SonicMotionCommandCfg()


@configclass
class SonicPolicyObsCfg(ManagerObservationGroupCfg):
    obs: mdp.SonicActorObservationCfg = mdp.SonicActorObservationCfg()
    g1_reference: mdp.SonicG1ReferenceObservationCfg = mdp.SonicG1ReferenceObservationCfg()
    smpl_reference: mdp.SonicSmplReferenceObservationCfg = mdp.SonicSmplReferenceObservationCfg()
    encoder_index: mdp.SonicEncoderIndexObservationCfg = mdp.SonicEncoderIndexObservationCfg()


@configclass
class SonicValueObsCfg(ManagerObservationGroupCfg):
    obs: mdp.SonicCriticObservationCfg = mdp.SonicCriticObservationCfg()


@configclass
class SonicObservationsCfg(ManagerObservationsCfg):
    policy: SonicPolicyObsCfg = SonicPolicyObsCfg()
    value: SonicValueObsCfg = SonicValueObsCfg()


@configclass
class SonicRewardsCfg(ManagerRewardsCfg):
    motion_global_ref_position_error_exp: mdp.GlobalRefPositionRewardCfg = mdp.GlobalRefPositionRewardCfg(
        weight=0.5, sigma=0.3
    )
    motion_global_ref_orientation_error_exp: mdp.GlobalRefOrientationRewardCfg = mdp.GlobalRefOrientationRewardCfg(
        weight=0.5, sigma=0.4
    )
    motion_relative_body_position_error_exp: mdp.RelativeBodyPositionRewardCfg = mdp.RelativeBodyPositionRewardCfg(
        weight=1.0, sigma=0.3
    )
    motion_relative_body_orientation_error_exp: mdp.RelativeBodyOrientationRewardCfg = (
        mdp.RelativeBodyOrientationRewardCfg(weight=1.0, sigma=0.4)
    )
    motion_global_body_lin_vel: mdp.GlobalBodyLinearVelocityRewardCfg = mdp.GlobalBodyLinearVelocityRewardCfg(
        weight=1.0, sigma=1.0
    )
    motion_global_body_ang_vel: mdp.GlobalBodyAngularVelocityRewardCfg = mdp.GlobalBodyAngularVelocityRewardCfg(
        weight=1.0, sigma=3.14
    )
    action_rate_l2: mdp.ActionRateRewardCfg = mdp.ActionRateRewardCfg(weight=-0.1)
    limits_dof_pos: mdp.DofLimitRewardCfg = mdp.DofLimitRewardCfg(weight=-10.0, soft_limit=0.9, cap=5.0)
    undesired_contacts: mdp.UndesiredContactsRewardCfg = mdp.UndesiredContactsRewardCfg(weight=-0.1, threshold=1.0)
    anti_shake: mdp.SonicAntiShakeRewardCfg = mdp.SonicAntiShakeRewardCfg(weight=-0.005)
    feet_acc: mdp.SonicFeetAccelerationRewardCfg = mdp.SonicFeetAccelerationRewardCfg(weight=-2.5e-6)
    tracking_vr_5point_local: mdp.SonicTrackingRewardCfg = mdp.SonicTrackingRewardCfg(weight=2.0)


@configclass
class SonicTerminationsCfg(ManagerTerminationsCfg):
    anchor_pos_z: mdp.SonicAnchorPosTerminationCfg = mdp.SonicAnchorPosTerminationCfg()
    anchor_ori: mdp.SonicAnchorOriTerminationCfg = mdp.SonicAnchorOriTerminationCfg()
    ee_body_pos_z: mdp.SonicEeBodyPosTerminationCfg = mdp.SonicEeBodyPosTerminationCfg()
    feet_pos: mdp.SonicFeetPosTerminationCfg = mdp.SonicFeetPosTerminationCfg()
    time_out: mdp.SonicTimeOutTerminationCfg = mdp.SonicTimeOutTerminationCfg()
    clip_end: mdp.SonicClipEndTerminationCfg = mdp.SonicClipEndTerminationCfg()


@configclass
class SonicResetCfg(ManagerResetCfg):
    body_pos: BodyPosResetCfg = BodyPosResetCfg()
    body_rot: BodyRotResetCfg = BodyRotResetCfg()
    body_lin_vel: BodyLinVelResetCfg = BodyLinVelResetCfg()
    body_rot_vel: BodyRotVelResetCfg = BodyRotVelResetCfg()
    body_dof_pos: BodyDofPosResetCfg = BodyDofPosResetCfg()
    actuator_dynamics: mdp.SonicActuatorDynamicsResetCfg = mdp.SonicActuatorDynamicsResetCfg()


@configclass
class SonicManagerEnvCfg(ManagerBasedEnvCfg):
    ctrl_dt: float = 0.02
    max_episode_seconds: float | None = 10.0
    queries: SimQueriesCfg = SimQueriesCfg(
        model={
            "actuator_kp": ActuatorKpQuery(),
            "robot_joint_position_limits": BodyJointPositionLimitsQuery(body=MISSING),
        }
    )
    commands: SonicCommandsCfg = SonicCommandsCfg()
    actions: SonicActionsCfg = SonicActionsCfg()
    observations: SonicObservationsCfg = SonicObservationsCfg()
    rewards: SonicRewardsCfg = SonicRewardsCfg()
    terminations: SonicTerminationsCfg = SonicTerminationsCfg()
    sim_reset: SonicResetCfg = SonicResetCfg()

    def __post_init__(self) -> None:
        robot = self.scene.objs.robot
        joint_names = tuple(robot.resolve_name(name) for name in self.commands.motion.joint_names)
        tracked_body_names = tuple(robot.resolve_name(name) for name in self.commands.motion.tracked_body_names)
        self.commands.motion.joint_names = joint_names
        self.commands.motion.tracked_body_names = tracked_body_names
        self.actions.joint_position.actuator_names = joint_names
        self.queries.data.update(
            {
                "robot_dof_pos": JointPositionQuery(joints=joint_names),
                "robot_dof_vel": JointVelocityQuery(joints=joint_names),
                "tracked_body_pos": BatchLinkPositionQuery(links=tracked_body_names),
                "tracked_body_quat": BatchLinkQuaternionQuery(links=tracked_body_names),
                "tracked_body_linear_velocity": BatchLinkLinearVelocityQuery(links=tracked_body_names),
                "tracked_body_angular_velocity": BatchLinkAngularVelocityQuery(links=tracked_body_names),
                "undesired_contact_forces": BatchLinkNetContactForceQuery(
                    links=tuple(
                        robot.resolve_name(name)
                        for name in (
                            "pelvis",
                            "left_hip_roll_link",
                            "left_hip_yaw_link",
                            "left_knee_link",
                            "right_hip_roll_link",
                            "right_hip_yaw_link",
                            "right_knee_link",
                            "torso_link",
                            "left_shoulder_yaw_link",
                            "right_shoulder_yaw_link",
                        )
                    )
                ),
            }
        )
        self.queries.model["robot_joint_position_limits"] = BodyJointPositionLimitsQuery(
            body=robot.resolved_base_link_name
        )

    def for_play(self) -> SonicManagerEnvCfg:
        cfg = deepcopy(self)
        cfg.max_episode_seconds = None
        cfg.commands.motion.adaptive_sampling_enabled = False
        cfg.commands.motion.start_at_timestep_zero_prob = 1.0
        cfg.commands.motion.hold_at_clip_end = False
        cfg.sim_reset.body_pos.noise_scale = 0.0
        cfg.sim_reset.body_rot.noise_scale = 0.0
        cfg.sim_reset.body_lin_vel.noise_scale = 0.0
        cfg.sim_reset.body_rot_vel.noise_scale = 0.0
        cfg.sim_reset.body_dof_pos.noise_scale = 0.0
        return cfg
