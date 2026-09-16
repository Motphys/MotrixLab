# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Manager-based configuration for whole-body tracking tasks."""

from __future__ import annotations

from copy import deepcopy

from omegaconf import MISSING

from motrix_env_core.base import SimCfg
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
from motrix_env_core.mdp.observations import (
    ActionsObsCfg,
    BodyAngularVelocityObsCfg,
    BodyLinearVelocityObsCfg,
    UniformNoiseCfg,
)
from motrix_env_core.mdp.rewards import ActionRateRewardCfg
from motrix_env_core.sim import (
    ActuatorKpQuery,
    BatchLinkAngularVelocityQuery,
    BatchLinkLinearVelocityQuery,
    BatchLinkPositionQuery,
    BatchLinkQuaternionQuery,
    BodyJointPositionLimitsQuery,
    JointPositionQuery,
    JointVelocityQuery,
)
from motrix_envs.locomotion.wbt.mdp.action import (
    WbtJointPositionActionCfg,
)
from motrix_envs.locomotion.wbt.mdp.command import (
    WbtMotionCommandCfg,
)
from motrix_envs.locomotion.wbt.mdp.observations import (
    DofPosRelObsCfg,
    DofVelObsCfg,
    MotionJointObsCfg,
    MotionReferenceOrientationObsCfg,
    MotionReferencePositionObsCfg,
    RobotBodyOrientationObsCfg,
    RobotBodyPositionInReferenceFrameObsCfg,
)
from motrix_envs.locomotion.wbt.mdp.reset import (
    BodyDofPosResetCfg,
    BodyLinVelResetCfg,
    BodyPosResetCfg,
    BodyRotResetCfg,
    BodyRotVelResetCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    DofLimitRewardCfg,
    GlobalBodyAngularVelocityRewardCfg,
    GlobalBodyLinearVelocityRewardCfg,
    GlobalRefOrientationRewardCfg,
    GlobalRefPositionRewardCfg,
    RelativeBodyOrientationRewardCfg,
    RelativeBodyPositionRewardCfg,
    UndesiredContactsRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.terminations import (
    BadBodyZTerminationCfg,
    BadDofPositionTerminationCfg,
    BadDofVelocityTerminationCfg,
    BadRefOrientationTerminationCfg,
    BadRefZTerminationCfg,
)


@configclass
class ActionsCfg(ManagerActionsCfg):
    """Typed action terms for WBT."""

    joint_position: WbtJointPositionActionCfg = WbtJointPositionActionCfg()


@configclass
class CommandsCfg(ManagerCommandsCfg):
    """Typed command terms for WBT."""

    motion: WbtMotionCommandCfg = MISSING


@configclass
class RewardsCfg(ManagerRewardsCfg):
    motion_global_ref_position_error_exp: GlobalRefPositionRewardCfg = GlobalRefPositionRewardCfg(
        weight=1.0,
        sigma=0.3,
    )
    motion_global_ref_orientation_error_exp: GlobalRefOrientationRewardCfg = GlobalRefOrientationRewardCfg(
        weight=0.5,
        sigma=0.4,
    )
    motion_relative_body_position_error_exp: RelativeBodyPositionRewardCfg = RelativeBodyPositionRewardCfg(
        weight=2.0,
        sigma=0.3,
    )
    motion_relative_body_orientation_error_exp: RelativeBodyOrientationRewardCfg = RelativeBodyOrientationRewardCfg(
        weight=1.0,
        sigma=0.4,
    )
    motion_global_body_lin_vel: GlobalBodyLinearVelocityRewardCfg = GlobalBodyLinearVelocityRewardCfg(
        weight=1.0,
        sigma=1.0,
    )
    motion_global_body_ang_vel: GlobalBodyAngularVelocityRewardCfg = GlobalBodyAngularVelocityRewardCfg(
        weight=1.0,
        sigma=3.14,
    )
    action_rate_l2: ActionRateRewardCfg = ActionRateRewardCfg(weight=-1.0)
    limits_dof_pos: DofLimitRewardCfg = DofLimitRewardCfg(
        weight=-10.0,
        soft_limit=0.9,
        # Diverges from Holosoma: Motrix NP can let an unstable joint state move far
        # beyond hard limits; cap this raw term so one exploded env cannot dominate Q targets.
        cap=5.0,
    )
    undesired_contacts: UndesiredContactsRewardCfg = UndesiredContactsRewardCfg(
        weight=-0.1,
        threshold=1.0,
    )


@configclass
class TerminationsCfg(ManagerTerminationsCfg):
    bad_ref_z: BadRefZTerminationCfg = BadRefZTerminationCfg(threshold=0.5)
    bad_ref_ori: BadRefOrientationTerminationCfg = BadRefOrientationTerminationCfg(threshold=0.8)
    bad_body_z: BadBodyZTerminationCfg = BadBodyZTerminationCfg(threshold=0.25)
    bad_dof_pos: BadDofPositionTerminationCfg = BadDofPositionTerminationCfg(threshold=0.5)
    bad_dof_vel: BadDofVelocityTerminationCfg = BadDofVelocityTerminationCfg(threshold=100.0)


@configclass
class ObservationsCfg(ManagerObservationsCfg):
    """Typed observation groups for WBT."""

    @configclass
    class PolicyCfg(ManagerObservationGroupCfg):
        """Policy observation group for WBT.

        Robot-scoped terms read the standard robot sim-query inputs; the dof-pos
        reference pose comes from the robot's ``RobotCfg.key_pose``.
        """

        motion_joint: MotionJointObsCfg = MotionJointObsCfg()
        motion_ref_ori_b: MotionReferenceOrientationObsCfg = MotionReferenceOrientationObsCfg(
            noise=UniformNoiseCfg(amplitude=0.05)
        )
        base_ang_vel: BodyAngularVelocityObsCfg = BodyAngularVelocityObsCfg(noise=UniformNoiseCfg(amplitude=0.2))
        dof_pos: DofPosRelObsCfg = DofPosRelObsCfg(noise=UniformNoiseCfg(amplitude=0.01))
        dof_vel: DofVelObsCfg = DofVelObsCfg(noise=UniformNoiseCfg(amplitude=0.5))
        actions: ActionsObsCfg = ActionsObsCfg()

    @configclass
    class ValueCfg(ManagerObservationGroupCfg):
        """Value (critic) observation group for WBT.

        Robot-scoped terms read the standard robot sim-query inputs; the dof-pos
        reference pose comes from the robot's ``RobotCfg.key_pose``.
        """

        motion_joint: MotionJointObsCfg = MotionJointObsCfg()
        motion_ref_pos_b: MotionReferencePositionObsCfg = MotionReferencePositionObsCfg()
        motion_ref_ori_b: MotionReferenceOrientationObsCfg = MotionReferenceOrientationObsCfg()
        robot_body_pos_b: RobotBodyPositionInReferenceFrameObsCfg = RobotBodyPositionInReferenceFrameObsCfg()
        robot_body_ori_b: RobotBodyOrientationObsCfg = RobotBodyOrientationObsCfg()
        base_lin_vel: BodyLinearVelocityObsCfg = BodyLinearVelocityObsCfg()
        base_ang_vel: BodyAngularVelocityObsCfg = BodyAngularVelocityObsCfg()
        dof_pos: DofPosRelObsCfg = DofPosRelObsCfg()
        dof_vel: DofVelObsCfg = DofVelObsCfg()
        actions: ActionsObsCfg = ActionsObsCfg()

    policy: PolicyCfg = PolicyCfg()
    value: ValueCfg = ValueCfg()


@configclass
class ResetCfg(ManagerResetCfg):
    """Typed reset terms for WBT."""

    body_pos: BodyPosResetCfg = BodyPosResetCfg()
    body_rot: BodyRotResetCfg = BodyRotResetCfg()
    body_lin_vel: BodyLinVelResetCfg = BodyLinVelResetCfg()
    body_rot_vel: BodyRotVelResetCfg = BodyRotVelResetCfg()
    body_dof_pos: BodyDofPosResetCfg = BodyDofPosResetCfg()


@configclass
class WbtEnvCfg(ManagerBasedEnvCfg):
    """Standalone manager-based configuration for whole-body tracking."""

    queries: SimQueriesCfg = SimQueriesCfg(
        model={
            "actuator_kp": ActuatorKpQuery(),
            "robot_joint_position_limits": BodyJointPositionLimitsQuery(body=MISSING),
        },
    )

    max_episode_seconds: float = 10.0
    sim_reset: ManagerResetCfg = ResetCfg()
    sim: SimCfg = SimCfg(dt=0.005)
    ctrl_dt: float = 0.02
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        robot = self.scene.objs.robot
        joint_names = tuple(robot.resolve_name(name) for name in robot.key_pose.joint_names)
        self.commands.motion.joint_names = joint_names
        self.queries.data["robot_dof_pos"] = JointPositionQuery(joints=joint_names)
        self.queries.data["robot_dof_vel"] = JointVelocityQuery(joints=joint_names)
        self.queries.model["robot_joint_position_limits"] = BodyJointPositionLimitsQuery(
            body=robot.resolved_base_link_name
        )
        if "default" not in robot.key_pose.poses:
            raise ValueError("WBT robot must define key pose 'default'")

    def _set_tracked_body_names(self, body_names: tuple[str, ...]) -> None:
        self.commands.motion.tracked_body_names = body_names
        self.queries.data["tracked_body_pos"] = BatchLinkPositionQuery(links=body_names)
        self.queries.data["tracked_body_quat"] = BatchLinkQuaternionQuery(links=body_names)
        self.queries.data["tracked_body_linear_velocity"] = BatchLinkLinearVelocityQuery(links=body_names)
        self.queries.data["tracked_body_angular_velocity"] = BatchLinkAngularVelocityQuery(links=body_names)

    def for_play(self) -> WbtEnvCfg:
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
