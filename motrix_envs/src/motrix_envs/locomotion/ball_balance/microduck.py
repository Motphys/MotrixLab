# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Pollen Robotics Microduck ball-balance preset and registration."""

from pathlib import Path

from omegaconf import MISSING

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import BodyCfg, MjcfFileCfg, SystemCameraCfg
from motrix_env_core.manager import (
    ManagerActionsCfg,
    ManagerBasedEnvCfg,
    ManagerEnv,
    ManagerObservationGroupCfg,
    ManagerObservationsCfg,
    ManagerResetCfg,
    ManagerRewardsCfg,
    ManagerTerminationsCfg,
    SimQueriesCfg,
)
from motrix_env_core.mdp.observations import (
    RobotBaseAngularVelocityObsCfg,
    RobotBaseLinearVelocityObsCfg,
    UniformNoiseCfg,
)
from motrix_env_core.sim import (
    ActuatorKpQuery,
    BatchLinkPositionQuery,
    BodyJointPositionLimitsQuery,
    BodyLinkNetContactForceQuery,
    JointPositionQuery,
    JointVelocityQuery,
    LinkLinearVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
)
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg
from motrix_envs.locomotion.ball_balance.mdp.observations import (
    BallPositionObsCfg,
    BallRelativePositionObsCfg,
    BallRelativeVelocityObsCfg,
    BallVelocityObsCfg,
    ProjectedGravityObsCfg,
)
from motrix_envs.locomotion.ball_balance.mdp.reset import (
    BallResetCfg,
    BodyDofPosResetCfg,
    BodyLinVelResetCfg,
    BodyPosResetCfg,
    BodyRotResetCfg,
    BodyRotVelResetCfg,
)
from motrix_envs.locomotion.ball_balance.mdp.rewards import (
    AliveRewardCfg,
    BallUnderFeetRewardCfg,
    BaseHeightRewardCfg,
    DofDefaultRewardCfg,
    UprightRewardCfg,
)
from motrix_envs.locomotion.ball_balance.mdp.terminations import (
    BadBaseZTerminationCfg,
    BadOrientationTerminationCfg,
    BallEscapedTerminationCfg,
)
from motrix_envs.locomotion.wbt.mdp.action import (
    WbtControlCfg,
    WbtJointPositionActionCfg,
)
from motrix_envs.locomotion.wbt.mdp.observations import (
    ActionsObsCfg,
    DofPosRelObsCfg,
    DofVelObsCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    ActionRateRewardCfg,
    DofLimitRewardCfg,
    UndesiredContactsRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.terminations import (
    BadDofPositionTerminationCfg,
    BadDofVelocityTerminationCfg,
)
from motrix_envs.robot import Microduck

_BALL_ASSET_DIR = Path(__file__).parent / "assets"
_BALL_RADIUS = 0.14
# Trunk rests at z=0.12 in the default stand pose; on top of the ball the
# feet stand at the ball apex (2 * radius), lifting the trunk accordingly.
_ROBOT_BASE_SPAWN_Z = 0.12 + 2.0 * _BALL_RADIUS
_MICRODUCK_FEET_LINKS = ("ankle_left", "ankle_right")


@configclass(kw_only=True)
class Basketball(BodyCfg):
    """Free sphere prop the robot balances on."""

    model: MjcfFileCfg = MjcfFileCfg(file=_BALL_ASSET_DIR / "basketball.xml")
    base_link_name: str = "basketball"


@configclass
class BallBalanceSceneObjsCfg(StandardSceneObjsCfg):
    """Standard robot scene plus the balance ball prop."""

    ball: BodyCfg = MISSING


@configclass
class ActionsCfg(ManagerActionsCfg):
    joint_position: WbtJointPositionActionCfg = WbtJointPositionActionCfg(
        control=WbtControlCfg(action_scale=0.5, action_scales_by_effort_limit_over_p_gain=False),
    )


@configclass
class RewardsCfg(ManagerRewardsCfg):
    alive: AliveRewardCfg = AliveRewardCfg(weight=1.0)
    upright: UprightRewardCfg = UprightRewardCfg(weight=4.0, sigma=0.2)
    base_height: BaseHeightRewardCfg = BaseHeightRewardCfg(weight=1.5, target_z=_ROBOT_BASE_SPAWN_Z, sigma=0.05)
    ball_under_feet: BallUnderFeetRewardCfg = BallUnderFeetRewardCfg(weight=3.0, sigma=0.05)
    dof_default: DofDefaultRewardCfg = DofDefaultRewardCfg(weight=0.5, sigma=0.5)
    action_rate_l2: ActionRateRewardCfg = ActionRateRewardCfg(weight=-0.5)
    limits_dof_pos: DofLimitRewardCfg = DofLimitRewardCfg(weight=-5.0, soft_limit=0.9, cap=5.0)
    undesired_contacts: UndesiredContactsRewardCfg = UndesiredContactsRewardCfg(weight=-0.2, threshold=0.5)


@configclass
class TerminationsCfg(ManagerTerminationsCfg):
    bad_base_z: BadBaseZTerminationCfg = BadBaseZTerminationCfg(threshold=0.22)
    bad_orientation: BadOrientationTerminationCfg = BadOrientationTerminationCfg(threshold=0.6)
    ball_escaped: BallEscapedTerminationCfg = BallEscapedTerminationCfg(threshold=0.20)
    bad_dof_pos: BadDofPositionTerminationCfg = BadDofPositionTerminationCfg(threshold=0.5)
    bad_dof_vel: BadDofVelocityTerminationCfg = BadDofVelocityTerminationCfg(threshold=100.0)


@configclass
class ObservationsCfg(ManagerObservationsCfg):
    @configclass
    class PolicyCfg(ManagerObservationGroupCfg):
        projected_gravity: ProjectedGravityObsCfg = ProjectedGravityObsCfg(noise=UniformNoiseCfg(amplitude=0.05))
        base_ang_vel: RobotBaseAngularVelocityObsCfg = RobotBaseAngularVelocityObsCfg(
            noise=UniformNoiseCfg(amplitude=0.1)
        )
        ball_pos_b: BallRelativePositionObsCfg = BallRelativePositionObsCfg(noise=UniformNoiseCfg(amplitude=0.02))
        ball_vel_b: BallRelativeVelocityObsCfg = BallRelativeVelocityObsCfg(noise=UniformNoiseCfg(amplitude=0.1))
        dof_pos: DofPosRelObsCfg = DofPosRelObsCfg(noise=UniformNoiseCfg(amplitude=0.01))
        dof_vel: DofVelObsCfg = DofVelObsCfg(noise=UniformNoiseCfg(amplitude=0.25))
        actions: ActionsObsCfg = ActionsObsCfg()

    @configclass
    class ValueCfg(ManagerObservationGroupCfg):
        projected_gravity: ProjectedGravityObsCfg = ProjectedGravityObsCfg()
        base_lin_vel: RobotBaseLinearVelocityObsCfg = RobotBaseLinearVelocityObsCfg()
        base_ang_vel: RobotBaseAngularVelocityObsCfg = RobotBaseAngularVelocityObsCfg()
        ball_pos_b: BallRelativePositionObsCfg = BallRelativePositionObsCfg()
        ball_vel_b: BallRelativeVelocityObsCfg = BallRelativeVelocityObsCfg()
        ball_pos: BallPositionObsCfg = BallPositionObsCfg()
        ball_vel: BallVelocityObsCfg = BallVelocityObsCfg()
        dof_pos: DofPosRelObsCfg = DofPosRelObsCfg()
        dof_vel: DofVelObsCfg = DofVelObsCfg()
        actions: ActionsObsCfg = ActionsObsCfg()

    policy: PolicyCfg = PolicyCfg()
    value: ValueCfg = ValueCfg()


@configclass
class ResetCfg(ManagerResetCfg):
    body_pos: BodyPosResetCfg = BodyPosResetCfg(spawn=(0.0, 0.0, _ROBOT_BASE_SPAWN_Z))
    body_rot: BodyRotResetCfg = BodyRotResetCfg()
    body_lin_vel: BodyLinVelResetCfg = BodyLinVelResetCfg()
    body_rot_vel: BodyRotVelResetCfg = BodyRotVelResetCfg()
    body_dof_pos: BodyDofPosResetCfg = BodyDofPosResetCfg()
    ball: BallResetCfg = BallResetCfg(
        ball_link_name="basketball",
        spawn=(0.0, 0.0, _BALL_RADIUS),
    )


@configclass
class MicroduckBallBalanceEnvCfg(ManagerBasedEnvCfg):
    """Manager-based configuration for Microduck balancing on a basketball."""

    queries: SimQueriesCfg = SimQueriesCfg(
        model={
            "actuator_kp": ActuatorKpQuery(),
            "robot_joint_position_limits": BodyJointPositionLimitsQuery(body=MISSING),
        },
    )

    max_episode_seconds: float = 20.0
    sim_reset: ManagerResetCfg = ResetCfg()
    sim: SimCfg = SimCfg(dt=0.005, solver_iterations=6, solver_tolerance=1e-4)
    ctrl_dt: float = 0.02
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    scene: StandardSceneCfg = StandardSceneCfg(
        system_camera=SystemCameraCfg(
            lookat=(0.0, 0.0, 0.2),
            distance=0.6,
            elevation=-15.0,
            azimuth=180.0,
        ),
        objs=BallBalanceSceneObjsCfg(robot=Microduck(), ball=Basketball()),
    )

    def __post_init__(self) -> None:
        robot = self.scene.objs.robot
        joint_names = tuple(robot.resolve_name(name) for name in robot.key_pose.joint_names)
        base_link = robot.resolved_base_link_name
        feet_links = tuple(robot.resolve_name(name) for name in _MICRODUCK_FEET_LINKS)
        self.queries.data["robot_dof_pos"] = JointPositionQuery(joints=joint_names)
        self.queries.data["robot_dof_vel"] = JointVelocityQuery(joints=joint_names)
        self.queries.data["robot_base_pos"] = LinkPositionQuery(link=base_link)
        self.queries.data["robot_base_quat"] = LinkQuaternionQuery(link=base_link)
        self.queries.data["feet_pos"] = BatchLinkPositionQuery(links=feet_links)
        self.queries.data["ball_pos"] = LinkPositionQuery(link="basketball")
        self.queries.data["ball_lin_vel"] = LinkLinearVelocityQuery(link="basketball")
        self.queries.data["undesired_contact_forces"] = BodyLinkNetContactForceQuery(
            body=base_link,
            exclude_links=feet_links,
        )
        self.queries.model["robot_joint_position_limits"] = BodyJointPositionLimitsQuery(body=base_link)


@registry.envcfg("microduck-ball-balance")
def make_microduck_ball_balance_cfg() -> MicroduckBallBalanceEnvCfg:
    """Balance on top of a basketball with Microduck.

    zh_CN: 让 Microduck 双脚站在篮球上并保持平衡。
    """
    return MicroduckBallBalanceEnvCfg()


registry.env("microduck-ball-balance")(ManagerEnv)
