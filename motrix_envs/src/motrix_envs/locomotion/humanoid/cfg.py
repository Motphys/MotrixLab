# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared configuration for the manager-based humanoid velocity-tracking task.

Robot presets (``g1`` / ``k1`` / ``microduck`` / ``dex_evt``) provide the
scene and model element names; all task parameters live on the manager term
cfgs declared here. The fused kernel owns observations, rewards,
terminations, the gait-phase command clock, the penalty-scale curriculum,
rough-terrain spawn sampling, and terrain-height lookups. Compile-time model
queries (height-field grid, key-pose foot frames via FK) provide the static
data those terms consume. No environment subclass is needed.
"""

from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import (
    HFieldTerrainCfg,
    NoiseTerrainGeneratorCfg,
    ProceduralHFieldAssetCfg,
    SystemCameraCfg,
)
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
    BodyJointPosRelObsCfg,
    BodyJointVelObsCfg,
    BodyLinearVelocityObsCfg,
    BodyProjectedGravityObsCfg,
    CommandObsCfg,
    UniformNoiseCfg,
)
from motrix_env_core.mdp.rewards import (
    AliveRewardCfg,
    TrackingAngVelZRewardCfg,
    TrackingLinVelXyRewardCfg,
)
from motrix_env_core.mdp.terminations import CollidingTerminationCfg
from motrix_env_core.sim import (
    ActuatorKpQuery,
    BodyJointPositionLimitsQuery,
    GeomSpecsQuery,
    HeightFieldDataQuery,
)
from motrix_envs.config.scene import StandardSceneAssetsCfg, StandardSceneCfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.command import WalkCommandCfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.observations import GaitPhaseObsCfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.reset import WalkStateResetCfg
from motrix_envs.locomotion.humanoid.walk_manager_mdp.rewards import (
    FeetPhaseRewardCfg,
    PenaltyActionRateRewardCfg,
    PenaltyAngVelXyRewardCfg,
    PenaltyCloseFeetXyRewardCfg,
    PenaltyFeetOriRewardCfg,
    PenaltyOrientationRewardCfg,
    PoseRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.action import WbtControlCfg, WbtJointPositionActionCfg
from motrix_envs.robot import HumanoidRobotCfg


@configclass
class TerrainSceneAssetsCfg(StandardSceneAssetsCfg):
    terrain: ProceduralHFieldAssetCfg = ProceduralHFieldAssetCfg(
        generator=NoiseTerrainGeneratorCfg(
            seed=0,
            height_scale=0.05,
            flip_y=True,
        ),
        size=(32.0, 32.0),
        shape=(320, 320),
    )


@configclass
class HumanoidWalkSceneCfg(StandardSceneCfg):
    """Standard humanoid-walk scene with consistent playback framing."""

    system_camera: SystemCameraCfg = SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0)


@configclass
class WalkActionsCfg(ManagerActionsCfg):
    """Position action term shared with the WBT task family."""

    joint_position: WbtJointPositionActionCfg = WbtJointPositionActionCfg(
        control=WbtControlCfg(action_scale=0.5, action_scales_by_effort_limit_over_p_gain=False)
    )


@configclass
class WalkCommandsCfg(ManagerCommandsCfg):
    """Velocity-command term: sampling, gait clock, and penalty curriculum."""

    walk: WalkCommandCfg = WalkCommandCfg()


@configclass
class WalkRewardsCfg(ManagerRewardsCfg):
    """Reward terms of the humanoid velocity-tracking task."""

    tracking_lin_vel: TrackingLinVelXyRewardCfg = TrackingLinVelXyRewardCfg(command_name="walk", weight=4.0)
    tracking_ang_vel: TrackingAngVelZRewardCfg = TrackingAngVelZRewardCfg(command_name="walk", weight=3.0)
    penalty_ang_vel_xy: PenaltyAngVelXyRewardCfg = PenaltyAngVelXyRewardCfg(weight=-1.0)
    penalty_orientation: PenaltyOrientationRewardCfg = PenaltyOrientationRewardCfg(weight=-10.0)
    penalty_action_rate: PenaltyActionRateRewardCfg = PenaltyActionRateRewardCfg(weight=-0.5)
    feet_phase: FeetPhaseRewardCfg = FeetPhaseRewardCfg(weight=5.0)
    pose: PoseRewardCfg = PoseRewardCfg(weight=-0.5)
    penalty_close_feet_xy: PenaltyCloseFeetXyRewardCfg = PenaltyCloseFeetXyRewardCfg(weight=-10.0)
    penalty_feet_ori: PenaltyFeetOriRewardCfg = PenaltyFeetOriRewardCfg(weight=-5.0)
    alive: AliveRewardCfg = AliveRewardCfg(weight=10.0)


@configclass
class WalkTerminationsCfg(ManagerTerminationsCfg):
    colliding: CollidingTerminationCfg = CollidingTerminationCfg()


@configclass
class WalkObservationsCfg(ManagerObservationsCfg):
    """Actor/critic observation layout of the humanoid velocity-tracking task."""

    @configclass
    class PolicyCfg(ManagerObservationGroupCfg):
        base_ang_vel: BodyAngularVelocityObsCfg = BodyAngularVelocityObsCfg(scale=0.25)
        projected_gravity: BodyProjectedGravityObsCfg = BodyProjectedGravityObsCfg()
        command: CommandObsCfg = CommandObsCfg(command_name="walk")
        joint_pos: BodyJointPosRelObsCfg = BodyJointPosRelObsCfg(scale=1.0, noise=UniformNoiseCfg(amplitude=0.01))
        joint_vel: BodyJointVelObsCfg = BodyJointVelObsCfg(scale=0.05, noise=UniformNoiseCfg(amplitude=0.1))
        actions: ActionsObsCfg = ActionsObsCfg()
        sin_phase: GaitPhaseObsCfg = GaitPhaseObsCfg(offset=0, size=2)
        cos_phase: GaitPhaseObsCfg = GaitPhaseObsCfg(offset=2, size=2)

    @configclass
    class ValueCfg(ManagerObservationGroupCfg):
        base_lin_vel: BodyLinearVelocityObsCfg = BodyLinearVelocityObsCfg(scale=2.0)
        base_ang_vel: BodyAngularVelocityObsCfg = BodyAngularVelocityObsCfg(scale=0.25)
        projected_gravity: BodyProjectedGravityObsCfg = BodyProjectedGravityObsCfg()
        command: CommandObsCfg = CommandObsCfg(command_name="walk")
        joint_pos: BodyJointPosRelObsCfg = BodyJointPosRelObsCfg(scale=1.0)
        joint_vel: BodyJointVelObsCfg = BodyJointVelObsCfg(scale=0.05)
        actions: ActionsObsCfg = ActionsObsCfg()
        sin_phase: GaitPhaseObsCfg = GaitPhaseObsCfg(offset=0, size=2)
        cos_phase: GaitPhaseObsCfg = GaitPhaseObsCfg(offset=2, size=2)

    policy: PolicyCfg = PolicyCfg()
    value: ValueCfg = ValueCfg()


@configclass
class WalkResetCfg(ManagerResetCfg):
    humanoid_state: WalkStateResetCfg = WalkStateResetCfg()


@configclass
class HumanoidVelocityTrackingManagerEnvCfg(ManagerBasedEnvCfg):
    """Robot-agnostic manager-based humanoid velocity-tracking configuration.

    Term cfgs own their parameters directly: reward weights and sigmas on the
    reward terms, gait and curriculum knobs on ``commands.walk``, contact
    termination geoms on ``terminations.colliding``. ``__post_init__`` only
    assembles the shared model queries and propagates the scene-level floor
    geom and control dt.
    """

    scene: HumanoidWalkSceneCfg = HumanoidWalkSceneCfg()
    max_episode_seconds: float = 20.0
    sim: SimCfg = SimCfg(dt=0.005)
    ctrl_dt: float = 0.02

    queries: SimQueriesCfg = SimQueriesCfg()
    sim_reset: WalkResetCfg = WalkResetCfg()
    actions: WalkActionsCfg = WalkActionsCfg()
    commands: WalkCommandsCfg = WalkCommandsCfg()
    observations: WalkObservationsCfg = WalkObservationsCfg()
    rewards: WalkRewardsCfg = WalkRewardsCfg()
    terminations: WalkTerminationsCfg = WalkTerminationsCfg()

    def __post_init__(self) -> None:
        robot = self.scene.objs.robot
        if not isinstance(robot, HumanoidRobotCfg):
            raise TypeError(f"humanoid walk scene robot must be HumanoidRobotCfg, got {type(robot).__name__}")
        if "default" not in robot.key_pose.poses:
            raise ValueError("humanoid walk robot must define key pose 'default'")

        ground_geom = self.terminations.colliding.ground_geom
        if not ground_geom:
            raise ValueError("terminations.colliding requires a non-empty ground_geom")
        termination_geoms = tuple(name for name in self.terminations.colliding.termination_geoms if name != ground_geom)
        # Reward and observation terms self-declare their data queries; the
        # task declares only what no term owns.
        self.queries.data = {}
        self.queries.model = {
            "geoms": GeomSpecsQuery(names=termination_geoms + (ground_geom,)),
            "actuator_kp": ActuatorKpQuery(),
            "robot_joint_position_limits": BodyJointPositionLimitsQuery(body=robot.resolved_base_link_name),
        }

        # Rough-terrain presets place an HField geom as the floor; export its
        # static grid so the fused kernel can look up ground heights itself.
        self.ground_heightfield_geom: str | None = None
        floor_obj = getattr(self.scene.objs, "floor", None)
        if isinstance(floor_obj, HFieldTerrainCfg):
            self.ground_heightfield_geom = ground_geom
            self.queries.model["ground_heightfield"] = HeightFieldDataQuery(geom=ground_geom)

        # Scene-level facts shared by several terms.
        self.rewards.feet_phase.ground_geom = ground_geom
        self.sim_reset.humanoid_state.ground_geom = ground_geom
        self.commands.walk.ctrl_dt = self.ctrl_dt
