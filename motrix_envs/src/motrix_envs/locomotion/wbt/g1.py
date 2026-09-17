# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree G1 whole-body tracking presets and registration."""

from dataclasses import InitVar
from pathlib import Path

from motrix_env_core import registry
from motrix_env_core.base import EnvCfg, SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SystemCameraCfg
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.rewards import ActionRateRewardCfg
from motrix_env_core.sim import BodyLinkNetContactForceQuery
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg
from motrix_envs.locomotion.wbt.cfg import CommandsCfg, RewardsCfg, TerminationsCfg, WbtEnvCfg
from motrix_envs.locomotion.wbt.mdp.command import (
    WbtMotionCommandCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    CurriculumActionRateRewardCfg,
    EeBodyPosZRewardCfg,
    FlightTuckRewardCfg,
    GlobalBodyAngularVelocityRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.terminations import (
    BadBodyZTerminationCfg,
    BadRefOrientationTerminationCfg,
)
from motrix_envs.robot import UnitreeG129Dof

_MOTION_DIR = Path(__file__).parent / "assets" / "motion" / "g1"
_G1_TRACKED_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)


@configclass(kw_only=True)
class G1WbtEnvCfg(WbtEnvCfg):
    """Whole-body tracking configuration specialized for Unitree G1."""

    motion_file: InitVar[str | None] = None
    commands: CommandsCfg = CommandsCfg(motion=WbtMotionCommandCfg())
    sim: SimCfg = SimCfg(dt=0.005, solver_iterations=3)
    scene: StandardSceneCfg = StandardSceneCfg(
        system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
        objs=StandardSceneObjsCfg(robot=UnitreeG129Dof()),
    )
    rewards: RewardsCfg = RewardsCfg(action_rate_l2=ActionRateRewardCfg(weight=-0.5))
    terminations: TerminationsCfg = TerminationsCfg(
        bad_body_z=BadBodyZTerminationCfg(
            threshold=0.25,
            body_names=(
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ),
        ),
    )

    def __post_init__(self, motion_file: str | None) -> None:
        super().__post_init__()
        if motion_file is not None:
            self.commands.motion.motion_file = motion_file
        self._set_tracked_body_names(_G1_TRACKED_BODY_NAMES)
        self.commands.motion.reference_body_name = "torso_link"

        self.queries.data["undesired_contact_forces"] = BodyLinkNetContactForceQuery(
            body=self.scene.objs.robot.resolved_base_link_name,
            exclude_links=(
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ),
        )


@registry.envcfg("g1-29dof-wbt-largebox")
def make_g129dof_wbt_largebox_cfg() -> EnvCfg:
    """Track a large-box carrying reference motion with Unitree G1.

    zh_CN: 让 Unitree G1 跟踪搬运大箱子的参考动作。
    """

    return G1WbtEnvCfg(motion_file=str(_MOTION_DIR / "sub3_largebox_003.npz"))


@registry.envcfg("g1-wbt-dance")
def make_g129dof_wbt_dance_cfg() -> EnvCfg:
    """Track the bundled G1 dance motion with the manager-based environment.

    zh_CN: 让 Unitree G1 跟踪内置舞蹈参考动作。
    """

    return G1WbtEnvCfg(motion_file=str(_MOTION_DIR / "dance1_subject2.npz"))


registry.env("g1-29dof-wbt-largebox")(ManagerEnv)


@configclass
class G1BackflipRewardsCfg(RewardsCfg):
    """Backflip rewards: add the UniLab-style end-effector height term.

    zh_CN: 后空翻奖励：附加末梢高度项，并放宽动作率惩罚。

    ``action_rate_l2`` is relaxed to -0.1 (UniLab's flip task uses -0.005): the
    launch is the most violent joint-acceleration window of the whole clip, and
    a heavy action-rate tax teaches the policy to stay gentle exactly when it
    must explode.
    """

    # Flat -0.005 scaled by the closed-loop curriculum: once mean episode
    # length clears 400 (reliable landing + hold) the multiplier ramps toward
    # 4x for anti-jitter; falling back under 300 relaxes it again.
    action_rate_l2: CurriculumActionRateRewardCfg = CurriculumActionRateRewardCfg(weight=-0.005)

    # Load-bearing for the landing breakthrough (length ~326 with, ~138
    # without): the undiluted tuck signal through the flight window.
    flight_tuck: FlightTuckRewardCfg = FlightTuckRewardCfg(
        weight=1.5,
        sigma=1.5,
        ground_z=0.15,
        body_names=(
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ),
        joint_names=(
            "left_knee_joint",
            "right_knee_joint",
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
        ),
    )

    motion_ee_body_pos_z: EeBodyPosZRewardCfg = EeBodyPosZRewardCfg(
        weight=2.0,
        sigma=0.3,
        body_names=(
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ),
    )
    # The stock sigma (pi rad/s) makes the reward nearly insensitive to angular
    # velocity error, so the policy has no gradient to rotate faster — the direct
    # cause of under-rotated, inverted touchdowns. Tighten it to sharpen the
    # rotation-speed signal during flight.
    motion_global_body_ang_vel: GlobalBodyAngularVelocityRewardCfg = GlobalBodyAngularVelocityRewardCfg(
        weight=1.0,
        sigma=1.0,
    )


@configclass(kw_only=True)
class G1BackflipWbtEnvCfg(G1WbtEnvCfg):
    """Backflip tracking with the official-gain G1 model.

    zh_CN: 后空翻跟踪：官方增益 G1 模型 + 放宽 body 高度跟踪终止。

    The stock menagerie gains (28-99 N*m/rad) sag so much under gravity that no
    learning signal survives the standing frames (perfect-tracking rollouts
    collapse identically with and without controls). The official Unitree RL
    gains (hip 100, knee 150, ankle 40) keep the stance phase stable and let the
    policy learn the feedforward offsets the flip needs. ``bad_body_z`` is
    restricted to the ankle links at 0.5 m: wrists and ankles legitimately swing
    far from the reference mid-flip. ``bad_ref_ori`` is effectively disabled
    (threshold 1e9, matching UniLab's anchor_ori): torso orientation error is
    expected to spike while airborne and the tracking rewards already penalize
    it. The extra ``motion_ee_body_pos_z`` reward (weight 2.0) mirrors UniLab's
    flip recipe, giving the launch/rotation signal an undiluted path.
    """

    scene: StandardSceneCfg = StandardSceneCfg(
        system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
        # Stock menagerie gains keep the per-joint action scales large
        # (scale = 0.25 * effort/kp); official gains shrank them ~2.5x,
        # limiting how far the policy can push targets per step during the
        # violent launch.
        objs=StandardSceneObjsCfg(robot=UnitreeG129Dof()),
    )

    commands: CommandsCfg = CommandsCfg(
        motion=WbtMotionCommandCfg(
            # Start-mode reset (UniLab's sampling_mode: start): mid-clip resets
            # zero the pelvis velocity at the most violent frames, which is a
            # near-guaranteed death and teaches nothing. Starting at frame 0
            # gives 75 standing/crouch frames to build phase state first.
            adaptive_sampling_enabled=False,
            start_at_timestep_zero_prob=1.0,
            # Anti-jitter curriculum: ramp the action-rate multiplier when the
            # mean episode length EMA clears 400, relax below 300.
            action_rate_curriculum=True,
            curriculum_high=400.0,
            curriculum_low=300.0,
        ),
    )

    terminations: TerminationsCfg = TerminationsCfg(
        bad_body_z=BadBodyZTerminationCfg(
            threshold=0.5,
            body_names=(
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ),
        ),
        bad_ref_ori=BadRefOrientationTerminationCfg(threshold=1.0e9),
    )

    rewards: G1BackflipRewardsCfg = G1BackflipRewardsCfg()


@registry.envcfg("g1-wbt-backflip")
def make_g129dof_wbt_backflip_cfg() -> EnvCfg:
    """Track the bundled G1 backflip reference motion.

    zh_CN: 让 Unitree G1 跟踪内置后空翻参考动作。
    """

    return G1BackflipWbtEnvCfg(motion_file=str(_MOTION_DIR / "flip_360_001__A304.npz"))


registry.env("g1-wbt-dance")(ManagerEnv)
registry.env("g1-wbt-backflip")(ManagerEnv)
