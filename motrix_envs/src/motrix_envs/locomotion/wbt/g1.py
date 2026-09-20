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
    EeBodyPosRewardCfg,
    EeBodyPosZRewardCfg,
    FlightRotationProgressRewardCfg,
    FlightTuckRewardCfg,
    GlobalBodyAngularVelocityRewardCfg,
    GlobalRefOrientationRewardCfg,
    GlobalRefPositionRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.terminations import (
    BadBodyZTerminationCfg,
    BadDofPositionTerminationCfg,
    BadDofVelocityTerminationCfg,
    BadMotionBodyPositionTerminationCfg,
    BadRefFullOrientationTerminationCfg,
    BadRefOrientationTerminationCfg,
    BadRefPositionPhasedTerminationCfg,
    BadRefZPhasedTerminationCfg,
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
    """Backflip rewards: v5 hybrid — v3's proven stack, holosoma-aligned base.

    zh_CN: 后空翻奖励：v3 验证过的组合（tuck/EE 高度/轻动作率课程）保留。

    The pure holosoma fastsac weights (v4) regressed the launch: the strict
    EE termination killed every takeoff attempt before the skill formed, and
    without ``flight_tuck`` / ``motion_ee_body_pos_z`` / the light
    curriculum-scaled ``action_rate_l2`` there was no remaining signal that
    rewards the launch itself. This hybrid keeps those three launch-critical
    terms (validated by the v1-v3 ablations) on top of the holosoma-aligned
    base weights.
    """

    # Flat -0.01 (gating removed): the action-rate sweep established -0.01 as
    # the only weight that unlocks rotation within a 40k budget (pitch 1.04,
    # vs 0.04-0.25 for every heavier flat or gated variant tested).
    action_rate_l2: ActionRateRewardCfg = ActionRateRewardCfg(weight=-0.01)

    # Dense rotation signal: linear pay per radian of reference-direction
    # pitch rate inside the flight window — the exp ang-vel kernel alone
    # cannot bootstrap rotation from partial attempts.
    flight_rotation_progress: FlightRotationProgressRewardCfg = FlightRotationProgressRewardCfg(weight=0.3)

    # Load-bearing for the launch/landing breakthrough (v1 ablation: episode
    # length ~326 with vs ~138 without). Tuck coverage extended to the leg
    # ABDUCTION joints (hip roll/yaw): play inspection of the 80k runs showed
    # the legs spreading apart mid-flight — the reference tuck holds hip_roll
    # within ±2° while the policy flew with open legs, inflating the moment
    # of inertia and stalling rotation at ~3 rad (half-turn plateau). Those
    # joints were in no reward's coverage: the tuck term only tracked
    # knee/hip_pitch and the all-body position mean dilutes ankle deviation
    # to 1/13. Sigma tightened 2.5 -> 1.0: at 2.5 a 1 rad joint error only
    # costs ~15% of the kernel — effectively unshaped; weight raised
    # 2.5 -> 4.0 for the physical lever (tighter tuck, smaller inertia,
    # the missing final third of the rotation).
    flight_tuck: FlightTuckRewardCfg = FlightTuckRewardCfg(
        weight=4.0,
        sigma=1.0,
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
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
        ),
    )

    # Load-bearing for launch bootstrap (ablated 2026-09-20): with weight 0
    # and only the full-3D EE term left, launch formation stalls ~20k iters
    # (max_z 0.09 at 30k vs 0.43 with this term) — the 3D error is dominated
    # by early xy drift and drowns the height signal. Endpoint at 40k nearly
    # recovers, but the mid-training stall costs the budget.
    motion_ee_body_pos_z: EeBodyPosZRewardCfg = EeBodyPosZRewardCfg(
        weight=0.0,
        sigma=0.3,
        body_names=(
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ),
    )

    # Full-3D EE tracking: the z-only variant left foot placement (xy) error
    # at ~0.12 m — diluted to 1/14 in the all-body mean, so landing accuracy
    # had no dedicated gradient. Kernel at the observed error is ~0.86,
    # comfortably inside the learning band.
    motion_ee_body_pos: EeBodyPosRewardCfg = EeBodyPosRewardCfg(
        weight=3.0,
        sigma=0.3,
        body_names=(
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ),
    )

    # holosoma semantics restored (weight 1.0, sigma 3.14): the v2-era
    # tightening to sigma 1.0 saturated the kernel against the ~1.5-2 rad/s
    # whole-body angular-velocity noise floor (kernel pinned at 0.02 for the
    # entire v7 run — no gradient, rotation never learned).
    motion_global_body_ang_vel: GlobalBodyAngularVelocityRewardCfg = GlobalBodyAngularVelocityRewardCfg(
        weight=1.0,
        sigma=3.14,
    )
    motion_global_ref_orientation_error_exp: GlobalRefOrientationRewardCfg = GlobalRefOrientationRewardCfg(
        weight=1.0,
        sigma=0.4,
    )
    # Planar drift needs an unsaturated gradient: at the observed 1.28 m drift
    # the stock sigma 0.3 kernel is exp(-18) ~ 0. Widening to 1.0 restores a
    # learnable signal (exp(-1.6) ~ 0.2 at current drift); near-field
    # precision is handled by the relative-body terms.
    motion_global_ref_position_error_exp: GlobalRefPositionRewardCfg = GlobalRefPositionRewardCfg(
        weight=1.0,
        sigma=1.0,
    )


@configclass(kw_only=True)
class G1BackflipWbtEnvCfg(G1WbtEnvCfg):
    """Backflip tracking with the official-gain G1 model.

    zh_CN: 后空翻跟踪：官方增益 G1 模型 + holosoma fastsac 配方（含跟踪终止）。
    The stock menagerie gains (28-99 N*m/rad) sag so much under gravity that no
    learning signal survives the standing frames (perfect-tracking rollouts
    collapse identically with and without controls). The official Unitree RL
    gains (hip 100, knee 150, ankle 40) keep the stance phase stable and let the
    policy learn the feedforward offsets the flip needs. Termination follows
    holosoma's ``BadTrackingZOnly`` contract: anchor 3D position error
    > 0.5 m (catches planar drift a z-only check cannot see), full anchor
    orientation error > 0.8 rad, and end-effector (ankle/wrist) position error
    vs the reference targets > 0.25 m — sloppy limb motion terminates the
    episode early, and the adaptive sampler re-samples those failing frames
    instead of the rollout surviving to timeout with garbage in the buffer.
    Rewards and reset are the holosoma fastsac recipe unchanged (centered
    noise reference-state teleport with velocities, adaptive failure-biased
    frame sampling).
    """

    scene: StandardSceneCfg = StandardSceneCfg(
        system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
        # Stock menagerie gains keep the per-joint action scales large
        # (scale = 0.25 * effort/kp); official gains shrank them ~2.5x,
        # limiting how far the policy can push targets per step during the
        # violent launch.
        objs=StandardSceneObjsCfg(robot=UnitreeG129Dof()),
    )

    # dt=0.01 experiment (80k iters, mixed sampling): ~2x collector
    # throughput but rotation never formed with 10% frame-0 starts and only
    # reached ~0.6 rad with 20% — the launch impulse that creates angular
    # momentum spans ~0.1 s of ground contact and is lost to the coarse
    # contact solve. dt=0.005 is required for the flip to be learnable.
    sim: SimCfg = SimCfg(dt=0.005, solver_iterations=3)
    render_spacing = 1.5

    commands: CommandsCfg = CommandsCfg(
        motion=WbtMotionCommandCfg(
            # UniLab SAC flip's "mixed" sampling over the whole clip: 20% of
            # episodes start at frame 0 (full skill from standing), 80% start
            # uniformly anywhere — mid-air frames included, as in holosoma's
            # RSI and UniLab's flip recipes. Frame-0-only start concentrated
            # resets on the run-up frames the early policy dies in, starving
            # the flight window and the landing hold of episode-start coverage.
            # Adaptive sampling was rejected in BOTH regimes: from scratch the
            # uniform_ratio sweep (0.1 / 0.5 / 0.8) lost to plain uniform at
            # every value (40k rotation 0.08 / 0.15 / 0.06 rad vs 0.48; the
            # 0.8 run held even at 60k), and fine-tuning the already-flipping
            # 80k policy for another 40k with adaptive sampling REGRESSED the
            # play-visible skill vs the mixed checkpoint. Failure-biased start
            # frames are strictly harmful for this task.
            # Mid-air resets are exact in velocity (see __post_init__: root
            # velocity reset noise zeroed), so teleports into the flight
            # window land on the reference ballistic trajectory. Judge skill
            # by frame-0-start episodes (play): training-time means stay
            # diluted by mid-clip starts.
            adaptive_sampling_enabled=False,
            start_at_timestep_zero_prob=0.2,
            flight_metrics_enabled=True,
            # Hold the final frame instead of wrap-rematerializing: an episode
            # that reaches the clip end keeps tracking the landing hold until
            # timeout, so "land and stay standing" is the terminal skill.
            hold_at_clip_end=True,
        ),
    )

    terminations: TerminationsCfg = TerminationsCfg(
        # v6: v3's launch-forcing stack (validated to produce full flips).
        # bad_ref_z at 0.35 is the load-bearing term: standing through the
        # flight window yields a pelvis-z error of ~0.44 against the 1.19 m
        # apex, so standing is fatal and jumping is mandatory. The v5 experiment
        # replaced it with the holosoma 3D/EE contract (0.5/0.8/0.45) which
        # standing survives (0.44 < 0.5) — the policy rationally unlearned the
        # launch. Strict contract terms stay as metrics-only here; enable them
        # (with real thresholds) only to fine-tune an already-flipping policy.
        # Phased anchor-z: loose (0.5) outside the flight window so the early
        # policy survives the crouch/run-up, tight (0.25) inside it — at the
        # 1.19 m apex the pelvis must reach >= 0.94 m, beyond any grounded pose
        # (tiptoe tops out ~0.9 m), so passing the window requires actually
        # leaving the ground.
        bad_ref_z=BadRefZPhasedTerminationCfg(threshold=0.5, threshold_in=0.25),
        bad_ref_ori=BadRefOrientationTerminationCfg(threshold=1.5),
        bad_body_z=BadBodyZTerminationCfg(
            threshold=0.5,
            body_names=(
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ),
        ),
        # Aligned-contract terms as metrics-only (disabled thresholds), except
        # the phased 3D position check: grounded drift beyond 0.8 m is a real
        # failure (v7 drifted 1.28 m with zero pressure), while mid-flight the
        # robot is ballistic so only divergence beyond 1.0 m dies.
        bad_ref_pos=BadRefPositionPhasedTerminationCfg(threshold=0.8, threshold_in=1.0),
        bad_ref_full_ori=BadRefFullOrientationTerminationCfg(threshold=1.0e9),
        bad_motion_body_pos=BadMotionBodyPositionTerminationCfg(
            threshold=1.0e9,
            body_names=(
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ),
        ),
        # Kept as NaN/divergence guards; they never fire on healthy rollouts.
        bad_dof_pos=BadDofPositionTerminationCfg(threshold=0.5),
        bad_dof_vel=BadDofVelocityTerminationCfg(threshold=100.0),
    )

    rewards: G1BackflipRewardsCfg = G1BackflipRewardsCfg()

    def __post_init__(self, motion_file: str | None) -> None:
        super().__post_init__(motion_file)
        # Zero the root velocity reset noise for this task only. Most reset
        # frames are grounded (reference velocities near zero), so the
        # holosoma velocity noise (lin ±0.5 m/s, ang ±0.52/0.78 rad/s)
        # degenerates into a random push on a standing robot: same start
        # frame, scattered launch ballistics. UniLab's flip recipes use exact
        # reference-state resets for exactly this reason, and mid-air RSI
        # teleports must land on the reference ballistic trajectory.
        # Position/rotation/dof noise stays — harmless robustness for
        # grounded starts.
        self.sim_reset.body_lin_vel.noise_scale = 0.0
        self.sim_reset.body_rot_vel.noise_scale = 0.0


@registry.envcfg("g1-wbt-backflip")
def make_g129dof_wbt_backflip_cfg() -> EnvCfg:
    """Track the bundled G1 backflip reference motion.

    zh_CN: 让 Unitree G1 跟踪内置后空翻参考动作。
    """

    return G1BackflipWbtEnvCfg(motion_file=str(_MOTION_DIR / "backflip.npz"))


registry.env("g1-wbt-dance")(ManagerEnv)
registry.env("g1-wbt-backflip")(ManagerEnv)
