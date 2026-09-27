# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""G1 backflip tracking task."""

from motrix_env_core import registry
from motrix_env_core.base import EnvCfg, SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SystemCameraCfg
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.rewards import ActionRateRewardCfg
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg
from motrix_envs.locomotion.wbt.cfg import CommandsCfg, RewardsCfg, TerminationsCfg
from motrix_envs.locomotion.wbt.mdp.command import WbtMotionCommandCfg
from motrix_envs.locomotion.wbt.mdp.rewards import (
    EeBodyPosRewardCfg,
    GlobalBodyAngularVelocityRewardCfg,
    GlobalRefOrientationRewardCfg,
    GlobalRefPositionRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.terminations import (
    BadBodyZTerminationCfg,
    BadDofPositionTerminationCfg,
    BadDofVelocityTerminationCfg,
    BadRefOrientationTerminationCfg,
    BadRefZTerminationCfg,
)
from motrix_envs.robot import UnitreeG129Dof

from .common import MOTION_DIR, G1WbtEnvCfg


@configclass
class G1BackflipRewardsCfg(RewardsCfg):
    """Backflip rewards: holosoma-aligned base plus a light action-rate term.

    zh_CN: 后空翻奖励：holosoma 对齐基础权重，外加轻量动作率项。

    The light ``action_rate_l2`` keeps per-step action churn from
    dominating while the position/orientation terms drive the flip.
    """

    # Deliberately light: heavier rates (or gating) suppress the rotation
    # signal before the flip forms.
    action_rate_l2: ActionRateRewardCfg = ActionRateRewardCfg(weight=-0.01)

    # Full-3D EE tracking: planar foot-placement error is diluted to 1/14 in
    # the all-body mean, so landing accuracy needs its own gradient; the
    # kernel sits comfortably inside the learning band at typical errors.
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

    # Wide sigma on purpose: against the ~1.5-2 rad/s whole-body
    # angular-velocity noise floor a tighter kernel saturates near zero and
    # the rotation signal loses its gradient.
    motion_global_body_ang_vel: GlobalBodyAngularVelocityRewardCfg = GlobalBodyAngularVelocityRewardCfg(
        weight=1.0,
        sigma=3.14,
    )
    motion_global_ref_orientation_error_exp: GlobalRefOrientationRewardCfg = GlobalRefOrientationRewardCfg(
        weight=1.0,
        sigma=0.4,
    )
    # Planar drift needs an unsaturated gradient: at meter-scale drift the
    # stock sigma 0.3 kernel is vanishingly small. Widening to 1.0 keeps a
    # learnable signal far out; near-field precision is handled by the
    # relative-body terms.
    motion_global_ref_position_error_exp: GlobalRefPositionRewardCfg = GlobalRefPositionRewardCfg(
        weight=1.0,
        sigma=1.0,
    )


@configclass(kw_only=True)
class G1BackflipWbtEnvCfg(G1WbtEnvCfg):
    """Backflip tracking with the built-in G1 model.

    zh_CN: 后空翻跟踪：内置 G1 模型 + holosoma fastsac 配方（z 向跟踪终止）。
    Uses the stock menagerie-gain G1 unchanged — the flip is learnable with
    the built-in actuator gains, whose large per-joint action scales
    (scale = 0.25 * effort/kp) let the policy push targets far during the
    violent launch. Termination is holosoma's ``BadTrackingZOnly`` shape:
    under the command's per-step re-anchoring, world-z is the only anchor
    axis with a physical reference (gravity), so ``bad_ref_z`` (0.5 m) is
    the sole anchor-position check, backed by the tilt-only orientation
    check and body-z/NaN guards. Rewards and reset are the holosoma
    fastsac recipe unchanged (centered-noise reference-state teleport with
    velocities, mixed frame sampling).
    """

    scene: StandardSceneCfg = StandardSceneCfg(
        system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
        objs=StandardSceneObjsCfg(robot=UnitreeG129Dof()),
    )

    # The launch impulse that creates angular momentum spans ~0.1 s of
    # ground contact; a coarser timestep loses it to the contact solve and
    # the flip stops being learnable, despite the higher collector
    # throughput.
    sim: SimCfg = SimCfg(dt=0.005, solver_iterations=3)
    render_spacing: float = 1.5

    commands: CommandsCfg = CommandsCfg(
        motion=WbtMotionCommandCfg(
            # Mixed sampling: 20% of episodes start at frame 0 (full skill
            # from standing), 80% start uniformly anywhere — mid-air frames
            # included. Frame-0-only start starves the flight window and the
            # landing hold of episode-start coverage; adaptive
            # failure-biased sampling is strictly harmful for this task.
            # Mid-air resets are exact in velocity, so teleports into the
            # flight window land on the reference ballistic trajectory; judge
            # skill by frame-0-start episodes (play), since training-time
            # means stay diluted by mid-clip starts.
            adaptive_sampling_enabled=False,
            start_at_timestep_zero_prob=0.2,
            # Hold the final frame instead of wrap-rematerializing: an episode
            # that reaches the clip end keeps tracking the landing hold until
            # timeout, so "land and stay standing" is the terminal skill.
            hold_at_clip_end=True,
        ),
    )

    terminations: TerminationsCfg = TerminationsCfg(
        # bad_ref_z is the launch-forcing term: standing through the flight
        # window yields a pelvis-z error of ~0.44 against the 1.19 m apex, so
        # standing is fatal and jumping is mandatory. Under the per-step
        # re-anchoring semantics world-z is the only anchor axis with a
        # physical reference (gravity), so it is also the only anchor-position
        # check the preset enables.
        bad_ref_z=BadRefZTerminationCfg(threshold=0.5),
        bad_ref_ori=BadRefOrientationTerminationCfg(threshold=1.5),
        bad_body_z=BadBodyZTerminationCfg(
            threshold=0.5,
            body_names=(
                "left_ankle_roll_link",
                "right_ankle_roll_link",
            ),
        ),
        # Kept as NaN/divergence guards; they never fire on healthy rollouts.
        bad_dof_pos=BadDofPositionTerminationCfg(threshold=0.5),
        bad_dof_vel=BadDofVelocityTerminationCfg(threshold=100.0),
    )

    rewards: G1BackflipRewardsCfg = G1BackflipRewardsCfg()


@registry.envcfg("g1-wbt-backflip")
def make_g129dof_wbt_backflip_cfg() -> EnvCfg:
    """Track the bundled G1 backflip reference motion.

    zh_CN: 让 Unitree G1 跟踪内置后空翻参考动作。
    """

    return G1BackflipWbtEnvCfg(motion_file=str(MOTION_DIR / "backflip.npz"))


registry.env("g1-wbt-backflip")(ManagerEnv)
