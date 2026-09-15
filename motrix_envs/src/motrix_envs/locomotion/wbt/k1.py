# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Booster K1 whole-body tracking preset and registration."""

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
from motrix_envs.locomotion.wbt.cfg import ActionsCfg, CommandsCfg, TerminationsCfg, WbtEnvCfg
from motrix_envs.locomotion.wbt.mdp.action import (
    WbtControlCfg,
    WbtJointPositionActionCfg,
)
from motrix_envs.locomotion.wbt.mdp.command import (
    WbtMotionCommandCfg,
)
from motrix_envs.locomotion.wbt.mdp.terminations import (
    BadBodyZTerminationCfg,
)
from motrix_envs.robot import BoosterK1

_MOTION_DIR = Path(__file__).parent / "assets" / "motion" / "k1"
_K1_TRACKED_BODY_NAMES = (
    "Trunk",
    "Left_Hip_Roll",
    "Left_Shank",
    "left_foot_link",
    "Right_Hip_Roll",
    "Right_Shank",
    "right_foot_link",
    "Left_Arm_2",
    "Left_Arm_3",
    "left_hand_link",
    "Right_Arm_2",
    "Right_Arm_3",
    "right_hand_link",
)


@configclass(kw_only=True)
class K1WbtEnvCfg(WbtEnvCfg):
    """Whole-body tracking configuration specialized for Booster K1."""

    motion_file: InitVar[str | None] = None
    commands: CommandsCfg = CommandsCfg(motion=WbtMotionCommandCfg())
    actions: ActionsCfg = ActionsCfg(
        joint_position=WbtJointPositionActionCfg(
            # K1 motion clips span large arm/leg offsets from the walk handoff pose.
            # Direct position scaling keeps the full joint range reachable;
            # the MJCF actuator forceranges still enforce K1 torque limits.
            control=WbtControlCfg(action_scale=1.0, action_scales_by_effort_limit_over_p_gain=False),
        ),
    )
    sim: SimCfg = SimCfg(dt=0.005, solver_iterations=6, solver_tolerance=1e-4)
    scene: StandardSceneCfg = StandardSceneCfg(
        system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
        objs=StandardSceneObjsCfg(robot=BoosterK1()),
    )
    terminations: TerminationsCfg = TerminationsCfg(
        bad_body_z=BadBodyZTerminationCfg(
            threshold=0.25,
            body_names=("left_foot_link", "right_foot_link", "left_hand_link", "right_hand_link"),
        ),
    )

    def __post_init__(self, motion_file: str | None) -> None:
        super().__post_init__()
        if motion_file is not None:
            self.commands.motion.motion_file = motion_file
        self._set_tracked_body_names(_K1_TRACKED_BODY_NAMES)
        self.commands.motion.reference_body_name = "Trunk"

        self.queries.data["undesired_contact_forces"] = BodyLinkNetContactForceQuery(
            body=self.scene.objs.robot.resolved_base_link_name,
            exclude_links=("left_foot_link", "right_foot_link"),
        )


@registry.envcfg("k1-wbt-freekick")
def make_k1_wbt_freekick_cfg() -> EnvCfg:
    """Track a free-kick reference motion with Booster K1.

    zh_CN: 让 Booster K1 跟踪任意球射门参考动作。
    """
    cfg = K1WbtEnvCfg(motion_file=str(_MOTION_DIR / "freekick_shoot_arc_02.npz"))
    cfg.rewards.action_rate_l2 = ActionRateRewardCfg(weight=-0.1)
    return cfg


registry.env("k1-wbt-freekick")(ManagerEnv)
