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
from motrix_envs.locomotion.wbt.mdp.terminations import (
    BadBodyZTerminationCfg,
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
registry.env("g1-wbt-dance")(ManagerEnv)
