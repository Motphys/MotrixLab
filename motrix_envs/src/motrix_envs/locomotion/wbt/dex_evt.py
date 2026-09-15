# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Dex-EVT whole-body tracking preset and registration."""

from dataclasses import InitVar
from pathlib import Path

from motrix_env_core import registry
from motrix_env_core.base import EnvCfg, SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, SystemCameraCfg
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.rewards import ActionRateRewardCfg
from motrix_env_core.sim import BodyLinkNetContactForceQuery
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg
from motrix_envs.locomotion.wbt.cfg import ActionsCfg, CommandsCfg, RewardsCfg, TerminationsCfg, WbtEnvCfg
from motrix_envs.locomotion.wbt.mdp.action import (
    WbtControlCfg,
    WbtJointPositionActionCfg,
)
from motrix_envs.locomotion.wbt.mdp.command import (
    WbtMotionCommandCfg,
)
from motrix_envs.locomotion.wbt.mdp.rewards import (
    GlobalRefPositionRewardCfg,
)
from motrix_envs.locomotion.wbt.mdp.terminations import (
    BadBodyZTerminationCfg,
)
from motrix_envs.robot import DexEvt

_MOTION_DIR = Path(__file__).parent / "assets" / "motion" / "dex_evt"
_DEX_EVT_TRACKED_BODY_NAMES = (
    "pelvis",
    "hip_roll_l_link",
    "knee_pitch_l_link",
    "ankle_roll_l_link",
    "hip_roll_r_link",
    "knee_pitch_r_link",
    "ankle_roll_r_link",
    "waist_pitch_link",
    "shoulder_roll_l_link",
    "elbow_pitch_l_link",
    "left_tcp_link",
    "shoulder_roll_r_link",
    "elbow_pitch_r_link",
    "right_tcp_link",
)


@configclass(kw_only=True)
class DexEvtWbtEnvCfg(WbtEnvCfg):
    """Whole-body tracking configuration specialized for Dex-EVT."""

    motion_file: InitVar[str | None] = None
    commands: CommandsCfg = CommandsCfg(motion=WbtMotionCommandCfg())
    actions: ActionsCfg = ActionsCfg(
        joint_position=WbtJointPositionActionCfg(
            control=WbtControlCfg(action_scale=1.0, action_scales_by_effort_limit_over_p_gain=False),
        ),
    )
    sim: SimCfg = SimCfg(dt=0.005, solver_iterations=6, solver_tolerance=0.0001)
    scene: StandardSceneCfg = StandardSceneCfg(
        system_camera=SystemCameraCfg(distance=7.0, elevation=-20.0, azimuth=180.0),
        objs=StandardSceneObjsCfg(
            floor=FlatTerrainCfg(material="mat_ground", friction=(0.8, 0.005, 0.0001)),
            robot=DexEvt(),
        ),
    )
    rewards: RewardsCfg = RewardsCfg(
        motion_global_ref_position_error_exp=GlobalRefPositionRewardCfg(weight=2.0, sigma=0.3),
        action_rate_l2=ActionRateRewardCfg(weight=-0.75),
    )
    terminations: TerminationsCfg = TerminationsCfg(
        bad_body_z=BadBodyZTerminationCfg(
            threshold=0.25,
            body_names=("ankle_roll_l_link", "ankle_roll_r_link", "elbow_pitch_l_link", "elbow_pitch_r_link"),
        ),
    )
    render_spacing: float = 1.5

    def __post_init__(self, motion_file: str | None) -> None:
        super().__post_init__()
        if motion_file is not None:
            self.commands.motion.motion_file = motion_file
        self._set_tracked_body_names(_DEX_EVT_TRACKED_BODY_NAMES)
        self.commands.motion.reference_body_name = "waist_pitch_link"

        # Only the feet may contact freely; arm and hand contacts are penalized.
        self.queries.data["undesired_contact_forces"] = BodyLinkNetContactForceQuery(
            body=self.scene.objs.robot.resolved_base_link_name,
            exclude_links=("ankle_roll_l_link", "ankle_roll_r_link"),
        )


@registry.envcfg("dex-evt-wbt-dance")
def make_dex_evt_wbt_dance_cfg() -> EnvCfg:
    """Track a bundled dance reference motion with Dex-EVT.

    zh_CN: 让 Dex-EVT 跟踪内置舞蹈参考动作。
    """
    return DexEvtWbtEnvCfg(motion_file=str(_MOTION_DIR / "dance1_easy.npz"))


registry.env("dex-evt-wbt-dance")(ManagerEnv)
