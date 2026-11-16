# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from motrix_robots.franka import PANDA_ASSET_DIR, FrankaPanda

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import MjcfFileCfg, SceneCfg, SceneObjsCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = Path(__file__).parent / "xmls" / "scene.xml"


@configclass
class InitState:
    """Arm targets and settled joint positions for an airborne near-hole start."""

    default_joint_pos = (0.00300, 0.15184, -0.00259, -2.50364, 0.00487, 2.70612, 0.78305, 0.012, 0.012)
    # Recorded with MotrixSim 1.0 at the default targets, with the socket
    # out of reach. Keeping the target bias compensates for gravity sag.
    settled_joint_pos = (0.00300, 0.18100, -0.00279, -2.52489, 0.00398, 2.69831, 0.78305, 0.00002, 0.00002)
    joint_pos_reset_noise_scale: float = 0.002


@configclass
class ResetConfig:
    # offset_cap is the OUTER radius. The reset() sampler stratifies into
    # 30/30/40 over (0, 0.3cap] / (0.3cap, 0.6cap] / (0.6cap, cap] — the
    # reference protocol. Stage-1 runs cap=0.002; stage-2 restores 0.010.
    offset_cap: float = 0.010
    bucket_weights: tuple = (0.3, 0.3)
    socket_top_z: float = 0.040
    socket_outer_radius: float = 0.040
    history_len: int = 30


@configclass
class ActionControllerCfg:
    action_scale: float = 0.0025
    # Anti-windup: |joint_target - qpos_arm| hard bound (rad).
    max_tracking_error: float = 0.10
    # The peg is rigidly attached to the hand. The policy controls only the arm.
    gripper_close_ctrl: float = 0.0


@configclass
class RewardConfig:
    reach_scale: float = 2.0
    xy_scale: float = 0.5
    depth_scale: float = 120.0
    # First-touch pay reduced: at 0.5 it combined with passive sag into a
    # reliable ~2-4/episode for mere plate-resting (stage-1 v2 success 0.002
    # while reward hit 11.2).
    contact_bonus: float = 0.2
    contact_residency: float = 0.01
    contact_maintenance: float = 0.002
    # Depth-linear insertion residency, slightly raised: real insertion pay
    # must dominate every passive leak combined.
    insert_residency: float = 0.03
    mouth_bonus: float = 1.0
    # Dominant terminal pay: the coordinated vertical press is HARD (single
    # joints carry lateral drift); the bonus must crush all farmable income.
    success_bonus: float = 30.0
    action_penalty: float = 0.0002
    drop_penalty: float = 10.0
    descent_scale: float = 100.0


@configclass
class TerminationConfig:
    target_depth: float = 0.015
    hold_steps: int = 10
    xy_tolerance: float = 0.0015
    tilt_tolerance_deg: float = 10.0
    penetration_limit: float = 0.001
    success_penetration_witness: float = 0.00005
    drop_drift: float = 0.015
    drop_angle_deg: float = 15.0


@registry.envcfg("panda-blind-insertion")
@registry.envcfg("panda-insert")
@configclass
class PandaBlindInsertEnvCfg(DirectEnvCfg):
    """Panda proprioceptive blind peg insertion (open asset baseline).

    A 34 mm square peg fits a 36 mm square hole with 1 mm clearance per
    side. The actor sees a proprioceptive history, including the joint
    tracking residual and gripper velocity. The critic also sees peg and
    socket geometry.
    """

    scene: SceneCfg = SceneCfg(
        file=model_file,
        objs=SceneObjsCfg(robot=FrankaPanda(model=MjcfFileCfg(file=PANDA_ASSET_DIR / "panda_welded_peg.xml"))),
    )
    sim: SimCfg = SimCfg(dt=0.001)
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 16.0
    render_spacing: float = 1.5
    init_state: InitState = InitState()
    reset: ResetConfig = ResetConfig()
    action: ActionControllerCfg = ActionControllerCfg()
    reward: RewardConfig = RewardConfig()
    termination: TerminationConfig = TerminationConfig()
