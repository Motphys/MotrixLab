# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.join(os.path.dirname(__file__), "scene.xml")


@configclass
class InitState:
    """Panda near-hole pose (IK-solved offline, see task docstring).

    Solved against "peg vertical, peg bottom ~8 mm above the socket top",
    starting from the Franka ready configuration. Joint order: 7 arm joints
    + 2 finger joints (closed on the peg).
    """

    default_joint_pos = np.array(
        [0.00300, 0.15184, -0.00259, -2.50364, 0.00487, 2.70612, 0.78305, 0.012, 0.012], np.float32
    )
    joint_pos_reset_noise_scale: float = 0.002


@configclass
class ResetConfig:
    # offset_cap is the OUTER radius. The reset() sampler stratifies into
    # 30/30/40 over (0, 0.3cap] / (0.3cap, 0.6cap] / (0.6cap, cap] — the
    # reference protocol. Stage-1 runs cap=0.002; stage-2 restores 0.010.
    offset_cap: float = 0.010
    bucket_weights: tuple = (0.3, 0.3)
    socket_top_z: float = 0.040
    # Square peg (22 mm) seated between flat pads at matched opening
    # (finger j = 0.0034 gives a 22 mm inner gap): flat-on-flat contact,
    # no roll-off. The peg sinks 6 mm below the pad centers so the grasp
    # plane sits above the peg center of mass.
    # 20 mm above the seating height: the 25-step grasp settle lets the
    # closing pads catch the peg and sink it onto a near-contact airborne
    # start. Fingers START at j=0.012 (39 mm) so the 34 mm peg seats
    # without interpenetration, then the ctrl=0 closure grips it.
    peg_hold_height: float = 0.020
    socket_outer_radius: float = 0.040
    hole_clearance: float = 0.001
    history_len: int = 30


@configclass
class ActionControllerCfg:
    action_scale: float = 0.0025
    # Anti-windup: |joint_target - qpos_arm| hard bound (rad).
    max_tracking_error: float = 0.10
    # Gripper holds the peg; the policy cannot open it. actuator8 is an
    # OPENING servo: F = 350 * (ctrl - j), so grip force = 350 * (j - ctrl)
    # with the peg blocking j at ~0.0125. Commanding full closure (ctrl=0)
    # yields ~4.4 N pad preload per finger — the sustained grip.
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

    Franka Panda (Apache-2.0 MJX model) + a fully parametric primitive
    socket: 24 wall boxes ring a 13 mm blind hole (1.0 mm radial clearance
    against the 12 mm peg). No meshes beyond the stock Panda visual set; no
    proprietary geometry. The actor sees a flattened 30-frame proprioceptive
    history including the PD tracking residual (impedance signal) and the
    gripper linear velocity; privileged peg/socket state is critic-only.
    """

    scene: SceneCfg = SceneCfg(file=model_file)
    sim: SimCfg = SimCfg(dt=0.001)
    ctrl_dt: float = 0.02
    max_episode_seconds: float = 16.0
    render_spacing: float = 1.5
    init_state: InitState = InitState()
    reset: ResetConfig = ResetConfig()
    action: ActionControllerCfg = ActionControllerCfg()
    reward: RewardConfig = RewardConfig()
    termination: TerminationConfig = TerminationConfig()
