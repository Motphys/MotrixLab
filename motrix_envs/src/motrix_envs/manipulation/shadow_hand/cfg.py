# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Shadow Hand Cube Reorientation Environment"""

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.direct.env import DirectEnvCfg

# Path to the repose_cube.xml model
model_file = os.path.join(os.path.dirname(__file__), "xmls", "repose_cube.xml")

_FINGERTIP_LINKS = (
    "rh_ffdistal",  # First finger (index) distal
    "rh_mfdistal",  # Middle finger distal
    "rh_rfdistal",  # Ring finger distal
    "rh_lfdistal",  # Little finger distal
    "rh_thdistal",  # Thumb distal
)


@registry.envcfg("shadow-hand-repose")
@configclass
class ShadowHandReposeEnvCfg(DirectEnvCfg):
    """Control a Shadow Hand to reorient a cube in-hand.

    zh_CN: 控制 Shadow Hand 在手中重定向立方体。
    """

    # ====================
    # Scene Configuration
    # ====================
    scene: SceneCfg = SceneCfg(file=model_file)

    # ====================
    # Simulation Parameters
    # ====================
    sim: SimCfg = SimCfg(dt=0.01, max_contacts=256, max_contact_constraints=768)
    sim_substeps: int = 1  # Number of simulation steps per control step
    ctrl_dt: float = 0.01 * sim_substeps

    max_episode_seconds: float = 10.0
    max_episode_steps: int = int(max_episode_seconds / ctrl_dt)

    # ====================
    # Robot Configuration
    # ====================
    num_hand_dofs: int = 24  # Total DOFs in Shadow Hand
    num_actuators: int = 20  # Actuated joints

    # Fingertip link names for forward kinematics
    fingertip_link_names: list[str] = (
        "rh_ffdistal",  # First finger (index) distal
        "rh_mfdistal",  # Middle finger distal
        "rh_rfdistal",  # Ring finger distal
        "rh_lfdistal",  # Little finger distal
        "rh_thdistal",  # Thumb distal
    )

    # ====================
    # Object Configuration
    # ====================
    cube_initial_pos: tuple[float, float, float] = (0.33, 0.00, 0.295)  # Initial cube position

    # ====================
    # Reward Parameters
    # ====================
    # Core reward components
    dist_reward_scale: float = -10.0  # Balanced for MotrixSim (推荐)
    rot_reward_scale: float = 1.0  # Moderate rotation reward
    rot_eps: float = 0.1  # Stable denominator

    action_penalty_scale: float = -0.0002

    # Success and failure criteria
    success_tolerance: float = 0.1  # ~8.6° (moderate challenge)
    reach_goal_bonus: float = 2.0  # Balanced incentive

    fall_dist: float = 0.24  # Reasonable manipulation space          # Distance threshold for dropping cube (meters)
    fall_penalty: float = 0.0  # Penalty for dropping the cube

    # In-hand distance threshold (only used for success check, not reward)
    in_hand_dist_threshold: float = 0.05  # Distance threshold for "in-hand" (5cm)

    # Success hold mechanism (uses max_consecutive_successes)
    max_consecutive_successes: int = 50  # Reset after holding success for this many steps

    # Averaging factor for consecutive successes tracking
    av_factor: float = 0.1

    # ====================
    # Reset Noise Parameters
    # ====================
    reset_position_noise: float = 0.01  # Increased robustness
    reset_dof_pos_noise: float = 0.2  # Higher generalization
    reset_dof_vel_noise: float = 0.0  # DOF velocity noise at reset

    # ====================
    # Observation Scaling
    # ====================
    vel_obs_scale: float = 0.2  # Scale factor for velocity observations

    # ====================
    # Action Processing
    # ====================
    act_moving_average: float = 1.0  # Action smoothing (1.0 = no smoothing)

    # ====================
    # Visualization
    # ====================
    # Offset for target visualization (relative to hand position)
    # Recommended: offset to upper-left to avoid occluding the real hand and cube
    viz_target_offset: tuple[float, float, float] = (
        0.0,  # Left (negative X)
        0.0,  # Forward/Up (negative Y)
        0.2,  # Up (positive Z)
    )

    # ====================
    # Domain Randomization (Optional)
    # ====================
    # Enable domain randomization (recommended to start with False)
    enable_domain_randomization: bool = False

    # Randomization parameters (only used if enable_domain_randomization=True)
    randomize_friction: bool = False
    friction_range: tuple[float, float] = (0.8, 12)

    randomize_mass: bool = False
    mass_range: tuple[float, float] = (0.8, 1.2)

    randomize_com: bool = False
    com_displacement_range: float = 0.01
