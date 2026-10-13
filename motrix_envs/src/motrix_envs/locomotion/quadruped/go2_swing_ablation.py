# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""The production Go2 F2-P1b flat-walk environment."""

from dataclasses import replace

import numpy as np
from motrix_robots.unitree import UnitreeGo2Robot

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import FlatTerrainCfg, KeyPoseCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.quadruped.cfg import QuadrupedSceneCfg, RewardConfig
from motrix_envs.locomotion.quadruped.go2_walk import Go2WalkDirectEnvCfg
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask


_F2_P1B_KEY_POSE = KeyPoseCfg(
    joint_names=[
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ],
    poses={"default": [
        0.1, 0.8, -1.5,
        -0.1, 0.8, -1.5,
        0.1, 1.0, -1.5,
        -0.1, 1.0, -1.5,
    ]},
)


@registry.envcfg("go2-walk-flat-f2-p1b")
@configclass
class Go2WalkF2P1BEnvCfg(Go2WalkDirectEnvCfg):
    """Go2 flat walk with the selected F2-P1b gait regularization."""

    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(
        objs=StandardSceneObjsCfg(
            floor=FlatTerrainCfg(
                material="mat_ground",
                friction=(0.6, 0.005, 0.0001),
            ),
            robot=UnitreeGo2Robot(key_pose=_F2_P1B_KEY_POSE),
        ),
    )
    reward_config: RewardConfig = replace(
        Go2WalkDirectEnvCfg().reward_config,
        scales=replace(Go2WalkDirectEnvCfg().reward_config.scales, lin_vel_z=-3.0),
    )
    rear_swing_width_target: float = 0.145
    rear_swing_width_scale: float = -0.05
    rear_swing_lateral_speed_scale: float = -0.01
    rear_swing_width_tolerance: float = 0.02
    rear_swing_lateral_speed_tolerance: float = 0.15
    front_swing_inner_width_limit: float = 0.13
    front_swing_width_limit: float = 0.16
    front_swing_width_tolerance: float = 0.02
    front_swing_width_scale: float = -0.02
    front_swing_lateral_speed_tolerance: float = 0.15
    front_swing_lateral_speed_scale: float = -0.002
    front_stance_inner_width_limit: float = 0.13
    front_stance_outer_width_limit: float = 0.16
    front_stance_width_tolerance: float = 0.02
    front_stance_width_scale: float = -0.01


class Go2F2P1BTask(QuadrupedWalkTask):
    """Add the selected F2-P1b swing and stance regularization terms."""

    def _compute_reward(self, state, linvel, gyro, dof_pos):
        previous_feet_pos = self._previous_feet_pos.copy()
        reward = super()._compute_reward(state, linvel, gyro, dof_pos)

        rear_swinging = self._feet_phase[:, 2:4] >= 0.6
        rear_abs_y = np.abs(self.feet_pos[:, 2:4, 1])
        rear_width = np.sum(
            np.square(
                (rear_abs_y - self.cfg.rear_swing_width_target) / self.cfg.rear_swing_width_tolerance
            )
            * rear_swinging,
            axis=1,
        ) / 2.0
        rear_y_velocity = (self.feet_pos[:, 2:4, 1] - previous_feet_pos[:, 2:4, 1]) / self.cfg.ctrl_dt
        rear_speed = np.sum(
            np.square(rear_y_velocity / self.cfg.rear_swing_lateral_speed_tolerance) * rear_swinging,
            axis=1,
        ) / 2.0

        front_swinging = self._feet_phase[:, :2] >= 0.6
        front_abs_y = np.abs(self.feet_pos[:, :2, 1])
        front_outer = np.maximum(front_abs_y - self.cfg.front_swing_width_limit, 0.0)
        front_inner = np.maximum(self.cfg.front_swing_inner_width_limit - front_abs_y, 0.0)
        front_width = np.sum(
            (
                np.square(front_outer / self.cfg.front_swing_width_tolerance)
                + np.square(front_inner / self.cfg.front_swing_width_tolerance)
            )
            * front_swinging,
            axis=1,
        ) / 2.0
        front_y_velocity = (self.feet_pos[:, :2, 1] - previous_feet_pos[:, :2, 1]) / self.cfg.ctrl_dt
        front_speed = np.sum(
            np.square(front_y_velocity / self.cfg.front_swing_lateral_speed_tolerance)
            * front_swinging
            * ((front_outer > 0.0) | (front_inner > 0.0)),
            axis=1,
        ) / 2.0

        front_support = self.feet_contact[:, :2]
        stance_outer = np.maximum(front_abs_y - self.cfg.front_stance_outer_width_limit, 0.0)
        stance_inner = np.maximum(self.cfg.front_stance_inner_width_limit - front_abs_y, 0.0)
        stance_width = np.sum(
            (
                np.square(stance_outer / self.cfg.front_stance_width_tolerance)
                + np.square(stance_inner / self.cfg.front_stance_width_tolerance)
            )
            * front_support,
            axis=1,
        ) / 2.0

        terms = {
            "rear_swing_width": rear_width * self.cfg.rear_swing_width_scale,
            "rear_swing_lateral_speed": rear_speed * self.cfg.rear_swing_lateral_speed_scale,
            "front_swing_width": front_width * self.cfg.front_swing_width_scale,
            "front_swing_lateral_speed": front_speed * self.cfg.front_swing_lateral_speed_scale,
            "front_swing_inner_width": -np.sum(
                np.square(front_inner / self.cfg.front_swing_width_tolerance) * front_swinging,
                axis=1,
            ) / 2.0 * abs(self.cfg.front_swing_width_scale),
            "front_stance_width": stance_width * self.cfg.front_stance_width_scale,
        }
        terms = {name: value.astype(np.float32, copy=False) for name, value in terms.items()}
        state.reward_terms.update(terms)
        return reward + sum(terms.values()) * self.cfg.ctrl_dt


registry.env("go2-walk-flat-f2-p1b")(Go2F2P1BTask)


__all__ = ["Go2WalkF2P1BEnvCfg", "Go2F2P1BTask"]
