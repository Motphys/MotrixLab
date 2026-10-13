# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Isolated Go2 rear-leg swing-width ablation."""

import numpy as np

from motrix_env_core import registry
from motrix_env_core.config import configclass
from motrix_envs.locomotion.quadruped.go2_legacy_pose import Go2WalkLegacyPoseEnvCfg
from motrix_envs.locomotion.quadruped.walk_np import QuadrupedWalkTask


@configclass
class Go2WalkSwingEnvCfg(Go2WalkLegacyPoseEnvCfg):
    """Experiment D plus rear-leg swing lateral regularization."""

    rear_swing_width_target: float = 0.145
    rear_swing_width_scale: float = -0.05
    rear_swing_lateral_speed_scale: float = -0.01
    rear_swing_width_tolerance: float = 0.02
    rear_swing_lateral_speed_tolerance: float = 0.15
    normalize_swing_penalties: bool = False


@registry.envcfg("go2-walk-flat-d-swing")
class _RegisteredGo2WalkSwingEnvCfg(Go2WalkSwingEnvCfg):
    pass


class Go2SwingAblationTask(QuadrupedWalkTask):
    """Add swing-only rear-foot lateral penalties to the shared walk task."""

    def _compute_reward(self, state, linvel, gyro, dof_pos):
        previous_feet_pos = self._previous_feet_pos.copy()
        swinging = self._feet_phase[:, 2:4] >= 0.6
        rear_abs_y = np.abs(self.feet_pos[:, 2:4, 1])
        width_error = np.square(rear_abs_y - self.cfg.rear_swing_width_target)
        foot_y_velocity = (self.feet_pos[:, 2:4, 1] - previous_feet_pos[:, 2:4, 1]) / self.cfg.ctrl_dt
        if self.cfg.normalize_swing_penalties:
            width_error = width_error / (self.cfg.rear_swing_width_tolerance**2)
            foot_y_velocity = foot_y_velocity / self.cfg.rear_swing_lateral_speed_tolerance
        width_penalty = np.sum(width_error * swinging, axis=1) / 2.0
        speed_penalty = np.sum(np.square(foot_y_velocity) * swinging, axis=1) / 2.0

        reward = super()._compute_reward(state, linvel, gyro, dof_pos)
        weighted_width = (width_penalty * self.cfg.rear_swing_width_scale).astype(np.float32, copy=False)
        weighted_speed = (speed_penalty * self.cfg.rear_swing_lateral_speed_scale).astype(np.float32, copy=False)
        state.reward_terms["rear_swing_width"] = weighted_width
        state.reward_terms["rear_swing_lateral_speed"] = weighted_speed
        return reward + (weighted_width + weighted_speed) * self.cfg.ctrl_dt


registry.env("go2-walk-flat-d-swing")(Go2SwingAblationTask)


@registry.envcfg("go2-walk-flat-d-swing-v2")
@configclass
class Go2WalkSwingV2EnvCfg(Go2WalkSwingEnvCfg):
    """D-swing with normalized rear swing penalties."""

    normalize_swing_penalties: bool = True


registry.env("go2-walk-flat-d-swing-v2")(Go2SwingAblationTask)


@registry.envcfg("go2-walk-flat-d-swing-v3")
@configclass
class Go2WalkSwingV3EnvCfg(Go2WalkSwingV2EnvCfg):
    """D-swing-v2 plus a weak front-foot over-extension penalty.

    The front limit is a one-sided hinge: normal front-foot width is free and
    only excessive lateral excursion is penalized.  This avoids pulling the
    front feet toward an arbitrary fixed target.
    """

    front_swing_width_limit: float = 0.16
    front_swing_width_tolerance: float = 0.02
    front_swing_width_scale: float = -0.02
    front_swing_lateral_speed_tolerance: float = 0.15
    front_swing_lateral_speed_scale: float = -0.002


class Go2SwingFrontRegularizedTask(Go2SwingAblationTask):
    """D-swing-v2 with weak, over-limit-only front swing regularization."""

    def _compute_reward(self, state, linvel, gyro, dof_pos):
        previous_feet_pos = self._previous_feet_pos.copy()
        swinging = self._feet_phase[:, :2] >= 0.6
        front_abs_y = np.abs(self.feet_pos[:, :2, 1])
        excess = np.maximum(front_abs_y - self.cfg.front_swing_width_limit, 0.0)
        width_penalty = (
            np.sum(
                np.square(excess / self.cfg.front_swing_width_tolerance) * swinging,
                axis=1,
            )
            / 2.0
        )
        front_y_velocity = (self.feet_pos[:, :2, 1] - previous_feet_pos[:, :2, 1]) / self.cfg.ctrl_dt
        # Only damp lateral speed while the foot is already outside the safe
        # lateral envelope; normal front swing remains unconstrained.
        speed_penalty = (
            np.sum(
                np.square(front_y_velocity / self.cfg.front_swing_lateral_speed_tolerance) * swinging * (excess > 0.0),
                axis=1,
            )
            / 2.0
        )

        reward = super()._compute_reward(state, linvel, gyro, dof_pos)
        weighted_width = (width_penalty * self.cfg.front_swing_width_scale).astype(np.float32, copy=False)
        weighted_speed = (speed_penalty * self.cfg.front_swing_lateral_speed_scale).astype(np.float32, copy=False)
        state.reward_terms["front_swing_width"] = weighted_width
        state.reward_terms["front_swing_lateral_speed"] = weighted_speed
        return reward + (weighted_width + weighted_speed) * self.cfg.ctrl_dt


registry.env("go2-walk-flat-d-swing-v3")(Go2SwingFrontRegularizedTask)


__all__ = [
    "Go2WalkSwingEnvCfg",
    "Go2WalkSwingV2EnvCfg",
    "Go2WalkSwingV3EnvCfg",
    "Go2SwingAblationTask",
    "Go2SwingFrontRegularizedTask",
]
