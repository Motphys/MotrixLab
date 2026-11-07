# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.sim import (
    DofPositionQuery,
    DofVelocityQuery,
    JointPositionWrite,
    LinkPositionQuery,
)
from motrix_env_core.sim.write import CtrlTargetsWrite, JointVelocityWrite

from .cfg import PointMassEnvCfg

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
    "point_mass_pos": LinkPositionQuery(link="point_mass"),
    "target_pos": LinkPositionQuery(link="target"),
}
_SIM_MODEL_QUERIES = {}


@registry.env("point_mass")
class PointMassEnv(DirectEnv):
    _cfg: PointMassEnvCfg

    def __init__(self, cfg: PointMassEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "point_mass_position": JointPositionWrite(("point_mass_x", "point_mass_y")),
                "point_mass_velocity": JointVelocityWrite(("point_mass_x", "point_mass_y")),
                "target_position": JointPositionWrite(("target_x", "target_y")),
                "target_velocity": JointVelocityWrite(("target_x", "target_y")),
            },
            reset=True,
        )
        self._point_position = self._reset_program.buffer("point_mass_position")
        self._point_velocity = self._reset_program.buffer("point_mass_velocity")
        self._target_position = self._reset_program.buffer("target_position")
        self._target_velocity = self._reset_program.buffer("target_velocity")
        self._action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)
        self._target_radius = cfg.target_radius

        # Target stay counter, used to control reset after 0.5 seconds of overlap
        self._in_target_steps = np.zeros(self._num_envs, dtype=np.int32)
        self._required_in_target_steps = int(0.5 / cfg.ctrl_dt)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState):
        actions = actions.detach().cpu().numpy()
        actions = np.clip(actions, -1.0, 1.0)
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = np.asarray(actions, dtype=np.float32)
        self._ctrl_writes.execute()
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        dof_pos = inputs["dof_pos"][:, :2]  # Only x and y positions
        dof_vel = inputs["dof_vel"][:, :2]  # Only x and y velocities

        # Get target position
        target_pos = inputs["target_pos"][:, :2]

        # Calculate distance and direction to target
        delta = target_pos - dof_pos
        distance = np.linalg.norm(delta, axis=-1, keepdims=True)

        obs = np.concatenate([dof_pos, dof_vel, target_pos, delta, distance], axis=-1)
        return state.replace(obs=obs)

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        inputs = self.sim_data
        # Get positions of point mass and target
        point_pos = inputs["point_mass_pos"][:, :2]
        target_pos = inputs["target_pos"][:, :2]
        dist_to_target = np.linalg.norm(point_pos - target_pos, axis=-1)

        # Calculate effective target radius for complete overlap
        # Blue ball radius is 0.05, red ball radius is 0.1
        # For complete overlap, center distance should be very small
        effective_target_radius = 0.02  # Smaller radius for complete overlap

        # Fine-grained distance reward - exponential function, reward grows faster as distance decreases
        distance_reward = np.exp(-10 * dist_to_target)  # Stronger exponential reward

        # Large bonus for complete target entry
        in_target = dist_to_target < effective_target_radius
        target_bonus = 100.0 * in_target  # Significantly increased reward

        # Continuous stay reward
        continuous_reward = 30.0 * in_target  # Increased continuous reward

        # Penalty for distance from target center - encourages complete overlap
        # When inside target, penalty increases with distance from center
        center_penalty = np.where(in_target, 10.0 * dist_to_target, 0.0)

        # Control penalty - increased penalty to encourage smoother movement
        dof_vel = inputs["dof_vel"][:, :2]
        vel_magnitude = np.linalg.norm(dof_vel, axis=-1)
        control_penalty = 0.1 * vel_magnitude  # Increased penalty to reduce excessive movement

        # Path optimization reward - encourages straight-line movement
        # Calculate alignment between velocity direction and target direction
        if dist_to_target.max() > 0:
            delta = target_pos - point_pos
            delta_norm = np.linalg.norm(delta, axis=-1, keepdims=True)
            delta_normalized = delta / delta_norm
            vel_normalized = dof_vel / (np.linalg.norm(dof_vel, axis=-1, keepdims=True) + 1e-6)
            direction_alignment = np.sum(delta_normalized * vel_normalized, axis=-1)
            path_reward = 0.5 * direction_alignment
        else:
            path_reward = 0.0

        # Total reward
        rwd = distance_reward + target_bonus + continuous_reward + path_reward - center_penalty - control_penalty

        # Update target stay steps
        self._in_target_steps = np.where(in_target, self._in_target_steps + 1, 0)

        # Check if stayed in target long enough
        in_target_long_enough = self._in_target_steps >= self._required_in_target_steps

        # Check termination conditions - terminate when reaching target for 0.5 seconds or when NaN encountered
        terminated = np.zeros((self._num_envs,), dtype=bool)
        terminated = np.logical_or(in_target_long_enough, terminated)
        terminated = np.logical_or(np.isnan(inputs["dof_pos"]).any(axis=-1), terminated)
        terminated = np.logical_or(np.isnan(rwd), terminated)

        state.reward = rwd
        state.terminated = terminated
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.detach().cpu().numpy()
        num_reset = len(env_ids)

        # Random initial position within a range for the point mass (only x, y)
        x_pos = np.random.uniform(-1.0, 1.0, size=num_reset).astype(np.float32)
        y_pos = np.random.uniform(-1.0, 1.0, size=num_reset).astype(np.float32)

        point_position = np.stack([x_pos, y_pos], axis=-1)
        point_velocity = np.zeros((num_reset, 2), dtype=np.float32)

        # Randomize target position using its slide joints
        target_x = np.random.uniform(-1.5, 1.5, size=num_reset).astype(np.float32)
        target_y = np.random.uniform(-1.5, 1.5, size=num_reset).astype(np.float32)

        target_position = np.stack([target_x, target_y], axis=-1)
        target_velocity = np.zeros((num_reset, 2), dtype=np.float32)

        self._point_position[env_ids] = point_position
        self._point_velocity[env_ids] = point_velocity
        self._target_position[env_ids] = target_position
        self._target_velocity[env_ids] = target_velocity
        self._reset_program.execute(env_ids)

        # Reset target stay counter for the environments being reset
        self._in_target_steps[env_ids] = 0

        self.sim_data.execute(np.asarray(env_ids, np.int64))
