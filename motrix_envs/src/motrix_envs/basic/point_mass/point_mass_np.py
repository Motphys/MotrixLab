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
        self._in_target_steps = torch.zeros(self._num_envs, dtype=torch.int64)
        self._required_in_target_steps = int(0.5 / cfg.ctrl_dt)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState):
        actions = torch.clamp(actions, -1.0, 1.0)
        self._ctrl_writes.buffer("ctrl").copy_(actions)
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
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True)

        obs = torch.cat([dof_pos, dof_vel, target_pos, delta, distance], dim=-1)
        return state.replace(obs=obs)

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        inputs = self.sim_data
        # Get positions of point mass and target
        point_pos = inputs["point_mass_pos"][:, :2]
        target_pos = inputs["target_pos"][:, :2]
        dist_to_target = torch.linalg.norm(point_pos - target_pos, dim=-1)

        # Calculate effective target radius for complete overlap
        # Blue ball radius is 0.05, red ball radius is 0.1
        # For complete overlap, center distance should be very small
        effective_target_radius = 0.02  # Smaller radius for complete overlap

        # Fine-grained distance reward - exponential function, reward grows faster as distance decreases
        distance_reward = torch.exp(-10 * dist_to_target)  # Stronger exponential reward

        # Large bonus for complete target entry
        in_target = dist_to_target < effective_target_radius
        target_bonus = 100.0 * in_target.float()  # Significantly increased reward

        # Continuous stay reward
        continuous_reward = 30.0 * in_target.float()  # Increased continuous reward

        # Penalty for distance from target center - encourages complete overlap
        # When inside target, penalty increases with distance from center
        center_penalty = torch.where(in_target, 10.0 * dist_to_target, torch.zeros_like(dist_to_target))

        # Control penalty - increased penalty to encourage smoother movement
        dof_vel = inputs["dof_vel"][:, :2]
        vel_magnitude = torch.linalg.norm(dof_vel, dim=-1)
        control_penalty = 0.1 * vel_magnitude  # Increased penalty to reduce excessive movement

        # Path optimization reward - encourages straight-line movement
        # Calculate alignment between velocity direction and target direction
        if bool(dist_to_target.max() > 0):
            delta = target_pos - point_pos
            delta_norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
            delta_normalized = delta / delta_norm
            vel_normalized = dof_vel / (torch.linalg.norm(dof_vel, dim=-1, keepdim=True) + 1e-6)
            direction_alignment = torch.sum(delta_normalized * vel_normalized, dim=-1)
            path_reward = 0.5 * direction_alignment
        else:
            path_reward = 0.0

        # Total reward
        rwd = distance_reward + target_bonus + continuous_reward + path_reward - center_penalty - control_penalty

        # Update target stay steps
        self._in_target_steps = torch.where(
            in_target, self._in_target_steps + 1, torch.zeros_like(self._in_target_steps)
        )

        # Check if stayed in target long enough
        in_target_long_enough = self._in_target_steps >= self._required_in_target_steps

        # Check termination conditions - terminate when reaching target for 0.5 seconds or when NaN encountered
        terminated = in_target_long_enough.clone()
        terminated |= inputs["dof_pos"].isnan().any(dim=-1)
        terminated |= rwd.isnan()

        state.reward[:] = rwd
        state.terminated[:] = terminated
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        num_reset = env_ids.numel()

        # Random initial position within a range for the point mass (only x, y)
        x_pos = (torch.rand(num_reset, dtype=torch.float32) * 2.0 - 1.0) * 1.0
        y_pos = (torch.rand(num_reset, dtype=torch.float32) * 2.0 - 1.0) * 1.0

        point_position = torch.stack([x_pos, y_pos], dim=-1)
        point_velocity = torch.zeros((num_reset, 2), dtype=torch.float32)

        # Randomize target position using its slide joints
        target_x = (torch.rand(num_reset, dtype=torch.float32) * 2.0 - 1.0) * 1.5
        target_y = (torch.rand(num_reset, dtype=torch.float32) * 2.0 - 1.0) * 1.5

        target_position = torch.stack([target_x, target_y], dim=-1)
        target_velocity = torch.zeros((num_reset, 2), dtype=torch.float32)

        self._point_position[env_ids] = point_position
        self._point_velocity[env_ids] = point_velocity
        self._target_position[env_ids] = target_position
        self._target_velocity[env_ids] = target_velocity
        self._reset_program.execute(env_ids)

        # Reset target stay counter for the environments being reset
        self._in_target_steps[env_ids] = 0

        self.sim_data.execute(env_ids)
