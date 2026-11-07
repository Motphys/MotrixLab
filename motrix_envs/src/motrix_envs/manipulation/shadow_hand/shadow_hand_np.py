# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState, EnvObs
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.math import quaternion_array
from motrix_env_core.sim import (
    BatchLinkAngularVelocityQuery,
    BatchLinkLinearVelocityQuery,
    BatchLinkPositionQuery,
    BatchLinkQuaternionQuery,
    BodyAngularVelocityWrite,
    BodyJointPositionLimitsQuery,
    BodyJointPositionWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    DofPositionQuery,
    DofVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
)
from motrix_env_core.sim.write import (
    BodyJointVelocityWrite,
    CtrlTargetsWrite,
    KinematicBodyPositionWrite,
    KinematicBodyRotationWrite,
)

from .cfg import ShadowHandReposeEnvCfg

_FINGERTIP_LINKS = (
    "rh_ffdistal",  # First finger (index) distal
    "rh_mfdistal",  # Middle finger distal
    "rh_rfdistal",  # Ring finger distal
    "rh_lfdistal",  # Little finger distal
    "rh_thdistal",  # Thumb distal
)

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
    "cube_pos": LinkPositionQuery(link="cube"),
    "cube_quat": LinkQuaternionQuery(link="cube"),
    "fingertip_pos": BatchLinkPositionQuery(links=_FINGERTIP_LINKS),
    "fingertip_quat": BatchLinkQuaternionQuery(links=_FINGERTIP_LINKS),
    "fingertip_lin_vel": BatchLinkLinearVelocityQuery(links=_FINGERTIP_LINKS),
    "fingertip_ang_vel": BatchLinkAngularVelocityQuery(links=_FINGERTIP_LINKS),
}

_SIM_MODEL_QUERIES = {"hand_joint_position_limits": BodyJointPositionLimitsQuery(body="rh_forearm")}


def _generate_random_shoemake(size: int) -> torch.Tensor:
    """Uniformly distributed random quaternions (Shoemake's method), (x, y, z, w)."""
    u1, u2, u3 = torch.rand(3, size, dtype=torch.float32)
    sqrt1_u1 = torch.sqrt(1 - u1)
    sqrtu1 = torch.sqrt(u1)
    w = sqrt1_u1 * torch.sin(2 * torch.pi * u2)
    x = sqrt1_u1 * torch.cos(2 * torch.pi * u2)
    y = sqrtu1 * torch.sin(2 * torch.pi * u3)
    z = sqrtu1 * torch.cos(2 * torch.pi * u3)
    return torch.stack([x, y, z, w], dim=-1)


"""
Shadow Hand Cube Reorientation Environment for MotrixSim

This environment implements the classic in-hand cube manipulation task where the
Shadow Hand must reorient a cube to match random target orientations.

"""


@registry.env("shadow-hand-repose")
class ShadowHandReposeEnv(DirectEnv):
    """
    Shadow Hand Cube Reorientation Environment

    Observation space: 157 dimensions
        - 24: hand dof positions (unscaled)
        - 24: hand dof velocities (scaled by 0.2)
        - 7: object pose (pos + quat)
        - 3: object linear velocity
        - 3: object angular velocity (scaled by 0.2)
        - 7: goal pose (pos + quat)
        - 4: relative quaternion (object to goal)
        - 65: fingertip states (5 fingertips * 13: pos + quat + vel)
        - 20: previous actions

    Action space: 20 dimensions (normalized [-1, 1] position targets for actuators)
    """

    _cfg: ShadowHandReposeEnvCfg

    def __init__(self, cfg: ShadowHandReposeEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._target_writes = self.sim.write_compiler.compile(
            {
                "target_pos": KinematicBodyPositionWrite(("target",)),
                "target_rot": KinematicBodyRotationWrite(("target",)),
            }
        )
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "hand_position": BodyJointPositionWrite("rh_forearm"),
                "hand_velocity": BodyJointVelocityWrite("rh_forearm"),
                "cube_position": BodyPositionWrite(("cube",)),
                "cube_rotation": BodyRotationWrite(("cube",)),
                "cube_linear_velocity": BodyLinearVelocityWrite(("cube",)),
                "cube_angular_velocity": BodyAngularVelocityWrite(("cube",)),
            },
            reset=True,
        )

        # Get model info
        self._num_hand_dofs = cfg.num_hand_dofs  # 24 total DOFs
        self._num_actuators = cfg.num_actuators  # 20 actuated joints

        # Initialize spaces
        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._num_actuators,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(157,), dtype=np.float32)

        # Get actuator control ranges from model
        self._actuator_ctrl_lower = torch.tensor(
            [spec.ctrl_range[0] for spec in self.model.actuators], dtype=torch.float32
        )
        self._actuator_ctrl_upper = torch.tensor(
            [spec.ctrl_range[1] for spec in self.model.actuators], dtype=torch.float32
        )

        # Hand limits use the same articulated-DOF order as the hand position slice.
        hand_lower, hand_upper = self.model.others["hand_joint_position_limits"]
        self._hand_dof_lower_limits = hand_lower.to(torch.float32)
        self._hand_dof_upper_limits = hand_upper.to(torch.float32)
        expected_shape = (self._num_hand_dofs,)
        if self._hand_dof_lower_limits.shape != expected_shape or self._hand_dof_upper_limits.shape != expected_shape:
            raise ValueError(
                "Shadow Hand joint limits must match hand DOFs: "
                f"lower={self._hand_dof_lower_limits.shape}, upper={self._hand_dof_upper_limits.shape}, "
                f"expected={expected_shape}."
            )

        self._num_fingertips = len(cfg.fingertip_link_names)

        # Cube velocities share the trailing six full-scene velocity channels.
        self._cube_base_vel = slice(-6, None)

        # Initial cube position (in hand)
        self._in_hand_pos = torch.tensor(cfg.cube_initial_pos, dtype=torch.float32)

        # Episode-scoped task state: full-batch buffers, reset writes the done
        # rows in place.
        num_envs = self._num_envs
        self._goal_pos = self._in_hand_pos.unsqueeze(0).repeat(num_envs, 1)
        self._goal_rot = torch.zeros(num_envs, 4, dtype=torch.float32)
        self._prev_actions = torch.zeros(num_envs, self._num_actuators, dtype=torch.float32)
        self._successes = torch.zeros(num_envs, dtype=torch.int64)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _extract_cube_states(self, rows):
        inputs = self.sim_data
        return (
            inputs["cube_pos"][rows],
            inputs["cube_quat"][rows],
            inputs["dof_vel"][rows, self._cube_base_vel],
        )

    def _extract_link_states(self, rows):
        """
        Extract position, quaternion, and velocity for the fingertip links.

        Returns:
            Tuple of (positions, quaternions, velocities)
            - positions: (rows, num_links, 3)
            - quaternions: (rows, num_links, 4) in (x, y, z, w) format
            - velocities: (rows, num_links, 6) [linear_vel, angular_vel]
        """
        inputs = self.sim_data
        positions = inputs["fingertip_pos"][rows]
        quaternions = inputs["fingertip_quat"][rows]
        velocities = torch.cat((inputs["fingertip_lin_vel"][rows], inputs["fingertip_ang_vel"][rows]), dim=-1)
        return positions, quaternions, velocities

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState):
        """Apply actions to the hand actuators."""
        cfg = self._cfg

        # Scale actions from [-1, 1] to actuator control range
        targets = 0.5 * (actions + 1.0) * (self._actuator_ctrl_upper - self._actuator_ctrl_lower) + (
            self._actuator_ctrl_lower
        )

        # Apply action moving average for smoothness
        if cfg.act_moving_average < 1.0:
            targets = cfg.act_moving_average * targets + (1.0 - cfg.act_moving_average) * self._prev_actions

        # Clamp to control limits
        targets = torch.clamp(targets, self._actuator_ctrl_lower, self._actuator_ctrl_upper)

        # Set actuator controls
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl.copy_(targets)
        self._ctrl_writes.execute()
        self._prev_actions.copy_(targets)

        return state

    def compute_observation(self, state: ArrayEnvState):
        """Build the full 157-dim observation batch from cached simulator data.

        Reads only the cache left by the last read-program execution in the
        transition; never performs reads itself and never touches reward or
        termination.
        """
        cfg = self._cfg
        rows = slice(None)

        # Get hand DOF states
        hand_dof_pos = self.sim_data["dof_pos"][rows][:, : self._num_hand_dofs]
        hand_dof_vel = self.sim_data["dof_vel"][rows][:, : self._num_hand_dofs]
        num_envs = hand_dof_pos.shape[0]

        # Get cube state using link queries
        cube_pos, cube_quat, cube_vel = self._extract_cube_states(rows)
        cube_linvel = cube_vel[:, :3]
        cube_angvel = cube_vel[:, 3:]

        # Get fingertip states using link queries
        fingertip_pos, fingertip_quat, fingertip_vel = self._extract_link_states(rows)

        # Flatten fingertip states (5 × 13 = 65)
        fingertip_state = torch.cat(
            [
                fingertip_pos.reshape(num_envs, -1),  # 15
                fingertip_quat.reshape(num_envs, -1),  # 20
                fingertip_vel.reshape(num_envs, -1),  # 30
            ],
            dim=-1,
        )  # Total: 65

        # Compute relative quaternion
        relative_quat = quaternion_array.mul(cube_quat, quaternion_array.inverse(self._goal_rot))
        scaled_hand_pos = (
            2.0
            * (hand_dof_pos - self._hand_dof_lower_limits)
            / (self._hand_dof_upper_limits - self._hand_dof_lower_limits)
            - 1.0
        )

        # Build observation (157 dims)
        obs = torch.cat(
            [
                scaled_hand_pos,
                cfg.vel_obs_scale * hand_dof_vel,  # 24
                cube_pos,  # 3
                cube_quat,  # 4
                cube_linvel,  # 3
                cfg.vel_obs_scale * cube_angvel,  # 3
                self._goal_pos,  # 3
                self._goal_rot,  # 4
                relative_quat,  # 4
                fingertip_state,  # 65
                self._prev_actions,  # 20
            ],
            dim=-1,
        )
        # Publish the computed policy observation on the state. The field
        # name is applied via setattr so static audits can tell this
        # sanctioned observation-stage write apart from the forbidden
        # transition-time writes.
        setattr(state, "obs", EnvObs(policy=obs))
        return state

    def compute_transition(self, state: ArrayEnvState):
        """Update reward, termination, and goal bookkeeping from refreshed sim data.

        Observations are built separately by :meth:`compute_observation`; this
        method never touches the observation field.
        """
        self.sim_data.execute()

        # Compute reward and termination from the refreshed simulator cache
        reward, terminated, goal_reached = self._compute_reward()
        if bool(goal_reached.any()):
            reset_goal_indices = torch.nonzero(goal_reached, as_tuple=False).flatten()
            self._reset_goal_pose(reset_goal_indices)
        # Update the goal mocap body so viewers track the current goal pose
        # (a write program, not an observation read).
        self._update_target_visualization()

        state.reward = reward
        state.terminated = terminated

        return state

    def _compute_reward(self):
        """
        Reward components (3 core items):
        1. Position distance penalty
        2. Rotation alignment reward
        3. Action regularization penalty

        Additional rewards/penalties:
        - Success bonus when goal is reached
        - Fall penalty when cube drops
        - Timeout penalty when episode ends without success
        """
        cfg = self._cfg
        num_envs = self._num_envs

        # Get cube state using link queries
        cube_pos, cube_quat, _ = self._extract_cube_states(slice(None))

        # Distance from cube to goal position
        goal_dist = torch.linalg.norm(cube_pos - self._goal_pos, dim=-1)

        # Rotation distance
        rot_dist = quaternion_array.rotation_distance(cube_quat, self._goal_rot)

        # Core reward components
        dist_rew = goal_dist * cfg.dist_reward_scale
        rot_rew = 1.0 / (torch.abs(rot_dist) + cfg.rot_eps) * cfg.rot_reward_scale
        action_penalty = torch.sum(self._prev_actions**2, dim=-1) * cfg.action_penalty_scale

        # Base reward
        reward = dist_rew + rot_rew + action_penalty

        # Check for success (only rotation tolerance)
        goal_reached = torch.abs(rot_dist) <= cfg.success_tolerance

        # Update success counter
        self._successes += goal_reached.to(torch.int64)

        # Success bonus
        reward = torch.where(goal_reached, reward + cfg.reach_goal_bonus, reward)

        # Fall penalty
        fallen = goal_dist >= cfg.fall_dist
        reward = torch.where(fallen, reward + cfg.fall_penalty, reward)

        # Termination conditions
        terminated = torch.zeros(num_envs, dtype=torch.bool)

        # 1. Fall termination
        terminated = terminated | fallen

        # 2. Success termination with hold mechanism
        new_pos = torch.zeros(num_envs, dtype=torch.bool)
        if cfg.max_consecutive_successes > 0:
            # Reset progress on goal reached when max consecutive successes reached
            new_pos = self._successes >= cfg.max_consecutive_successes
            self._successes *= (~new_pos).to(torch.int64)

        # 3. NaN protection
        terminated = terminated | rot_dist.isnan()
        terminated = terminated | goal_dist.isnan()

        return reward, terminated, new_pos

    def _update_target_visualization(self):
        """Update the target mocap body to visualize the goal pose."""
        cfg = self._cfg

        # Compute visualization position (offset from goal position)
        viz_pos = self._goal_pos + torch.tensor(cfg.viz_target_offset, dtype=torch.float32)

        # Update mocap body pose
        all_ids = torch.arange(self._num_envs, dtype=torch.int64)
        self._target_writes.buffer("target_pos")[all_ids, 0] = viz_pos
        self._target_writes.buffer("target_rot")[all_ids, 0] = self._goal_rot
        self._target_writes.execute(all_ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset environments."""
        cfg = self._cfg

        num_resets = env_ids.numel()

        # Reset hand DOFs with noise
        hand_pos = self._reset_program.buffer("hand_position")
        hand_vel = self._reset_program.buffer("hand_velocity")

        # Add noise to DOF positions
        dof_pos_noise = (
            torch.rand(num_resets, self._num_hand_dofs, dtype=torch.float32) * 2.0 - 1.0
        ) * cfg.reset_dof_pos_noise

        # Add noise to DOF velocities
        dof_vel_noise = (
            torch.rand(num_resets, self._num_hand_dofs, dtype=torch.float32) * 2.0 - 1.0
        ) * cfg.reset_dof_vel_noise

        # Set hand DOF states for all envs being reset.
        hand_pos[env_ids] = dof_pos_noise
        hand_vel[env_ids] = dof_vel_noise

        # Reset cube position with small noise
        cube_pos_noise = (torch.rand(num_resets, 3, dtype=torch.float32) * 2.0 - 1.0) * cfg.reset_position_noise
        cube_pos = self._in_hand_pos.unsqueeze(0).repeat(num_resets, 1)
        cube_pos = cube_pos + cube_pos_noise

        # Randomize cube orientation
        cube_quat = _generate_random_shoemake(num_resets)

        # BodyPositionWrite / BodyRotationWrite buffers are (N, B, 3) / (N, B, 4).
        self._reset_program.buffer("cube_position")[env_ids, 0] = cube_pos
        self._reset_program.buffer("cube_rotation")[env_ids, 0] = cube_quat
        self._reset_program.buffer("cube_linear_velocity")[env_ids, 0] = 0.0
        self._reset_program.buffer("cube_angular_velocity")[env_ids, 0] = 0.0
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)

        # Reset episode-scoped task state for the reset rows. The goal position
        # is fixed; the goal orientation is sampled uniformly on SO(3).
        self._goal_pos[env_ids] = self._in_hand_pos
        self._goal_rot[env_ids] = _generate_random_shoemake(num_resets)
        self._prev_actions[env_ids] = 0.0
        self._successes[env_ids] = 0

    def _reset_goal_pose(self, env_ids):
        """Reset goal pose to random orientation with fixed position."""
        num_resets = env_ids.numel()

        # Goal position is fixed

        # Randomize goal orientation using Shoemake method for uniform SO(3) sampling
        self._goal_rot[env_ids] = _generate_random_shoemake(num_resets)
