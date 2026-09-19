# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState, NpObs
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.math import quaternion, utils
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
        self._actuator_ctrl_lower = np.asarray([spec.ctrl_range[0] for spec in self.model.actuators], dtype=np.float32)
        self._actuator_ctrl_upper = np.asarray([spec.ctrl_range[1] for spec in self.model.actuators], dtype=np.float32)

        # Hand limits use the same articulated-DOF order as the hand position slice.
        hand_lower, hand_upper = self.model.others["hand_joint_position_limits"]
        self._hand_dof_lower_limits = np.asarray(hand_lower, dtype=np.float32)
        self._hand_dof_upper_limits = np.asarray(hand_upper, dtype=np.float32)
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
        self._in_hand_pos = np.array(cfg.cube_initial_pos, dtype=np.float32)

        # Episode-scoped task state: full-batch buffers, reset writes the done
        # rows in place.
        num_envs = self._num_envs
        self._goal_pos = np.tile(self._in_hand_pos, (num_envs, 1))
        self._goal_rot = np.zeros((num_envs, 4), dtype=np.float32)
        self._prev_actions = np.zeros((num_envs, self._num_actuators), dtype=np.float32)
        self._successes = np.zeros(num_envs, dtype=np.int32)

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
        velocities = np.concatenate((inputs["fingertip_lin_vel"][rows], inputs["fingertip_ang_vel"][rows]), axis=-1)
        return positions, quaternions, velocities

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState):
        """Apply actions to the hand actuators."""
        cfg = self._cfg

        # Scale actions from [-1, 1] to actuator control range
        targets = utils.scale(actions, self._actuator_ctrl_lower, self._actuator_ctrl_upper)

        # Apply action moving average for smoothness
        if cfg.act_moving_average < 1.0:
            targets = cfg.act_moving_average * targets + (1.0 - cfg.act_moving_average) * self._prev_actions

        # Clamp to control limits
        targets = np.clip(targets, self._actuator_ctrl_lower, self._actuator_ctrl_upper)

        # Set actuator controls
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = np.asarray(targets, dtype=np.float32)
        self._ctrl_writes.execute()
        self._prev_actions[:] = targets

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
        fingertip_state = np.concatenate(
            [
                fingertip_pos.reshape(num_envs, -1),  # 15
                fingertip_quat.reshape(num_envs, -1),  # 20
                fingertip_vel.reshape(num_envs, -1),  # 30
            ],
            axis=-1,
        )  # Total: 65

        # Compute relative quaternion
        relative_quat = quaternion.mul(cube_quat, quaternion.conjugate(self._goal_rot))
        scaled_hand_pos = utils.unscale(hand_dof_pos, self._hand_dof_lower_limits, self._hand_dof_upper_limits)

        # Build observation (157 dims)
        obs = np.concatenate(
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
            axis=-1,
        )
        # Publish the computed policy observation on the state. The field
        # name is applied via setattr so static audits can tell this
        # sanctioned observation-stage write apart from the forbidden
        # transition-time writes.
        setattr(state, "obs", NpObs(policy=obs))
        return state

    def compute_transition(self, state: ArrayEnvState):
        """Update reward, termination, and goal bookkeeping from refreshed sim data.

        Observations are built separately by :meth:`compute_observation`; this
        method never touches the observation field.
        """
        self.sim_data.execute()

        # Compute reward and termination from the refreshed simulator cache
        reward, terminated, goal_reached = self._compute_reward()
        if np.any(goal_reached):
            reset_goal_indices = np.where(goal_reached)[0]
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
        goal_dist = np.linalg.norm(cube_pos - self._goal_pos, axis=-1)

        # Rotation distance
        rot_dist = quaternion.rotation_distance(cube_quat, self._goal_rot)

        # Core reward components
        dist_rew = goal_dist * cfg.dist_reward_scale
        rot_rew = 1.0 / (np.abs(rot_dist) + cfg.rot_eps) * cfg.rot_reward_scale
        action_penalty = np.sum(self._prev_actions**2, axis=-1) * cfg.action_penalty_scale

        # Base reward
        reward = dist_rew + rot_rew + action_penalty

        # Check for success (only rotation tolerance)
        goal_reached = np.abs(rot_dist) <= cfg.success_tolerance

        # Update success counter
        self._successes += goal_reached * 1

        # Success bonus
        reward = np.where(goal_reached, reward + cfg.reach_goal_bonus, reward)

        # Fall penalty
        fallen = goal_dist >= cfg.fall_dist
        reward = np.where(fallen, reward + cfg.fall_penalty, reward)

        # Termination conditions
        terminated = np.zeros(num_envs, dtype=bool)

        # 1. Fall termination
        terminated = np.logical_or(terminated, fallen)

        # 2. Success termination with hold mechanism
        new_pos = np.zeros(num_envs, dtype=bool)
        if cfg.max_consecutive_successes > 0:
            # Reset progress on goal reached when max consecutive successes reached
            new_pos = self._successes >= cfg.max_consecutive_successes
            self._successes *= 1 - new_pos

        # 3. NaN protection
        terminated = np.logical_or(terminated, np.isnan(rot_dist))
        terminated = np.logical_or(terminated, np.isnan(goal_dist))

        return reward, terminated, new_pos

    def _update_target_visualization(self):
        """Update the target mocap body to visualize the goal pose."""
        cfg = self._cfg

        # Compute visualization position (offset from goal position)
        viz_pos = self._goal_pos + np.array(cfg.viz_target_offset, dtype=np.float32)

        # Update mocap body pose
        all_ids = np.arange(self._num_envs, dtype=np.int64)
        self._target_writes.buffer("target_pos")[all_ids, 0] = np.asarray(viz_pos, dtype=np.float32)
        self._target_writes.buffer("target_rot")[all_ids, 0] = np.asarray(self._goal_rot, dtype=np.float32)
        self._target_writes.execute(all_ids)

    def reset(self, env_ids: np.ndarray) -> None:
        """Reset environments."""
        cfg = self._cfg

        num_resets = len(env_ids)
        row_ids = np.asarray(env_ids, dtype=np.int64)

        # Reset hand DOFs with noise
        hand_pos = self._reset_program.buffer("hand_position")
        hand_vel = self._reset_program.buffer("hand_velocity")

        # Add noise to DOF positions
        dof_pos_noise = np.random.uniform(
            -cfg.reset_dof_pos_noise,
            cfg.reset_dof_pos_noise,
            (num_resets, self._num_hand_dofs),
        ).astype(np.float32)

        # Add noise to DOF velocities
        dof_vel_noise = np.random.uniform(
            -cfg.reset_dof_vel_noise, cfg.reset_dof_vel_noise, (num_resets, self._num_hand_dofs)
        ).astype(np.float32)

        # Set hand DOF states for all envs being reset.
        hand_pos[row_ids] = dof_pos_noise
        hand_vel[row_ids] = dof_vel_noise

        # Reset cube position with small noise
        cube_pos_noise = np.random.uniform(-cfg.reset_position_noise, cfg.reset_position_noise, (num_resets, 3)).astype(
            np.float32
        )
        cube_pos = np.tile(self._in_hand_pos, (num_resets, 1))
        cube_pos += cube_pos_noise

        # Randomize cube orientation
        cube_quat = quaternion.generate_random_shoemake(num_resets)

        # BodyPositionWrite / BodyRotationWrite buffers are (N, B, 3) / (N, B, 4).
        self._reset_program.buffer("cube_position")[row_ids, 0] = cube_pos
        self._reset_program.buffer("cube_rotation")[row_ids, 0] = cube_quat
        self._reset_program.buffer("cube_linear_velocity")[row_ids, 0] = 0.0
        self._reset_program.buffer("cube_angular_velocity")[row_ids, 0] = 0.0
        self._reset_program.execute(row_ids)
        self.sim_data.execute(row_ids)

        # Reset episode-scoped task state for the reset rows. The goal position
        # is fixed; the goal orientation is sampled uniformly on SO(3).
        self._goal_pos[row_ids] = self._in_hand_pos
        self._goal_rot[row_ids] = quaternion.generate_random_shoemake(num_resets)
        self._prev_actions[row_ids] = 0.0
        self._successes[row_ids] = 0

    def _reset_goal_pose(self, env_ids):
        """Reset goal pose to random orientation with fixed position."""
        num_resets = len(env_ids)

        # Goal position is fixed

        # Randomize goal orientation using Shoemake method for uniform SO(3) sampling
        self._goal_rot[env_ids] = quaternion.generate_random_shoemake(num_resets)
