# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.math import quaternion
from motrix_env_core.sim import (
    ActuatorCtrlQuery,
    BodyAngularVelocityWrite,
    BodyJointPositionQuery,
    BodyJointPositionWrite,
    BodyJointVelocityQuery,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    GeomPairCollidingQuery,
    GeomSpecsQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
    SensorValuesQuery,
)
from motrix_env_core.sim.write import (
    BodyJointVelocityWrite,
    CtrlTargetsWrite,
    KinematicBodyPositionWrite,
    KinematicBodyRotationWrite,
)

from .cfg import AnymalCEnvCfg


def _sim_data_queries(cfg: AnymalCEnvCfg):
    return {
        "robot_joint_pos": BodyJointPositionQuery(body=cfg.asset.body_name),
        "robot_joint_vel": BodyJointVelocityQuery(body=cfg.asset.body_name),
        "actuator_ctrls": ActuatorCtrlQuery(),
        "root_pos": LinkPositionQuery(link="base"),
        "root_quat": LinkQuaternionQuery(link="base"),
        "base_linvel": SensorValuesQuery(sensors=("base_linvel",)),
        "base_gyro": SensorValuesQuery(sensors=("base_gyro",)),
        "termination_colliding": GeomPairCollidingQuery(pairs=(("base", "ground"),)),
    }


def _sim_model_queries(cfg: AnymalCEnvCfg):
    return {
        "geoms": GeomSpecsQuery(
            names=(cfg.asset.ground_name, *cfg.asset.terminate_after_contacts_on, *cfg.asset.foot_names)
        )
    }


@registry.env("anymal_c_navigation_flat")
class AnymalCEnv(DirectEnv):
    _cfg: AnymalCEnvCfg

    def __init__(self, cfg: AnymalCEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_sim_model_queries(cfg))
        self.sim_data = self.sim.compile_reads(_sim_data_queries(cfg))
        self._heading_writes = self.sim.write_compiler.compile(
            {
                "robot_pos": KinematicBodyPositionWrite(("robot_heading_arrow",)),
                "robot_rot": KinematicBodyRotationWrite(("robot_heading_arrow",)),
                "desired_pos": KinematicBodyPositionWrite(("desired_heading_arrow",)),
                "desired_rot": KinematicBodyRotationWrite(("desired_heading_arrow",)),
            },
        )
        self._target_writes = self.sim.write_compiler.compile(
            {
                "target_pos": KinematicBodyPositionWrite(("target_marker",)),
                "target_rot": KinematicBodyRotationWrite(("target_marker",)),
            }
        )
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "base_position": BodyPositionWrite((cfg.asset.body_name,)),
                "base_rotation": BodyRotationWrite((cfg.asset.body_name,)),
                "base_linear_velocity": BodyLinearVelocityWrite((cfg.asset.body_name,)),
                "base_angular_velocity": BodyAngularVelocityWrite((cfg.asset.body_name,)),
                "joints_position": BodyJointPositionWrite(cfg.asset.body_name),
                "joints_velocity": BodyJointVelocityWrite(cfg.asset.body_name),
            },
            reset=True,
        )
        self._reset_position = self._reset_program.buffer("base_position")[:, 0]
        self._reset_rotation = self._reset_program.buffer("base_rotation")[:, 0]
        self._reset_linear_velocity = self._reset_program.buffer("base_linear_velocity")[:, 0]
        self._reset_angular_velocity = self._reset_program.buffer("base_angular_velocity")[:, 0]
        self._reset_joint_position = self._reset_program.buffer("joints_position")
        self._reset_joint_velocity = self._reset_program.buffer("joints_velocity")

        self._init_contact_geometry()

        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # Observation space: linvel(3) + gyro(3) + gravity(3) + joint_pos(12) + joint_vel(12) + last_actions(12) +
        # commands(3) + position_error(2) + heading_error(1) + distance(1) + reached_flag(1) + stop_ready_flag(1) = 54
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(54,), dtype=np.float32)
        self._num_action = self.num_actuators

        self._init_base_pose = self.model.init_dof_pos[:7].copy()

        self._init_buffer()

    def _init_buffer(self):
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        # PD parameters controlled by kp and kv in XML

        # Normalization coefficients
        self.commands_scale = np.array(
            [cfg.normalization.lin_vel, cfg.normalization.lin_vel, cfg.normalization.ang_vel], dtype=np.float32
        )

        # Set default joint angles
        actuator_names = [spec.name for spec in self.model.actuators]
        for i in range(self.num_actuators):
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in actuator_names[i]:
                    self.default_angles[i] = angle

        self._init_joint_position = self.default_angles.copy()

        # Episode-scoped navigation state: full-batch buffers, reset writes the
        # done rows in place.
        num_envs = self._num_envs
        self._pose_commands = np.zeros((num_envs, 3), dtype=np.float32)
        self._last_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)
        self._current_actions = np.zeros((num_envs, self._num_action), dtype=np.float32)
        self._ever_reached = np.zeros(num_envs, dtype=bool)
        self._min_distance = np.zeros(num_envs, dtype=np.float32)

    def _init_contact_geometry(self):
        """Initialize geometry name pairs required for contact detection"""
        cfg = self._cfg
        self.ground_name = cfg.asset.ground_name

        # Initialize contact detection pairs
        self._init_termination_contact()
        self._init_foot_contact()

    def _init_termination_contact(self):
        """Initialize termination contact detection"""
        cfg = self._cfg
        # Find base geometries
        base_names = []
        for base_name in cfg.asset.terminate_after_contacts_on:
            if base_name in self.model.others["geoms"]:
                base_names.append(base_name)
            else:
                print(f"Warning: Geom '{base_name}' not found in model")

        # Create base-ground contact detection pairs
        if base_names:
            self.termination_contact = tuple((name, self.ground_name) for name in base_names)
            self.num_termination_check = len(self.termination_contact)
        else:
            # Use empty inventory
            self.termination_contact = ()
            self.num_termination_check = 0
            print("Warning: No base contacts configured for termination")

    def _init_foot_contact(self):
        """Initialize foot contact detection"""
        cfg = self._cfg
        foot_names = []
        for foot_name in cfg.asset.foot_names:
            if foot_name in self.model.others["geoms"]:
                foot_names.append(foot_name)
            else:
                print(f"Warning: Foot geom '{foot_name}' not found in model")

        # Create foot-ground contact detection pairs (kept for parity with the
        # legacy inventory; no observation reads them yet)
        if foot_names:
            self.foot_contact_check = tuple((name, self.ground_name) for name in foot_names)
            self.num_foot_check = len(self.foot_contact_check)
        else:
            self.foot_contact_check = ()
            self.num_foot_check = 0
            print("Warning: No foot contacts configured")

    def get_dof_pos(self):
        return self.sim_data["robot_joint_pos"]

    def get_dof_vel(self):
        return self.sim_data["robot_joint_vel"]

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState):
        # Save current action for incremental control
        self._last_actions[:] = self._current_actions
        self._current_actions[:] = actions

        # Position control mode: directly input target angles
        actions_scaled = actions * self._cfg.control_config.action_scale
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = np.asarray(self.default_angles + actions_scaled, dtype=np.float32)
        self._ctrl_writes.execute()
        return state

    def _navigation_state(self, pose_commands: np.ndarray):
        """Derive navigation commands from cached sim reads and pose commands.

        Pure physics-derived quantities shared by ``compute_transition``
        (reward / markers) and ``compute_observation``: position error, wrapped
        heading error, distance to target, reached mask, desired velocities.
        """
        root_pos = self.sim_data["root_pos"]
        root_quat = self.sim_data["root_quat"]

        robot_position = root_pos[:, :2]
        robot_heading = quaternion.get_yaw(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]

        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)

        position_threshold = 0.3
        heading_threshold = np.deg2rad(15)
        reached_all = np.logical_and(distance_to_target < position_threshold, np.abs(heading_diff) < heading_threshold)

        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)  # Simple P controller
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)

        # Angular velocity command calculation + deadband
        deadband_yaw = np.deg2rad(8)
        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        desired_yaw_rate = np.where(np.abs(heading_diff) < deadband_yaw, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)

        velocity_commands = np.concatenate([desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1)
        return position_error, heading_diff, distance_to_target, reached_all, desired_vel_xy, velocity_commands

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        """Build the full observation from cached sim reads and episode state."""
        inputs = self.sim_data
        gyro = inputs["base_gyro"]
        projected_gravity = self._compute_projected_gravity(inputs["root_quat"])

        # Navigation quantities recomputed from the same cached reads, so
        # freshly reset rows observe their post-reset command state.
        (position_error, heading_diff, distance_to_target, reached_all, _, velocity_commands) = self._navigation_state(
            self._pose_commands
        )

        # Normalize observations
        noisy_linvel = inputs["base_linvel"][:, :3] * self._cfg.normalization.lin_vel
        noisy_gyro = gyro * self._cfg.normalization.ang_vel
        noisy_joint_angle = (inputs["robot_joint_pos"] - self.default_angles) * self._cfg.normalization.dof_pos
        noisy_joint_vel = inputs["robot_joint_vel"] * self._cfg.normalization.dof_vel
        command_normalized = velocity_commands * self.commands_scale
        last_actions = self._current_actions

        # Calculate task-related observations
        position_error_normalized = position_error / 5.0  # Normalize to reasonable range
        heading_error_normalized = heading_diff / np.pi  # Normalize to [-1, 1]
        distance_normalized = np.clip(distance_to_target / 5.0, 0, 1)  # Normalize distance
        reached_flag = reached_all.astype(np.float32)  # Whether target is reached

        # Calculate if zero_ang standard is met: reached and angular velocity close to zero
        stop_ready = np.logical_and(reached_all, np.abs(gyro[:, 2]) < 5e-2)
        stop_ready_flag = stop_ready.astype(np.float32)

        obs = np.concatenate(
            [
                noisy_linvel,  # 3
                noisy_gyro,  # 3
                projected_gravity,  # 3
                noisy_joint_angle,  # 12
                noisy_joint_vel,  # 12
                last_actions,  # 12
                command_normalized,  # 3
                position_error_normalized,  # 2 - Position error vector to target
                heading_error_normalized[:, np.newaxis],  # 1 - Heading error
                distance_normalized[:, np.newaxis],  # 1 - Distance to target
                reached_flag[:, np.newaxis],  # 1 - Whether reached
                stop_ready_flag[:, np.newaxis],  # 1 - Whether stop standard is met
            ],
            axis=-1,
        )
        assert obs.shape == (self._num_envs, 54)
        return state.replace(obs=obs)

    def compute_transition(self, state: ArrayEnvState):
        self.sim_data.execute()

        # Get root state and joint states from the refreshed cache
        root_pos = self.sim_data["root_pos"]
        base_lin_vel = self.sim_data["base_linvel"][:, :3]

        # Navigation quantities derived directly from the cached reads
        (_, _, _, _, desired_vel_xy, velocity_commands) = self._navigation_state(self._pose_commands)

        # Update target position marker
        num_envs = self._num_envs
        self._update_target_marker(np.arange(num_envs, dtype=np.int64), self._pose_commands)
        # Update arrow visualization (no physical effect)
        base_lin_vel_xy = base_lin_vel[:, :2]
        self._update_heading_arrows(np.arange(num_envs, dtype=np.int64), root_pos, desired_vel_xy, base_lin_vel_xy)

        # Calculate reward
        state.reward = self._compute_reward(velocity_commands)

        # Calculate termination conditions
        state = self._compute_terminated(state)

        return state

    def _update_heading_arrows(
        self, env_ids: np.ndarray, robot_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray
    ):
        """
        Update arrow positions (mocap bodies, no physical effect)
        robot_pos: [num_envs, 3] - Robot position
        desired_vel_xy: [num_envs, 2] - Desired linear velocity (ground coordinates)
        base_lin_vel_xy: [num_envs, 2] - Actual linear velocity (ground coordinates)
        """

        arrow_height = 0.76  # Arrow height (base=0.56 + 0.2)
        cur_yaw = np.where(
            np.linalg.norm(base_lin_vel_xy, axis=1) > 1e-3,
            np.arctan2(base_lin_vel_xy[:, 1], base_lin_vel_xy[:, 0]),
            0.0,
        )
        robot_arrow_pos = robot_pos.copy()
        robot_arrow_pos[:, 2] = arrow_height
        robot_arrow_quat = quaternion.from_euler(0, 0, cur_yaw).astype(np.float32)
        self._heading_writes.buffer("robot_pos")[env_ids, 0] = robot_arrow_pos.astype(np.float32)
        self._heading_writes.buffer("robot_rot")[env_ids, 0] = robot_arrow_quat

        des_yaw = np.where(
            np.linalg.norm(desired_vel_xy, axis=1) > 1e-6, np.arctan2(desired_vel_xy[:, 1], desired_vel_xy[:, 0]), 0.0
        )
        desired_arrow_quat = quaternion.from_euler(0, 0, des_yaw).astype(np.float32)
        self._heading_writes.buffer("desired_pos")[env_ids, 0] = robot_arrow_pos.astype(np.float32)
        self._heading_writes.buffer("desired_rot")[env_ids, 0] = desired_arrow_quat
        # Both heading markers go to the backend in one crossing.
        self._heading_writes.execute(env_ids)

    def _compute_reward(self, velocity_commands: np.ndarray) -> np.ndarray:
        """
        Velocity tracking reward mechanism
        velocity_commands: [num_envs, 3] - (vx, vy, vyaw)
        """
        num_envs = self._num_envs
        # Calculate termination condition penalties
        termination_penalty = np.zeros(num_envs, dtype=np.float32)

        # Check if DOF velocity exceeds limit
        dof_vel = self.get_dof_vel()
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_overflow = vel_max > self._cfg.max_dof_vel
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)
        termination_penalty = np.where(vel_overflow | vel_extreme, -20.0, termination_penalty)

        # Robot base contacts ground penalty
        termination_check = self.sim_data["termination_colliding"]
        base_contact = termination_check.any(axis=1)
        termination_penalty = np.where(base_contact, -20.0, termination_penalty)

        # Side flip penalty
        root_quat = self.sim_data["root_quat"]
        proj_g = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)
        gz = proj_g[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip_mask = tilt_angle > np.deg2rad(75)
        termination_penalty = np.where(side_flip_mask, -20.0, termination_penalty)

        # 1. Linear velocity tracking reward
        base_lin_vel = self.sim_data["base_linvel"]
        lin_vel_error = np.sum(np.square(velocity_commands[:, :2] - base_lin_vel[:, :2]), axis=1)
        tracking_lin_vel = np.exp(-lin_vel_error / 0.25)  # tracking_sigma = 0.25

        # 2. Angular velocity tracking reward / heading error penalty (mixed strategy)
        gyro = self.sim_data["base_gyro"]
        ang_vel_error = np.square(velocity_commands[:, 2] - gyro[:, 2])
        tracking_ang_vel = np.exp(-ang_vel_error / 0.25)

        # Get robot position and heading for arrival determination, derived
        # directly from the cached reads and pose commands
        (_, _, distance_to_target, reached_all, _, _) = self._navigation_state(self._pose_commands)

        # One-time reward for first time reaching position
        first_time_reach = np.logical_and(reached_all, ~self._ever_reached)
        self._ever_reached |= reached_all
        arrival_bonus = np.where(first_time_reach, 10.0, 0.0)

        # Distance approach reward: incentivize getting closer to target
        # Use historical minimum distance to calculate progress
        distance_improvement = self._min_distance - distance_to_target
        np.minimum(self._min_distance, distance_to_target, out=self._min_distance)
        approach_reward = np.clip(distance_improvement * 4.0, -1.0, 1.0)  # Reward 5 points for every 1 meter closer

        # 3. Orientation stability reward (penalize deviation from normal standing posture)
        # When standing normally, projected_gravity ≈ [0, 0, -1]
        projected_gravity = self._compute_projected_gravity(root_quat)
        orientation_penalty = (
            np.square(projected_gravity[:, 0])
            + np.square(projected_gravity[:, 1])
            + np.square(projected_gravity[:, 2] + 1.0)
        )

        # Arrival and stop determination (reward bonus)
        speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
        zero_ang_mask = np.abs(gyro[:, 2]) < 0.05  # Relax to 0.05 rad/s ≈ 2.86°/s
        zero_ang_bonus = np.where(np.logical_and(reached_all, zero_ang_mask), 6.0, 0.0)
        stop_base = 2 * (0.8 * np.exp(-((speed_xy / 0.2) ** 2)) + 1.2 * np.exp(-((np.abs(gyro[:, 2]) / 0.1) ** 4)))
        stop_bonus = np.where(reached_all, stop_base + zero_ang_bonus, 0.0)

        # 4. Z-axis linear velocity penalty
        lin_vel_z_penalty = np.square(base_lin_vel[:, 2])

        # 5. XY-axis angular velocity penalty
        ang_vel_xy_penalty = np.sum(np.square(gyro[:, :2]), axis=1)

        # 6. Torque penalty
        torque_penalty = np.sum(np.square(self.sim_data["actuator_ctrls"]), axis=1)

        # 7. Joint velocity penalty
        joint_vel = self.get_dof_vel()
        dof_vel_penalty = np.sum(np.square(joint_vel), axis=1)

        # 8. Action change penalty
        action_diff = self._current_actions - self._last_actions
        action_rate_penalty = np.sum(np.square(action_diff), axis=1)

        # Combined reward
        # After reaching: stop all positive rewards, only keep stop reward and penalties
        reward = np.where(
            reached_all,
            # After reaching: only stop reward and penalties
            (
                stop_bonus
                + arrival_bonus
                - 2.0 * lin_vel_z_penalty
                - 0.05 * ang_vel_xy_penalty
                - 0.0 * orientation_penalty
                - 0.00001 * torque_penalty
                - 0.0 * dof_vel_penalty
                - 0.001 * action_rate_penalty
                + termination_penalty  # Termination condition penalty
            ),
            # Not reached: normal rewards
            (
                1.5 * tracking_lin_vel  # Increase linear velocity tracking weight
                + 0.3 * tracking_ang_vel  # Decrease angular velocity weight
                + approach_reward  # Approach reward
                - 2.0 * lin_vel_z_penalty
                - 0.05 * ang_vel_xy_penalty
                - 0.0 * orientation_penalty
                - 0.00001 * torque_penalty
                - 0.0 * dof_vel_penalty
                - 0.001 * action_rate_penalty
                + termination_penalty  # Termination condition penalty
            ),
        )

        return reward

    def _update_target_marker(self, env_ids: np.ndarray, pose_commands: np.ndarray):
        """
        Update position and orientation of target marker
        """
        num_envs = pose_commands.shape[0]
        arrow_pos = pose_commands.copy()
        arrow_pos[:, 2] = 0.05
        arrow_pos = np.column_stack([pose_commands[:, 0], pose_commands[:, 1], np.full((num_envs, 1), 0.5)])
        arrow_quat = quaternion.from_euler(0, 0, pose_commands[:, 2]).astype(np.float32)
        self._target_writes.buffer("target_pos")[env_ids, 0] = arrow_pos.astype(np.float32)
        self._target_writes.buffer("target_rot")[env_ids, 0] = arrow_quat
        self._target_writes.execute(env_ids)

    def _compute_terminated(self, state: ArrayEnvState) -> ArrayEnvState:
        terminated = np.zeros(self._num_envs, dtype=bool)

        # Check if DOF velocity exceeds limit (prevent inf/numerical divergence)
        dof_vel = self.get_dof_vel()
        vel_max = np.abs(dof_vel).max(axis=1)
        vel_overflow = vel_max > self._cfg.max_dof_vel
        # Extreme velocity/NaN/Inf protection
        vel_extreme = (np.isnan(dof_vel).any(axis=1)) | (np.isinf(dof_vel).any(axis=1)) | (vel_max > 1e6)
        terminated = np.logical_or(terminated, vel_overflow)
        terminated = np.logical_or(terminated, vel_extreme)

        # Robot base contacts ground termination
        termination_check = self.sim_data["termination_colliding"]
        base_contact = termination_check.any(axis=1)
        terminated = np.logical_or(terminated, base_contact)

        # Side flip termination: tilt angle exceeds 75°
        root_quat = self.sim_data["root_quat"]
        proj_g = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(proj_g[:, :2], axis=1)
        gz = proj_g[:, 2]
        tilt_angle = np.arctan2(gxy, np.abs(gz))
        side_flip_mask = tilt_angle > np.deg2rad(75)
        terminated = np.logical_or(terminated, side_flip_mask)

        return state.replace(terminated=terminated)

    def reset(self, env_ids: np.ndarray) -> None:
        cfg: AnymalCEnvCfg = self._cfg
        num_envs = len(env_ids)

        # First generate robot initial position (in world coordinates)
        pos_range = cfg.init_state.pos_randomization_range
        robot_init_x = np.random.uniform(
            pos_range[0],
            pos_range[2],  # x_min, x_max
            num_envs,
        ).astype(np.float32)
        robot_init_y = np.random.uniform(
            pos_range[1],
            pos_range[3],  # y_min, y_max
            num_envs,
        ).astype(np.float32)
        robot_init_pos = np.stack([robot_init_x, robot_init_y], axis=1)  # [num_envs, 2]

        # Generate target position: offset relative to robot initial position
        # pose_command_range now represents offset range relative to robot
        target_offset = np.random.uniform(
            low=cfg.commands.pose_command_range[:2], high=cfg.commands.pose_command_range[3:5], size=(num_envs, 2)
        ).astype(np.float32)
        target_positions = robot_init_pos + target_offset  # Target position in world coordinates

        # Generate target heading (absolute heading, random in horizontal direction)
        target_headings = np.random.uniform(
            low=cfg.commands.pose_command_range[2], high=cfg.commands.pose_command_range[5], size=(num_envs, 1)
        ).astype(np.float32)

        pose_commands = np.concatenate([target_positions, target_headings], axis=1)

        # Set initial state without perturbing the base quaternion.
        base_pose = np.tile(self._init_base_pose, (num_envs, 1))
        base_pose[:, 0] = robot_init_x
        base_pose[:, 1] = robot_init_y
        # Keep the configured/default Z height and reset all velocities to zero.
        _reset_pose = base_pose
        self._reset_position[env_ids] = _reset_pose[:, :3]
        self._reset_rotation[env_ids] = _reset_pose[:, 3:7]
        self._reset_linear_velocity[env_ids] = 0.0
        self._reset_angular_velocity[env_ids] = 0.0
        self._reset_joint_position[env_ids] = self._init_joint_position
        self._reset_joint_velocity[env_ids] = 0.0
        self._reset_program.execute(env_ids)

        # Update target position marker (after the row reset so its default
        # mocap pose restoration cannot wipe it)
        self._update_target_marker(np.asarray(env_ids, dtype=np.int64), pose_commands)
        self.sim_data.execute(np.asarray(env_ids, dtype=np.int64))

        # Get root state for the reset rows only: the cached views are
        # full-batch, while pose_commands and the arrow writes below are
        # sized to this reset batch.
        rows = np.asarray(env_ids, dtype=np.int64)
        root_pos = self.sim_data["root_pos"][rows]
        root_quat = self.sim_data["root_quat"][rows]
        root_vel = self.sim_data["base_linvel"][rows]

        base_lin_vel = root_vel[:, :3]

        # Calculate velocity commands (consistent with update_state)
        robot_position = root_pos[:, :2]
        robot_heading = quaternion.get_yaw(root_quat)
        target_position = pose_commands[:, :2]
        target_heading = pose_commands[:, 2]

        position_error = target_position - robot_position
        distance_to_target = np.linalg.norm(position_error, axis=1)

        # Position threshold: considered reached within 0.1 meters
        position_threshold = 0.1
        reached_position = distance_to_target < position_threshold

        desired_vel_xy = np.clip(position_error * 1.0, -1.0, 1.0)
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)  # Velocity is 0 after reaching

        # Actual linear velocity XY
        base_lin_vel_xy = base_lin_vel[:, :2]

        # Update arrow visualization (no physical effect)
        self._update_heading_arrows(np.asarray(env_ids, dtype=np.int64), root_pos, desired_vel_xy, base_lin_vel_xy)

        heading_diff = target_heading - robot_heading
        heading_diff = np.where(heading_diff > np.pi, heading_diff - 2 * np.pi, heading_diff)
        heading_diff = np.where(heading_diff < -np.pi, heading_diff + 2 * np.pi, heading_diff)

        # Heading threshold: considered reached within 15 degrees
        heading_threshold = np.deg2rad(15)
        reached_heading = np.abs(heading_diff) < heading_threshold

        desired_yaw_rate = np.clip(heading_diff * 1.0, -1.0, 1.0)
        reached_all = np.logical_and(reached_position, reached_heading)
        desired_yaw_rate = np.where(reached_all, 0.0, desired_yaw_rate)  # Velocity is 0 after reaching
        desired_vel_xy = np.where(reached_all[:, np.newaxis], 0.0, desired_vel_xy)  # Velocity is 0 after reaching

        # Ensure desired_yaw_rate is 1D array
        if desired_yaw_rate.ndim > 1:
            desired_yaw_rate = desired_yaw_rate.flatten()

        # Write episode-scoped state for the reset rows
        self._pose_commands[rows] = pose_commands
        self._last_actions[rows] = 0.0
        self._current_actions[rows] = 0.0
        self._ever_reached[rows] = False
        self._min_distance[rows] = distance_to_target

    def _compute_projected_gravity(self, quat: np.ndarray) -> np.ndarray:
        gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return quaternion.rotate_vector(quat, gravity)
