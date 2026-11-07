# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.sim import (
    BodyAngularVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    DofPositionQuery,
    DofVelocityQuery,
    GeomPositionQuery,
    GeomSpecsQuery,
    JointPositionWrite,
)
from motrix_env_core.sim.write import (
    CtrlTargetsWrite,
    JointVelocityWrite,
    KinematicBodyPositionWrite,
    KinematicBodyRotationWrite,
)

from .cfg import BounceBallEnvCfg

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
    "paddle_pos": GeomPositionQuery(geom="blocker"),
}
_SIM_MODEL_QUERIES = {"geoms": GeomSpecsQuery(names=("blocker",))}


@registry.env("bounce_ball")
class BounceBallEnv(DirectEnv):
    _cfg: BounceBallEnvCfg

    def __init__(self, cfg: BounceBallEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._marker_writes = self.sim.write_compiler.compile(
            {
                "height_pos": KinematicBodyPositionWrite(("target_height_marker",)),
                "height_rot": KinematicBodyRotationWrite(("target_height_marker",)),
                "paddle_pos": KinematicBodyPositionWrite(("paddle_home_marker",)),
                "paddle_rot": KinematicBodyRotationWrite(("paddle_home_marker",)),
            },
        )
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "arm_position": JointPositionWrite(tuple(f"Joint{i}" for i in range(1, 7))),
                "arm_velocity": JointVelocityWrite(tuple(f"Joint{i}" for i in range(1, 7))),
                "ball_position": BodyPositionWrite(("ball_link",)),
                "ball_rotation": BodyRotationWrite(("ball_link",)),
                "ball_linear_velocity": BodyLinearVelocityWrite(("ball_link",)),
                "ball_angular_velocity": BodyAngularVelocityWrite(("ball_link",)),
            },
            reset=True,
        )
        self._arm_reset_position = self._reset_program.buffer("arm_position")
        self._arm_reset_velocity = self._reset_program.buffer("arm_velocity")
        self._ball_reset_position = self._reset_program.buffer("ball_position")[:, 0]
        self._ball_reset_rotation = self._reset_program.buffer("ball_rotation")[:, 0]
        self._ball_reset_linear_velocity = self._reset_program.buffer("ball_linear_velocity")[:, 0]
        self._ball_reset_angular_velocity = self._reset_program.buffer("ball_angular_velocity")[:, 0]

        # Action space: 6D joint position control
        self._action_space = gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)

        # Episode-scoped task state: full-batch buffers, reset writes the
        # done rows in place.
        num_envs = self._num_envs
        self._ball_was_upward = torch.zeros(num_envs, dtype=torch.bool)
        self._consecutive_bounces = torch.zeros(num_envs, dtype=torch.int32)
        self._target_heights = torch.zeros(num_envs, dtype=torch.float32)
        self._current_actions = torch.zeros((num_envs, 6), dtype=torch.float32)
        self._last_actions = torch.zeros((num_envs, 6), dtype=torch.float32)

        # Observation space: joint states + paddle position + target height (29D)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (29,), dtype=np.float32)

        # Initial arm joint positions
        self._init_arm_qpos = torch.tensor(self._cfg.arm_init_qpos, dtype=torch.float32) * torch.pi / 180.0

        # Action scaling
        self._action_scale = torch.tensor(self._cfg.action_scale, dtype=torch.float32)
        self._action_bias = torch.tensor(self._cfg.action_bias, dtype=torch.float32)

        # Ball initial conditions
        self._ball_init_pos = torch.tensor(self._cfg.ball_init_pos, dtype=torch.float32)
        self._ball_init_vel = torch.tensor(self._cfg.ball_init_vel, dtype=torch.float32)

        # Constants for marker poses
        self._ball_radius = 0.019  # Ball radius in meters
        # Target marker base pose: [x, y, z_placeholder, qx, qy, qz, qw]
        self._target_marker_base_pose = torch.tensor(
            [
                self._cfg.target_ball_x,
                self._cfg.target_ball_y,
                0.0,  # z will be set per environment
                0.0,
                0.0,
                0.0,
                1.0,  # identity quaternion
            ],
            dtype=torch.float32,
        )
        # Paddle home marker pose: [x, y, z, qx, qy, qz, qw]
        self._paddle_home_marker_pose = torch.tensor(
            [
                self._cfg.target_ball_x,
                self._cfg.target_ball_y,
                self._cfg.paddle_home_position_z,
                0.0,
                0.0,
                0.0,
                1.0,  # identity quaternion
            ],
            dtype=torch.float32,
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Denormalize action from [-1, 1] to joint position changes"""
        return self._action_scale * action + self._action_bias

    def _compute_reward(
        self,
        dof_pos: torch.Tensor,
        dof_vel: torch.Tensor,
        paddle_pos: torch.Tensor,
        consecutive_bounces: torch.Tensor,
        bounce_detected: torch.Tensor,
        target_heights: torch.Tensor,
        current_actions: torch.Tensor,
        last_actions: torch.Tensor,
    ) -> tuple:
        """
        Compute reward based on ball position, velocity, and paddle alignment.

        The reward function uses a composite design with multiple reward and penalty terms
        to guide the robot to learn a stable ball bouncing strategy.

        Returns:
            tuple: (total_reward, reward_details) where reward_details contains
                   individual reward components for analysis.
        """
        # Extract ball state: the ball's free joint contributes (x, y, z) at dof 6:9
        ball_x = dof_pos[:, 6]
        ball_y = dof_pos[:, 7]
        ball_z = dof_pos[:, 8]
        ball_vz = dof_vel[:, 8]

        # Extract paddle position
        paddle_xy = paddle_pos[:, :2]
        paddle_z = paddle_pos[:, 2]

        # Target positions
        target_ball_x = self._cfg.target_ball_x
        target_ball_y = self._cfg.target_ball_y
        target_height = target_heights

        # Physics constant
        g = self._cfg.gravity

        # ============================================================================
        # 1. Horizontal position reward (weighted by vertical distance)
        # Core reward ensuring ball stays directly above paddle
        # ============================================================================
        x_position_error = torch.abs(ball_x - target_ball_x)
        y_position_error = torch.abs(ball_y - target_ball_y)
        xy_position_error = torch.sqrt(x_position_error**2 + y_position_error**2)

        vertical_dist = torch.abs(ball_z - paddle_z)
        vertical_weight = torch.exp(-vertical_dist / self._cfg.vertical_weight_scale)

        weighted_horizontal_scale = self._cfg.weighted_horizontal_base_scale * (
            1.0 + self._cfg.weighted_horizontal_weight_factor * vertical_weight
        )
        weighted_position_reward = torch.exp(-(xy_position_error**2) / (2 * weighted_horizontal_scale**2))

        # ============================================================================
        # 2. Out-of-position penalty
        # Strong penalty for severe deviation to prevent ball flying out of control
        # ============================================================================
        out_of_position_penalty = -2.0 / (
            1.0
            + torch.exp(
                -(xy_position_error - self._cfg.out_of_position_threshold) / self._cfg.out_of_position_sharpness
            )
        )

        # ============================================================================
        # 3. Velocity matching reward
        # Based on projectile motion physics, encourages ball trajectory to have
        # desired velocity at target height for stable control
        # ============================================================================
        height_diff = target_height - ball_z
        desired_velocity_at_target = self._cfg.desired_velocity_at_target

        # Calculate velocity at target height using physics
        upward_below_condition = (ball_vz > 0) & (ball_z < target_height)
        velocity_squared_at_target_upward = torch.where(
            upward_below_condition, ball_vz**2 - 2 * g * height_diff, torch.zeros_like(ball_vz)
        )
        velocity_at_target_upward = torch.sqrt(velocity_squared_at_target_upward.clamp(min=0.0))

        downward_above_condition = (ball_vz < 0) & (ball_z > target_height)
        velocity_squared_at_target_downward = torch.where(
            downward_above_condition, ball_vz**2 + 2 * g * torch.abs(height_diff), torch.zeros_like(ball_vz)
        )
        velocity_at_target_downward = -torch.sqrt(velocity_squared_at_target_downward.clamp(min=0.0))

        near_target_condition = torch.abs(height_diff) < 0.05
        velocity_at_target_near = torch.where(near_target_condition, ball_vz, torch.zeros_like(ball_vz))

        # Smooth combination
        upward_motion = 1.0 / (1.0 + torch.exp(-ball_vz / 0.2))
        below_target = 1.0 / (1.0 + torch.exp(-height_diff / 0.02))
        downward_motion = 1.0 - upward_motion
        above_target = 1.0 - below_target
        at_target_weight = torch.exp(-(height_diff**2) / (2 * 0.01**2))

        velocity_at_target = (
            velocity_at_target_upward * upward_motion * below_target
            + velocity_at_target_downward * downward_motion * above_target
            + velocity_at_target_near * at_target_weight
        )

        velocity_error = torch.abs(velocity_at_target - desired_velocity_at_target)
        velocity_matching_reward = torch.exp(-(velocity_error**2) / (2 * self._cfg.velocity_error_sigma**2))

        # ============================================================================
        # 4. Height reward
        # Directly encourages ball to approach target height, core task objective
        # ============================================================================
        height_error = torch.abs(ball_z - target_height)
        height_reward = torch.exp(-(height_error**2) / (2 * self._cfg.height_error_sigma**2))

        # Height progress bonus
        height_progress_bonus = (
            (ball_z - self._cfg.height_progress_threshold).clamp(min=0.0) * self._cfg.height_progress_scale
        )

        # ============================================================================
        # 5. Controlled upward velocity reward
        # Only rewards upward velocity when ball position is good, avoiding random hitting
        # Guides strategy to learn precise hitting force
        # ============================================================================
        positioning_quality = torch.exp(-(xy_position_error**2) / (2 * self._cfg.positioning_quality_sigma**2))

        ideal_launch_velocity = torch.sqrt(2 * g * height_diff.clamp(min=0.0))
        ideal_launch_velocity = ideal_launch_velocity.clamp(
            self._cfg.ideal_velocity_min, self._cfg.ideal_velocity_max
        )

        upward_velocity_quality = torch.exp(
            -((ball_vz - ideal_launch_velocity) ** 2) / (2 * self._cfg.upward_velocity_sigma**2)
        )

        upward_mask = 1.0 / (1.0 + torch.exp(-ball_vz / self._cfg.upward_mask_scale))

        controlled_upward_reward = (
            positioning_quality
            * upward_velocity_quality
            * upward_mask
            * (ball_vz * 1.0).clamp(0.0, self._cfg.controlled_upward_clip_max)
        )

        # ============================================================================
        # 6. Velocity penalties
        # Prevents ball velocity from being too fast or falling freely
        # Ensures ball motion stays within controllable range
        # ============================================================================
        excessive_upward_penalty = -1.0 / (
            1.0 + torch.exp(-(ball_vz - self._cfg.excessive_upward_threshold) / self._cfg.excessive_upward_sharpness)
        )

        downward_penalty_magnitude = -ball_vz * (-ball_vz * self._cfg.downward_velocity_scale).clamp(
            0.0, self._cfg.downward_velocity_clip_max
        )
        downward_penalty_trigger = 1.0 / (1.0 + torch.exp((ball_vz - self._cfg.downward_velocity_threshold) / 0.2))
        downward_velocity_penalty = downward_penalty_magnitude * downward_penalty_trigger

        # ============================================================================
        # 7. Consecutive bounces reward
        # Encourages multiple consecutive successful bounces for stable long-term control
        # Uses logarithmic function to avoid infinite reward growth
        # ============================================================================
        bounce_positioning_quality = torch.exp(-(xy_position_error**2) / (2 * self._cfg.bounce_positioning_sigma**2))

        bounce_log_reward = torch.log(consecutive_bounces.to(torch.float32) + 1.0) * self._cfg.bounce_log_scale

        bounce_activation = (consecutive_bounces > 0).to(torch.float32) * bounce_positioning_quality
        consecutive_bounces_reward = bounce_log_reward * bounce_activation

        # High bounce count bonus
        high_bounce_activation = 1.0 / (
            1.0
            + torch.exp(
                -(consecutive_bounces.to(torch.float32) - self._cfg.high_bounce_threshold)
                / self._cfg.high_bounce_sharpness
            )
        )
        high_bounce_bonus = (
            consecutive_bounces.to(torch.float32)
            * self._cfg.high_bounce_scale
            * bounce_positioning_quality
            * high_bounce_activation
        )

        # ============================================================================
        # 8. Paddle-ball horizontal alignment
        # Encourages paddle to actively move directly below ball
        # Extra reward at bounce moment to reinforce correct hitting behavior
        # ============================================================================
        ball_xy = torch.stack([ball_x, ball_y], dim=1)
        paddle_ball_xy_error = torch.norm(ball_xy - paddle_xy, dim=1)

        vertical_proximity_weight = torch.exp(-vertical_dist / self._cfg.vertical_proximity_scale)

        paddle_alignment_quality = torch.exp(-(paddle_ball_xy_error**2) / (2 * self._cfg.paddle_alignment_sigma**2)) * (
            1.0 + self._cfg.paddle_alignment_weight_factor * vertical_proximity_weight
        )

        bounce_boost = bounce_detected.to(torch.float32) * self._cfg.bounce_boost_factor + 1.0
        paddle_center_reward = paddle_alignment_quality * bounce_boost * self._cfg.paddle_center_scale

        # ============================================================================
        # 9. Paddle home position reward
        # Encourages paddle to return to home position when ball is far away
        # Makes paddle motion more energy-efficient and natural
        # ============================================================================
        paddle_home_z = self._cfg.paddle_home_position_z
        paddle_height_deviation = torch.abs(paddle_z - paddle_home_z)

        # Distance-based dynamic factor
        distance_factor = 1.0 + self._cfg.distance_factor_scale / (
            1.0
            + torch.exp(
                -(vertical_dist - self._cfg.distance_factor_threshold) / self._cfg.distance_factor_sharpness
            )
        )

        home_position_reward = (
            torch.exp(-(paddle_height_deviation**2) / (2 * self._cfg.home_position_sigma**2)) * distance_factor
        )

        # Height violation penalty
        max_deviation = self._cfg.max_paddle_height_deviation
        height_violation = (paddle_height_deviation - max_deviation).clamp(min=0.0)
        height_violation_penalty = -height_violation * self._cfg.height_violation_scale

        # ============================================================================
        # 10. Action and velocity penalties
        # Penalizes drastic action changes and excessive joint velocities
        # Encourages smooth and energy-efficient control
        # ============================================================================
        action_diff = current_actions - last_actions
        action_penalty = torch.sum(torch.square(action_diff), dim=-1)

        joint_vel = self.sim_data["dof_vel"][:, :6]
        joint_vel_penalty = torch.sum(torch.square(joint_vel), dim=-1)

        # ============================================================================
        # Total reward
        # ============================================================================
        action_penalty_rate = self._cfg.action_penalty_rate
        joint_vel_penalty_rate = self._cfg.joint_vel_penalty_rate

        total_reward = (
            weighted_position_reward * self._cfg.weighted_position_weight
            + velocity_matching_reward * self._cfg.velocity_matching_weight
            + height_reward * self._cfg.height_reward_weight
            + height_progress_bonus * self._cfg.height_progress_weight
            + controlled_upward_reward * self._cfg.controlled_upward_weight
            + consecutive_bounces_reward * self._cfg.consecutive_bounces_weight
            + high_bounce_bonus * self._cfg.high_bounce_weight
            + paddle_center_reward * self._cfg.paddle_center_weight
            + home_position_reward * self._cfg.home_position_weight
            + out_of_position_penalty * self._cfg.out_of_position_weight
            + excessive_upward_penalty * self._cfg.excessive_upward_weight
            + downward_velocity_penalty * self._cfg.downward_velocity_weight
            + height_violation_penalty * self._cfg.height_violation_weight
            - action_penalty_rate * action_penalty
            - joint_vel_penalty_rate * joint_vel_penalty
        )

        # ============================================================================
        # Reward details for debugging
        # ============================================================================
        reward_details = {
            "x_position_error": x_position_error,
            "y_position_error": y_position_error,
            "xy_position_error": xy_position_error,
            "ball_z": ball_z,
            "paddle_z": paddle_z,
            "vertical_dist": vertical_dist,
            "vertical_weight": vertical_weight,
            "ball_vz": ball_vz,
            "height_diff": height_diff,
            "height_error": height_error,
            "velocity_at_target": velocity_at_target,
            "desired_velocity_at_target": torch.full_like(ball_vz, desired_velocity_at_target),
            "velocity_error": velocity_error,
            "ideal_launch_velocity": ideal_launch_velocity,
            "positioning_quality": positioning_quality,
            "upward_velocity_quality": upward_velocity_quality,
            "bounce_positioning_quality": bounce_positioning_quality,
            "paddle_ball_xy_error": paddle_ball_xy_error,
            "vertical_proximity_weight": vertical_proximity_weight,
            "bounce_detected": bounce_detected.to(torch.float32),
            "consecutive_bounces": consecutive_bounces.to(torch.float32),
            "action_penalty": action_penalty,
            "joint_vel_penalty": joint_vel_penalty,
            "distance_factor": distance_factor,
            "weighted_position_reward": weighted_position_reward * 2.0,
            "velocity_matching_reward": velocity_matching_reward * 2.0,
            "height_reward": height_reward * 4.5,
            "height_progress_bonus": height_progress_bonus * 1.0,
            "controlled_upward_reward": controlled_upward_reward * 1.5,
            "consecutive_bounces_reward": consecutive_bounces_reward * 0.8,
            "high_bounce_bonus": high_bounce_bonus * 0.3,
            "paddle_center_reward": paddle_center_reward * 0.6,
            "home_position_reward": home_position_reward * 1.5,
            "out_of_position_penalty": out_of_position_penalty * 1.0,
            "excessive_upward_penalty": excessive_upward_penalty * 1.0,
            "downward_velocity_penalty": downward_velocity_penalty * 1.0,
            "height_violation_penalty": height_violation_penalty * 1.0,
            "action_penalty_weighted": -action_penalty_rate * action_penalty,
            "joint_vel_penalty_weighted": -joint_vel_penalty_rate * joint_vel_penalty,
            "paddle_height_deviation": paddle_height_deviation,
            "total_reward": total_reward,
        }

        return total_reward, reward_details

    def _compute_terminated(
        self, dof_pos: torch.Tensor, dof_vel: torch.Tensor, target_heights: torch.Tensor
    ) -> torch.Tensor:
        """Check if episode should terminate based on DOF states"""
        # Extract ball position from DOF (indices 6-8 for x,y,z)
        ball_x = dof_pos[:, 6]  # Ball x position
        ball_y = dof_pos[:, 7]  # Ball y position
        ball_z = dof_pos[:, 8]  # Ball z position

        # Terminate if ball falls below ground or goes significantly higher than target
        terminated = (ball_z < 0.05) | (ball_z > target_heights + 1.0)

        # Also terminate if ball goes too far horizontally
        terminated |= (torch.abs(ball_x) > 1.5) | (torch.abs(ball_y) > 1.5)

        # Terminate if joint velocity is too high
        # Limit: 360 degrees/second = 2*pi rad/s ≈ 6.28 rad/s
        joint_vel = dof_vel[:, :6]  # Joint velocities (first 6 DOFs are arm joints)
        max_joint_vel = 2.0 * torch.pi  # 360 degrees/second in radians
        terminated |= torch.abs(joint_vel).max(dim=-1).values > max_joint_vel

        return terminated

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        """Apply action to control paddle position"""
        # Save last actions for penalty calculation
        self._last_actions.copy_(self._current_actions)
        self._current_actions.copy_(actions)

        # Get current joint positions
        current_joint_pos = self.sim_data["dof_pos"][:, :6]  # First 6 DOFs are arm joints

        # Denormalize actions to get actual position changes
        delta_positions = self._denormalize_action(actions)

        # Calculate target positions = current positions + position changes
        target_positions = current_joint_pos + delta_positions

        # Apply target positions as actuator controls (position control)
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl.copy_(target_positions)
        self._ctrl_writes.execute()
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        # Observation: joint states + paddle position + target height (29D)
        # Concatenate: DOF pos (13) + DOF vel (12) + paddle xyz (3) + target height (1)
        obs = torch.cat(
            [inputs["dof_pos"], inputs["dof_vel"], inputs["paddle_pos"], self._target_heights[:, None]], dim=-1
        )
        return state.replace(obs=obs)

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        """Update state with rewards and termination flags"""
        self.sim_data.execute()
        inputs = self.sim_data

        # Episode-scoped bounce tracking and target heights live in env buffers.
        consecutive_bounces = self._consecutive_bounces
        ball_was_upward = self._ball_was_upward
        target_heights = self._target_heights

        # Detect bounces and update consecutive bounce count.
        # The ball's free joint contributes (x, y, z) at dof 6:9.
        current_ball_z = inputs["dof_pos"][:, 8]  # Ball z position
        current_ball_vz = inputs["dof_vel"][:, 8]  # Ball z velocity

        # Detect bounces: ball moving upward after being near paddle height
        near_paddle = (current_ball_z < 0.4) & (current_ball_z > 0.15)
        moving_upward = current_ball_vz > 0.01

        # A bounce is detected when ball was going down and now goes up near paddle height
        bounce_detected = ~ball_was_upward & moving_upward & near_paddle

        # Update consecutive bounce count
        consecutive_bounces = torch.where(bounce_detected, consecutive_bounces + 1, consecutive_bounces)

        # Reset count if ball is falling too much (not bouncing properly)
        falling = (current_ball_vz < -0.5) & (current_ball_z < 0.4)
        consecutive_bounces[falling] = 0
        self._consecutive_bounces = consecutive_bounces
        self._ball_was_upward = moving_upward.clone()

        # Compute reward and termination from simulator quantities
        reward, reward_details = self._compute_reward(
            inputs["dof_pos"],
            inputs["dof_vel"],
            inputs["paddle_pos"],
            consecutive_bounces,
            bounce_detected=bounce_detected,
            target_heights=target_heights,
            current_actions=self._current_actions,
            last_actions=self._last_actions,
        )
        terminated = self._compute_terminated(inputs["dof_pos"], inputs["dof_vel"], target_heights=target_heights)

        state.reward[:] = reward
        state.terminated[:] = terminated

        # Store reward details for debugging
        if self._cfg.store_reward_details:
            state.reward_terms = reward_details

        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset environment to initial state with randomized target heights"""
        cfg: BounceBallEnvCfg = self._cfg
        num_reset = env_ids.numel()

        # Randomize target heights for the environments being reset
        if cfg.randomize_target_height:
            min_height, max_height = cfg.target_height_range
            new_target_heights = (torch.rand(num_reset, dtype=torch.float32) * (max_height - min_height) + min_height)
        else:
            # Use mean of target_height_range when not randomizing
            default_height = sum(cfg.target_height_range) / 2.0
            new_target_heights = torch.full((num_reset,), default_height, dtype=torch.float32)

        # Add noise to initial arm joint positions only (not ball)
        arm_noise_pos = (torch.rand(num_reset, 6, dtype=torch.float32) * 2.0 - 1.0) * cfg.reset_noise_scale
        arm_vel = (torch.rand(num_reset, 6, dtype=torch.float32) * 2.0 - 1.0) * cfg.reset_noise_scale

        arm_pos = self._init_arm_qpos.tile((num_reset, 1)) + arm_noise_pos
        ball_pos = self._ball_init_pos + (torch.rand(num_reset, 3, dtype=torch.float32) * 2.0 - 1.0) * 0.01

        self._arm_reset_position[env_ids] = arm_pos
        self._arm_reset_velocity[env_ids] = arm_vel
        self._ball_reset_position[env_ids] = ball_pos
        self._ball_reset_rotation[env_ids] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self._ball_reset_linear_velocity[env_ids] = 0.0
        self._ball_reset_angular_velocity[env_ids] = 0.0
        self._reset_program.execute(env_ids)

        # Update target height marker position (thin cylinder disc)
        # Marker center aligns with ball top: marker_z = target_height + ball_radius
        target_marker_poses = self._target_marker_base_pose.tile((num_reset, 1)).clone()
        target_marker_poses[:, 2] = new_target_heights + self._ball_radius  # Set z position

        self._marker_writes.buffer("height_pos")[env_ids, 0] = target_marker_poses[:, :3]
        self._marker_writes.buffer("height_rot")[env_ids, 0] = target_marker_poses[:, 3:7]

        # Set paddle home marker mocap body pose
        paddle_home_marker_poses = self._paddle_home_marker_pose.tile((num_reset, 1))
        self._marker_writes.buffer("paddle_pos")[env_ids, 0] = paddle_home_marker_poses[:, :3]
        self._marker_writes.buffer("paddle_rot")[env_ids, 0] = paddle_home_marker_poses[:, 3:7]
        # Both markers go to the backend in one crossing.
        self._marker_writes.execute(env_ids)

        # Refresh cached reads for the reset rows so the follow-up
        # compute_observation (which never executes) sees post-reset state.
        self.sim_data.execute(env_ids)

        # Write episode-scoped state for the reset rows
        self._consecutive_bounces[env_ids] = 0
        self._ball_was_upward[env_ids] = False
        self._target_heights[env_ids] = new_target_heights
        self._current_actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
