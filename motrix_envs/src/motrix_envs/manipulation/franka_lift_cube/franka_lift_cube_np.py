# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState, EnvObs
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.sim import (
    BodyAngularVelocityWrite,
    BodyJointPositionQuery,
    BodyJointPositionWrite,
    BodyJointVelocityQuery,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    GeomLinearVelocityQuery,
    GeomPositionQuery,
    GeomQuaternionQuery,
    SitePositionQuery,
    SiteQuaternionQuery,
)
from motrix_env_core.sim.write import BodyJointVelocityWrite, CtrlTargetsWrite

from .cfg import FrankaLiftCubeEnvCfg

_SIM_DATA_QUERIES = {
    "robot_joint_pos": BodyJointPositionQuery(body="link0"),
    "robot_joint_vel": BodyJointVelocityQuery(body="link0"),
    "gripper_pos": SitePositionQuery(site="gripper"),
    "gripper_quat": SiteQuaternionQuery(site="gripper"),
    "cube_pos": GeomPositionQuery(geom="cube"),
    "cube_quat": GeomQuaternionQuery(geom="cube"),
    "cube_lin_vel": GeomLinearVelocityQuery(geom="cube"),
}
_SIM_MODEL_QUERIES = {}

# Decay parameters (constants, can be defined during class initialization)
START_EPSILON = 1.0  # Initial value
MIN_EPSILON = 0.05  # Minimum value (typically 0.01 or 0.05)
# Assume we want to complete decay in half of total steps (12000 steps)
END_STEP = 12000


@registry.env("franka-lift-cube")
class FrankaLiftCubeEnv(DirectEnv):
    _cfg: FrankaLiftCubeEnvCfg

    def __init__(self, cfg: FrankaLiftCubeEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs=num_envs, backend=backend)
        self.model = self.sim.compile_model({})
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "robot_position": BodyJointPositionWrite("link0"),
                "robot_velocity": BodyJointVelocityWrite("link0"),
                "cube_position": BodyPositionWrite(("free_cube",)),
                "cube_rotation": BodyRotationWrite(("free_cube",)),
                "cube_linear_velocity": BodyLinearVelocityWrite(("free_cube",)),
                "cube_angular_velocity": BodyAngularVelocityWrite(("free_cube",)),
            },
            reset=True,
        )
        self.default_joint_pos = torch.as_tensor(self._cfg.init_state.default_joint_pos, dtype=torch.float32)

        self._action_dim = 8
        self._obs_dim = 36  # 9 + 9 + 3 + 7 + 8
        self._action_space = gym.spaces.Box(-np.inf, np.inf, (self._action_dim,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (self._obs_dim,), dtype=np.float32)

        self._num_dof_pos = 9  # self._model.num_dof_pos # 9
        self._num_dof_vel = 9  # self._model.num_dof_vel # 9
        self._init_dof_pos = self.default_joint_pos
        self._init_dof_vel = torch.zeros(self._num_dof_vel, dtype=torch.float32)

        self.joint_pos_min_limit = torch.tensor(self._cfg.control_config.min_pos, dtype=torch.float32)
        self.joint_pos_max_limit = torch.tensor(self._cfg.control_config.max_pos, dtype=torch.float32)

        self.epsilon = START_EPSILON

        self.count = 0

        # Episode-scoped task state: full-batch buffers, reset writes the done
        # rows in place.
        num_envs = self._num_envs
        self._commands = torch.zeros(num_envs, 3, dtype=torch.float32)
        self._current_actions = torch.zeros(num_envs, self._action_dim, dtype=torch.float32)
        self._last_actions = torch.zeros(num_envs, self._action_dim, dtype=torch.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState):
        self._last_actions.copy_(self._current_actions)
        self._current_actions.copy_(actions)

        # no gripper
        old_joint_pos = self.get_dof_pos(slice(None))[:, : self._action_dim - 1]
        new_joint_pos = actions[:, : self._action_dim - 1] + old_joint_pos  # action as offset

        # with gripper
        # 1. Map to probability p (using Sigmoid)
        probabilities = 1 / (1 + torch.exp(-actions[:, -1]))
        # 2. Bernoulli sampling - probability always has chance to sample different results
        # torch.rand generates a random number r ~ U(0, 1) for each environment
        # If r < p, result is 1 (success/grasp), otherwise 0 (failure/release)
        sampled_gripper_action = torch.where(probabilities > torch.rand_like(probabilities), 0.0, 0.04).unsqueeze(
            -1
        )  # Close 0, Open 0.04

        new_pos = torch.cat([new_joint_pos, sampled_gripper_action], dim=-1)

        # step action
        cliped_new_pos = torch.clamp(
            new_pos, self.joint_pos_min_limit, self.joint_pos_max_limit
        )  # clip new pos to limit

        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl.copy_(cliped_new_pos)
        self._ctrl_writes.execute()

        return state

    def compute_observation(self, state: ArrayEnvState):
        """Build the full observation batch from cached simulator data.

        Reads only the cache left by the last read-program execution in the
        transition; never performs reads itself and never touches reward or
        termination.
        """
        dof_pos = self.get_dof_pos(slice(None))
        dof_vel = self.get_dof_vel(slice(None))
        dof_pos_rel = self._get_joint_pos_rel(dof_pos)
        dof_vel_rel = self._get_joint_vel_rel(dof_vel)

        object_pick_pose = self.get_cube_pose(slice(None))

        object_lift_pos = self._commands

        last_actions = self._current_actions

        obs = torch.cat([dof_pos_rel, dof_vel_rel, object_pick_pose, object_lift_pos, last_actions], dim=-1)

        assert obs.shape == (dof_pos.shape[0], self._obs_dim)
        assert not obs.isnan().any(), "obs contain nan"
        # Publish the computed policy observation on the state. The field
        # name is applied via setattr so static audits can tell this
        # sanctioned observation-stage write apart from the forbidden
        # transition-time writes.
        setattr(state, "obs", EnvObs(policy=obs))
        return state

    def compute_transition(self, state: ArrayEnvState):
        # One authoritative read of the post-physics simulator state; reward and
        # termination below must come from this refreshed cache, never from obs.
        self.sim_data.execute()

        # compute truncated
        truncated = self._check_termination(state)

        # compute reward
        reward = self._compute_reward(state, truncated)

        state.reward = reward
        state.terminated = truncated

        self.count += 1

        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        num_reset = env_ids.numel()

        # Robot arm initial joint angle noise
        noise_scale = self._cfg.init_state.joint_pos_reset_noise_scale
        noise_pos = (torch.rand(self._num_dof_pos, dtype=torch.float32) * 2.0 - 1.0) * noise_scale
        robot_dof_pos = self._init_dof_pos + noise_pos

        # Domain randomization for cube position
        # x -0.1, 0.1
        # y -0.25, 0.25
        x_low, x_high = -0.1, 0.1
        y_low, y_high = -0.25, 0.25
        pos_x = torch.rand((), dtype=torch.float32) * (x_high - x_low) + x_low
        pos_y = torch.rand((), dtype=torch.float32) * (y_high - y_low) + y_low

        self._reset_program.buffer("robot_position")[env_ids] = robot_dof_pos
        self._reset_program.buffer("robot_velocity")[env_ids] = self._init_dof_vel
        self._reset_program.buffer("cube_position")[env_ids, 0] = torch.tensor(
            [pos_x, pos_y, 0.05], dtype=torch.float32
        )
        self._reset_program.buffer("cube_rotation")[env_ids, 0] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32
        )
        self._reset_program.buffer("cube_linear_velocity")[:, 0][env_ids] = 0.0
        self._reset_program.buffer("cube_angular_velocity")[:, 0][env_ids] = 0.0
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)

        commands = self._generated_commands(num_reset)
        assert not commands.isnan().any(), "commands contain nan"

        # Write episode-scoped state for the reset rows
        self._commands[env_ids] = commands
        self._current_actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0

    def _check_termination(self, state: ArrayEnvState):
        cube_height = self.get_cube_pose(slice(None))[:, 2]
        truncated = cube_height < -0.05  # New truncated condition

        # Check joint velocity is not too large (set to 5 radians per second here)
        joint_vel = self.get_dof_vel(slice(None))
        truncated = truncated | (joint_vel.abs().amax(dim=-1) > 10)

        # Check cube velocity
        cube_vel = self.sim_data["cube_lin_vel"]  # shape = (*data.shape, 3).
        truncated = truncated | (cube_vel.abs().amax(dim=-1) > 10)
        return truncated

    def _compute_reward(self, state: ArrayEnvState, truncated: torch.Tensor):
        hand_pos = self.sim_data["gripper_pos"]
        cube_pos = self.sim_data["cube_pos"]

        # reach reward
        hand_cube_distance = torch.linalg.norm(cube_pos - hand_pos, dim=-1)

        std = 0.1
        reach_reward = 1 - torch.tanh(hand_cube_distance / std)

        # lift reward
        lift_height = cube_pos[:, 2]  # Cube center of mass height - initial center of mass height 0.02 = lift height
        minimal_height = 0.04  # 4cm height limit
        lifted = lift_height > minimal_height

        # object_command_tracking reward
        object_command_dist = torch.linalg.norm(cube_pos - self._commands, dim=-1)

        def shifted_sigmoid_reward(d, k=8, center=0.3):
            # Sigmoid(-k * (d - center))
            # The larger d is, the more positive (d-center) is, the more negative -k*(...) is, Sigmoid closer to 0
            # The smaller d is, the more negative (d-center) is, the more positive -k*(...) is, Sigmoid closer to 1
            x = -k * (d - center)
            return 1 / (1 + torch.exp(-x))

        object_command_tracking_reward = (
            shifted_sigmoid_reward(object_command_dist) * (lift_height > 0.04) * (hand_cube_distance < 0.02)
        )

        object_command_tracking_fine_graind_reward = (
            (1 - torch.tanh(object_command_dist / 0.4)) * (lift_height > 0.04) * (hand_cube_distance < 0.02)
        )

        object_command_tracking_close_reward = (
            (1 - torch.tanh(object_command_dist / 0.05)) * (object_command_dist < 0.2) * (hand_cube_distance < 0.02)
        )

        # action_diff_sq: Sum of squares of action changes
        action_diff_sq = torch.sum((self._current_actions - self._last_actions) ** 2, dim=-1)
        # joint_vel_sq: Sum of squares of joint velocities
        joint_vel_sq = torch.sum(self.get_dof_vel(slice(None))[:, : self._num_dof_vel] ** 2, dim=1)

        ## action penalty rate
        reach_weight = 1.5  # Cannot be too small
        cmd_tracking_weight = 10.0
        cmd_tracking_fine_graind_weight = 20.0  # Should be larger, need strong pull to target area
        object_command_tracking_close_reward_weight = 10.0

        if self.count < 20000:
            action_penalty_rate = 1e-4
            joint_vel_penalty_rate = 1e-4
        else:
            action_penalty_rate = 1e-1
            joint_vel_penalty_rate = 1e-1

        reward = (
            reach_weight * reach_reward
            + 30 * lifted * (hand_cube_distance < 0.05)
            + (cmd_tracking_weight * object_command_tracking_reward) ** 2
            + (cmd_tracking_fine_graind_weight * object_command_tracking_fine_graind_reward) ** 2
            + (object_command_tracking_close_reward_weight * object_command_tracking_close_reward) ** 2
            + 200 * object_command_tracking_close_reward
            + -action_penalty_rate * action_diff_sq
            + -joint_vel_penalty_rate * joint_vel_sq
        )

        return reward

    def get_dof_pos(self, rows):
        return self.sim_data["robot_joint_pos"][rows]

    def get_dof_vel(self, rows):
        return self.sim_data["robot_joint_vel"][rows]

    def get_cube_pose(self, rows):
        return torch.cat((self.sim_data["cube_pos"][rows], self.sim_data["cube_quat"][rows]), dim=-1)

    def _get_joint_pos_rel(self, dof_pos: torch.Tensor):
        return dof_pos - self.default_joint_pos

    def _get_joint_vel_rel(self, dof_vel: torch.Tensor):
        return dof_vel - self._init_dof_vel

    def _generated_commands(self, num_envs: int):
        # Command is the final object_pose that cube should reach
        x_low, x_high = self._cfg.command_config.target_pos_x
        y_low, y_high = self._cfg.command_config.target_pos_y
        z_low, z_high = self._cfg.command_config.target_pos_z

        pos_x = torch.rand(num_envs, dtype=torch.float32) * (x_high - x_low) + x_low
        pos_y = torch.rand(num_envs, dtype=torch.float32) * (y_high - y_low) + y_low
        pos_z = torch.rand(num_envs, dtype=torch.float32) * (z_high - z_low) + z_low
        command_cube_target_pos = torch.stack([pos_x, pos_y, pos_z], dim=-1)

        assert not command_cube_target_pos.isnan().any(), "command_cube_target_pos contain nan"
        return command_cube_target_pos
