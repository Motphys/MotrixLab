# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState, EnvObs
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.math import quaternion
from motrix_env_core.sim import (
    BodyJointPositionQuery,
    BodyJointPositionWrite,
    BodyJointVelocityQuery,
    GeomPositionQuery,
    JointPositionQuery,
    JointPositionWrite,
    JointVelocityQuery,
    SitePositionQuery,
    SiteQuaternionQuery,
)
from motrix_env_core.sim.write import BodyJointVelocityWrite, CtrlTargetsWrite, JointVelocityWrite

from .cfg import FrankaOpenCabinetEnvCfg

_SIM_DATA_QUERIES = {
    "robot_joint_pos": BodyJointPositionQuery(body="link0"),
    "robot_joint_vel": BodyJointVelocityQuery(body="link0"),
    "drawer_pos": JointPositionQuery(joints=("drawer_top_joint",)),
    "drawer_vel": JointVelocityQuery(joints=("drawer_top_joint",)),
    "gripper_pos": SitePositionQuery(site="gripper"),
    "gripper_quat": SiteQuaternionQuery(site="gripper"),
    "handle_pos": SitePositionQuery(site="drawer_top_handle"),
    "handle_quat": SiteQuaternionQuery(site="drawer_top_handle"),
    "left_finger_pos": GeomPositionQuery(geom="left_finger_pad"),
    "right_finger_pos": GeomPositionQuery(geom="right_finger_pad"),
}

_SIM_MODEL_QUERIES = {}


@registry.env("franka-open-cabinet")
class FrankaOpenCabinetEnv(DirectEnv):
    _cfg: FrankaOpenCabinetEnvCfg

    def __init__(self, cfg: FrankaOpenCabinetEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs=num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "robot_position": BodyJointPositionWrite("link0"),
                "robot_velocity": BodyJointVelocityWrite("link0"),
                "cabinet_position": JointPositionWrite(
                    ("door_right_joint", "door_left_joint", "drawer_top_joint", "drawer_bottom_joint")
                ),
                "cabinet_velocity": JointVelocityWrite(
                    ("door_right_joint", "door_left_joint", "drawer_top_joint", "drawer_bottom_joint")
                ),
            },
            reset=True,
        )
        self.robot_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ]
        self.robot_default_joint_pos = np.array(
            [
                0.0 * np.pi,
                -30 / 180 * np.pi,
                0 * np.pi,
                -156 / 180 * np.pi,
                0.0 * np.pi,
                186 / 180 * np.pi,
                -45 / 180 * np.pi,
                0.04,
                0.04,
            ],
            np.float32,
        )

        self._action_dim = 8
        self._obs_dim = 25  # 8 + 8 + 7 + 1 + 1
        self._action_space = gym.spaces.Box(-np.inf, np.inf, (self._action_dim,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (self._obs_dim,), dtype=np.float32)

        self._num_dof_pos = 9  # self._model.num_dof_pos # 9
        self._num_dof_vel = 9  # self._model.num_dof_vel # 9
        self._init_dof_pos = self.robot_default_joint_pos
        self._init_dof_vel = np.zeros(self._num_dof_vel, dtype=np.float32)
        # Initialize properties
        self.robot_joint_pos_min_limit = np.asarray(
            [spec.ctrl_range[0] for spec in self.model.actuators], dtype=np.float32
        )
        self.robot_joint_pos_max_limit = np.asarray(
            [spec.ctrl_range[1] for spec in self.model.actuators], dtype=np.float32
        )

        self.count = 0
        # Set print options to 2 decimal places
        np.set_printoptions(precision=2)

        # Episode-scoped task state: full-batch buffers, reset writes the done
        # rows in place.
        num_envs = self._num_envs
        self._current_actions = np.zeros((num_envs, self._action_dim), dtype=np.float32)
        self._last_actions = np.zeros((num_envs, self._action_dim), dtype=np.float32)
        self._current_gripper_action = np.zeros(num_envs, dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState):
        assert not np.isnan(actions).any(), "actions contain nan"

        self._last_actions[:] = self._current_actions
        self._current_actions[:] = actions

        # no gripper
        old_joint_pos = self.get_robot_joint_pos(slice(None))[:, : self._action_dim - 1]
        new_joint_pos = actions[:, : self._action_dim - 1] + old_joint_pos  # action as offset

        # with gripper
        # 1. Map to probability p using Sigmoid
        probabilities = 1 / (1 + np.exp(-actions[:, -1]))
        # 2. Bernoulli sampling - probability can sample different results
        # np.random.uniform(0, 1, size) generates random number r ~ U(0, 1) for each environment
        # If r < p, result is 1 (success/grasp), otherwise 0 (failure/release)
        sampled_gripper_action = np.where(probabilities > np.random.rand(*probabilities.shape), 0, 0.04)[
            :, None
        ]  # 0 for closed, 0.04 for open
        self._current_gripper_action[:] = sampled_gripper_action.squeeze(-1)

        new_pos = np.concatenate([new_joint_pos, sampled_gripper_action], axis=-1)

        # step action
        cliped_new_pos = np.clip(
            new_pos, self.robot_joint_pos_min_limit, self.robot_joint_pos_max_limit, dtype=np.float32
        )  # clip new pos to limit

        # actuator1~8 by order
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = cliped_new_pos
        self._ctrl_writes.execute()

        return state

    def compute_observation(self, state: ArrayEnvState):
        """Build the full observation batch from cached simulator data.

        Reads only the cache left by the last read-program execution in the
        transition; never performs reads itself and never touches reward or
        termination.
        """
        rows = slice(None)
        num_envs = self.num_envs

        # dof_pos: (num_envs, 8) range: [-1 ~ 1]
        dof_pos = self.get_robot_joint_pos(rows)  # shape: (num_envs, 8)
        dof_pos_rel = self._get_robot_joint_pos_rel(dof_pos)[:, : self._action_dim]

        dof_lower_limits = np.tile(self.robot_joint_pos_min_limit, (num_envs, 1))
        dof_upper_limits = np.tile(self.robot_joint_pos_max_limit, (num_envs, 1))

        dof_pos_scaled = 2.0 * dof_pos_rel / (dof_upper_limits - dof_lower_limits) - 1.0
        # relative vel: (num_envs, 8) range approximately (-pi ~ pi) / 2 (divided by 2 for smaller values)
        dof_vel = self.get_robot_joint_vel(rows)
        dof_vel_rel = self._get_robot_joint_vel_rel(dof_vel)[:, : self._action_dim] / 2

        # relative orientation: (num_envs, 1)
        robot_grasp_pose = self._grasp_pose(rows, "gripper")
        drawer_grasp_pose = self._grasp_pose(rows, "handle")
        to_target = drawer_grasp_pose - robot_grasp_pose

        # Cabinet joint
        drawer_top_joint_pos = self.sim_data["drawer_pos"][rows]
        drawer_top_joint_vel = self.sim_data["drawer_vel"][rows]

        obs = np.concatenate(
            [dof_pos_scaled, dof_vel_rel, to_target, drawer_top_joint_pos, drawer_top_joint_vel], axis=-1
        )

        assert obs.shape == (num_envs, self._obs_dim)
        assert not np.isnan(obs).any(), "obs contain nan"
        # Publish the computed policy observation on the state. The field
        # name is applied via setattr so static audits can tell this
        # sanctioned observation-stage write apart from the forbidden
        # transition-time writes.
        setattr(state, "obs", EnvObs(policy=np.clip(obs, -5, 5)))
        return state

    def compute_transition(self, state: ArrayEnvState):
        """Update reward and termination from the refreshed simulator cache.

        Observations are built separately by :meth:`compute_observation`.
        """
        self.sim_data.execute()
        # compute truncated
        truncated = self._check_termination(state)

        # compute reward
        reward = self._compute_reward(state, truncated)

        state.reward = reward
        state.terminated = truncated

        self.count += 1

        return state

    def reset(self, env_ids) -> None:
        num_reset = len(env_ids)
        row_ids = np.asarray(env_ids, dtype=np.int64)

        noise_pos = np.random.uniform(
            -0.125,  # -cfg.reset_noise_scale,
            0.125,  # cfg.reset_noise_scale,
            (num_reset, self._num_dof_pos),
        )

        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1)) + noise_pos  # Add noise in range [-0.125, 0.125]
        self._reset_program.buffer("robot_position")[row_ids] = np.asarray(dof_pos, dtype=np.float32)
        self._reset_program.buffer("robot_velocity")[row_ids] = 0.0
        self._reset_program.buffer("cabinet_position")[row_ids] = 0.0
        self._reset_program.buffer("cabinet_velocity")[row_ids] = 0.0
        self._reset_program.execute(row_ids)
        self.sim_data.execute(row_ids)

        # Write episode-scoped state for the reset rows
        self._current_actions[row_ids] = 0.0
        self._last_actions[row_ids] = 0.0
        self._current_gripper_action[row_ids] = 0.0

    def _compute_reward(self, state: ArrayEnvState, truncated: np.ndarray):
        robot_grasp_pose = self._grasp_pose(slice(None), "gripper")
        drawer_grasp_pose = self._grasp_pose(slice(None), "handle")

        gripper_drawer_dist = np.linalg.norm(drawer_grasp_pose[:, :3] - robot_grasp_pose[:, :3], axis=-1)

        ## distance reward
        std = 0.1
        dist_reward = 1 - np.tanh(gripper_drawer_dist / std)
        dist_reward *= 10

        ## matching orientation reward
        quat_reward = quaternion.similarity(robot_grasp_pose[:, -4:], drawer_grasp_pose[:, -4:])

        ## close gripper reward
        # When gripper distance < 0.025, closing gripper gets reward
        # When gripper distance > 0.025, closing gripper gets penalty
        # When gripper distance > 0.025 or < 0.025, opening gripper gets no reward
        open_gripper = np.where(gripper_drawer_dist < 0.025, 100.0, -20) * (
            0.04 - self._current_gripper_action
        )  # dist_reward * 0 or 0.04

        ## open drawer reward
        open_dist = self.sim_data["drawer_pos"][:, 0]
        open_dist = np.clip(open_dist, 0, 1)
        open_reward = (np.exp(open_dist) - 1) * 20

        wrong_open = np.logical_and(
            open_dist > 0, gripper_drawer_dist > 0.03
        )  # Drawer opened but gripper not on handle
        open_reward = (
            np.bitwise_not(wrong_open) * open_reward
        )  # No reward for forced opening (can't force open after increasing MJCF resistance)
        quat_reward = np.where(open_reward > 0, 1.0, quat_reward)

        ##################### Penalty Terms #####################"
        ## Action penalty
        ## Joint velocity penalty - sometimes some joints rotate more while others rotate less
        action_penalty = np.sum(np.square(self._current_actions - self._last_actions), axis=-1)
        joint_vel_penalty = np.sum(np.square(self.sim_data["robot_joint_vel"][:, : self._action_dim]), axis=-1)

        ## finger position penalty
        lfinger_dist = self.sim_data["left_finger_pos"][:, 2] - drawer_grasp_pose[:, 2]
        rfinger_dist = drawer_grasp_pose[:, 2] - self.sim_data["right_finger_pos"][:, 2]
        finger_dist_penalty = np.zeros_like(lfinger_dist)
        finger_dist_penalty += np.where(lfinger_dist < 0, lfinger_dist, np.zeros_like(lfinger_dist))
        finger_dist_penalty += np.where(rfinger_dist < 0, rfinger_dist, np.zeros_like(rfinger_dist))

        ##################### Coefficient Schedule #####################"

        ## action penalty rate
        if self.count < 8000:
            action_penalty_rate = 1e-3
            joint_vel_penalty_rate = 0 * 10  # Keep very small at the beginning
        else:
            action_penalty_rate = 2e-3
            joint_vel_penalty_rate = 2e-7

        ##################### Reward Calculation #####################"

        step2_reward = dist_reward + quat_reward + open_gripper + open_reward + finger_dist_penalty

        # Final reward
        reward = step2_reward - action_penalty_rate * action_penalty - joint_vel_penalty_rate * joint_vel_penalty

        # Apply truncation penalty
        reward = np.where(truncated, reward - np.array(10.0), reward)

        return reward

    def _check_termination(self, state: ArrayEnvState):
        # Check if robot arm extends too far forward causing collision
        robot_grasp_pos_x = self._grasp_pose(slice(None), "gripper")[:, 0]
        drawer_grasp_pos_x = self._grasp_pose(slice(None), "handle")[:, 0]
        truncated = robot_grasp_pos_x - drawer_grasp_pos_x < -0.03

        # Check that joint velocity doesn't exceed threshold of 5 rad/s
        joint_vel = self.get_robot_joint_vel(slice(None))
        truncated = np.logical_or(truncated, np.abs(joint_vel).max(axis=-1) > 5)
        return truncated

    def _grasp_pose(self, rows, which: str):
        return np.concatenate((self.sim_data[f"{which}_pos"][rows], self.sim_data[f"{which}_quat"][rows]), axis=-1)

    def get_robot_joint_pos(self, rows):
        return self.sim_data["robot_joint_pos"][rows][:, : self._num_dof_pos]

    def get_robot_joint_vel(self, rows):
        return self.sim_data["robot_joint_vel"][rows][:, : self._num_dof_pos]

    def _get_robot_joint_pos_rel(self, dof_pos: np.ndarray):
        return dof_pos - self.robot_default_joint_pos

    def _get_robot_joint_vel_rel(self, dof_vel: np.ndarray):
        return dof_vel - self._init_dof_vel
