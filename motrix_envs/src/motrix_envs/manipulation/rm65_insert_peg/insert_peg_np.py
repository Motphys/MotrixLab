# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState, NpObs
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.math import quaternion
from motrix_env_core.sim import (
    BodyJointPositionQuery,
    BodyJointPositionWrite,
    BodyJointVelocityQuery,
    BodyJointVelocityWrite,
    DofPositionQuery,
    GeomLinearVelocityQuery,
    GeomPositionQuery,
    GeomQuaternionQuery,
    SitePositionQuery,
)
from motrix_env_core.sim.write import (
    CtrlTargetsWrite,
    JointAngularVelocityWrite,
    JointPositionWrite,
    JointQuaternionWrite,
    JointVelocityWrite,
)

from .cfg import PegInsertEnvCfg

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "robot_joint_pos": BodyJointPositionQuery(body="base_link"),
    "robot_joint_vel": BodyJointVelocityQuery(body="base_link"),
    "peg_pos": GeomPositionQuery(geom="peg"),
    "peg_quat": GeomQuaternionQuery(geom="peg"),
    "peg_lin_vel": GeomLinearVelocityQuery(geom="peg"),
    "socket_pos": GeomPositionQuery(geom="socket"),
    "hand_pos": SitePositionQuery(site="gripper"),
    "left_finger_pos": SitePositionQuery(site="left_finger_pad"),
    "right_finger_pos": SitePositionQuery(site="right_finger_pad"),
}

_SIM_MODEL_QUERIES = {}


@registry.env("rm65_insert_peg")
@registry.env("peg-insert")
class PegInsertEnv(DirectEnv):
    _cfg: PegInsertEnvCfg

    def __init__(self, cfg: PegInsertEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs=num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "robot_position": BodyJointPositionWrite("base_link"),
                "robot_velocity": BodyJointVelocityWrite("base_link"),
                "peg_translation": JointPositionWrite(("peg_x", "peg_y", "peg_z")),
                "peg_translation_velocity": JointVelocityWrite(("peg_x", "peg_y", "peg_z")),
                "peg_orientation": JointQuaternionWrite(("peg_rot",)),
                "peg_angular_velocity": JointAngularVelocityWrite(("peg_rot",)),
            },
            reset=True,
        )
        self._peg_reset_translation = self._reset_program.buffer("peg_translation")
        self._peg_reset_orientation = self._reset_program.buffer("peg_orientation")[:, 0]
        self.default_joint_pos = self._cfg.init_state.default_joint_pos

        self._action_dim = 7
        self._num_dof_pos = 8
        self._num_dof_vel = 8

        self._obs_dim = 45
        self._action_space = gym.spaces.Box(-1, 1, (self._action_dim,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (self._obs_dim,), dtype=np.float32)

        self._init_dof_pos = np.concatenate([self.default_joint_pos, [0.0]])
        self._init_dof_vel = np.zeros(self._num_dof_vel, dtype=np.float32)

        self.joint_pos_min_limit = np.array(self._cfg.control_config.min_pos, dtype=np.float32)
        self.joint_pos_max_limit = np.array(self._cfg.control_config.max_pos, dtype=np.float32)

        self.action_scale = self._cfg.action_config.action_scale
        self.gripper_closed = self._cfg.action_config.gripper_closed
        self.gripper_open = self._cfg.action_config.gripper_open

        # Episode-scoped task state: full-batch buffers, reset writes the done
        # rows in place.
        self._current_actions = np.zeros((num_envs, self._action_dim), dtype=np.float32)
        self._last_actions = np.zeros_like(self._current_actions)
        self._socket_pos = np.zeros((num_envs, 3), dtype=np.float32)
        self._initial_peg_pos = np.zeros((num_envs, 3), dtype=np.float32)
        self._current_gripper_action = np.zeros(num_envs, dtype=np.float32)
        self._gripper_is_closed = np.zeros(num_envs, dtype=bool)
        self._grasp_success = np.zeros(num_envs, dtype=bool)
        self._phase = np.zeros(num_envs, dtype=np.int32)
        self._prev_hand_to_peg_dist = np.zeros(num_envs, dtype=np.float32)
        self._prev_peg_to_socket_dist = np.zeros(num_envs, dtype=np.float32)
        self._prev_gripper_closure = np.zeros(num_envs, dtype=np.float32)
        self._prev_midpoint_xy_dist = np.zeros(num_envs, dtype=np.float32)
        self._prev_pregrasp_height_error = np.zeros(num_envs, dtype=np.float32)
        self._prev_peg_height = np.zeros(num_envs, dtype=np.float32)
        self._prev_socket_xy_dist = np.zeros(num_envs, dtype=np.float32)
        self._prev_insert_depth = np.zeros(num_envs, dtype=np.float32)
        self._prev_socket_entry_gap = np.full(num_envs, 0.2, dtype=np.float32)
        self._consecutive_capture_steps = np.zeros(num_envs, dtype=np.int32)
        self._consecutive_grasp_steps = np.zeros(num_envs, dtype=np.int32)
        self._consecutive_pregrasp_open_steps = np.zeros(num_envs, dtype=np.int32)

    def _sample_peg_xy(self, socket_xy: np.ndarray) -> np.ndarray:
        peg_init = self._cfg.peg_init_config
        num_reset = socket_xy.shape[0]

        peg_xy = np.empty((num_reset, 2), dtype=np.float32)
        peg_xy[:, 0] = np.random.uniform(peg_init.x_range[0], peg_init.x_range[1], size=(num_reset,))
        peg_xy[:, 1] = np.random.uniform(peg_init.y_range[0], peg_init.y_range[1], size=(num_reset,))

        min_socket_xy_dist = float(getattr(peg_init, "min_socket_xy_dist", 0.0))
        max_socket_xy_dist = float(getattr(peg_init, "max_socket_xy_dist", np.inf))
        max_sample_attempts = int(getattr(peg_init, "max_sample_attempts", 1))

        if min_socket_xy_dist <= 0.0 and not np.isfinite(max_socket_xy_dist):
            return peg_xy

        peg_to_socket_xy_dist = np.linalg.norm(peg_xy - socket_xy, axis=-1)
        valid = (peg_to_socket_xy_dist >= min_socket_xy_dist) & (peg_to_socket_xy_dist <= max_socket_xy_dist)

        attempts = 0
        while not np.all(valid) and attempts < max_sample_attempts:
            invalid = ~valid
            peg_xy[invalid, 0] = np.random.uniform(peg_init.x_range[0], peg_init.x_range[1], size=(invalid.sum(),))
            peg_xy[invalid, 1] = np.random.uniform(peg_init.y_range[0], peg_init.y_range[1], size=(invalid.sum(),))
            peg_to_socket_xy_dist = np.linalg.norm(peg_xy - socket_xy, axis=-1)
            valid = (peg_to_socket_xy_dist >= min_socket_xy_dist) & (peg_to_socket_xy_dist <= max_socket_xy_dist)
            attempts += 1

        if not np.all(valid):
            invalid = ~valid
            fallback_radius = np.random.uniform(
                min_socket_xy_dist,
                max_socket_xy_dist if np.isfinite(max_socket_xy_dist) else min_socket_xy_dist + 0.05,
                size=(invalid.sum(),),
            )
            fallback_angle = np.random.uniform(-np.pi, np.pi, size=(invalid.sum(),))
            peg_xy[invalid, 0] = socket_xy[invalid, 0] + fallback_radius * np.cos(fallback_angle)
            peg_xy[invalid, 1] = socket_xy[invalid, 1] + fallback_radius * np.sin(fallback_angle)
            peg_xy[:, 0] = np.clip(peg_xy[:, 0], peg_init.x_range[0], peg_init.x_range[1])
            peg_xy[:, 1] = np.clip(peg_xy[:, 1], peg_init.y_range[0], peg_init.y_range[1])

        return peg_xy.astype(np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState):
        self._last_actions[:] = self._current_actions
        self._current_actions[:] = actions

        old_joint_pos = self.get_dof_pos(slice(None))[:, : self._action_dim - 1]

        new_joint_pos = actions[:, : self._action_dim - 1] * self.action_scale + old_joint_pos

        inputs = self.sim_data
        peg_pos = inputs["peg_pos"]
        peg_quat = inputs["peg_quat"]

        left_finger_pos = inputs["left_finger_pos"]
        right_finger_pos = inputs["right_finger_pos"]

        finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)

        midpoint_xy_dist = np.linalg.norm(
            finger_midpoint[:, :2] - peg_pos[:, :2],
            axis=-1,
        )

        midpoint_z_offset = finger_midpoint[:, 2] - peg_pos[:, 2]

        peg_axis = quaternion.rotate_vector(
            peg_quat,
            np.array(
                [0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        )

        peg_uprightness = np.clip(
            peg_axis[:, 2],
            -1.0,
            1.0,
        )

        close_allowed = (
            (midpoint_xy_dist < 0.032)
            & (midpoint_z_offset > 0.003)
            & (midpoint_z_offset < 0.05)
            & (peg_uprightness > 0.90)
        )

        gripper_raw = actions[:, -1]

        close_request = gripper_raw > 0.2

        grasp_locked = self._grasp_success

        gripper_is_closed = np.where(
            grasp_locked,
            True,
            close_request & close_allowed,
        )

        gripper_action = np.where(
            gripper_is_closed,
            self.gripper_closed,
            self.gripper_open,
        ).astype(np.float32)

        self._gripper_is_closed[:] = gripper_is_closed
        self._current_gripper_action[:] = gripper_action

        new_pos = np.concatenate(
            [
                new_joint_pos,
                gripper_action[:, None],
            ],
            axis=-1,
        )

        cliped_new_pos = np.clip(
            new_pos,
            self.joint_pos_min_limit,
            self.joint_pos_max_limit,
        )

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

        dof_pos = self.get_dof_pos(rows)
        dof_vel = self.get_dof_vel(rows)
        dof_pos_rel = dof_pos[:, : self._num_dof_pos] - self._init_dof_pos
        dof_vel_rel = dof_vel[:, : self._num_dof_vel]

        inputs = self.sim_data
        peg_pos = inputs["peg_pos"][rows]
        peg_axis = quaternion.rotate_vector(inputs["peg_quat"][rows], np.array([0.0, 0.0, 1.0], dtype=np.float32))

        socket_pos = self._socket_pos

        hand_pos = inputs["hand_pos"][rows]
        left_finger_pos = inputs["left_finger_pos"][rows]
        right_finger_pos = inputs["right_finger_pos"][rows]
        finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)

        hand_to_peg = peg_pos - hand_pos
        hand_to_peg_dist = np.linalg.norm(hand_to_peg, axis=-1, keepdims=True)
        hand_to_peg_dir = hand_to_peg / (hand_to_peg_dist + 1e-6)

        midpoint_to_peg = peg_pos - finger_midpoint
        midpoint_to_peg_dist = np.linalg.norm(midpoint_to_peg, axis=-1, keepdims=True)
        midpoint_to_peg_dir = midpoint_to_peg / (midpoint_to_peg_dist + 1e-6)
        midpoint_z_offset = (finger_midpoint[:, 2] - peg_pos[:, 2])[:, None]
        left_to_peg_dist = np.linalg.norm(peg_pos - left_finger_pos, axis=-1, keepdims=True)
        right_to_peg_dist = np.linalg.norm(peg_pos - right_finger_pos, axis=-1, keepdims=True)
        finger_balance = np.abs(left_to_peg_dist - right_to_peg_dist)
        finger_gap = np.linalg.norm(left_finger_pos - right_finger_pos, axis=-1, keepdims=True)
        gripper_joint_pos = dof_pos[:, 6:7]
        gripper_closure = np.clip(-gripper_joint_pos / max(abs(self.gripper_closed), 1e-6), 0.0, 1.0)

        peg_to_socket = socket_pos - peg_pos
        peg_to_socket_dist = np.linalg.norm(peg_to_socket, axis=-1, keepdims=True)
        peg_to_socket_dir = peg_to_socket / (peg_to_socket_dist + 1e-6)

        gripper_state = self._current_gripper_action
        grasp_success = self._grasp_success.astype(np.float32)
        peg_to_socket_dir = peg_to_socket_dir * grasp_success[:, None]
        peg_to_socket_dist = peg_to_socket_dist * grasp_success[:, None]

        obs = np.concatenate(
            [
                dof_pos_rel,
                dof_vel_rel,
                peg_pos,
                hand_pos,
                hand_to_peg_dir,
                hand_to_peg_dist,
                midpoint_to_peg_dir,
                midpoint_to_peg_dist,
                midpoint_z_offset,
                finger_balance,
                left_to_peg_dist,
                right_to_peg_dist,
                finger_gap,
                gripper_closure,
                peg_axis,
                peg_to_socket_dir,
                peg_to_socket_dist,
                gripper_state[:, None],
                grasp_success[:, None],
            ],
            axis=-1,
        )

        if np.isnan(obs).any():
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        assert obs.shape == (peg_pos.shape[0], self._obs_dim), f"Expected {self._obs_dim}, got {obs.shape}"
        # Publish the computed policy observation on the state. The field
        # name is applied via setattr so static audits can tell this
        # sanctioned observation-stage write apart from the forbidden
        # transition-time writes.
        setattr(state, "obs", NpObs(policy=obs.astype(np.float32)))
        return state

    def compute_transition(self, state: ArrayEnvState):
        """Update reward and termination from the refreshed simulator cache.

        Observations are built separately by :meth:`compute_observation`.
        """
        self.sim_data.execute()

        terminated = self._check_termination(state)
        reward = self._compute_reward(state, terminated)

        state.reward = reward
        state.terminated = terminated

        return state

    def reset(self, env_ids) -> None:
        num_reset = len(env_ids)
        row_ids = np.asarray(env_ids, dtype=np.int64)

        noise_pos = np.random.uniform(
            -self._cfg.init_state.joint_pos_reset_noise_scale,
            self._cfg.init_state.joint_pos_reset_noise_scale,
            self._num_dof_pos,
        )
        robot_dof_pos = self._init_dof_pos + noise_pos

        # Restore the semantic reset targets first and read the socket position
        # at the default pose, matching the legacy reset-then-sample sequence.
        robot_pos = self._reset_program.buffer("robot_position")
        robot_pos[row_ids] = 0.0
        robot_pos[row_ids, : self._num_dof_pos] = self._init_dof_pos
        self._reset_program.buffer("robot_velocity")[row_ids] = 0.0
        self._peg_reset_translation[row_ids] = 0.0
        self._peg_reset_orientation[row_ids] = 0.0
        self._peg_reset_orientation[row_ids, 3] = 1.0  # identity quaternion (xyzw)
        self._reset_program.execute(row_ids)
        self.sim_data.execute(row_ids)

        peg_init = self._cfg.peg_init_config
        socket_pos = self.sim_data["socket_pos"][row_ids].copy()
        peg_xy = self._sample_peg_xy(socket_pos[:, :2])
        peg_x = peg_xy[:, 0]
        peg_y = peg_xy[:, 1]
        peg_z = np.random.uniform(
            peg_init.z_pos - peg_init.z_noise_scale,
            peg_init.z_pos + peg_init.z_noise_scale,
            (num_reset,),
        ).astype(np.float32)

        robot_pos[row_ids] = 0.0
        robot_pos[row_ids, : self._num_dof_pos] = np.asarray(robot_dof_pos, dtype=np.float32)
        self._reset_program.buffer("robot_velocity")[row_ids] = 0.0
        self._peg_reset_translation[row_ids] = np.stack([peg_x, peg_y, peg_z], axis=-1)
        self._peg_reset_orientation[row_ids] = 0.0
        self._peg_reset_orientation[row_ids, 3] = 1.0  # identity quaternion (xyzw)
        self._reset_program.execute(row_ids)
        self.sim_data.execute(row_ids)

        inputs = self.sim_data
        socket_pos = inputs["socket_pos"][row_ids].copy()
        peg_pos = inputs["peg_pos"][row_ids].copy()
        hand_pos = inputs["hand_pos"][row_ids].copy()
        left_finger_pos = inputs["left_finger_pos"][row_ids].copy()
        right_finger_pos = inputs["right_finger_pos"][row_ids].copy()

        finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)
        midpoint_to_peg_xy_dist = np.linalg.norm(peg_pos[:, :2] - finger_midpoint[:, :2], axis=-1)

        midpoint_z_offset = finger_midpoint[:, 2] - peg_pos[:, 2]

        safe_height = self._cfg.peg_config.safe_approach_height
        grasp_height = self._cfg.peg_config.grasp_target_height

        height_blend = np.clip(
            (midpoint_to_peg_xy_dist - 0.025) / (0.08 - 0.025),
            0.0,
            1.0,
        )

        target_pregrasp_height = grasp_height + (safe_height - grasp_height) * height_blend

        pregrasp_height_error = np.abs(midpoint_z_offset - target_pregrasp_height)

        gripper_joint_pos = self.get_dof_pos(row_ids)[:, 6]
        gripper_closure = np.clip(-gripper_joint_pos / max(abs(self.gripper_closed), 1e-6), 0.0, 1.0)
        socket_xy_dist = np.linalg.norm(peg_pos[:, :2] - socket_pos[:, :2], axis=-1)

        # Seed the episode-scoped state buffers for the rows being reset; the
        # step pipeline owns them afterwards (never per-step state).
        self._current_actions[row_ids] = 0.0
        self._last_actions[row_ids] = 0.0
        self._socket_pos[row_ids] = socket_pos
        self._initial_peg_pos[row_ids] = peg_pos
        self._current_gripper_action[row_ids] = 0.0
        self._gripper_is_closed[row_ids] = False
        self._grasp_success[row_ids] = False
        self._phase[row_ids] = 0
        self._prev_hand_to_peg_dist[row_ids] = np.linalg.norm(peg_pos - hand_pos, axis=-1)
        self._prev_peg_to_socket_dist[row_ids] = np.linalg.norm(peg_pos - socket_pos, axis=-1)
        self._prev_gripper_closure[row_ids] = gripper_closure
        self._prev_midpoint_xy_dist[row_ids] = midpoint_to_peg_xy_dist
        self._prev_pregrasp_height_error[row_ids] = pregrasp_height_error
        self._prev_peg_height[row_ids] = peg_pos[:, 2]
        self._prev_socket_xy_dist[row_ids] = socket_xy_dist
        self._prev_insert_depth[row_ids] = 0.0
        self._prev_socket_entry_gap[row_ids] = 0.2
        self._consecutive_capture_steps[row_ids] = 0
        self._consecutive_grasp_steps[row_ids] = 0
        self._consecutive_pregrasp_open_steps[row_ids] = 0

    def _check_termination(self, state: ArrayEnvState):
        inputs = self.sim_data
        peg_pos = inputs["peg_pos"]

        terminated = peg_pos[:, 2] < -0.15

        prev_grasp = self._grasp_success
        if np.any(prev_grasp):
            hand_pos = inputs["hand_pos"]
            hand_to_peg_dist = np.linalg.norm(peg_pos - hand_pos, axis=-1)
            dropped_after_grasp = prev_grasp & (hand_to_peg_dist > 0.12) & (peg_pos[:, 2] < 0.09)
            terminated = np.logical_or(terminated, dropped_after_grasp)

        joint_vel = self.get_dof_vel(slice(None))
        terminated = np.logical_or(terminated, np.abs(joint_vel).max(axis=-1) > 30)

        peg_vel = inputs["peg_lin_vel"]
        terminated = np.logical_or(terminated, np.abs(peg_vel).max(axis=-1) > 15)

        return terminated

    def _collect_reward_features(self) -> dict:
        """Collect the current geometry and gripper state shared by all reward terms."""
        inputs = self.sim_data
        peg_pos = inputs["peg_pos"]
        peg_quat = inputs["peg_quat"]
        socket_pos = self._socket_pos
        hand_pos = inputs["hand_pos"]
        left_finger_pos = inputs["left_finger_pos"]
        right_finger_pos = inputs["right_finger_pos"]

        hand_to_peg = peg_pos - hand_pos
        hand_to_peg_dist = np.linalg.norm(hand_to_peg, axis=-1)
        finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)
        midpoint_to_peg = peg_pos - finger_midpoint
        midpoint_xy_dist = np.linalg.norm(midpoint_to_peg[:, :2], axis=-1)
        midpoint_z_offset = finger_midpoint[:, 2] - peg_pos[:, 2]
        left_to_peg_dist = np.linalg.norm(peg_pos - left_finger_pos, axis=-1)
        right_to_peg_dist = np.linalg.norm(peg_pos - right_finger_pos, axis=-1)
        midpoint_to_peg_dist = np.linalg.norm(peg_pos - finger_midpoint, axis=-1)
        finger_balance = np.abs(left_to_peg_dist - right_to_peg_dist)
        finger_gap = np.linalg.norm(left_finger_pos - right_finger_pos, axis=-1)

        safe_height = self._cfg.peg_config.safe_approach_height
        grasp_height = self._cfg.peg_config.grasp_target_height

        height_blend = np.clip(
            (midpoint_xy_dist - 0.025) / (0.08 - 0.025),
            0.0,
            1.0,
        )

        target_pregrasp_height = grasp_height + (safe_height - grasp_height) * height_blend

        pregrasp_height_error = np.abs(midpoint_z_offset - target_pregrasp_height)

        close_quality = (
            (1 - np.tanh(midpoint_xy_dist / 0.02))
            * (1 - np.tanh(pregrasp_height_error / 0.02))
            * (1 - np.tanh(finger_balance / 0.03))
        )

        peg_to_socket = socket_pos - peg_pos
        peg_to_socket_dist = np.linalg.norm(peg_to_socket, axis=-1)
        xy_dist = np.linalg.norm(peg_pos[:, :2] - socket_pos[:, :2], axis=-1)
        socket_top_z = socket_pos[:, 2] + self._cfg.peg_config.socket_depth * 0.5
        peg_bottom_z = peg_pos[:, 2] - self._cfg.peg_config.peg_length * 0.5
        raw_insert_depth = np.clip(socket_top_z - peg_bottom_z, 0.0, self._cfg.peg_config.socket_depth)

        gripper_state = self._current_gripper_action
        is_gripper_command_closed = gripper_state < -0.5
        is_gripper_command_open = gripper_state > -0.3
        gripper_joint_pos = self.get_dof_pos(slice(None))[:, 6]
        gripper_closure = np.clip(-gripper_joint_pos / max(abs(self.gripper_closed), 1e-6), 0.0, 1.0)
        close_commitment = np.clip(
            0.55 * gripper_closure + 0.45 * np.clip((0.10 - finger_gap) / 0.06, 0.0, 1.0),
            0.0,
            1.0,
        )
        actual_gripper_narrow = (gripper_closure > 0.42) | (finger_gap < 0.065)

        initial_peg_pos = self._initial_peg_pos
        lift_height = np.maximum(0.0, peg_pos[:, 2] - initial_peg_pos[:, 2])
        peg_xy_shift = np.linalg.norm(peg_pos[:, :2] - initial_peg_pos[:, :2], axis=-1)
        peg_drop_height = np.maximum(0.0, initial_peg_pos[:, 2] - peg_pos[:, 2])
        peg_axis = quaternion.rotate_vector(peg_quat, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        peg_uprightness = np.clip(peg_axis[:, 2], -1.0, 1.0)
        peg_linear_vel = inputs["peg_lin_vel"]
        peg_speed = np.linalg.norm(peg_linear_vel, axis=-1)

        return {
            "peg_pos": peg_pos,
            "socket_pos": socket_pos,
            "hand_to_peg_dist": hand_to_peg_dist,
            "midpoint_xy_dist": midpoint_xy_dist,
            "midpoint_z_offset": midpoint_z_offset,
            "left_to_peg_dist": left_to_peg_dist,
            "right_to_peg_dist": right_to_peg_dist,
            "midpoint_to_peg_dist": midpoint_to_peg_dist,
            "finger_balance": finger_balance,
            "finger_gap": finger_gap,
            "pregrasp_height_error": pregrasp_height_error,
            "close_quality": close_quality,
            "peg_to_socket_dist": peg_to_socket_dist,
            "xy_dist": xy_dist,
            "socket_top_z": socket_top_z,
            "peg_bottom_z": peg_bottom_z,
            "raw_insert_depth": raw_insert_depth,
            "gripper_state": gripper_state,
            "is_gripper_command_closed": is_gripper_command_closed,
            "is_gripper_command_open": is_gripper_command_open,
            "gripper_closure": gripper_closure,
            "close_commitment": close_commitment,
            "actual_gripper_narrow": actual_gripper_narrow,
            "lift_height": lift_height,
            "peg_xy_shift": peg_xy_shift,
            "peg_drop_height": peg_drop_height,
            "peg_uprightness": peg_uprightness,
            "peg_linear_vel": peg_linear_vel,
            "peg_speed": peg_speed,
        }

    def _update_reward_progress(self, features: dict) -> dict:
        """Update one-step progress signals that depend on the previous simulator state."""
        # Snapshot the pre-step grasp state: _update_task_tracking overwrites
        # the buffer in place, and downstream reads (first_grasp,
        # pregrasp_open_active) must still see the prior values.
        prev_grasp = self._grasp_success.copy()

        gripper_closure_progress = features["gripper_closure"] - self._prev_gripper_closure
        self._prev_gripper_closure[:] = features["gripper_closure"]

        self._prev_hand_to_peg_dist[:] = features["hand_to_peg_dist"]

        xy_progress = self._prev_midpoint_xy_dist - features["midpoint_xy_dist"]
        self._prev_midpoint_xy_dist[:] = features["midpoint_xy_dist"]

        height_progress = self._prev_pregrasp_height_error - features["pregrasp_height_error"]
        self._prev_pregrasp_height_error[:] = features["pregrasp_height_error"]

        peg_height_progress = features["peg_pos"][:, 2] - self._prev_peg_height
        self._prev_peg_height[:] = features["peg_pos"][:, 2]

        self._prev_peg_to_socket_dist[:] = features["peg_to_socket_dist"]

        xy_socket_progress = self._prev_socket_xy_dist - features["xy_dist"]
        self._prev_socket_xy_dist[:] = features["xy_dist"]

        return {
            "prev_grasp": prev_grasp,
            "gripper_closure_progress": gripper_closure_progress,
            "xy_progress": xy_progress,
            "height_progress": height_progress,
            "peg_height_progress": peg_height_progress,
            "xy_socket_progress": xy_socket_progress,
        }

    def _update_task_tracking(self, features: dict, progress: dict) -> dict:
        """Update reward-related task phase state such as grasp, transport, and insertion."""
        prev_grasp = progress["prev_grasp"]

        pregrasp_ready = (
            (features["midpoint_xy_dist"] < 0.032)
            & (features["midpoint_z_offset"] > 0.003)
            & (features["midpoint_z_offset"] < 0.05)
            & (features["finger_balance"] < 0.055)
        )
        grasp_channel_ready = (
            (features["midpoint_xy_dist"] < 0.028)
            & (features["midpoint_z_offset"] > -0.002)
            & (features["midpoint_z_offset"] < 0.04)
            & (features["left_to_peg_dist"] < 0.065)
            & (features["right_to_peg_dist"] < 0.10)
            & (features["finger_balance"] < 0.05)
        )
        grasp_candidate = (
            features["is_gripper_command_closed"]
            & features["actual_gripper_narrow"]
            & grasp_channel_ready
            & (features["peg_uprightness"] > 0.9)
        )

        consecutive_capture = np.where(grasp_candidate, self._consecutive_capture_steps + 1, 0)
        self._consecutive_capture_steps[:] = consecutive_capture

        grasp_candidate = (
            features["is_gripper_command_closed"]
            & features["actual_gripper_narrow"]
            & grasp_channel_ready
            & (features["peg_uprightness"] > 0.9)
        )

        confirmed_grasp = consecutive_capture >= 3

        ever_grasped = prev_grasp | confirmed_grasp

        self._grasp_success[:] = ever_grasped

        is_grasping = (
            ever_grasped
            & features["is_gripper_command_closed"]
            & features["actual_gripper_narrow"]
            & (features["midpoint_to_peg_dist"] < 0.07)
        )

        consecutive_grasp = np.where(
            is_grasping,
            self._consecutive_grasp_steps + 1,
            0,
        )

        self._consecutive_grasp_steps[:] = consecutive_grasp

        pregrasp_open_active = pregrasp_ready & features["is_gripper_command_open"] & (~prev_grasp)
        consecutive_pregrasp_open = np.where(pregrasp_open_active, self._consecutive_pregrasp_open_steps + 1, 0)
        self._consecutive_pregrasp_open_steps[:] = consecutive_pregrasp_open

        insert_ready = ever_grasped & (features["xy_dist"] < 0.008) & (features["peg_uprightness"] > 0.97)
        insert_depth = np.where(insert_ready, features["raw_insert_depth"], 0.0)
        insert_depth_progress = insert_depth - self._prev_insert_depth
        self._prev_insert_depth[:] = insert_depth

        socket_entry_gap = np.maximum(0.0, features["peg_bottom_z"] - features["socket_top_z"])
        socket_entry_gap_progress = self._prev_socket_entry_gap - socket_entry_gap
        self._prev_socket_entry_gap[:] = socket_entry_gap

        touch_socket = (
            ever_grasped & (insert_depth > 0.008) & (features["xy_dist"] < 0.006) & (features["peg_uprightness"] > 0.97)
        )
        success = (
            (insert_depth > self._cfg.peg_config.insertion_threshold)
            & (features["xy_dist"] < self._cfg.peg_config.success_threshold)
            & (features["peg_uprightness"] > 0.98)
        )

        phase = self._phase
        phase[:] = 0
        phase[(~pregrasp_ready) & (~ever_grasped)] = 1
        phase[pregrasp_ready | grasp_channel_ready | ever_grasped] = 2
        phase[touch_socket] = 3
        phase[success] = 4

        return {
            "pregrasp_ready": pregrasp_ready,
            "grasp_channel_ready": grasp_channel_ready,
            "grasp_candidate": grasp_candidate,
            "consecutive_capture": consecutive_capture,
            "is_grasping": is_grasping,
            "consecutive_grasp": consecutive_grasp,
            "consecutive_pregrasp_open": consecutive_pregrasp_open,
            "prev_grasp": prev_grasp,
            "ever_grasped": ever_grasped,
            "first_grasp": ever_grasped & (~prev_grasp),
            "insert_depth": insert_depth,
            "insert_depth_progress": insert_depth_progress,
            "socket_entry_gap": socket_entry_gap,
            "socket_entry_gap_progress": socket_entry_gap_progress,
            "touch_socket": touch_socket,
            "success": success,
        }

    def _compute_pregrasp_rewards(self, reward_cfg, features: dict, progress: dict, tracking: dict) -> dict:
        """Reward reaching the peg and shaping a stable pre-grasp before closing."""
        num_envs = features["xy_dist"].shape[0]

        xy_align_reward = reward_cfg.reach_weight * 0.3 * (1 - np.tanh(features["midpoint_xy_dist"] / 0.05))

        height_align_reward = (
            reward_cfg.hand_approach_bonus * 0.3 * (1 - np.tanh(features["pregrasp_height_error"] / 0.04))
        )

        xy_progress_reward = (
            reward_cfg.approach_weight * 0.6 * np.tanh(np.clip(progress["xy_progress"], -0.02, 0.02) / 0.01)
        )

        height_progress_reward = (
            reward_cfg.hand_approach_bonus * 1.2 * np.tanh(np.clip(progress["height_progress"], -0.02, 0.02) / 0.01)
        )

        find_peg_reward = np.where(
            features["is_gripper_command_open"] & (~tracking["ever_grasped"]),
            xy_align_reward + height_align_reward + xy_progress_reward + height_progress_reward,
            0.0,
        )

        hover_above_peg_penalty = np.zeros(num_envs, dtype=np.float32)

        # hover_above_peg_mask = (
        #     features["is_gripper_command_open"]
        #     & (~tracking["ever_grasped"])
        #     & (features["midpoint_xy_dist"] < 0.04)
        #     & (features["midpoint_z_offset"] > 0.045)
        # )
        # hover_above_peg_penalty[hover_above_peg_mask] = (
        #     -reward_cfg.hand_approach_bonus
        #     * 1.8
        #     * np.tanh((features["midpoint_z_offset"][hover_above_peg_mask] - 0.045) / 0.02)
        # )

        pregrasp_open_reward = np.zeros(num_envs, dtype=np.float32)
        pregrasp_open_mask = (
            tracking["pregrasp_ready"]
            & (~tracking["grasp_channel_ready"])
            & features["is_gripper_command_open"]
            & (~tracking["ever_grasped"])
        )
        pregrasp_open_reward[pregrasp_open_mask] = 0.0

        pregrasp_stall_penalty = np.zeros(num_envs, dtype=np.float32)
        pregrasp_stall_mask = pregrasp_open_mask & (tracking["consecutive_pregrasp_open"] > 4)
        pregrasp_stall_penalty[pregrasp_stall_mask] = (
            -4.0
            - 10.0 * np.tanh((tracking["consecutive_pregrasp_open"][pregrasp_stall_mask] - 4).astype(np.float32) / 6.0)
            - reward_cfg.hand_approach_bonus * 0.8 * features["close_quality"][pregrasp_stall_mask]
        )

        ready_but_open_penalty = np.zeros(num_envs, dtype=np.float32)
        ready_but_open_mask = (
            tracking["grasp_channel_ready"] & features["is_gripper_command_open"] & (~tracking["ever_grasped"])
        )
        ready_but_open_penalty[ready_but_open_mask] = (
            -24.0 - reward_cfg.gripper_close_bonus * 6.0 * features["close_quality"][ready_but_open_mask]
        )

        close_in_grasp_zone_reward = np.zeros(num_envs, dtype=np.float32)
        close_zone_mask = (
            tracking["pregrasp_ready"] & features["is_gripper_command_closed"] & (~tracking["ever_grasped"])
        )
        close_in_grasp_zone_reward[close_zone_mask] = 42.0 + reward_cfg.gripper_close_bonus * 14.0 * features[
            "close_quality"
        ][close_zone_mask] * (0.3 + 0.7 * features["close_commitment"][close_zone_mask])

        premature_close_penalty = np.zeros(num_envs, dtype=np.float32)
        premature_close_mask = (
            features["is_gripper_command_closed"] & (~tracking["pregrasp_ready"]) & (~tracking["ever_grasped"])
        )
        premature_close_penalty[premature_close_mask] = (
            -20.0
            - 10.0 * np.tanh(features["midpoint_xy_dist"][premature_close_mask] / 0.04)
            - 8.0 * np.tanh(np.maximum(0.0, features["midpoint_z_offset"][premature_close_mask] - 0.045) / 0.02)
        )

        gripper_close_progress_reward = np.zeros(num_envs, dtype=np.float32)
        close_progress_mask = (
            features["is_gripper_command_closed"] & tracking["pregrasp_ready"] & (~tracking["ever_grasped"])
        )
        gripper_close_progress_reward[close_progress_mask] = reward_cfg.gripper_close_bonus * (
            4.0 * features["close_commitment"][close_progress_mask]
            + 9.0 * np.tanh(np.clip(progress["gripper_closure_progress"][close_progress_mask], 0.0, 0.12) / 0.03)
        )

        grasp_channel_reward = np.zeros(num_envs, dtype=np.float32)
        grasp_channel_mask = features["is_gripper_command_closed"] & tracking["grasp_channel_ready"]
        grasp_channel_reward[grasp_channel_mask] = 40.0 + reward_cfg.gripper_close_bonus * 12.0 * features[
            "close_quality"
        ][grasp_channel_mask] * (0.2 + 0.8 * features["close_commitment"][grasp_channel_mask])

        stable_close_reward = np.zeros(num_envs, dtype=np.float32)
        stable_close_mask = grasp_channel_mask & (~tracking["ever_grasped"])
        stable_close_reward[stable_close_mask] = (
            reward_cfg.grasp_weight
            * 2.4
            * features["close_quality"][stable_close_mask]
            * features["close_commitment"][stable_close_mask]
            * np.clip(features["peg_uprightness"][stable_close_mask], 0.0, 1.0)
            * (1 - np.tanh(features["peg_xy_shift"][stable_close_mask] / 0.01))
        )

        return {
            "find_peg_reward": find_peg_reward,
            "hover_above_peg_penalty": hover_above_peg_penalty,
            "pregrasp_open_reward": pregrasp_open_reward,
            "pregrasp_stall_penalty": pregrasp_stall_penalty,
            "ready_but_open_penalty": ready_but_open_penalty,
            "close_in_grasp_zone_reward": close_in_grasp_zone_reward,
            "premature_close_penalty": premature_close_penalty,
            "gripper_close_progress_reward": gripper_close_progress_reward,
            "grasp_channel_reward": grasp_channel_reward,
            "stable_close_reward": stable_close_reward,
        }

    def _compute_grasp_transport_rewards(self, reward_cfg, features: dict, progress: dict, tracking: dict) -> dict:
        """Reward successful capture, lifting, and transporting the peg toward the socket."""
        num_envs = features["xy_dist"].shape[0]

        capture_reward = np.zeros(num_envs, dtype=np.float32)
        capture_mask = tracking["grasp_candidate"] & (~tracking["ever_grasped"])
        capture_reward[capture_mask] = reward_cfg.grasp_weight * (
            1.2 * features["close_quality"][capture_mask] * (0.35 + 0.65 * features["close_commitment"][capture_mask])
            + 0.8 * np.tanh(tracking["consecutive_capture"][capture_mask].astype(np.float32) / 2.0)
        )

        lift_intent_reward = np.zeros(num_envs, dtype=np.float32)
        lift_intent_reward[capture_mask] = (
            reward_cfg.lift_weight
            * 2.2
            * np.tanh(np.clip(progress["peg_height_progress"][capture_mask], 0.0, 0.008) / 0.003)
        )

        grasp_reward = np.zeros(num_envs, dtype=np.float32)
        grasp_reward[tracking["ever_grasped"]] = reward_cfg.grasp_weight * 0.4 * features["close_commitment"][
            tracking["ever_grasped"]
        ] + reward_cfg.lift_weight * 0.7 * np.tanh(
            np.minimum(features["lift_height"][tracking["ever_grasped"]], 0.03) / 0.02
        )
        grasp_reward[tracking["first_grasp"]] += reward_cfg.grasp_success_bonus

        stable_grasp_mask = tracking["ever_grasped"] & (tracking["consecutive_grasp"] > 5)
        grasp_reward[stable_grasp_mask] += 10.0

        grip_hold_reward = np.zeros(num_envs, dtype=np.float32)
        hold_mask = tracking["ever_grasped"] & features["is_gripper_command_closed"]
        grip_hold_reward[hold_mask] = reward_cfg.gripper_close_bonus * 1.2 * features["close_commitment"][
            hold_mask
        ] + reward_cfg.grasp_weight * 0.2 * (1 - np.tanh(features["midpoint_to_peg_dist"][hold_mask] / 0.025))

        release_after_grasp_penalty = np.zeros(num_envs, dtype=np.float32)
        release_mask = tracking["ever_grasped"] & (
            (~features["is_gripper_command_closed"]) | (features["close_commitment"] < 0.35)
        )
        release_after_grasp_penalty[release_mask] = -80.0 - reward_cfg.grasp_weight * 2.0 * np.tanh(
            np.maximum(0.0, 0.35 - features["close_commitment"][release_mask]) / 0.15
        )

        carry_upright_reward = np.zeros(num_envs, dtype=np.float32)
        carry_upright_reward[tracking["ever_grasped"]] = (
            reward_cfg.orientation_weight
            * 2.5
            * np.clip(
                (features["peg_uprightness"][tracking["ever_grasped"]] - 0.9) / 0.1,
                0.0,
                1.0,
            )
        )

        carry_tilt_penalty = np.zeros(num_envs, dtype=np.float32)
        carry_tilt_penalty[tracking["ever_grasped"]] = (
            -reward_cfg.orientation_weight
            * 3.0
            * np.tanh(np.maximum(0.0, 0.9 - features["peg_uprightness"][tracking["ever_grasped"]]) / 0.08)
        )

        carry_target_z = features["socket_top_z"] + self._cfg.peg_config.peg_length * 0.5 + 0.01
        carry_height_error = np.abs(features["peg_pos"][:, 2] - carry_target_z)
        carry_height_reward = np.zeros(num_envs, dtype=np.float32)
        carry_height_mask = tracking["ever_grasped"] & (features["xy_dist"] > 0.045)

        carry_height_reward[carry_height_mask] = (
            reward_cfg.lift_weight * 1.4 * (1 - np.tanh(carry_height_error[carry_height_mask] / 0.03))
        )

        move_to_socket_reward = np.zeros(num_envs, dtype=np.float32)
        move_to_socket_reward[tracking["ever_grasped"]] = (
            reward_cfg.peg_to_socket_approach_weight
            * 2.2
            * (1 - np.tanh(features["xy_dist"][tracking["ever_grasped"]] / 0.03))
        )
        move_to_socket_reward[tracking["ever_grasped"]] += (
            reward_cfg.peg_to_socket_approach_weight
            * 3.2
            * np.tanh(np.clip(progress["xy_socket_progress"][tracking["ever_grasped"]], -0.02, 0.02) / 0.006)
        )

        transport_stall_penalty = np.zeros(num_envs, dtype=np.float32)
        transport_stall_mask = (
            tracking["ever_grasped"] & (features["xy_dist"] > 0.04) & (progress["xy_socket_progress"] < 0.0015)
        )
        transport_stall_penalty[transport_stall_mask] = (
            -reward_cfg.peg_to_socket_approach_weight
            * 1.4
            * np.tanh((features["xy_dist"][transport_stall_mask] - 0.04) / 0.03)
        )

        lift_reward = np.zeros(num_envs, dtype=np.float32)
        lift_stage_mask = (
            features["is_gripper_command_closed"]
            & (tracking["pregrasp_ready"] | tracking["grasp_channel_ready"])
            & (~tracking["ever_grasped"])
        )
        lift_reward[lift_stage_mask] = (
            reward_cfg.lift_weight * 4.5 * np.tanh(features["lift_height"][lift_stage_mask] / 0.03)
        )

        carry_lift_reward = np.zeros(num_envs, dtype=np.float32)
        carry_lift_mask = tracking["ever_grasped"] & (features["lift_height"] < 0.035)
        carry_lift_reward[carry_lift_mask] = (
            reward_cfg.lift_weight
            * 2.2
            * (1 - np.tanh(np.maximum(0.0, 0.035 - features["lift_height"][carry_lift_mask]) / 0.02))
        )

        lift_progress_reward = np.zeros(num_envs, dtype=np.float32)
        lift_progress_reward[lift_stage_mask] = (
            reward_cfg.lift_weight
            * 5.0
            * np.tanh(np.clip(progress["peg_height_progress"][lift_stage_mask], -0.01, 0.01) / 0.004)
        )

        move_away_penalty = np.where(
            (progress["xy_socket_progress"] < 0) & tracking["ever_grasped"],
            -reward_cfg.peg_to_socket_approach_weight
            * np.tanh(np.clip(-progress["xy_socket_progress"], 0.0, 0.02) / 0.01),
            0.0,
        )

        return {
            "capture_reward": capture_reward,
            "lift_intent_reward": lift_intent_reward,
            "grasp_reward": grasp_reward,
            "grip_hold_reward": grip_hold_reward,
            "release_after_grasp_penalty": release_after_grasp_penalty,
            "carry_upright_reward": carry_upright_reward,
            "carry_tilt_penalty": carry_tilt_penalty,
            "carry_height_reward": carry_height_reward,
            "move_to_socket_reward": move_to_socket_reward,
            "transport_stall_penalty": transport_stall_penalty,
            "lift_reward": lift_reward,
            "carry_lift_reward": carry_lift_reward,
            "lift_progress_reward": lift_progress_reward,
            "move_away_penalty": move_away_penalty,
        }

    def _compute_insert_rewards(self, reward_cfg, features: dict, tracking: dict) -> dict:
        """Reward precise socket approach and insertion once the peg is grasped."""
        num_envs = features["xy_dist"].shape[0]
        aligned_for_insert = (
            tracking["ever_grasped"] & (features["xy_dist"] < 0.01) & (features["peg_uprightness"] > 0.96)
        )

        insert_progress_reward = np.zeros(num_envs, dtype=np.float32)
        insert_progress_reward[aligned_for_insert] = (
            reward_cfg.insert_weight
            * 2.2
            * np.tanh(np.clip(tracking["insert_depth_progress"][aligned_for_insert], -0.01, 0.01) / 0.003)
        )

        preinsert_descend_reward = np.zeros(num_envs, dtype=np.float32)
        preinsert_descend_reward[aligned_for_insert] = (
            reward_cfg.insert_weight * 0.15 * (1 - np.tanh(tracking["socket_entry_gap"][aligned_for_insert] / 0.015))
        )

        descend_progress_reward = np.zeros(num_envs, dtype=np.float32)
        descend_progress_reward[aligned_for_insert] = (
            reward_cfg.insert_weight
            * 2.4
            * np.tanh(np.clip(tracking["socket_entry_gap_progress"][aligned_for_insert], -0.02, 0.02) / 0.004)
        )

        hover_near_socket_penalty = np.zeros(num_envs, dtype=np.float32)
        hover_near_socket_mask = (
            tracking["ever_grasped"] & (features["xy_dist"] < 0.05) & (tracking["socket_entry_gap"] > 0.03)
        )
        hover_near_socket_penalty[hover_near_socket_mask] = (
            -reward_cfg.insert_weight
            * 0.7
            * np.tanh((tracking["socket_entry_gap"][hover_near_socket_mask] - 0.03) / 0.03)
        )

        high_hold_near_socket_penalty = np.zeros(num_envs, dtype=np.float32)
        high_hold_mask = (
            tracking["ever_grasped"]
            & (features["xy_dist"] < 0.06)
            & (features["peg_pos"][:, 2] > features["socket_top_z"] + 0.09)
        )
        high_hold_near_socket_penalty[high_hold_mask] = (
            -reward_cfg.lift_weight
            * 2.0
            * np.tanh(
                (features["peg_pos"][high_hold_mask, 2] - (features["socket_top_z"][high_hold_mask] + 0.09)) / 0.025
            )
        )

        premature_descend_penalty = np.zeros(num_envs, dtype=np.float32)
        premature_descend_mask = (
            tracking["ever_grasped"]
            & (features["xy_dist"] > 0.028)
            & (features["peg_bottom_z"] < features["socket_top_z"] + 0.015)
        )
        premature_descend_penalty[premature_descend_mask] = (
            -reward_cfg.align_weight
            * 1.2
            * np.tanh(
                (
                    features["socket_top_z"][premature_descend_mask]
                    + 0.015
                    - features["peg_bottom_z"][premature_descend_mask]
                )
                / 0.02
            )
        )

        touch_reward = np.zeros(num_envs, dtype=np.float32)
        touch_reward[tracking["touch_socket"]] = 120.0

        insert_reward = tracking["insert_depth"] * reward_cfg.insert_weight * 6.0
        depth_insert_bonus = reward_cfg.depth_insert_bonus * np.tanh(tracking["insert_depth"] / 0.01)

        align_reward = reward_cfg.align_weight * (1 - np.tanh(features["xy_dist"] / 0.015))
        align_reward *= (tracking["ever_grasped"] | tracking["touch_socket"] | tracking["success"]).astype(np.float32)

        precision_reward = np.zeros(num_envs, dtype=np.float32)
        precision_reward[tracking["touch_socket"]] = reward_cfg.precision_weight * (
            1 - np.tanh(features["xy_dist"][tracking["touch_socket"]] / 0.008)
        )

        return {
            "insert_progress_reward": insert_progress_reward,
            "preinsert_descend_reward": preinsert_descend_reward,
            "descend_progress_reward": descend_progress_reward,
            "hover_near_socket_penalty": hover_near_socket_penalty,
            "high_hold_near_socket_penalty": high_hold_near_socket_penalty,
            "premature_descend_penalty": premature_descend_penalty,
            "touch_reward": touch_reward,
            "insert_reward": insert_reward,
            "depth_insert_bonus": depth_insert_bonus,
            "align_reward": align_reward,
            "precision_reward": precision_reward,
            "success_bonus": reward_cfg.success_bonus * tracking["success"].astype(np.float32),
        }

    def _compute_safety_penalties(self, reward_cfg, features: dict, tracking: dict) -> dict:
        """Penalize unstable contacts, drops, and excessive shaking throughout the task."""
        num_envs = features["xy_dist"].shape[0]

        peg_tilt_excess = np.maximum(0.0, 0.93 - features["peg_uprightness"])
        peg_shift_excess = np.maximum(0.0, features["peg_xy_shift"] - 0.006)
        peg_drop_excess = np.maximum(0.0, features["peg_drop_height"] - 0.002)
        peg_speed_excess = np.maximum(0.0, features["peg_speed"] - 0.18)

        pregrasp_knock_penalty = np.zeros(num_envs, dtype=np.float32)
        pregrasp_knock_mask = (~tracking["ever_grasped"]) & (
            (features["midpoint_xy_dist"] < 0.05) | tracking["pregrasp_ready"] | tracking["grasp_channel_ready"]
        )
        pregrasp_knock_penalty[pregrasp_knock_mask] = (
            -reward_cfg.orientation_weight * 1.8 * np.tanh(peg_tilt_excess[pregrasp_knock_mask] / 0.05)
            - reward_cfg.precision_weight * 0.8 * np.tanh(peg_shift_excess[pregrasp_knock_mask] / 0.015)
            - reward_cfg.lift_weight * 1.0 * np.tanh(peg_drop_excess[pregrasp_knock_mask] / 0.01)
            - reward_cfg.anti_shake_weight * 1800.0 * np.tanh(peg_speed_excess[pregrasp_knock_mask] / 0.25)
        )

        drop_penalty = np.zeros(num_envs, dtype=np.float32)
        peg_dropped = tracking["ever_grasped"] & (~tracking["is_grasping"])
        drop_penalty[peg_dropped] = -100.0

        carry_shake_penalty = np.zeros(num_envs, dtype=np.float32)

        carry_mask = tracking["ever_grasped"] & (~tracking["success"])

        carry_speed_limit = np.where(
            features["xy_dist"] < 0.06,
            0.04,
            0.12,
        )

        carry_speed_excess = np.maximum(
            0.0,
            features["peg_speed"] - carry_speed_limit,
        )

        carry_shake_penalty[carry_mask] = -35.0 * np.tanh(carry_speed_excess[carry_mask] / 0.12)

        return {
            "pregrasp_knock_penalty": pregrasp_knock_penalty,
            "drop_penalty": drop_penalty,
            "carry_shake_penalty": carry_shake_penalty,
        }

    def _compute_reward(self, state: ArrayEnvState, terminated: np.ndarray):
        # 1) Gather geometry and actuation features shared across all reward terms.
        reward_cfg = self._cfg.reward_config
        features = self._collect_reward_features()

        # 2) Update history-dependent progress signals and the task phase tracker.
        progress = self._update_reward_progress(features)
        tracking = self._update_task_tracking(features, progress)

        # 3) Compute reward components by task stage. The final sum order below
        # intentionally mirrors the pre-refactor implementation for reproducibility.
        pregrasp_rewards = self._compute_pregrasp_rewards(reward_cfg, features, progress, tracking)
        grasp_transport_rewards = self._compute_grasp_transport_rewards(reward_cfg, features, progress, tracking)
        insert_rewards = self._compute_insert_rewards(reward_cfg, features, tracking)
        safety_penalties = self._compute_safety_penalties(reward_cfg, features, tracking)

        action_diff_sq = np.sum(np.square(self._current_actions - self._last_actions), axis=-1)
        joint_vel_sq = np.sum(np.square(self.get_dof_vel(slice(None))[:, : self._num_dof_vel]), axis=1)
        peg_vel_sq = np.sum(np.square(features["peg_linear_vel"]), axis=-1)

        reward_component_groups = (
            (
                pregrasp_rewards,
                (
                    "find_peg_reward",
                    "hover_above_peg_penalty",
                    "pregrasp_open_reward",
                    "pregrasp_stall_penalty",
                    "ready_but_open_penalty",
                    "close_in_grasp_zone_reward",
                    "gripper_close_progress_reward",
                    "grasp_channel_reward",
                    "stable_close_reward",
                    "premature_close_penalty",
                ),
            ),
            (
                grasp_transport_rewards,
                (
                    "capture_reward",
                    "lift_intent_reward",
                    "grasp_reward",
                    "grip_hold_reward",
                    "carry_upright_reward",
                    "carry_height_reward",
                    "move_to_socket_reward",
                    "transport_stall_penalty",
                    "lift_reward",
                    "carry_lift_reward",
                    "lift_progress_reward",
                    "release_after_grasp_penalty",
                    "carry_tilt_penalty",
                    "move_away_penalty",
                ),
            ),
            (
                insert_rewards,
                (
                    "insert_progress_reward",
                    "preinsert_descend_reward",
                    "descend_progress_reward",
                    "hover_near_socket_penalty",
                    "high_hold_near_socket_penalty",
                    "premature_descend_penalty",
                    "touch_reward",
                    "insert_reward",
                    "depth_insert_bonus",
                    "align_reward",
                    "precision_reward",
                    "success_bonus",
                ),
            ),
            (
                safety_penalties,
                ("pregrasp_knock_penalty", "drop_penalty", "carry_shake_penalty"),
            ),
        )
        reward = np.zeros(self._num_envs, dtype=np.float32)
        reward_details = {}
        for reward_dict, reward_keys in reward_component_groups:
            for reward_key in reward_keys:
                reward_component = reward_dict[reward_key]
                reward += reward_component
                reward_details[reward_key] = reward_component
        action_penalty = -reward_cfg.action_penalty_weight * action_diff_sq
        joint_vel_penalty = -reward_cfg.joint_vel_penalty_weight * joint_vel_sq
        anti_shake_penalty = -reward_cfg.anti_shake_weight * peg_vel_sq
        termination_penalty = np.where(terminated, -50.0, 0.0)
        reward_details["action_penalty"] = action_penalty
        reward_details["joint_vel_penalty"] = joint_vel_penalty
        reward_details["anti_shake_penalty"] = anti_shake_penalty
        reward_details["termination_penalty"] = termination_penalty
        reward += action_penalty + joint_vel_penalty + anti_shake_penalty + termination_penalty

        reward = np.clip(reward, -100.0, 3000.0)
        reward_details["total"] = reward.astype(np.float32)
        state.reward_terms = reward_details

        return reward.astype(np.float32)

    def get_dof_pos(self, rows):
        return self.sim_data["robot_joint_pos"][rows]

    def get_dof_vel(self, rows):
        return self.sim_data["robot_joint_vel"][rows]
