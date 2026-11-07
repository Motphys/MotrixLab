# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState, EnvObs
from motrix_env_core.direct.env import DirectEnv
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

_Z_AXIS = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)


def _rotate_vector(quats: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by unit (x, y, z, w) quaternions: v + 2*im×(im×v − w·v)."""
    im = quats[..., :3]
    w = quats[..., 3]
    vec = vector.expand_as(im)
    term = torch.linalg.cross(im, vec) - w.unsqueeze(-1) * vec
    return vector + 2.0 * torch.linalg.cross(im, term)


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
        self.default_joint_pos = torch.as_tensor(self._cfg.init_state.default_joint_pos, dtype=torch.float32)

        self._action_dim = 7
        self._num_dof_pos = 8
        self._num_dof_vel = 8

        self._obs_dim = 45
        self._action_space = gym.spaces.Box(-1, 1, (self._action_dim,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (self._obs_dim,), dtype=np.float32)

        self._init_dof_pos = torch.cat([self.default_joint_pos, torch.zeros(1, dtype=torch.float32)])
        self._init_dof_vel = torch.zeros(self._num_dof_vel, dtype=torch.float32)

        self.joint_pos_min_limit = torch.tensor(self._cfg.control_config.min_pos, dtype=torch.float32)
        self.joint_pos_max_limit = torch.tensor(self._cfg.control_config.max_pos, dtype=torch.float32)

        self.action_scale = self._cfg.action_config.action_scale
        self.gripper_closed = self._cfg.action_config.gripper_closed
        self.gripper_open = self._cfg.action_config.gripper_open

        # Episode-scoped task state: full-batch buffers, reset writes the done
        # rows in place.
        self._current_actions = torch.zeros(num_envs, self._action_dim, dtype=torch.float32)
        self._last_actions = torch.zeros_like(self._current_actions)
        self._socket_pos = torch.zeros(num_envs, 3, dtype=torch.float32)
        self._initial_peg_pos = torch.zeros(num_envs, 3, dtype=torch.float32)
        self._current_gripper_action = torch.zeros(num_envs, dtype=torch.float32)
        self._gripper_is_closed = torch.zeros(num_envs, dtype=torch.bool)
        self._grasp_success = torch.zeros(num_envs, dtype=torch.bool)
        self._phase = torch.zeros(num_envs, dtype=torch.int32)
        self._prev_hand_to_peg_dist = torch.zeros(num_envs, dtype=torch.float32)
        self._prev_peg_to_socket_dist = torch.zeros(num_envs, dtype=torch.float32)
        self._prev_gripper_closure = torch.zeros(num_envs, dtype=torch.float32)
        self._prev_midpoint_xy_dist = torch.zeros(num_envs, dtype=torch.float32)
        self._prev_pregrasp_height_error = torch.zeros(num_envs, dtype=torch.float32)
        self._prev_peg_height = torch.zeros(num_envs, dtype=torch.float32)
        self._prev_socket_xy_dist = torch.zeros(num_envs, dtype=torch.float32)
        self._prev_insert_depth = torch.zeros(num_envs, dtype=torch.float32)
        self._prev_socket_entry_gap = torch.full((num_envs,), 0.2, dtype=torch.float32)
        self._consecutive_capture_steps = torch.zeros(num_envs, dtype=torch.int32)
        self._consecutive_grasp_steps = torch.zeros(num_envs, dtype=torch.int32)
        self._consecutive_pregrasp_open_steps = torch.zeros(num_envs, dtype=torch.int32)

    def _sample_peg_xy(self, socket_xy: torch.Tensor) -> torch.Tensor:
        peg_init = self._cfg.peg_init_config
        num_reset = socket_xy.shape[0]

        def _uniform(low: float, high: float, size) -> torch.Tensor:
            return torch.rand(size, dtype=torch.float32) * (high - low) + low

        peg_xy = torch.empty(num_reset, 2, dtype=torch.float32)
        peg_xy[:, 0] = _uniform(peg_init.x_range[0], peg_init.x_range[1], (num_reset,))
        peg_xy[:, 1] = _uniform(peg_init.y_range[0], peg_init.y_range[1], (num_reset,))

        min_socket_xy_dist = float(getattr(peg_init, "min_socket_xy_dist", 0.0))
        max_socket_xy_dist = float(getattr(peg_init, "max_socket_xy_dist", np.inf))
        max_sample_attempts = int(getattr(peg_init, "max_sample_attempts", 1))

        if min_socket_xy_dist <= 0.0 and not np.isfinite(max_socket_xy_dist):
            return peg_xy

        peg_to_socket_xy_dist = torch.linalg.norm(peg_xy - socket_xy, dim=-1)
        valid = (peg_to_socket_xy_dist >= min_socket_xy_dist) & (peg_to_socket_xy_dist <= max_socket_xy_dist)

        attempts = 0
        while not bool(valid.all()) and attempts < max_sample_attempts:
            invalid = ~valid
            num_invalid = int(invalid.sum())
            peg_xy[invalid, 0] = _uniform(peg_init.x_range[0], peg_init.x_range[1], (num_invalid,))
            peg_xy[invalid, 1] = _uniform(peg_init.y_range[0], peg_init.y_range[1], (num_invalid,))
            peg_to_socket_xy_dist = torch.linalg.norm(peg_xy - socket_xy, dim=-1)
            valid = (peg_to_socket_xy_dist >= min_socket_xy_dist) & (peg_to_socket_xy_dist <= max_socket_xy_dist)
            attempts += 1

        if not bool(valid.all()):
            invalid = ~valid
            num_invalid = int(invalid.sum())
            fallback_radius = _uniform(
                min_socket_xy_dist,
                max_socket_xy_dist if np.isfinite(max_socket_xy_dist) else min_socket_xy_dist + 0.05,
                (num_invalid,),
            )
            fallback_angle = _uniform(-np.pi, np.pi, (num_invalid,))
            peg_xy[invalid, 0] = socket_xy[invalid, 0] + fallback_radius * torch.cos(fallback_angle)
            peg_xy[invalid, 1] = socket_xy[invalid, 1] + fallback_radius * torch.sin(fallback_angle)
            peg_xy[:, 0] = peg_xy[:, 0].clamp(peg_init.x_range[0], peg_init.x_range[1])
            peg_xy[:, 1] = peg_xy[:, 1].clamp(peg_init.y_range[0], peg_init.y_range[1])

        return peg_xy

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState):
        self._last_actions.copy_(self._current_actions)
        self._current_actions.copy_(actions)

        old_joint_pos = self.get_dof_pos(slice(None))[:, : self._action_dim - 1]

        new_joint_pos = actions[:, : self._action_dim - 1] * self.action_scale + old_joint_pos

        inputs = self.sim_data
        peg_pos = inputs["peg_pos"]
        peg_quat = inputs["peg_quat"]

        left_finger_pos = inputs["left_finger_pos"]
        right_finger_pos = inputs["right_finger_pos"]

        finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)

        midpoint_xy_dist = torch.linalg.norm(
            finger_midpoint[:, :2] - peg_pos[:, :2],
            dim=-1,
        )

        midpoint_z_offset = finger_midpoint[:, 2] - peg_pos[:, 2]

        peg_axis = _rotate_vector(peg_quat, _Z_AXIS)

        peg_uprightness = peg_axis[:, 2].clamp(-1.0, 1.0)

        close_allowed = (
            (midpoint_xy_dist < 0.032)
            & (midpoint_z_offset > 0.003)
            & (midpoint_z_offset < 0.05)
            & (peg_uprightness > 0.90)
        )

        gripper_raw = actions[:, -1]

        close_request = gripper_raw > 0.2

        grasp_locked = self._grasp_success

        gripper_is_closed = grasp_locked | (close_request & close_allowed)

        gripper_action = torch.where(
            gripper_is_closed,
            torch.full_like(gripper_raw, self.gripper_closed),
            torch.full_like(gripper_raw, self.gripper_open),
        )

        self._gripper_is_closed.copy_(gripper_is_closed)
        self._current_gripper_action.copy_(gripper_action)

        new_pos = torch.cat(
            [
                new_joint_pos,
                gripper_action.unsqueeze(-1),
            ],
            dim=-1,
        )

        cliped_new_pos = torch.clamp(
            new_pos,
            self.joint_pos_min_limit,
            self.joint_pos_max_limit,
        )

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
        rows = slice(None)

        dof_pos = self.get_dof_pos(rows)
        dof_vel = self.get_dof_vel(rows)
        dof_pos_rel = dof_pos[:, : self._num_dof_pos] - self._init_dof_pos
        dof_vel_rel = dof_vel[:, : self._num_dof_vel]

        inputs = self.sim_data
        peg_pos = inputs["peg_pos"][rows]
        peg_axis = _rotate_vector(inputs["peg_quat"][rows], _Z_AXIS)

        socket_pos = self._socket_pos

        hand_pos = inputs["hand_pos"][rows]
        left_finger_pos = inputs["left_finger_pos"][rows]
        right_finger_pos = inputs["right_finger_pos"][rows]
        finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)

        hand_to_peg = peg_pos - hand_pos
        hand_to_peg_dist = torch.linalg.norm(hand_to_peg, dim=-1, keepdim=True)
        hand_to_peg_dir = hand_to_peg / (hand_to_peg_dist + 1e-6)

        midpoint_to_peg = peg_pos - finger_midpoint
        midpoint_to_peg_dist = torch.linalg.norm(midpoint_to_peg, dim=-1, keepdim=True)
        midpoint_to_peg_dir = midpoint_to_peg / (midpoint_to_peg_dist + 1e-6)
        midpoint_z_offset = (finger_midpoint[:, 2] - peg_pos[:, 2]).unsqueeze(-1)
        left_to_peg_dist = torch.linalg.norm(peg_pos - left_finger_pos, dim=-1, keepdim=True)
        right_to_peg_dist = torch.linalg.norm(peg_pos - right_finger_pos, dim=-1, keepdim=True)
        finger_balance = torch.abs(left_to_peg_dist - right_to_peg_dist)
        finger_gap = torch.linalg.norm(left_finger_pos - right_finger_pos, dim=-1, keepdim=True)
        gripper_joint_pos = dof_pos[:, 6:7]
        gripper_closure = (-gripper_joint_pos / max(abs(self.gripper_closed), 1e-6)).clamp(0.0, 1.0)

        peg_to_socket = socket_pos - peg_pos
        peg_to_socket_dist = torch.linalg.norm(peg_to_socket, dim=-1, keepdim=True)
        peg_to_socket_dir = peg_to_socket / (peg_to_socket_dist + 1e-6)

        gripper_state = self._current_gripper_action
        grasp_success = self._grasp_success.to(torch.float32)
        peg_to_socket_dir = peg_to_socket_dir * grasp_success.unsqueeze(-1)
        peg_to_socket_dist = peg_to_socket_dist * grasp_success.unsqueeze(-1)

        obs = torch.cat(
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
                gripper_state.unsqueeze(-1),
                grasp_success.unsqueeze(-1),
            ],
            dim=-1,
        )

        if bool(obs.isnan().any()):
            obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        assert obs.shape == (peg_pos.shape[0], self._obs_dim), f"Expected {self._obs_dim}, got {obs.shape}"
        # Publish the computed policy observation on the state. The field
        # name is applied via setattr so static audits can tell this
        # sanctioned observation-stage write apart from the forbidden
        # transition-time writes.
        setattr(state, "obs", EnvObs(policy=obs))
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

    def reset(self, env_ids: torch.Tensor) -> None:
        num_reset = env_ids.numel()

        noise_scale = self._cfg.init_state.joint_pos_reset_noise_scale
        noise_pos = (torch.rand(self._num_dof_pos, dtype=torch.float32) * 2.0 - 1.0) * noise_scale
        robot_dof_pos = self._init_dof_pos + noise_pos

        # Restore the semantic reset targets first and read the socket position
        # at the default pose, matching the legacy reset-then-sample sequence.
        robot_pos = self._reset_program.buffer("robot_position")
        robot_pos[env_ids] = 0.0
        robot_pos[env_ids, : self._num_dof_pos] = self._init_dof_pos
        self._reset_program.buffer("robot_velocity")[env_ids] = 0.0
        self._peg_reset_translation[env_ids] = 0.0
        self._peg_reset_orientation[env_ids] = 0.0
        self._peg_reset_orientation[env_ids, 3] = 1.0  # identity quaternion (xyzw)
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)

        peg_init = self._cfg.peg_init_config
        socket_pos = self.sim_data["socket_pos"][env_ids].clone()
        peg_xy = self._sample_peg_xy(socket_pos[:, :2])
        peg_x = peg_xy[:, 0]
        peg_y = peg_xy[:, 1]
        peg_z = torch.rand(num_reset, dtype=torch.float32) * (2 * peg_init.z_noise_scale) + (
            peg_init.z_pos - peg_init.z_noise_scale
        )

        robot_pos[env_ids] = 0.0
        robot_pos[env_ids, : self._num_dof_pos] = robot_dof_pos
        self._reset_program.buffer("robot_velocity")[env_ids] = 0.0
        self._peg_reset_translation[env_ids] = torch.stack([peg_x, peg_y, peg_z], dim=-1)
        self._peg_reset_orientation[env_ids] = 0.0
        self._peg_reset_orientation[env_ids, 3] = 1.0  # identity quaternion (xyzw)
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)

        inputs = self.sim_data
        socket_pos = inputs["socket_pos"][env_ids].clone()
        peg_pos = inputs["peg_pos"][env_ids].clone()
        hand_pos = inputs["hand_pos"][env_ids].clone()
        left_finger_pos = inputs["left_finger_pos"][env_ids].clone()
        right_finger_pos = inputs["right_finger_pos"][env_ids].clone()

        finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)
        midpoint_to_peg_xy_dist = torch.linalg.norm(peg_pos[:, :2] - finger_midpoint[:, :2], dim=-1)

        midpoint_z_offset = finger_midpoint[:, 2] - peg_pos[:, 2]

        safe_height = self._cfg.peg_config.safe_approach_height
        grasp_height = self._cfg.peg_config.grasp_target_height

        height_blend = ((midpoint_to_peg_xy_dist - 0.025) / (0.08 - 0.025)).clamp(0.0, 1.0)

        target_pregrasp_height = grasp_height + (safe_height - grasp_height) * height_blend

        pregrasp_height_error = torch.abs(midpoint_z_offset - target_pregrasp_height)

        gripper_joint_pos = self.get_dof_pos(env_ids)[:, 6]
        gripper_closure = (-gripper_joint_pos / max(abs(self.gripper_closed), 1e-6)).clamp(0.0, 1.0)
        socket_xy_dist = torch.linalg.norm(peg_pos[:, :2] - socket_pos[:, :2], dim=-1)

        # Seed the episode-scoped state buffers for the rows being reset; the
        # step pipeline owns them afterwards (never per-step state).
        self._current_actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._socket_pos[env_ids] = socket_pos
        self._initial_peg_pos[env_ids] = peg_pos
        self._current_gripper_action[env_ids] = 0.0
        self._gripper_is_closed[env_ids] = False
        self._grasp_success[env_ids] = False
        self._phase[env_ids] = 0
        self._prev_hand_to_peg_dist[env_ids] = torch.linalg.norm(peg_pos - hand_pos, dim=-1)
        self._prev_peg_to_socket_dist[env_ids] = torch.linalg.norm(peg_pos - socket_pos, dim=-1)
        self._prev_gripper_closure[env_ids] = gripper_closure
        self._prev_midpoint_xy_dist[env_ids] = midpoint_to_peg_xy_dist
        self._prev_pregrasp_height_error[env_ids] = pregrasp_height_error
        self._prev_peg_height[env_ids] = peg_pos[:, 2]
        self._prev_socket_xy_dist[env_ids] = socket_xy_dist
        self._prev_insert_depth[env_ids] = 0.0
        self._prev_socket_entry_gap[env_ids] = 0.2
        self._consecutive_capture_steps[env_ids] = 0
        self._consecutive_grasp_steps[env_ids] = 0
        self._consecutive_pregrasp_open_steps[env_ids] = 0

    def _check_termination(self, state: ArrayEnvState):
        inputs = self.sim_data
        peg_pos = inputs["peg_pos"]

        terminated = peg_pos[:, 2] < -0.15

        prev_grasp = self._grasp_success
        if bool(prev_grasp.any()):
            hand_pos = inputs["hand_pos"]
            hand_to_peg_dist = torch.linalg.norm(peg_pos - hand_pos, dim=-1)
            dropped_after_grasp = prev_grasp & (hand_to_peg_dist > 0.12) & (peg_pos[:, 2] < 0.09)
            terminated = terminated | dropped_after_grasp

        joint_vel = self.get_dof_vel(slice(None))
        terminated = terminated | (joint_vel.abs().amax(dim=-1) > 30)

        peg_vel = inputs["peg_lin_vel"]
        terminated = terminated | (peg_vel.abs().amax(dim=-1) > 15)

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
        hand_to_peg_dist = torch.linalg.norm(hand_to_peg, dim=-1)
        finger_midpoint = 0.5 * (left_finger_pos + right_finger_pos)
        midpoint_to_peg = peg_pos - finger_midpoint
        midpoint_xy_dist = torch.linalg.norm(midpoint_to_peg[:, :2], dim=-1)
        midpoint_z_offset = finger_midpoint[:, 2] - peg_pos[:, 2]
        left_to_peg_dist = torch.linalg.norm(peg_pos - left_finger_pos, dim=-1)
        right_to_peg_dist = torch.linalg.norm(peg_pos - right_finger_pos, dim=-1)
        midpoint_to_peg_dist = torch.linalg.norm(peg_pos - finger_midpoint, dim=-1)
        finger_balance = torch.abs(left_to_peg_dist - right_to_peg_dist)
        finger_gap = torch.linalg.norm(left_finger_pos - right_finger_pos, dim=-1)

        safe_height = self._cfg.peg_config.safe_approach_height
        grasp_height = self._cfg.peg_config.grasp_target_height

        height_blend = ((midpoint_xy_dist - 0.025) / (0.08 - 0.025)).clamp(0.0, 1.0)

        target_pregrasp_height = grasp_height + (safe_height - grasp_height) * height_blend

        pregrasp_height_error = torch.abs(midpoint_z_offset - target_pregrasp_height)

        close_quality = (
            (1 - torch.tanh(midpoint_xy_dist / 0.02))
            * (1 - torch.tanh(pregrasp_height_error / 0.02))
            * (1 - torch.tanh(finger_balance / 0.03))
        )

        peg_to_socket = socket_pos - peg_pos
        peg_to_socket_dist = torch.linalg.norm(peg_to_socket, dim=-1)
        xy_dist = torch.linalg.norm(peg_pos[:, :2] - socket_pos[:, :2], dim=-1)
        socket_top_z = socket_pos[:, 2] + self._cfg.peg_config.socket_depth * 0.5
        peg_bottom_z = peg_pos[:, 2] - self._cfg.peg_config.peg_length * 0.5
        raw_insert_depth = (socket_top_z - peg_bottom_z).clamp(0.0, self._cfg.peg_config.socket_depth)

        gripper_state = self._current_gripper_action
        is_gripper_command_closed = gripper_state < -0.5
        is_gripper_command_open = gripper_state > -0.3
        gripper_joint_pos = self.get_dof_pos(slice(None))[:, 6]
        gripper_closure = (-gripper_joint_pos / max(abs(self.gripper_closed), 1e-6)).clamp(0.0, 1.0)
        close_commitment = (0.55 * gripper_closure + 0.45 * ((0.10 - finger_gap) / 0.06).clamp(0.0, 1.0)).clamp(
            0.0, 1.0
        )
        actual_gripper_narrow = (gripper_closure > 0.42) | (finger_gap < 0.065)

        initial_peg_pos = self._initial_peg_pos
        lift_height = torch.clamp(peg_pos[:, 2] - initial_peg_pos[:, 2], min=0.0)
        peg_xy_shift = torch.linalg.norm(peg_pos[:, :2] - initial_peg_pos[:, :2], dim=-1)
        peg_drop_height = torch.clamp(initial_peg_pos[:, 2] - peg_pos[:, 2], min=0.0)
        peg_axis = _rotate_vector(peg_quat, _Z_AXIS)
        peg_uprightness = peg_axis[:, 2].clamp(-1.0, 1.0)
        peg_linear_vel = inputs["peg_lin_vel"]
        peg_speed = torch.linalg.norm(peg_linear_vel, dim=-1)

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
        prev_grasp = self._grasp_success.clone()

        gripper_closure_progress = features["gripper_closure"] - self._prev_gripper_closure
        self._prev_gripper_closure.copy_(features["gripper_closure"])

        self._prev_hand_to_peg_dist.copy_(features["hand_to_peg_dist"])

        xy_progress = self._prev_midpoint_xy_dist - features["midpoint_xy_dist"]
        self._prev_midpoint_xy_dist.copy_(features["midpoint_xy_dist"])

        height_progress = self._prev_pregrasp_height_error - features["pregrasp_height_error"]
        self._prev_pregrasp_height_error.copy_(features["pregrasp_height_error"])

        peg_height_progress = features["peg_pos"][:, 2] - self._prev_peg_height
        self._prev_peg_height.copy_(features["peg_pos"][:, 2])

        self._prev_peg_to_socket_dist.copy_(features["peg_to_socket_dist"])

        xy_socket_progress = self._prev_socket_xy_dist - features["xy_dist"]
        self._prev_socket_xy_dist.copy_(features["xy_dist"])

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

        consecutive_capture = torch.where(grasp_candidate, self._consecutive_capture_steps + 1, 0)
        self._consecutive_capture_steps.copy_(consecutive_capture)

        confirmed_grasp = consecutive_capture >= 3

        ever_grasped = prev_grasp | confirmed_grasp

        self._grasp_success.copy_(ever_grasped)

        is_grasping = (
            ever_grasped
            & features["is_gripper_command_closed"]
            & features["actual_gripper_narrow"]
            & (features["midpoint_to_peg_dist"] < 0.07)
        )

        consecutive_grasp = torch.where(
            is_grasping,
            self._consecutive_grasp_steps + 1,
            0,
        )

        self._consecutive_grasp_steps.copy_(consecutive_grasp)

        pregrasp_open_active = pregrasp_ready & features["is_gripper_command_open"] & (~prev_grasp)
        consecutive_pregrasp_open = torch.where(pregrasp_open_active, self._consecutive_pregrasp_open_steps + 1, 0)
        self._consecutive_pregrasp_open_steps.copy_(consecutive_pregrasp_open)

        insert_ready = ever_grasped & (features["xy_dist"] < 0.008) & (features["peg_uprightness"] > 0.97)
        insert_depth = torch.where(insert_ready, features["raw_insert_depth"], torch.zeros_like(features["xy_dist"]))
        insert_depth_progress = insert_depth - self._prev_insert_depth
        self._prev_insert_depth.copy_(insert_depth)

        socket_entry_gap = torch.clamp(features["peg_bottom_z"] - features["socket_top_z"], min=0.0)
        socket_entry_gap_progress = self._prev_socket_entry_gap - socket_entry_gap
        self._prev_socket_entry_gap.copy_(socket_entry_gap)

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

        xy_align_reward = reward_cfg.reach_weight * 0.3 * (1 - torch.tanh(features["midpoint_xy_dist"] / 0.05))

        height_align_reward = (
            reward_cfg.hand_approach_bonus * 0.3 * (1 - torch.tanh(features["pregrasp_height_error"] / 0.04))
        )

        xy_progress_reward = (
            reward_cfg.approach_weight * 0.6 * torch.tanh(progress["xy_progress"].clamp(-0.02, 0.02) / 0.01)
        )

        height_progress_reward = (
            reward_cfg.hand_approach_bonus * 1.2 * torch.tanh(progress["height_progress"].clamp(-0.02, 0.02) / 0.01)
        )

        find_peg_reward = torch.where(
            features["is_gripper_command_open"] & (~tracking["ever_grasped"]),
            xy_align_reward + height_align_reward + xy_progress_reward + height_progress_reward,
            torch.zeros(num_envs, dtype=torch.float32),
        )

        hover_above_peg_penalty = torch.zeros(num_envs, dtype=torch.float32)

        # hover_above_peg_mask = (
        #     features["is_gripper_command_open"]
        #     & (~tracking["ever_grasped"])
        #     & (features["midpoint_xy_dist"] < 0.04)
        #     & (features["midpoint_z_offset"] > 0.045)
        # )
        # hover_above_peg_penalty[hover_above_peg_mask] = (
        #     -reward_cfg.hand_approach_bonus
        #     * 1.8
        #     * torch.tanh((features["midpoint_z_offset"][hover_above_peg_mask] - 0.045) / 0.02)
        # )

        pregrasp_open_reward = torch.zeros(num_envs, dtype=torch.float32)
        pregrasp_open_mask = (
            tracking["pregrasp_ready"]
            & (~tracking["grasp_channel_ready"])
            & features["is_gripper_command_open"]
            & (~tracking["ever_grasped"])
        )
        pregrasp_open_reward[pregrasp_open_mask] = 0.0

        pregrasp_stall_penalty = torch.zeros(num_envs, dtype=torch.float32)
        pregrasp_stall_mask = pregrasp_open_mask & (tracking["consecutive_pregrasp_open"] > 4)
        pregrasp_stall_penalty[pregrasp_stall_mask] = (
            -4.0
            - 10.0 * torch.tanh(tracking["consecutive_pregrasp_open"][pregrasp_stall_mask].to(torch.float32) / 6.0)
            - reward_cfg.hand_approach_bonus * 0.8 * features["close_quality"][pregrasp_stall_mask]
        )

        ready_but_open_penalty = torch.zeros(num_envs, dtype=torch.float32)
        ready_but_open_mask = (
            tracking["grasp_channel_ready"] & features["is_gripper_command_open"] & (~tracking["ever_grasped"])
        )
        ready_but_open_penalty[ready_but_open_mask] = (
            -24.0 - reward_cfg.gripper_close_bonus * 6.0 * features["close_quality"][ready_but_open_mask]
        )

        close_in_grasp_zone_reward = torch.zeros(num_envs, dtype=torch.float32)
        close_zone_mask = (
            tracking["pregrasp_ready"] & features["is_gripper_command_closed"] & (~tracking["ever_grasped"])
        )
        close_in_grasp_zone_reward[close_zone_mask] = 42.0 + reward_cfg.gripper_close_bonus * 14.0 * features[
            "close_quality"
        ][close_zone_mask] * (0.3 + 0.7 * features["close_commitment"][close_zone_mask])

        premature_close_penalty = torch.zeros(num_envs, dtype=torch.float32)
        premature_close_mask = (
            features["is_gripper_command_closed"] & (~tracking["pregrasp_ready"]) & (~tracking["ever_grasped"])
        )
        premature_close_penalty[premature_close_mask] = (
            -20.0
            - 10.0 * torch.tanh(features["midpoint_xy_dist"][premature_close_mask] / 0.04)
            - 8.0 * torch.tanh(torch.clamp(features["midpoint_z_offset"][premature_close_mask] - 0.045, min=0.0) / 0.02)
        )

        gripper_close_progress_reward = torch.zeros(num_envs, dtype=torch.float32)
        close_progress_mask = (
            features["is_gripper_command_closed"] & tracking["pregrasp_ready"] & (~tracking["ever_grasped"])
        )
        gripper_close_progress_reward[close_progress_mask] = reward_cfg.gripper_close_bonus * (
            4.0 * features["close_commitment"][close_progress_mask]
            + 9.0 * torch.tanh(progress["gripper_closure_progress"][close_progress_mask].clamp(0.0, 0.12) / 0.03)
        )

        grasp_channel_reward = torch.zeros(num_envs, dtype=torch.float32)
        grasp_channel_mask = features["is_gripper_command_closed"] & tracking["grasp_channel_ready"]
        grasp_channel_reward[grasp_channel_mask] = 40.0 + reward_cfg.gripper_close_bonus * 12.0 * features[
            "close_quality"
        ][grasp_channel_mask] * (0.2 + 0.8 * features["close_commitment"][grasp_channel_mask])

        stable_close_reward = torch.zeros(num_envs, dtype=torch.float32)
        stable_close_mask = grasp_channel_mask & (~tracking["ever_grasped"])
        stable_close_reward[stable_close_mask] = (
            reward_cfg.grasp_weight
            * 2.4
            * features["close_quality"][stable_close_mask]
            * features["close_commitment"][stable_close_mask]
            * features["peg_uprightness"][stable_close_mask].clamp(0.0, 1.0)
            * (1 - torch.tanh(features["peg_xy_shift"][stable_close_mask] / 0.01))
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

        capture_reward = torch.zeros(num_envs, dtype=torch.float32)
        capture_mask = tracking["grasp_candidate"] & (~tracking["ever_grasped"])
        capture_reward[capture_mask] = reward_cfg.grasp_weight * (
            1.2 * features["close_quality"][capture_mask] * (0.35 + 0.65 * features["close_commitment"][capture_mask])
            + 0.8 * torch.tanh(tracking["consecutive_capture"][capture_mask].to(torch.float32) / 2.0)
        )

        lift_intent_reward = torch.zeros(num_envs, dtype=torch.float32)
        lift_intent_reward[capture_mask] = (
            reward_cfg.lift_weight
            * 2.2
            * torch.tanh(progress["peg_height_progress"][capture_mask].clamp(0.0, 0.008) / 0.003)
        )

        grasp_reward = torch.zeros(num_envs, dtype=torch.float32)
        grasp_reward[tracking["ever_grasped"]] = reward_cfg.grasp_weight * 0.4 * features["close_commitment"][
            tracking["ever_grasped"]
        ] + reward_cfg.lift_weight * 0.7 * torch.tanh(
            torch.clamp(features["lift_height"][tracking["ever_grasped"]], max=0.03) / 0.02
        )
        grasp_reward[tracking["first_grasp"]] += reward_cfg.grasp_success_bonus

        stable_grasp_mask = tracking["ever_grasped"] & (tracking["consecutive_grasp"] > 5)
        grasp_reward[stable_grasp_mask] += 10.0

        grip_hold_reward = torch.zeros(num_envs, dtype=torch.float32)
        hold_mask = tracking["ever_grasped"] & features["is_gripper_command_closed"]
        grip_hold_reward[hold_mask] = reward_cfg.gripper_close_bonus * 1.2 * features["close_commitment"][
            hold_mask
        ] + reward_cfg.grasp_weight * 0.2 * (1 - torch.tanh(features["midpoint_to_peg_dist"][hold_mask] / 0.025))

        release_after_grasp_penalty = torch.zeros(num_envs, dtype=torch.float32)
        release_mask = tracking["ever_grasped"] & (
            (~features["is_gripper_command_closed"]) | (features["close_commitment"] < 0.35)
        )
        release_after_grasp_penalty[release_mask] = -80.0 - reward_cfg.grasp_weight * 2.0 * torch.tanh(
            torch.clamp(0.35 - features["close_commitment"][release_mask], min=0.0) / 0.15
        )

        carry_upright_reward = torch.zeros(num_envs, dtype=torch.float32)
        carry_upright_reward[tracking["ever_grasped"]] = (
            reward_cfg.orientation_weight
            * 2.5
            * ((features["peg_uprightness"][tracking["ever_grasped"]] - 0.9) / 0.1).clamp(0.0, 1.0)
        )

        carry_tilt_penalty = torch.zeros(num_envs, dtype=torch.float32)
        carry_tilt_penalty[tracking["ever_grasped"]] = (
            -reward_cfg.orientation_weight
            * 3.0
            * torch.tanh(torch.clamp(0.9 - features["peg_uprightness"][tracking["ever_grasped"]], min=0.0) / 0.08)
        )

        carry_target_z = features["socket_top_z"] + self._cfg.peg_config.peg_length * 0.5 + 0.01
        carry_height_error = torch.abs(features["peg_pos"][:, 2] - carry_target_z)
        carry_height_reward = torch.zeros(num_envs, dtype=torch.float32)
        carry_height_mask = tracking["ever_grasped"] & (features["xy_dist"] > 0.045)

        carry_height_reward[carry_height_mask] = (
            reward_cfg.lift_weight * 1.4 * (1 - torch.tanh(carry_height_error[carry_height_mask] / 0.03))
        )

        move_to_socket_reward = torch.zeros(num_envs, dtype=torch.float32)
        move_to_socket_reward[tracking["ever_grasped"]] = (
            reward_cfg.peg_to_socket_approach_weight
            * 2.2
            * (1 - torch.tanh(features["xy_dist"][tracking["ever_grasped"]] / 0.03))
        )
        move_to_socket_reward[tracking["ever_grasped"]] += (
            reward_cfg.peg_to_socket_approach_weight
            * 3.2
            * torch.tanh(progress["xy_socket_progress"][tracking["ever_grasped"]].clamp(-0.02, 0.02) / 0.006)
        )

        transport_stall_penalty = torch.zeros(num_envs, dtype=torch.float32)
        transport_stall_mask = (
            tracking["ever_grasped"] & (features["xy_dist"] > 0.04) & (progress["xy_socket_progress"] < 0.0015)
        )
        transport_stall_penalty[transport_stall_mask] = (
            -reward_cfg.peg_to_socket_approach_weight
            * 1.4
            * torch.tanh((features["xy_dist"][transport_stall_mask] - 0.04) / 0.03)
        )

        lift_reward = torch.zeros(num_envs, dtype=torch.float32)
        lift_stage_mask = (
            features["is_gripper_command_closed"]
            & (tracking["pregrasp_ready"] | tracking["grasp_channel_ready"])
            & (~tracking["ever_grasped"])
        )
        lift_reward[lift_stage_mask] = (
            reward_cfg.lift_weight * 4.5 * torch.tanh(features["lift_height"][lift_stage_mask] / 0.03)
        )

        carry_lift_reward = torch.zeros(num_envs, dtype=torch.float32)
        carry_lift_mask = tracking["ever_grasped"] & (features["lift_height"] < 0.035)
        carry_lift_reward[carry_lift_mask] = (
            reward_cfg.lift_weight
            * 2.2
            * (1 - torch.tanh(torch.clamp(0.035 - features["lift_height"][carry_lift_mask], min=0.0) / 0.02))
        )

        lift_progress_reward = torch.zeros(num_envs, dtype=torch.float32)
        lift_progress_reward[lift_stage_mask] = (
            reward_cfg.lift_weight
            * 5.0
            * torch.tanh(progress["peg_height_progress"][lift_stage_mask].clamp(-0.01, 0.01) / 0.004)
        )

        move_away_penalty = torch.where(
            (progress["xy_socket_progress"] < 0) & tracking["ever_grasped"],
            -reward_cfg.peg_to_socket_approach_weight
            * torch.tanh(torch.clamp(-progress["xy_socket_progress"], min=0.0, max=0.02) / 0.01),
            torch.zeros(num_envs, dtype=torch.float32),
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

        insert_progress_reward = torch.zeros(num_envs, dtype=torch.float32)
        insert_progress_reward[aligned_for_insert] = (
            reward_cfg.insert_weight
            * 2.2
            * torch.tanh(tracking["insert_depth_progress"][aligned_for_insert].clamp(-0.01, 0.01) / 0.003)
        )

        preinsert_descend_reward = torch.zeros(num_envs, dtype=torch.float32)
        preinsert_descend_reward[aligned_for_insert] = (
            reward_cfg.insert_weight * 0.15 * (1 - torch.tanh(tracking["socket_entry_gap"][aligned_for_insert] / 0.015))
        )

        descend_progress_reward = torch.zeros(num_envs, dtype=torch.float32)
        descend_progress_reward[aligned_for_insert] = (
            reward_cfg.insert_weight
            * 2.4
            * torch.tanh(tracking["socket_entry_gap_progress"][aligned_for_insert].clamp(-0.02, 0.02) / 0.004)
        )

        hover_near_socket_penalty = torch.zeros(num_envs, dtype=torch.float32)
        hover_near_socket_mask = (
            tracking["ever_grasped"] & (features["xy_dist"] < 0.05) & (tracking["socket_entry_gap"] > 0.03)
        )
        hover_near_socket_penalty[hover_near_socket_mask] = (
            -reward_cfg.insert_weight
            * 0.7
            * torch.tanh((tracking["socket_entry_gap"][hover_near_socket_mask] - 0.03) / 0.03)
        )

        high_hold_near_socket_penalty = torch.zeros(num_envs, dtype=torch.float32)
        high_hold_mask = (
            tracking["ever_grasped"]
            & (features["xy_dist"] < 0.06)
            & (features["peg_pos"][:, 2] > features["socket_top_z"] + 0.09)
        )
        high_hold_near_socket_penalty[high_hold_mask] = (
            -reward_cfg.lift_weight
            * 2.0
            * torch.tanh(
                (features["peg_pos"][high_hold_mask, 2] - (features["socket_top_z"][high_hold_mask] + 0.09)) / 0.025
            )
        )

        premature_descend_penalty = torch.zeros(num_envs, dtype=torch.float32)
        premature_descend_mask = (
            tracking["ever_grasped"]
            & (features["xy_dist"] > 0.028)
            & (features["peg_bottom_z"] < features["socket_top_z"] + 0.015)
        )
        premature_descend_penalty[premature_descend_mask] = (
            -reward_cfg.align_weight
            * 1.2
            * torch.tanh(
                (
                    features["socket_top_z"][premature_descend_mask]
                    + 0.015
                    - features["peg_bottom_z"][premature_descend_mask]
                )
                / 0.02
            )
        )

        touch_reward = torch.zeros(num_envs, dtype=torch.float32)
        touch_reward[tracking["touch_socket"]] = 120.0

        insert_reward = tracking["insert_depth"] * reward_cfg.insert_weight * 6.0
        depth_insert_bonus = reward_cfg.depth_insert_bonus * torch.tanh(tracking["insert_depth"] / 0.01)

        align_reward = reward_cfg.align_weight * (1 - torch.tanh(features["xy_dist"] / 0.015))
        align_reward = align_reward * (tracking["ever_grasped"] | tracking["touch_socket"] | tracking["success"]).to(
            torch.float32
        )

        precision_reward = torch.zeros(num_envs, dtype=torch.float32)
        precision_reward[tracking["touch_socket"]] = reward_cfg.precision_weight * (
            1 - torch.tanh(features["xy_dist"][tracking["touch_socket"]] / 0.008)
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
            "success_bonus": reward_cfg.success_bonus * tracking["success"].to(torch.float32),
        }

    def _compute_safety_penalties(self, reward_cfg, features: dict, tracking: dict) -> dict:
        """Penalize unstable contacts, drops, and excessive shaking throughout the task."""
        num_envs = features["xy_dist"].shape[0]

        peg_tilt_excess = torch.clamp(0.93 - features["peg_uprightness"], min=0.0)
        peg_shift_excess = torch.clamp(features["peg_xy_shift"] - 0.006, min=0.0)
        peg_drop_excess = torch.clamp(features["peg_drop_height"] - 0.002, min=0.0)
        peg_speed_excess = torch.clamp(features["peg_speed"] - 0.18, min=0.0)

        pregrasp_knock_penalty = torch.zeros(num_envs, dtype=torch.float32)
        pregrasp_knock_mask = (~tracking["ever_grasped"]) & (
            (features["midpoint_xy_dist"] < 0.05) | tracking["pregrasp_ready"] | tracking["grasp_channel_ready"]
        )
        pregrasp_knock_penalty[pregrasp_knock_mask] = (
            -reward_cfg.orientation_weight * 1.8 * torch.tanh(peg_tilt_excess[pregrasp_knock_mask] / 0.05)
            - reward_cfg.precision_weight * 0.8 * torch.tanh(peg_shift_excess[pregrasp_knock_mask] / 0.015)
            - reward_cfg.lift_weight * 1.0 * torch.tanh(peg_drop_excess[pregrasp_knock_mask] / 0.01)
            - reward_cfg.anti_shake_weight * 1800.0 * torch.tanh(peg_speed_excess[pregrasp_knock_mask] / 0.25)
        )

        drop_penalty = torch.zeros(num_envs, dtype=torch.float32)
        peg_dropped = tracking["ever_grasped"] & (~tracking["is_grasping"])
        drop_penalty[peg_dropped] = -100.0

        carry_shake_penalty = torch.zeros(num_envs, dtype=torch.float32)

        carry_mask = tracking["ever_grasped"] & (~tracking["success"])

        carry_speed_limit = torch.where(
            features["xy_dist"] < 0.06,
            0.04,
            0.12,
        )

        carry_speed_excess = torch.clamp(features["peg_speed"] - carry_speed_limit, min=0.0)

        carry_shake_penalty[carry_mask] = -35.0 * torch.tanh(carry_speed_excess[carry_mask] / 0.12)

        return {
            "pregrasp_knock_penalty": pregrasp_knock_penalty,
            "drop_penalty": drop_penalty,
            "carry_shake_penalty": carry_shake_penalty,
        }

    def _compute_reward(self, state: ArrayEnvState, terminated: torch.Tensor):
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

        action_diff_sq = torch.sum((self._current_actions - self._last_actions) ** 2, dim=-1)
        joint_vel_sq = torch.sum(self.get_dof_vel(slice(None))[:, : self._num_dof_vel] ** 2, dim=1)
        peg_vel_sq = torch.sum(features["peg_linear_vel"] ** 2, dim=-1)

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
        reward = torch.zeros(self._num_envs, dtype=torch.float32)
        reward_details = {}
        for reward_dict, reward_keys in reward_component_groups:
            for reward_key in reward_keys:
                reward_component = reward_dict[reward_key]
                reward += reward_component
                reward_details[reward_key] = reward_component
        action_penalty = -reward_cfg.action_penalty_weight * action_diff_sq
        joint_vel_penalty = -reward_cfg.joint_vel_penalty_weight * joint_vel_sq
        anti_shake_penalty = -reward_cfg.anti_shake_weight * peg_vel_sq
        termination_penalty = torch.where(terminated, -50.0, 0.0)
        reward_details["action_penalty"] = action_penalty
        reward_details["joint_vel_penalty"] = joint_vel_penalty
        reward_details["anti_shake_penalty"] = anti_shake_penalty
        reward_details["termination_penalty"] = termination_penalty
        reward += action_penalty + joint_vel_penalty + anti_shake_penalty + termination_penalty

        reward = reward.clamp(-100.0, 3000.0)
        reward_details["total"] = reward
        state.reward_terms = reward_details

        return reward

    def get_dof_pos(self, rows):
        return self.sim_data["robot_joint_pos"][rows]

    def get_dof_vel(self, rows):
        return self.sim_data["robot_joint_vel"][rows]
