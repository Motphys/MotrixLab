# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct import reward
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.sim import (
    ActuatorCtrlQuery,
    BodyAngularVelocityWrite,
    BodyJointPositionWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    DofPositionLimitsQuery,
    DofPositionQuery,
    DofVelocityQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
    SensorValuesQuery,
)
from motrix_env_core.sim.write import BodyJointVelocityWrite, CtrlTargetsWrite
from motrix_envs.basic.humanoid.cfg import HumanoidWalkCfg

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
    "actuator_ctrls": ActuatorCtrlQuery(),
    "torso_subtreelinvel": SensorValuesQuery(sensors=("torso_subtreelinvel",)),
    "torso_pos": LinkPositionQuery(link="torso"),
    "torso_quat": LinkQuaternionQuery(link="torso"),
    "head_pos": LinkPositionQuery(link="head"),
    "head_quat": LinkQuaternionQuery(link="head"),
    "pelvis_pos": LinkPositionQuery(link="pelvis"),
    "pelvis_quat": LinkQuaternionQuery(link="pelvis"),
    "left_hand_pos": LinkPositionQuery(link="left_hand"),
    "right_hand_pos": LinkPositionQuery(link="right_hand"),
    "left_foot_pos": LinkPositionQuery(link="left_foot"),
    "right_foot_pos": LinkPositionQuery(link="right_foot"),
}
_SIM_MODEL_QUERIES = {"dof_position_limits": DofPositionLimitsQuery()}


def _quat_to_rotation_mats(quats: np.ndarray) -> np.ndarray:
    """Convert (x, y, z, w) quaternions, shape (..., 4), to rotation mats (..., 3, 3)."""
    x, y, z, w = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
    mats = np.empty(quats.shape[:-1] + (3, 3), dtype=quats.dtype)
    mats[..., 0, 0] = 1 - 2 * (y * y + z * z)
    mats[..., 0, 1] = 2 * (x * y - z * w)
    mats[..., 0, 2] = 2 * (x * z + y * w)
    mats[..., 1, 0] = 2 * (x * y + z * w)
    mats[..., 1, 1] = 1 - 2 * (x * x + z * z)
    mats[..., 1, 2] = 2 * (y * z - x * w)
    mats[..., 2, 0] = 2 * (x * z - y * w)
    mats[..., 2, 1] = 2 * (y * z + x * w)
    mats[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return mats


@registry.env("dm-humanoid-stand")
@registry.env("dm-humanoid-walk")
@registry.env("dm-humanoid-run")
class Humanoid3DEnv(DirectEnv):
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: HumanoidWalkCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "torso_position": BodyPositionWrite(("torso",)),
                "torso_rotation": BodyRotationWrite(("torso",)),
                "torso_linear_velocity": BodyLinearVelocityWrite(("torso",)),
                "torso_angular_velocity": BodyAngularVelocityWrite(("torso",)),
                "joints_position": BodyJointPositionWrite("torso"),
                "joints_velocity": BodyJointVelocityWrite("torso"),
            },
            reset=True,
        )
        self._torso_reset_position = self._reset_program.buffer("torso_position")[:, 0]
        self._torso_reset_rotation = self._reset_program.buffer("torso_rotation")[:, 0]
        self._torso_reset_linear_velocity = self._reset_program.buffer("torso_linear_velocity")[:, 0]
        self._torso_reset_angular_velocity = self._reset_program.buffer("torso_angular_velocity")[:, 0]
        self._joint_reset_position = self._reset_program.buffer("joints_position")
        self._joint_reset_velocity = self._reset_program.buffer("joints_velocity")
        self._init_obs_space()
        self._init_action_space()

        self._move_speed = float(cfg.move_speed)
        self._stand_height = float(cfg.stand_height)
        self._target_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._target_direction_xy = self._target_direction[:2].copy()

        self._qpos_low, self._qpos_high = self._build_qpos_limits()
        self._cache_derived_constants(cfg)
        self._init_joint_randomization_config(cfg)

    def _build_qpos_limits(self) -> tuple[np.ndarray, np.ndarray]:
        num_dof_pos = self.num_dof_pos
        lower, upper = self.model.others["dof_position_limits"]
        expected_shape = (num_dof_pos,)
        if lower.shape != expected_shape or upper.shape != expected_shape:
            raise ValueError(
                "Humanoid DOF position limits must match global DOF positions: "
                f"lower={lower.shape}, upper={upper.shape}, expected={expected_shape}."
            )
        return lower, upper

    def _cache_derived_constants(self, cfg: HumanoidWalkCfg) -> None:
        t_cfg = cfg.termination_config
        self._head_height_min = self._stand_height * 0.95
        self._pelvis_height_min = 0.6 * self._stand_height
        self._pelvis_height_margin = 0.6 * self._stand_height
        self._term_head_height_min = float(t_cfg.head_height_factor) * self._stand_height
        self._term_torso_upright_threshold = float(t_cfg.torso_upright_threshold)
        self._term_extreme_vel_threshold = float(t_cfg.extreme_vel_threshold)

    def _init_obs_space(self):
        num_joint_angles = self.num_dof_pos - 7
        num_head_height = 1
        num_extremities = 12
        num_torso_vertical = 3
        num_com_vel = 3
        num_qvel = self.num_dof_vel
        num_target_local = 3
        num_obs = (
            num_joint_angles
            + num_head_height
            + num_extremities
            + num_torso_vertical
            + num_com_vel
            + num_qvel
            + num_target_local
        )
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

    def _init_action_space(self):
        ctrl_ranges = np.asarray([spec.ctrl_range for spec in self.model.actuators], dtype=np.float32)
        self._action_space = gym.spaces.Box(
            ctrl_ranges[:, 0],
            ctrl_ranges[:, 1],
            (self.num_actuators,),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        actions = actions.detach().cpu().numpy()
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = np.asarray(actions, dtype=np.float32)
        self._ctrl_writes.execute()
        return state

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        state = self.update_terminated(state)
        state = self.update_reward(state)
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        joint_angles = inputs["dof_pos"][:, 7:]
        head_height = self._get_head_height(slice(None))[:, None]

        torso_rot = _quat_to_rotation_mats(inputs["torso_quat"])
        torso_pos = inputs["torso_pos"]
        parts = [
            inputs["left_hand_pos"],
            inputs["left_foot_pos"],
            inputs["right_hand_pos"],
            inputs["right_foot_pos"],
        ]
        extremities = []
        for p in parts:
            torso_to_limb = p - torso_pos
            v_body = np.einsum("ni,nij->nj", torso_to_limb, torso_rot)
            extremities.append(v_body)
        extremities = np.concatenate(extremities, axis=-1)

        torso_vertical = torso_rot[:, 2, :]

        com_vel = inputs["torso_subtreelinvel"]

        qvel = inputs["dof_vel"]

        target_world = np.ones((self._num_envs, 3), dtype=np.float32) * self._target_direction[None, :]
        target_direction_local = np.einsum("ni,nij->nj", target_world, torso_rot)

        obs = np.concatenate(
            [joint_angles, head_height, extremities, torso_vertical, com_vel, qvel, target_direction_local], axis=-1
        )
        return state.replace(obs=obs)

    def update_terminated(self, state: ArrayEnvState) -> ArrayEnvState:
        head_height = self._get_head_height(slice(None))
        torso_upright = self._get_torso_upright(slice(None))
        terminated = self._compute_terminated(head_height, torso_upright)

        return state.replace(
            terminated=terminated,
        )

    def update_reward(self, state: ArrayEnvState) -> ArrayEnvState:
        terminated = state.terminated
        head_height = self._get_head_height(slice(None))
        pelvis_height = self._get_pelvis_height(slice(None))
        torso_upright = self._get_torso_upright(slice(None))
        rwd, reward_components = self._compute_reward(head_height, torso_upright, pelvis_height)
        rwd, reward_components = self._apply_termination_mask(terminated, rwd, reward_components)
        state.reward_terms = reward_components
        return state.replace(reward=rwd)

    def _apply_termination_mask(
        self,
        terminated: np.ndarray,
        rwd: np.ndarray,
        reward_components: dict,
    ) -> tuple[np.ndarray, dict]:
        rwd = np.where(terminated, 0.0, rwd).astype(np.float32)
        for k, v in reward_components.items():
            reward_components[k] = np.where(terminated, 0.0, v).astype(np.float32)
        return rwd, reward_components

    def reset(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.detach().cpu().numpy()
        self._randomize_joints(env_ids)

    def _get_head_height(self, rows) -> np.ndarray:
        return self.sim_data["head_pos"][rows][:, 2]

    def _get_pelvis_height(self, rows) -> np.ndarray:
        return self.sim_data["pelvis_pos"][rows][:, 2]

    def _get_torso_upright(self, rows) -> np.ndarray:
        return _quat_to_rotation_mats(self.sim_data["torso_quat"][rows])[:, 2, 2]

    def _compute_reward(
        self,
        head_height: np.ndarray,
        torso_upright: np.ndarray,
        pelvis_height: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        posture_reward = self._compute_posture_reward(head_height, torso_upright, pelvis_height)
        speed_reward, energy_reward = self._compute_speed_and_energy_reward()
        gait_reward = self._compute_gait_reward()

        rwd = (posture_reward * speed_reward * energy_reward * gait_reward).astype(np.float32)

        comps = {
            "energy": energy_reward.astype(np.float32),
            "speed": speed_reward.astype(np.float32),
            "posture": posture_reward.astype(np.float32),
            "gait": gait_reward.astype(np.float32),
        }
        return rwd, comps

    def _compute_posture_reward(
        self,
        head_height: np.ndarray,
        torso_upright: np.ndarray,
        pelvis_height: np.ndarray,
    ) -> np.ndarray:
        stand_reward = reward.tolerance(
            head_height,
            bounds=(self._head_height_min, float("inf")),
            margin=0.5,
        ).flatten()

        upright_reward = reward.tolerance(
            torso_upright,
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=0.9,
        ).flatten()

        pelvis_height_reward = reward.tolerance(
            pelvis_height,
            bounds=(self._pelvis_height_min, float("inf")),
            sigmoid="linear",
            margin=self._pelvis_height_margin,
        ).flatten()

        return stand_reward * upright_reward * pelvis_height_reward

    def _compute_speed_and_energy_reward(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_dir_xy = self._target_direction_xy

        ctrls = self.sim_data["actuator_ctrls"]
        com_vel = self.sim_data["torso_subtreelinvel"]

        if self._move_speed <= 0.0:
            energy_reward = np.exp(-1.0 * np.mean(np.square(ctrls), axis=-1))
            actual_speed = np.linalg.norm(com_vel[:, :2], axis=-1)
            speed_reward = reward.tolerance(
                actual_speed,
                bounds=(self._move_speed, self._move_speed),
                margin=1.0,
                value_at_margin=0.01,
            ).flatten()
        elif self._move_speed <= 3.0:
            energy_reward = np.exp(-0.5 * np.mean(np.square(ctrls), axis=-1))
            actual_speed = np.sum(com_vel[:, :2] * target_dir_xy, axis=-1)
            speed_reward = reward.tolerance(
                actual_speed,
                bounds=(self._move_speed, self._move_speed),
                margin=self._move_speed,
                value_at_margin=0.0,
                sigmoid="linear",
            ).flatten()
        else:
            energy_reward = np.exp(-0.3 * np.mean(np.square(ctrls), axis=-1))
            actual_speed = np.sum(com_vel[:, :2] * target_dir_xy, axis=-1)
            speed_reward = reward.tolerance(
                actual_speed,
                bounds=(self._move_speed, float("inf")),
                margin=self._move_speed,
                value_at_margin=0.0,
                sigmoid="linear",
            ).flatten()

        return speed_reward, energy_reward

    def _compute_heading_reward(
        self,
        forward_vec: np.ndarray,
        target_dir: np.ndarray,
        bounds,
        margin,
    ) -> np.ndarray:
        dot = np.sum(forward_vec * target_dir, axis=-1)
        return reward.tolerance(
            dot,
            bounds=bounds,
            margin=margin,
            value_at_margin=0.0,
            sigmoid="linear",
        ).flatten()

    def _compute_gait_reward(self) -> np.ndarray:
        inputs = self.sim_data
        target_dir = self._target_direction

        torso_rot = _quat_to_rotation_mats(inputs["torso_quat"])
        head_rot = _quat_to_rotation_mats(inputs["head_quat"])
        pelvis_rot = _quat_to_rotation_mats(inputs["pelvis_quat"])

        torso_forward = torso_rot[:, 0, 0:3]
        torso_heading_reward = self._compute_heading_reward(torso_forward, target_dir, bounds=(0.9, 1.0), margin=0.3)

        head_forward = head_rot[:, 0, 0:3]
        head_heading_reward = self._compute_heading_reward(head_forward, target_dir, bounds=(0.9, 1.0), margin=0.3)

        pelvis_forward = pelvis_rot[:, 0, 0:3]
        pelvis_yaw_reward = self._compute_heading_reward(pelvis_forward, target_dir, bounds=(0.9, 1.0), margin=0.3)

        pelvis_up = pelvis_rot[:, 2, 2]
        pelvis_level_reward = reward.tolerance(
            pelvis_up,
            bounds=(0.9, 1.0),
            margin=0.3,
            sigmoid="linear",
            value_at_margin=0.0,
        ).flatten()

        left_foot_pos = inputs["left_foot_pos"]
        right_foot_pos = inputs["right_foot_pos"]
        max_foot_h = np.maximum(left_foot_pos[:, 2], right_foot_pos[:, 2])
        feet_height_reward = reward.tolerance(
            max_foot_h,
            bounds=(0.0, 0.3),
            margin=0.5,
            sigmoid="quadratic",
            value_at_margin=0.0,
        ).flatten()

        return torso_heading_reward * head_heading_reward * pelvis_yaw_reward * pelvis_level_reward * feet_height_reward

    def _compute_terminated(
        self,
        head_height: np.ndarray,
        torso_upright: np.ndarray,
    ) -> np.ndarray:
        inputs = self.sim_data
        qpos = inputs["dof_pos"]
        qvel = inputs["dof_vel"]
        bad = ~np.isfinite(qpos).all(axis=-1) | ~np.isfinite(qvel).all(axis=-1)
        too_low = head_height < self._term_head_height_min
        too_tilted = torso_upright < self._term_torso_upright_threshold
        extreme_vel = np.abs(qvel).max(axis=-1) > self._term_extreme_vel_threshold
        return bad | too_low | too_tilted | extreme_vel

    def _init_joint_randomization_config(self, cfg: HumanoidWalkCfg) -> None:
        init_cfg = cfg.init_state
        self._reset_height = self._stand_height * init_cfg.reset_height_factor
        self._reset_qvel_range = init_cfg.reset_qvel_range
        self._reset_actuator_range = init_cfg.reset_actuator_range

        self._hip_yaw_range = tuple(np.deg2rad(x) for x in init_cfg.hip_yaw_range)
        self._hip_roll_range = tuple(np.deg2rad(x) for x in init_cfg.hip_roll_range)
        self._hip_pitch_range = tuple(np.deg2rad(x) for x in init_cfg.hip_pitch_range)

        self._symmetric_leg_pairs_rad = [
            (left_idx, right_idx, tuple(np.deg2rad(x) for x in deg_range))
            for left_idx, right_idx, deg_range in init_cfg.symmetric_leg_pairs
        ]
        self._symmetric_arm_pairs = init_cfg.symmetric_arm_pairs
        self._arm_margin_factor = init_cfg.arm_margin_factor
        self._symmetric_arm_used_indices = set()
        for left_idx, right_idx in self._symmetric_arm_pairs:
            self._symmetric_arm_used_indices.add(left_idx)
            self._symmetric_arm_used_indices.add(right_idx)

    def _randomize_joints(self, env_ids: np.ndarray) -> None:
        # qpos layout (humanoid.xml): 0-6 free (x,y,z,qw,qx,qy,qz), 7=abdomen_z, 8=abdomen_y, 9=abdomen_x,
        # 10-15 right leg (hip_x,z,y, knee, ankle_y,x), 16-21 left leg, 22-24 right arm, 25-27 left arm (num_dof_pos=28)
        n = len(env_ids)
        num_dof_pos = int(self.num_dof_pos)
        num_dof_vel = int(self.num_dof_vel)
        num_actuators = int(self.num_actuators)
        low, high = self._qpos_low, self._qpos_high

        qpos = np.zeros((n, num_dof_pos), dtype=np.float32)
        qpos[:, 2] = self._reset_height
        qpos[:, 3] = 1.0

        # qpos 7=abdomen_z (yaw), 8=abdomen_y (pitch), 9=abdomen_x (roll) per humanoid.xml
        qpos[:, 7] = np.random.uniform(self._hip_yaw_range[0], self._hip_yaw_range[1], size=(n,))
        qpos[:, 8] = np.random.uniform(self._hip_pitch_range[0], self._hip_pitch_range[1], size=(n,))
        qpos[:, 9] = np.random.uniform(self._hip_roll_range[0], self._hip_roll_range[1], size=(n,))

        self._randomize_symmetric_legs(qpos, n, num_dof_pos, low, high)
        self._randomize_symmetric_arms(qpos, n, num_dof_pos, low, high)
        self._randomize_remaining_joints(qpos, n, num_dof_pos, low, high)

        qvel = np.random.uniform(-self._reset_qvel_range, self._reset_qvel_range, size=(n, num_dof_vel)).astype(
            np.float32
        )
        actuator_ctrls = np.random.uniform(
            -self._reset_actuator_range, self._reset_actuator_range, size=(n, num_actuators)
        ).astype(np.float32)

        torso_pose = qpos[:, :7].copy()
        torso_pose[:, 3:7] = np.concatenate([qpos[:, 4:7], qpos[:, 3:4]], axis=1)
        _torso_pose = torso_pose
        self._torso_reset_position[env_ids] = _torso_pose[:, :3]
        self._torso_reset_rotation[env_ids] = _torso_pose[:, 3:7]
        self._torso_reset_linear_velocity[env_ids] = qvel[:, :3]
        self._torso_reset_angular_velocity[env_ids] = qvel[:, 3:6]
        self._joint_reset_position[env_ids] = qpos[:, 7:]
        self._joint_reset_velocity[env_ids] = qvel[:, 6:]
        self._reset_program.execute(env_ids)
        ctrl_targets = self.sim_data["actuator_ctrls"].copy()
        ctrl_targets[env_ids] = actuator_ctrls
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = ctrl_targets
        self._ctrl_writes.execute()
        self.sim_data.execute(np.asarray(env_ids, dtype=np.int64))

    def _randomize_symmetric_legs(
        self, qpos: np.ndarray, n: int, num_dof_pos: int, low: np.ndarray, high: np.ndarray
    ) -> None:
        for left_idx, right_idx, (min_rad, max_rad) in self._symmetric_leg_pairs_rad:
            if left_idx < num_dof_pos:
                qpos[:, left_idx] = np.random.uniform(
                    np.clip(min_rad, low[left_idx], high[left_idx]),
                    np.clip(max_rad, low[left_idx], high[left_idx]),
                    size=(n,),
                )
            if right_idx < num_dof_pos:
                right_min_rad = -max_rad
                right_max_rad = -min_rad
                qpos[:, right_idx] = np.random.uniform(
                    np.clip(right_min_rad, low[right_idx], high[right_idx]),
                    np.clip(right_max_rad, low[right_idx], high[right_idx]),
                    size=(n,),
                )

    def _randomize_symmetric_arms(
        self, qpos: np.ndarray, n: int, num_dof_pos: int, low: np.ndarray, high: np.ndarray
    ) -> None:
        # Default range when model joint_limits are missing (low/high are ±inf);
        # np.random.uniform requires finite bounds.
        default_lo, default_hi = -np.pi, np.pi
        for left_idx, right_idx in self._symmetric_arm_pairs:
            if left_idx < num_dof_pos and right_idx < num_dof_pos:
                lo_l = low[left_idx] if np.isfinite(low[left_idx]) else default_lo
                hi_l = high[left_idx] if np.isfinite(high[left_idx]) else default_hi
                lo_r = low[right_idx] if np.isfinite(low[right_idx]) else default_lo
                hi_r = high[right_idx] if np.isfinite(high[right_idx]) else default_hi

                left_range = hi_l - lo_l
                left_margin = left_range * self._arm_margin_factor

                left_min = lo_l + left_margin
                left_max = hi_l - left_margin

                right_min = -left_max
                right_max = -left_min

                right_min_clipped = max(right_min, lo_r)
                right_max_clipped = min(right_max, hi_r)

                if left_min < left_max:
                    qpos[:, left_idx] = np.random.uniform(left_min, left_max, size=(n,))
                else:
                    qpos[:, left_idx] = np.random.uniform(lo_l, hi_l, size=(n,))

                if right_min_clipped < right_max_clipped:
                    qpos[:, right_idx] = np.random.uniform(right_min_clipped, right_max_clipped, size=(n,))
                else:
                    qpos[:, right_idx] = np.random.uniform(lo_r, hi_r, size=(n,))

    def _randomize_remaining_joints(
        self, qpos: np.ndarray, n: int, num_dof_pos: int, low: np.ndarray, high: np.ndarray
    ) -> None:
        used_indices = self._symmetric_arm_used_indices
        default_lo, default_hi = -np.pi, np.pi

        # 22 = first arm joint (right_shoulder1) in humanoid.xml qpos order;
        # arms 22-27 are covered by symmetric_arm_pairs
        for i in range(22, num_dof_pos):
            if i not in used_indices:
                lo = low[i] if np.isfinite(low[i]) else default_lo
                hi = high[i] if np.isfinite(high[i]) else default_hi
                qpos[:, i] = np.random.uniform(lo, hi, size=(n,))
