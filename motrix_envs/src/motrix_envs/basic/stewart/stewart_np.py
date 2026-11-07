# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.math import quaternion
from motrix_env_core.sim import (
    ActuatorCtrlQuery,
    BodyAngularVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    JointPositionWrite,
    LinkPositionQuery,
    LinkQuaternionQuery,
)
from motrix_env_core.sim.write import CtrlTargetsWrite, JointVelocityWrite
from motrix_envs.basic.stewart.cfg import StewartBaseEnvCfg

_STEWART_LEGS = tuple(f"leg{j}{i}" for i in range(3) for j in range(2))
_STEWART_TOP_CONNECTS = tuple(f"top_connect{j}{i}" for i in range(3) for j in range(2))
_SIM_DATA_QUERIES = {
    "actuator_ctrls": ActuatorCtrlQuery(),
    "top_pos": LinkPositionQuery(link="top"),
    "top_quat": LinkQuaternionQuery(link="top"),
    "ball_pos": LinkPositionQuery(link="ball"),
    "stage_pos": LinkPositionQuery(link="disturb_stage"),
    "stage_quat": LinkQuaternionQuery(link="disturb_stage"),
    **{f"{name}_pos": LinkPositionQuery(link=name) for name in _STEWART_LEGS},
    **{f"{name}_pos": LinkPositionQuery(link=name) for name in _STEWART_TOP_CONNECTS},
}
_SIM_MODEL_QUERIES = {}


def _normalize_actions(actions: np.ndarray, num_envs: int, action_dim: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        if num_envs != 1 or actions.shape[0] != action_dim:
            raise ValueError(f"Expected action shape ({num_envs}, {action_dim}) or ({action_dim},).")
        actions = actions.reshape(1, action_dim)
    if actions.shape != (num_envs, action_dim):
        raise ValueError(f"Expected action shape ({num_envs}, {action_dim}), got {actions.shape}.")
    return np.clip(actions, -1.0, 1.0).astype(np.float32)


def _identity_quat(shape: tuple[int, ...]) -> np.ndarray:
    quat = np.zeros((*shape, 4), dtype=np.float32)
    quat[..., 3] = 1.0
    return quat


def _normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    identity = _identity_quat(quat.shape[:-1])
    safe = norm > 1e-8
    return np.where(safe, quat / np.maximum(norm, 1e-8), identity).astype(np.float32)


def _broadcast_quat_vec(quat: np.ndarray, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    quat = _normalize_quat(quat)
    vec = np.asarray(vec, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (..., 4), got {quat.shape}.")
    if vec.shape[-1] != 3:
        raise ValueError(f"Expected vector shape (..., 3), got {vec.shape}.")

    broadcast_shape = np.broadcast_shapes(quat.shape[:-1], vec.shape[:-1])
    quat_flat = np.broadcast_to(quat, (*broadcast_shape, 4)).reshape(-1, 4)
    vec_flat = np.broadcast_to(vec, (*broadcast_shape, 3)).reshape(-1, 3)
    return quat_flat, vec_flat, broadcast_shape


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat(quat)
    flip = quat[..., 3] < 0.0
    quat = np.where(flip[..., None], -quat, quat)
    xyz = quat[..., :3]
    w = np.clip(quat[..., 3], -1.0, 1.0)
    xyz_norm = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(xyz_norm, w)
    scale = np.where(xyz_norm > 1e-8, angle / np.maximum(xyz_norm, 1e-8), 2.0)
    return (xyz * scale[..., None]).astype(np.float32)


@registry.env("stewart")
@registry.env("stewart-static")
@registry.env("stewart-disturb-xy")
class StewartEnv(DirectEnv):
    _cfg: StewartBaseEnvCfg

    def __init__(self, cfg: StewartBaseEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model({})
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        # Body-state writes are per-body programs: the disturbance stage is
        # written every substep during control, the ball once at reset.
        self._body_state_writes = {
            body: self.sim.write_compiler.compile(
                {
                    f"{body}_position": BodyPositionWrite((body,)),
                    f"{body}_rotation": BodyRotationWrite((body,)),
                    f"{body}_linear_velocity": BodyLinearVelocityWrite((body,)),
                    f"{body}_angular_velocity": BodyAngularVelocityWrite((body,)),
                }
            )
            for body in ("disturb_stage", "ball")
        }
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "stage_position": BodyPositionWrite(("disturb_stage",)),
                "stage_rotation": BodyRotationWrite(("disturb_stage",)),
                "stage_linear_velocity": BodyLinearVelocityWrite(("disturb_stage",)),
                "stage_angular_velocity": BodyAngularVelocityWrite(("disturb_stage",)),
                "top_position": BodyPositionWrite(("top",)),
                "top_rotation": BodyRotationWrite(("top",)),
                "top_linear_velocity": BodyLinearVelocityWrite(("top",)),
                "top_angular_velocity": BodyAngularVelocityWrite(("top",)),
                "ball_position": BodyPositionWrite(("ball",)),
                "ball_rotation": BodyRotationWrite(("ball",)),
                "ball_linear_velocity": BodyLinearVelocityWrite(("ball",)),
                "ball_angular_velocity": BodyAngularVelocityWrite(("ball",)),
                "legs_position": JointPositionWrite(tuple(f"slide{leg[3:]}" for leg in _STEWART_LEGS)),
                "legs_velocity": JointVelocityWrite(tuple(f"slide{leg[3:]}" for leg in _STEWART_LEGS)),
            },
            reset=True,
        )
        self._cfg = cfg

        self._action_dim = 2
        self._obs_dim = 15 + (10 if cfg.disturbance_enabled and cfg.disturbance_include_obs else 0)
        self._action_space = gym.spaces.Box(-1.0, 1.0, (self._action_dim,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (self._obs_dim,), dtype=np.float32)

        self._model_action_dim = self.num_actuators
        if self._model_action_dim != 6:
            raise ValueError(f"Stewart model mismatch: expected 6 actuators, got {self._model_action_dim}")

        self._actuator_low = np.asarray([spec.ctrl_range[0] for spec in self.model.actuators], dtype=np.float32)
        self._actuator_high = np.asarray([spec.ctrl_range[1] for spec in self.model.actuators], dtype=np.float32)

        # Snapshot the default-pose geometry the same way the legacy env read
        # it from a fresh SceneData: write default rows (reset runs FK) and
        # read the compiled queries once before any episode reset.
        default_ids = np.arange(self._num_envs, dtype=np.int64)
        self.sim_data.execute(default_ids)
        inputs = self.sim_data
        self._top_pos_init = inputs["top_pos"][0].copy().astype(np.float32)
        stage_pose = np.concatenate([inputs["stage_pos"][0], inputs["stage_quat"][0]], axis=-1).astype(np.float32)
        self._stage_pos_init = stage_pose[:3].astype(np.float32)
        self._stage_quat_init = _normalize_quat(stage_pose[3:7])

        top_connect_offsets = []
        leg_length_init = []
        for leg, top_connect in zip(_STEWART_LEGS, _STEWART_TOP_CONNECTS):
            bottom_pos = inputs[f"{leg}_pos"][0].copy().astype(np.float32)
            top_connect_pos = inputs[f"{top_connect}_pos"][0].copy().astype(np.float32)
            top_connect_offsets.append(top_connect_pos - self._top_pos_init)
            leg_length_init.append(np.linalg.norm(top_connect_pos - bottom_pos))
        self._top_connect_offsets = np.asarray(top_connect_offsets, dtype=np.float32)
        self._leg_length_init = np.asarray(leg_length_init, dtype=np.float32)

        # Episode-scoped task state: full-batch buffers, reset writes the done
        # rows in place.
        num_envs = self._num_envs
        self._target_pos = np.zeros((num_envs, 3), dtype=np.float32)
        self._target_quat = _identity_quat((num_envs,))
        self._target_tilt_cmd = np.zeros((num_envs, 2), dtype=np.float32)
        self._prev_rel = np.zeros((num_envs, 3), dtype=np.float32)
        self._filtered_rel_vel = np.zeros((num_envs, 3), dtype=np.float32)
        self._last_rel_vel = np.zeros((num_envs, 3), dtype=np.float32)
        self._prev_top_quat = _identity_quat((num_envs,))
        self._filtered_top_ang_vel = np.zeros((num_envs, 3), dtype=np.float32)
        self._last_top_ang_vel = np.zeros((num_envs, 3), dtype=np.float32)
        self._initial_rel_xy = np.zeros(num_envs, dtype=np.float32)
        self._prev_zero_vel_rel_xy = np.zeros(num_envs, dtype=np.float32)
        self._still_steps = np.zeros(num_envs, dtype=np.int32)
        self._still_window_active = np.zeros(num_envs, dtype=bool)
        self._policy_action = np.zeros((num_envs, self._action_dim), dtype=np.float32)
        self._prev_action_exec = np.zeros((num_envs, self._action_dim), dtype=np.float32)
        self._action_exec = np.zeros((num_envs, self._action_dim), dtype=np.float32)
        self._disturb_time = np.zeros(num_envs, dtype=np.float32)
        self._disturb_pos = np.zeros((num_envs, 3), dtype=np.float32)
        self._disturb_lin_vel = np.zeros((num_envs, 3), dtype=np.float32)
        self._disturb_rot_deg = np.zeros((num_envs, 2), dtype=np.float32)
        self._disturb_ang_vel_deg = np.zeros((num_envs, 2), dtype=np.float32)
        self._disturb_pos_alpha = np.ones((num_envs, 3), dtype=np.float32)
        self._disturb_pos_limit_scale = np.ones((num_envs, 3), dtype=np.float32)
        self._disturb_pos_noise_scale = np.zeros((num_envs, 3), dtype=np.float32)
        self._disturb_pos_jitter_scale = np.zeros((num_envs, 3), dtype=np.float32)
        self._disturb_rot_alpha = np.ones((num_envs, 2), dtype=np.float32)
        self._disturb_rot_limit_scale = np.ones((num_envs, 2), dtype=np.float32)
        self._disturb_rot_noise_scale = np.zeros((num_envs, 2), dtype=np.float32)
        self._disturb_rot_jitter_scale = np.zeros((num_envs, 2), dtype=np.float32)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        actions = actions.detach().cpu().numpy()
        self._policy_action[:] = _normalize_actions(actions, self._num_envs, self._action_dim)
        return state

    def _smooth_actions(self, raw_actions: np.ndarray) -> None:
        alpha = float(self._cfg.action_smooth)
        action_exec = (alpha * raw_actions + (1.0 - alpha) * self._prev_action_exec).astype(np.float32)
        self._prev_action_exec[:] = action_exec
        self._action_exec[:] = action_exec

    def _write_body_state(
        self,
        body_name: str,
        env_ids: np.ndarray,
        pos: np.ndarray,
        quat: np.ndarray,
        lin_vel: np.ndarray,
        ang_vel: np.ndarray,
    ) -> None:
        env_ids = np.asarray(env_ids, dtype=np.int64)
        program = self._body_state_writes[body_name]
        program.buffer(f"{body_name}_position")[env_ids, 0] = np.asarray(pos, np.float32)
        program.buffer(f"{body_name}_rotation")[env_ids, 0] = _normalize_quat(quat)
        program.buffer(f"{body_name}_linear_velocity")[env_ids, 0] = np.ascontiguousarray(lin_vel, dtype=np.float32)
        program.buffer(f"{body_name}_angular_velocity")[env_ids, 0] = np.ascontiguousarray(ang_vel, dtype=np.float32)
        program.execute(env_ids)

    def _compute_rel_xy(self, rows) -> np.ndarray:
        inputs = self.sim_data
        top_pos = inputs["top_pos"][rows]
        top_quat = inputs["top_quat"][rows]
        ball_pos = inputs["ball_pos"][rows]
        quat_flat, rel_flat, rel_shape = _broadcast_quat_vec(top_quat, ball_pos - top_pos)
        rel = quaternion.rotate_inverse(quat_flat, rel_flat).reshape(*rel_shape, 3).astype(np.float32)
        return np.linalg.norm(rel[:, :2], axis=-1).astype(np.float32)

    def _compute_leg_ctrls(self, rows, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        bottom_positions = np.stack(
            [self.sim_data[f"{leg}_pos"][rows] for leg in _STEWART_LEGS],
            axis=1,
        )
        quat_flat, offset_flat, offset_shape = _broadcast_quat_vec(
            target_quat[:, None, :], self._top_connect_offsets[None, :, :]
        )
        rotated_offsets = quaternion.rotate_vector(quat_flat, offset_flat).reshape(*offset_shape, 3).astype(np.float32)
        expected = target_pos[:, None, :] + rotated_offsets
        leg_lengths = np.linalg.norm(expected - bottom_positions, axis=-1) - self._leg_length_init[None, :]
        return np.clip(leg_lengths, self._actuator_low, self._actuator_high).astype(np.float32)

    def _write_ctrl_rows(self, env_ids: np.ndarray, values: np.ndarray) -> None:
        # The ctrl write program is full-width; its buffer preserves the other rows' targets.
        ctrls = self.sim_data["actuator_ctrls"].copy()
        ctrls[env_ids] = values
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = ctrls
        self._ctrl_writes.execute()

    def _apply_pose_delta(self, actions: np.ndarray) -> None:
        rel_xy_now = self._compute_rel_xy(slice(None))
        if self._cfg.center_control_radius > 0.0 and self._cfg.center_control_min_gain < 1.0:
            ratio = np.clip(rel_xy_now / max(self._cfg.center_control_radius, 1e-6), 0.0, 1.0)
            gain = self._cfg.center_control_min_gain + (1.0 - self._cfg.center_control_min_gain) * ratio
        else:
            gain = np.ones((actions.shape[0],), dtype=np.float32)
        effective_actions = actions * gain[:, None]

        target_pos = np.tile(self._top_pos_init, (actions.shape[0], 1)).astype(np.float32)
        target_roll_deg = effective_actions[:, 0] * self._cfg.target_rotation_limit_deg
        target_pitch_deg = effective_actions[:, 1] * self._cfg.target_rotation_limit_deg
        target_tilt_cmd = np.stack([target_roll_deg, target_pitch_deg], axis=-1).astype(np.float32)
        target_euler_deg = np.concatenate([target_tilt_cmd, np.zeros((actions.shape[0], 1), dtype=np.float32)], axis=-1)
        target_euler_rad = np.deg2rad(target_euler_deg).astype(np.float32)
        target_quat = quaternion.from_euler(
            target_euler_rad[..., 0], target_euler_rad[..., 1], target_euler_rad[..., 2]
        ).astype(np.float32)

        self._target_pos[:] = target_pos
        self._target_quat[:] = target_quat
        self._target_tilt_cmd[:] = target_tilt_cmd

    def _clear_disturbance_state(self, env_ids: np.ndarray) -> None:
        self._disturb_time[env_ids] = 0.0
        self._disturb_pos[env_ids] = 0.0
        self._disturb_lin_vel[env_ids] = 0.0
        self._disturb_rot_deg[env_ids] = 0.0
        self._disturb_ang_vel_deg[env_ids] = 0.0
        self._disturb_pos_alpha[env_ids] = 1.0
        self._disturb_pos_limit_scale[env_ids] = 1.0
        self._disturb_pos_noise_scale[env_ids] = 0.0
        self._disturb_pos_jitter_scale[env_ids] = 0.0
        self._disturb_rot_alpha[env_ids] = 1.0
        self._disturb_rot_limit_scale[env_ids] = 1.0
        self._disturb_rot_noise_scale[env_ids] = 0.0
        self._disturb_rot_jitter_scale[env_ids] = 0.0

    def _reset_episode_disturbance(self, env_ids: np.ndarray) -> None:
        self._clear_disturbance_state(env_ids)
        if (not self._cfg.disturbance_enabled) or self._cfg.disturbance_scale <= 0.0:
            return

        num = len(env_ids)
        freq_min = max(1e-4, float(self._cfg.disturb_freq_min_hz))
        freq_max = max(freq_min, float(self._cfg.disturb_freq_max_hz))

        pos_freq = np.random.uniform(freq_min, freq_max, size=(num, 3)).astype(np.float32)
        rot_freq = np.random.uniform(freq_min, freq_max, size=(num, 2)).astype(np.float32)
        self._disturb_pos_alpha[env_ids] = np.exp(-2.0 * np.pi * pos_freq * self._cfg.ctrl_dt).astype(np.float32)
        self._disturb_rot_alpha[env_ids] = np.exp(-2.0 * np.pi * rot_freq * self._cfg.ctrl_dt).astype(np.float32)
        pos_limit_scale = np.ones((num, 3), dtype=np.float32)
        if self._cfg.disturb_pos_xy_max > 0.0:
            xy_min_ratio = np.clip(
                self._cfg.disturb_pos_xy_min / max(self._cfg.disturb_pos_xy_max, 1e-8),
                0.0,
                1.0,
            ).astype(np.float32)
            pos_limit_scale[:, :2] = np.random.uniform(xy_min_ratio, 1.00, size=(num, 2)).astype(np.float32)
        if self._cfg.disturb_pos_z_max > 0.0:
            pos_limit_scale[:, 2] = np.random.uniform(0.75, 1.00, size=(num,)).astype(np.float32)
        self._disturb_pos_limit_scale[env_ids] = pos_limit_scale
        self._disturb_rot_limit_scale[env_ids] = np.random.uniform(0.75, 1.00, size=(num, 2)).astype(np.float32)
        self._disturb_pos_noise_scale[env_ids] = np.random.uniform(0.35, 0.60, size=(num, 3)).astype(np.float32)
        self._disturb_rot_noise_scale[env_ids] = np.random.uniform(0.35, 0.60, size=(num, 2)).astype(np.float32)
        self._disturb_pos_jitter_scale[env_ids] = np.random.uniform(0.02, 0.06, size=(num, 3)).astype(np.float32)
        self._disturb_rot_jitter_scale[env_ids] = np.random.uniform(0.03, 0.08, size=(num, 2)).astype(np.float32)

    def _update_disturbance_state(self, advance: bool) -> None:
        num = self._num_envs
        if advance:
            self._disturb_time += self._cfg.ctrl_dt
        if (not self._cfg.disturbance_enabled) or self._cfg.disturbance_scale <= 0.0:
            self._disturb_pos[:] = 0.0
            self._disturb_lin_vel[:] = 0.0
            self._disturb_rot_deg[:] = 0.0
            self._disturb_ang_vel_deg[:] = 0.0
            return

        ramp = np.ones((num,), dtype=np.float32)
        if self._cfg.disturb_ramp_seconds > 1e-8:
            ramp = np.clip(self._disturb_time / self._cfg.disturb_ramp_seconds, 0.0, 1.0).astype(np.float32)

        base_pos_limit = self._cfg.disturbance_scale * np.array(
            [self._cfg.disturb_pos_xy_max, self._cfg.disturb_pos_xy_max, self._cfg.disturb_pos_z_max],
            dtype=np.float32,
        )
        base_rot_limit = self._cfg.disturbance_scale * np.array(
            [self._cfg.disturb_rot_max_deg, self._cfg.disturb_rot_max_deg],
            dtype=np.float32,
        )
        pos_limit = ramp[:, None] * base_pos_limit[None, :] * self._disturb_pos_limit_scale
        rot_limit = ramp[:, None] * base_rot_limit[None, :] * self._disturb_rot_limit_scale
        pos_noise_std = ramp[:, None] * base_pos_limit[None, :] * self._disturb_pos_noise_scale
        rot_noise_std = ramp[:, None] * base_rot_limit[None, :] * self._disturb_rot_noise_scale
        pos_jitter_std = ramp[:, None] * base_pos_limit[None, :] * self._disturb_pos_jitter_scale
        rot_jitter_std = ramp[:, None] * base_rot_limit[None, :] * self._disturb_rot_jitter_scale

        prev_pos = self._disturb_pos.copy()
        prev_rot = self._disturb_rot_deg.copy()
        pos_noise = np.random.standard_normal((num, 3)).astype(np.float32)
        rot_noise = np.random.standard_normal((num, 2)).astype(np.float32)
        pos_jitter = np.random.standard_normal((num, 3)).astype(np.float32)
        rot_jitter = np.random.standard_normal((num, 2)).astype(np.float32)
        pos_blend = np.sqrt(np.maximum(1.0 - self._disturb_pos_alpha**2, 0.0)).astype(np.float32)
        rot_blend = np.sqrt(np.maximum(1.0 - self._disturb_rot_alpha**2, 0.0)).astype(np.float32)

        disturb_pos = (
            self._disturb_pos_alpha * prev_pos + pos_blend * pos_noise_std * pos_noise + pos_jitter_std * pos_jitter
        )
        disturb_rot_deg = (
            self._disturb_rot_alpha * prev_rot + rot_blend * rot_noise_std * rot_noise + rot_jitter_std * rot_jitter
        )
        disturb_pos = np.clip(disturb_pos, -pos_limit, pos_limit)
        disturb_rot_deg = np.clip(disturb_rot_deg, -rot_limit, rot_limit)

        self._disturb_pos[:] = disturb_pos.astype(np.float32)
        self._disturb_rot_deg[:] = disturb_rot_deg.astype(np.float32)
        self._disturb_lin_vel[:] = ((disturb_pos - prev_pos) / max(self._cfg.ctrl_dt, 1e-8)).astype(np.float32)
        self._disturb_ang_vel_deg[:] = ((disturb_rot_deg - prev_rot) / max(self._cfg.ctrl_dt, 1e-8)).astype(np.float32)

    def _apply_disturbance_to_stage(self, env_ids: np.ndarray) -> None:
        num = len(env_ids)
        disturb_rot_deg = np.concatenate(
            [self._disturb_rot_deg[env_ids], np.zeros((num, 1), dtype=np.float32)],
            axis=-1,
        )
        disturb_rot_rad = np.deg2rad(disturb_rot_deg).astype(np.float32)
        disturb_rot_quat = quaternion.from_euler(
            disturb_rot_rad[..., 0], disturb_rot_rad[..., 1], disturb_rot_rad[..., 2]
        ).astype(np.float32)
        stage_quat = quaternion.mul(
            disturb_rot_quat, np.broadcast_to(self._stage_quat_init, disturb_rot_quat.shape)
        ).astype(np.float32)
        roll_rad, pitch_rad, yaw_rad = quaternion.get_euler_xyz(_normalize_quat(stage_quat))
        roll_deg = np.rad2deg(roll_rad).astype(np.float32)
        pitch_deg = np.rad2deg(pitch_rad).astype(np.float32)
        yaw_deg = np.rad2deg(yaw_rad).astype(np.float32)
        roll_deg = np.clip(roll_deg, -self._cfg.disturb_rot_limit_deg, self._cfg.disturb_rot_limit_deg)
        pitch_deg = np.clip(pitch_deg, -self._cfg.disturb_rot_limit_deg, self._cfg.disturb_rot_limit_deg)
        stage_euler_deg = np.stack([roll_deg, pitch_deg, yaw_deg], axis=-1).astype(np.float32)
        stage_euler_rad = np.deg2rad(stage_euler_deg).astype(np.float32)
        stage_quat = quaternion.from_euler(
            stage_euler_rad[..., 0], stage_euler_rad[..., 1], stage_euler_rad[..., 2]
        ).astype(np.float32)
        stage_pos = self._stage_pos_init[None, :] + self._disturb_pos[env_ids]
        stage_ang_vel = np.deg2rad(
            np.concatenate(
                [self._disturb_ang_vel_deg[env_ids], np.zeros((num, 1), dtype=np.float32)],
                axis=-1,
            )
        ).astype(np.float32)
        self._write_body_state(
            "disturb_stage",
            env_ids,
            stage_pos,
            stage_quat,
            self._disturb_lin_vel[env_ids],
            stage_ang_vel,
        )

    def _get_disturbance_obs(self) -> np.ndarray | None:
        if not (self._cfg.disturbance_enabled and self._cfg.disturbance_include_obs):
            return None
        rot_obs_scale = max(float(self._cfg.disturb_rot_limit_deg), 1e-6)
        ang_vel_obs_scale = max(float(self._cfg.disturb_ang_vel_obs_scale_deg_per_s), 1e-6)
        return np.concatenate(
            [
                self._disturb_pos,
                self._disturb_lin_vel,
                self._disturb_rot_deg / rot_obs_scale,
                self._disturb_ang_vel_deg / ang_vel_obs_scale,
            ],
            axis=-1,
        ).astype(np.float32)

    def _update_ball_kinematics(self) -> dict[str, np.ndarray]:
        """Advance the ball-on-platform state estimates for the full batch.

        Reads fresh simulator quantities (already refreshed by the
        ``sim_data.execute()`` at the top of ``compute_transition``), updates
        the filtered velocity estimates tracked on the instance, and returns
        the physical quantities shared by reward and termination.
        """
        inputs = self.sim_data
        top_pos = inputs["top_pos"]
        ball_pos = inputs["ball_pos"]
        top_quat = _normalize_quat(inputs["top_quat"])

        quat_flat, rel_flat, rel_shape = _broadcast_quat_vec(top_quat, ball_pos - top_pos)
        rel = quaternion.rotate_inverse(quat_flat, rel_flat).reshape(*rel_shape, 3).astype(np.float32)
        rel_vel = (rel - self._prev_rel) / self._cfg.ctrl_dt
        self._prev_rel[:] = rel
        filtered_rel_vel = self._cfg.vel_smooth * rel_vel + (1.0 - self._cfg.vel_smooth) * self._filtered_rel_vel
        self._filtered_rel_vel[:] = filtered_rel_vel.astype(np.float32)
        self._last_rel_vel[:] = filtered_rel_vel.astype(np.float32)

        quat_delta = quaternion.mul(top_quat, quaternion.conjugate(self._prev_top_quat)).astype(np.float32)
        top_ang_vel = _quat_to_rotvec(quat_delta) / self._cfg.ctrl_dt
        filtered_top_ang_vel = (
            self._cfg.vel_smooth * top_ang_vel + (1.0 - self._cfg.vel_smooth) * self._filtered_top_ang_vel
        )
        self._filtered_top_ang_vel[:] = filtered_top_ang_vel.astype(np.float32)
        self._last_top_ang_vel[:] = filtered_top_ang_vel.astype(np.float32)
        self._prev_top_quat[:] = top_quat.astype(np.float32)

        roll_rad, pitch_rad, _ = quaternion.get_euler_xyz(top_quat)
        roll_deg = np.rad2deg(roll_rad).astype(np.float32)
        pitch_deg = np.rad2deg(pitch_rad).astype(np.float32)
        return {
            "top_pos": top_pos.astype(np.float32),
            "ball_pos": ball_pos.astype(np.float32),
            "rel_xy": np.linalg.norm(rel[:, :2], axis=-1).astype(np.float32),
            "tilt_deg": np.maximum(np.abs(roll_deg), np.abs(pitch_deg)).astype(np.float32),
        }

    def _prepare_control_step(self) -> None:
        self._smooth_actions(self._policy_action)
        self._apply_pose_delta(self._action_exec)
        self._update_disturbance_state(advance=True)
        all_ids = np.arange(self._num_envs, dtype=np.int64)
        self._apply_disturbance_to_stage(all_ids)
        self.sim_data.execute(all_ids)
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = self._compute_leg_ctrls(slice(None), self._target_pos, self._target_quat)
        self._ctrl_writes.execute()

    def _simulate_control_step(self) -> None:
        all_ids = np.arange(self._num_envs, dtype=np.int64)
        for _ in range(self._cfg.sim_substeps):
            self._apply_disturbance_to_stage(all_ids)
            self.sim.step(1)

        self._apply_disturbance_to_stage(all_ids)

    def physics_step(self) -> None:
        # Stewart interleaves floating-base disturbance writes with per-substep
        # physics and FK-refreshed leg reads, so it owns the whole control step;
        # the post-step simulator refresh happens in compute_transition.
        self._prepare_control_step()
        self._simulate_control_step()

    def _update_stillness(self, rel_xy: np.ndarray, vel_xy: np.ndarray) -> np.ndarray:
        still_xy_enter = float(self._cfg.still_xy)
        still_vel_enter = float(self._cfg.still_vel)
        still_xy_exit = float(self._cfg.still_xy * self._cfg.still_xy_hysteresis)
        still_vel_exit = float(self._cfg.still_vel * self._cfg.still_vel_hysteresis)

        still_window_active = self._still_window_active
        still_steps = self._still_steps

        keep_mask = still_window_active & (rel_xy <= still_xy_exit) & (vel_xy <= still_vel_exit)
        break_mask = still_window_active & ~keep_mask
        enter_mask = (~still_window_active) & (rel_xy <= still_xy_enter) & (vel_xy <= still_vel_enter)
        idle_mask = (~still_window_active) & ~enter_mask

        still_steps[keep_mask] += 1
        still_window_active[break_mask] = False
        still_steps[break_mask] = 0
        still_window_active[enter_mask] = True
        still_steps[enter_mask] = 1
        still_steps[idle_mask] = 0

        return still_steps

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        top_quat = _normalize_quat(inputs["top_quat"])

        roll_rad, pitch_rad, _ = quaternion.get_euler_xyz(top_quat)
        roll_deg = np.rad2deg(roll_rad).astype(np.float32)
        pitch_deg = np.rad2deg(pitch_rad).astype(np.float32)
        quat_flat, ang_vel_flat, ang_vel_shape = _broadcast_quat_vec(top_quat, self._filtered_top_ang_vel)
        top_ang_vel_local = (
            quaternion.rotate_inverse(quat_flat, ang_vel_flat).reshape(*ang_vel_shape, 3).astype(np.float32)
        )
        obs_parts = [
            self._prev_rel,
            self._filtered_rel_vel,
            np.stack(
                [
                    roll_deg / self._cfg.target_rotation_limit_deg,
                    pitch_deg / self._cfg.target_rotation_limit_deg,
                ],
                axis=-1,
            ).astype(np.float32),
            top_ang_vel_local.astype(np.float32),
            (self._target_tilt_cmd / max(self._cfg.target_rotation_limit_deg, 1e-6)).astype(np.float32),
            self._action_exec,
        ]
        disturb_obs = self._get_disturbance_obs()
        if disturb_obs is not None:
            obs_parts.append(disturb_obs)
        return state.replace(obs=np.concatenate(obs_parts, axis=-1).astype(np.float32))

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        state_cache = self._update_ball_kinematics()

        top_pos = state_cache["top_pos"]
        ball_pos = state_cache["ball_pos"]
        rel_xy = state_cache["rel_xy"]
        tilt_deg = state_cache["tilt_deg"]

        rel_vel = self._last_rel_vel
        top_ang_vel = self._last_top_ang_vel
        vel_xy = np.linalg.norm(rel_vel[:, :2], axis=-1).astype(np.float32)
        top_ang_mag = np.linalg.norm(top_ang_vel, axis=-1).astype(np.float32)

        fall_z = top_pos[:, 2] - np.sin(np.deg2rad(30.0)).astype(np.float32) * self._cfg.platform_radius
        fallen = (rel_xy > self._cfg.platform_radius) | (ball_pos[:, 2] < fall_z)

        center_score = np.clip(1.0 - rel_xy / max(self._cfg.platform_radius, 1e-6), 0.0, 1.0).astype(np.float32)
        term_center = (self._cfg.k_center * center_score).astype(np.float32)

        prev_zero_vel_rel_xy = self._prev_zero_vel_rel_xy
        zero_reference = np.where(np.isfinite(prev_zero_vel_rel_xy), prev_zero_vel_rel_xy, self._initial_rel_xy)
        zero_event = vel_xy <= self._cfg.zero_vel_thresh
        zero_closer_mask = zero_event & (rel_xy < zero_reference)
        zero_improve = np.maximum(zero_reference - rel_xy, 0.0)
        zero_improve_norm = np.clip(zero_improve / max(self._cfg.platform_radius, 1e-6), 0.0, 1.0)
        term_zero_vel_closer = np.where(zero_closer_mask, self._cfg.k_progress * zero_improve_norm, 0.0).astype(
            np.float32
        )

        self._prev_zero_vel_rel_xy[zero_event] = rel_xy[zero_event]

        still_steps = self._update_stillness(rel_xy, vel_xy)
        success = still_steps >= self._cfg.still_steps_needed
        term_still_bonus = np.where(success, self._cfg.k_still, 0.0).astype(np.float32)

        reward = (term_center + term_zero_vel_closer + term_still_bonus).astype(np.float32)

        term_terminal = np.zeros((self._num_envs,), dtype=np.float32)
        reward = np.where(fallen, self._cfg.fall_penalty, reward).astype(np.float32)
        term_terminal[fallen] = self._cfg.fall_penalty

        # Timeout flag for logging only; the framework owns truncation via its
        # terminated/truncated bookkeeping.
        timeout = np.zeros((self._num_envs,), dtype=bool)
        if self._cfg.max_episode_steps is not None and self._cfg.max_episode_steps > 0:
            timeout = (state.episode_steps.numpy() + 1) >= self._cfg.max_episode_steps
            timeout &= ~(fallen | success)

        terminated = (fallen | success).astype(bool)

        state.metrics = {
            "rel_xy": rel_xy.astype(np.float32),
            "vel_xy": vel_xy.astype(np.float32),
            "tilt_deg": tilt_deg.astype(np.float32),
            "top_ang_mag": top_ang_mag.astype(np.float32),
            "still_steps": still_steps.astype(np.float32),
            "success": success.astype(np.float32),
            "fallen": fallen.astype(np.float32),
            "timeout": timeout.astype(np.float32),
            "disturb_pos_norm": np.linalg.norm(self._disturb_pos, axis=-1).astype(np.float32),
            "disturb_rot_norm_deg": np.linalg.norm(self._disturb_rot_deg, axis=-1).astype(np.float32),
        }
        state.reward_terms = {
            "center": term_center.astype(np.float32),
            "zero_vel_reference": zero_reference.astype(np.float32),
            "zero_vel_improve": zero_improve.astype(np.float32),
            "zero_vel_closer": term_zero_vel_closer.astype(np.float32),
            "still_bonus": term_still_bonus.astype(np.float32),
            "terminal": term_terminal.astype(np.float32),
        }

        return state.replace(reward=reward, terminated=terminated)

    def reset(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.detach().cpu().numpy()
        num = len(env_ids)
        row_ids = np.asarray(env_ids, dtype=np.int64)
        zeros3 = np.zeros((num, 3), dtype=np.float32)

        roll_deg = np.random.uniform(self._cfg.min_init_tilt_deg, self._cfg.init_tilt_deg, size=(num,)).astype(
            np.float32
        )
        roll_deg *= np.random.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(num,))
        pitch_deg = np.random.uniform(self._cfg.min_init_tilt_deg, self._cfg.init_tilt_deg, size=(num,)).astype(
            np.float32
        )
        pitch_deg *= np.random.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(num,))

        target_pos = np.tile(self._top_pos_init, (num, 1)).astype(np.float32)
        target_euler_deg = np.stack([roll_deg, pitch_deg, np.zeros((num,), dtype=np.float32)], axis=-1).astype(
            np.float32
        )
        target_euler_rad = np.deg2rad(target_euler_deg).astype(np.float32)
        target_quat = quaternion.from_euler(
            target_euler_rad[..., 0], target_euler_rad[..., 1], target_euler_rad[..., 2]
        ).astype(np.float32)

        # Write episode-scoped state for the reset rows
        self._target_pos[env_ids] = target_pos
        self._target_quat[env_ids] = target_quat
        self._target_tilt_cmd[env_ids] = np.stack([roll_deg, pitch_deg], axis=-1)
        self._prev_rel[env_ids] = 0.0
        self._filtered_rel_vel[env_ids] = 0.0
        self._last_rel_vel[env_ids] = 0.0
        self._filtered_top_ang_vel[env_ids] = 0.0
        self._last_top_ang_vel[env_ids] = 0.0
        self._still_steps[env_ids] = 0
        self._still_window_active[env_ids] = False
        self._policy_action[env_ids] = 0.0
        self._prev_action_exec[env_ids] = 0.0
        self._action_exec[env_ids] = 0.0

        stage_pose = np.concatenate(
            [np.tile(self._stage_pos_init, (num, 1)), _normalize_quat(np.tile(self._stage_quat_init, (num, 1)))],
            axis=-1,
        )
        top_pose = np.concatenate([target_pos, _normalize_quat(target_quat)], axis=-1)
        hidden_ball_pos = np.tile(np.array([0.0, 0.0, -10.0], dtype=np.float32), (num, 1))
        ball_pose = np.concatenate([hidden_ball_pos, _identity_quat((num,))], axis=-1)
        for name, pose in (("stage", stage_pose), ("top", top_pose), ("ball", ball_pose)):
            self._reset_program.buffer(f"{name}_position")[env_ids, 0] = pose[:, :3]
            self._reset_program.buffer(f"{name}_rotation")[env_ids, 0] = pose[:, 3:7]
            self._reset_program.buffer(f"{name}_linear_velocity")[env_ids, 0] = 0.0
            self._reset_program.buffer(f"{name}_angular_velocity")[env_ids, 0] = 0.0
        self._reset_program.buffer("legs_position")[env_ids] = 0.0
        self._reset_program.buffer("legs_velocity")[env_ids] = 0.0
        self._reset_program.execute(env_ids)

        # Kinematic leg bring-up: place the slide joints at the configuration
        # that realizes the commanded top pose. The row reset already leaves
        # kinematics consistent (FK runs as part of the write), so the leg
        # lengths follow analytically from the refreshed geometry — reset
        # never advances physics.
        self.sim_data.execute(row_ids)
        leg_lengths = self._compute_leg_ctrls(env_ids, target_pos, target_quat)
        self._reset_program.buffer("legs_position")[env_ids] = leg_lengths
        self._reset_program.execute(env_ids)

        radius = (
            self._cfg.platform_radius
            * self._cfg.init_ball_radius_ratio
            * np.sqrt(np.random.uniform(0.0, 1.0, size=(num,)).astype(np.float32))
        )
        theta = np.random.uniform(0.0, 2.0 * np.pi, size=(num,)).astype(np.float32)
        local_xy = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=-1)
        local_pos = np.concatenate(
            [
                local_xy,
                np.full((num, 1), self._cfg.top_surface_offset + self._cfg.ball_radius, dtype=np.float32),
            ],
            axis=-1,
        )
        quat_flat, local_pos_flat, local_pos_shape = _broadcast_quat_vec(target_quat, local_pos)
        world_pos = target_pos + quaternion.rotate_vector(quat_flat, local_pos_flat).reshape(
            *local_pos_shape, 3
        ).astype(np.float32)
        self._write_body_state("ball", env_ids, world_pos, _identity_quat((num,)), zeros3, zeros3)
        self._write_ctrl_rows(env_ids, np.zeros((num, self._model_action_dim), dtype=np.float32))
        self.sim_data.execute(row_ids)

        ball_pos = self.sim_data["ball_pos"][env_ids]
        top_pos = self.sim_data["top_pos"][env_ids]
        self._prev_top_quat[env_ids] = _normalize_quat(self.sim_data["top_quat"][env_ids])
        quat_flat, rel_flat, rel_shape = _broadcast_quat_vec(self._prev_top_quat[env_ids], ball_pos - top_pos)
        self._prev_rel[env_ids] = (
            quaternion.rotate_inverse(quat_flat, rel_flat).reshape(*rel_shape, 3).astype(np.float32)
        )
        initial_rel_xy = self._compute_rel_xy(env_ids)
        self._initial_rel_xy[env_ids] = initial_rel_xy
        self._prev_zero_vel_rel_xy[env_ids] = initial_rel_xy

        self._reset_episode_disturbance(env_ids)
        self._apply_disturbance_to_stage(env_ids)
        self.sim_data.execute(row_ids)
