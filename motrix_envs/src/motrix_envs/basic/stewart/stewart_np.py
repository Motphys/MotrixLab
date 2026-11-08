# Copyright Motphys Technology Co., Ltd. 2025, 2026

import math

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct.env import DirectEnv
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


def _normalize_actions(actions: torch.Tensor, num_envs: int, action_dim: int) -> torch.Tensor:
    actions = actions.to(torch.float32)
    if actions.ndim == 1:
        if num_envs != 1 or actions.shape[0] != action_dim:
            raise ValueError(f"Expected action shape ({num_envs}, {action_dim}) or ({action_dim},).")
        actions = actions.reshape(1, action_dim)
    if tuple(actions.shape) != (num_envs, action_dim):
        raise ValueError(f"Expected action shape ({num_envs}, {action_dim}), got {tuple(actions.shape)}.")
    return actions.clamp(-1.0, 1.0)


def _identity_quat(shape: tuple[int, ...]) -> torch.Tensor:
    quat = torch.zeros((*shape, 4), dtype=torch.float32)
    quat[..., 3] = 1.0
    return quat


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = quat.to(torch.float32)
    norm = torch.norm(quat, dim=-1, keepdim=True)
    identity = _identity_quat(tuple(quat.shape[:-1]))
    safe = norm > 1e-8
    return torch.where(safe, quat / torch.clamp(norm, min=1e-8), identity)


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = lhs[..., 0], lhs[..., 1], lhs[..., 2], lhs[..., 3]
    x2, y2, z2, w2 = rhs[..., 0], rhs[..., 1], rhs[..., 2], rhs[..., 3]
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([-quat[..., :3], quat[..., 3:4]], dim=-1)


def _rotate_vector(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Forward-rotate vectors by (x, y, z, w) quaternions; inputs broadcast."""
    w = quat[..., 3:4]
    im = quat[..., :3]
    t = 2.0 * torch.cross(im, vec, dim=-1)
    return vec + w * t + torch.cross(im, t, dim=-1)


def _rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Inverse-rotate vectors by (x, y, z, w) quaternions; inputs broadcast."""
    w = quat[..., 3:4]
    im = quat[..., :3]
    term = torch.cross(im, vec, dim=-1) - w * vec
    return vec + 2.0 * torch.cross(im, term, dim=-1)


def _from_euler(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return torch.stack([qx, qy, qz, qw], dim=-1).to(torch.float32)


def _euler_xyz(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert (x, y, z, w) quaternions to roll/pitch/yaw Euler angles."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1.0,
        torch.copysign(torch.full_like(sinp, math.pi / 2.0), sinp),
        torch.asin(sinp.clamp(-1.0, 1.0)),
    )

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def _quat_to_rotvec(quat: torch.Tensor) -> torch.Tensor:
    quat = _normalize_quat(quat)
    flip = quat[..., 3] < 0.0
    quat = torch.where(flip[..., None], -quat, quat)
    xyz = quat[..., :3]
    w = quat[..., 3].clamp(-1.0, 1.0)
    xyz_norm = torch.norm(xyz, dim=-1)
    angle = 2.0 * torch.atan2(xyz_norm, w)
    scale = torch.where(xyz_norm > 1e-8, angle / torch.clamp(xyz_norm, min=1e-8), torch.full_like(xyz_norm, 2.0))
    return xyz * scale[..., None]


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

        self._actuator_low = torch.tensor([spec.ctrl_range[0] for spec in self.model.actuators], dtype=torch.float32)
        self._actuator_high = torch.tensor([spec.ctrl_range[1] for spec in self.model.actuators], dtype=torch.float32)

        # Snapshot the default-pose geometry the same way the legacy env read
        # it from a fresh SceneData: write default rows (reset runs FK) and
        # read the compiled queries once before any episode reset.
        default_ids = torch.arange(self._num_envs, dtype=torch.int64)
        self.sim_data.execute(default_ids)
        inputs = self.sim_data
        self._top_pos_init = inputs["top_pos"][0].clone()
        stage_pose = torch.cat([inputs["stage_pos"][0], inputs["stage_quat"][0]], dim=-1)
        self._stage_pos_init = stage_pose[:3].clone()
        self._stage_quat_init = _normalize_quat(stage_pose[3:7])

        top_connect_offsets = []
        leg_length_init = []
        for leg, top_connect in zip(_STEWART_LEGS, _STEWART_TOP_CONNECTS):
            bottom_pos = inputs[f"{leg}_pos"][0].clone()
            top_connect_pos = inputs[f"{top_connect}_pos"][0].clone()
            top_connect_offsets.append(top_connect_pos - self._top_pos_init)
            leg_length_init.append(torch.norm(top_connect_pos - bottom_pos))
        self._top_connect_offsets = torch.stack(top_connect_offsets, dim=0)
        self._leg_length_init = torch.stack(leg_length_init, dim=0)

        # Episode-scoped task state: full-batch buffers, reset writes the done
        # rows in place.
        num_envs = self._num_envs
        self._target_pos = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._target_quat = _identity_quat((num_envs,))
        self._target_tilt_cmd = torch.zeros((num_envs, 2), dtype=torch.float32)
        self._prev_rel = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._filtered_rel_vel = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._last_rel_vel = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._prev_top_quat = _identity_quat((num_envs,))
        self._filtered_top_ang_vel = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._last_top_ang_vel = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._initial_rel_xy = torch.zeros(num_envs, dtype=torch.float32)
        self._prev_zero_vel_rel_xy = torch.zeros(num_envs, dtype=torch.float32)
        self._still_steps = torch.zeros(num_envs, dtype=torch.int32)
        self._still_window_active = torch.zeros(num_envs, dtype=torch.bool)
        self._policy_action = torch.zeros((num_envs, self._action_dim), dtype=torch.float32)
        self._prev_action_exec = torch.zeros((num_envs, self._action_dim), dtype=torch.float32)
        self._action_exec = torch.zeros((num_envs, self._action_dim), dtype=torch.float32)
        self._disturb_time = torch.zeros(num_envs, dtype=torch.float32)
        self._disturb_pos = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._disturb_lin_vel = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._disturb_rot_deg = torch.zeros((num_envs, 2), dtype=torch.float32)
        self._disturb_ang_vel_deg = torch.zeros((num_envs, 2), dtype=torch.float32)
        self._disturb_pos_alpha = torch.ones((num_envs, 3), dtype=torch.float32)
        self._disturb_pos_limit_scale = torch.ones((num_envs, 3), dtype=torch.float32)
        self._disturb_pos_noise_scale = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._disturb_pos_jitter_scale = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._disturb_rot_alpha = torch.ones((num_envs, 2), dtype=torch.float32)
        self._disturb_rot_limit_scale = torch.ones((num_envs, 2), dtype=torch.float32)
        self._disturb_rot_noise_scale = torch.zeros((num_envs, 2), dtype=torch.float32)
        self._disturb_rot_jitter_scale = torch.zeros((num_envs, 2), dtype=torch.float32)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        self._policy_action[:] = _normalize_actions(actions, self._num_envs, self._action_dim)
        return state

    def _smooth_actions(self, raw_actions: torch.Tensor) -> None:
        alpha = float(self._cfg.action_smooth)
        action_exec = alpha * raw_actions + (1.0 - alpha) * self._prev_action_exec
        self._prev_action_exec[:] = action_exec
        self._action_exec[:] = action_exec

    def _write_body_state(
        self,
        body_name: str,
        env_ids: torch.Tensor,
        pos: torch.Tensor,
        quat: torch.Tensor,
        lin_vel: torch.Tensor,
        ang_vel: torch.Tensor,
    ) -> None:
        program = self._body_state_writes[body_name]
        program.buffer(f"{body_name}_position")[env_ids, 0] = pos.to(torch.float32)
        program.buffer(f"{body_name}_rotation")[env_ids, 0] = _normalize_quat(quat)
        program.buffer(f"{body_name}_linear_velocity")[env_ids, 0] = lin_vel.to(torch.float32)
        program.buffer(f"{body_name}_angular_velocity")[env_ids, 0] = ang_vel.to(torch.float32)
        program.execute(env_ids)

    def _compute_rel_xy(self, rows) -> torch.Tensor:
        inputs = self.sim_data
        top_pos = inputs["top_pos"][rows]
        top_quat = inputs["top_quat"][rows]
        ball_pos = inputs["ball_pos"][rows]
        rel = _rotate_inverse(_normalize_quat(top_quat), ball_pos - top_pos)
        return torch.norm(rel[:, :2], dim=-1)

    def _compute_leg_ctrls(self, rows, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        bottom_positions = torch.stack(
            [self.sim_data[f"{leg}_pos"][rows] for leg in _STEWART_LEGS],
            dim=1,
        )
        rotated_offsets = _rotate_vector(target_quat[:, None, :], self._top_connect_offsets[None, :, :])
        expected = target_pos[:, None, :] + rotated_offsets
        leg_lengths = torch.norm(expected - bottom_positions, dim=-1) - self._leg_length_init[None, :]
        return leg_lengths.clamp(self._actuator_low, self._actuator_high)

    def _write_ctrl_rows(self, env_ids: torch.Tensor, values: torch.Tensor) -> None:
        # The ctrl write program is full-width; its buffer preserves the other rows' targets.
        ctrls = self.sim_data["actuator_ctrls"].clone()
        ctrls[env_ids] = values
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl.copy_(ctrls)
        self._ctrl_writes.execute()

    def _apply_pose_delta(self, actions: torch.Tensor) -> None:
        rel_xy_now = self._compute_rel_xy(slice(None))
        if self._cfg.center_control_radius > 0.0 and self._cfg.center_control_min_gain < 1.0:
            ratio = (rel_xy_now / max(self._cfg.center_control_radius, 1e-6)).clamp(0.0, 1.0)
            gain = self._cfg.center_control_min_gain + (1.0 - self._cfg.center_control_min_gain) * ratio
        else:
            gain = torch.ones((actions.shape[0],), dtype=torch.float32)
        effective_actions = actions * gain[:, None]

        target_pos = self._top_pos_init.unsqueeze(0).expand(actions.shape[0], -1).contiguous()
        target_roll_deg = effective_actions[:, 0] * self._cfg.target_rotation_limit_deg
        target_pitch_deg = effective_actions[:, 1] * self._cfg.target_rotation_limit_deg
        target_tilt_cmd = torch.stack([target_roll_deg, target_pitch_deg], dim=-1)
        target_euler_deg = torch.cat([target_tilt_cmd, torch.zeros((actions.shape[0], 1), dtype=torch.float32)], dim=-1)
        target_euler_rad = target_euler_deg * torch.pi / 180.0
        target_quat = _from_euler(target_euler_rad[..., 0], target_euler_rad[..., 1], target_euler_rad[..., 2])

        self._target_pos[:] = target_pos
        self._target_quat[:] = target_quat
        self._target_tilt_cmd[:] = target_tilt_cmd

    def _clear_disturbance_state(self, env_ids: torch.Tensor) -> None:
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

    def _reset_episode_disturbance(self, env_ids: torch.Tensor) -> None:
        self._clear_disturbance_state(env_ids)
        if (not self._cfg.disturbance_enabled) or self._cfg.disturbance_scale <= 0.0:
            return

        num = env_ids.numel()
        freq_min = max(1e-4, float(self._cfg.disturb_freq_min_hz))
        freq_max = max(freq_min, float(self._cfg.disturb_freq_max_hz))

        pos_freq = torch.empty(num, 3, dtype=torch.float32).uniform_(freq_min, freq_max)
        rot_freq = torch.empty(num, 2, dtype=torch.float32).uniform_(freq_min, freq_max)
        self._disturb_pos_alpha[env_ids] = torch.exp(-2.0 * torch.pi * pos_freq * self._cfg.ctrl_dt)
        self._disturb_rot_alpha[env_ids] = torch.exp(-2.0 * torch.pi * rot_freq * self._cfg.ctrl_dt)
        pos_limit_scale = torch.ones((num, 3), dtype=torch.float32)
        if self._cfg.disturb_pos_xy_max > 0.0:
            xy_min_ratio = min(max(self._cfg.disturb_pos_xy_min / max(self._cfg.disturb_pos_xy_max, 1e-8), 0.0), 1.0)
            pos_limit_scale[:, :2] = torch.empty(num, 2, dtype=torch.float32).uniform_(float(xy_min_ratio), 1.0)
        if self._cfg.disturb_pos_z_max > 0.0:
            pos_limit_scale[:, 2] = torch.empty(num, dtype=torch.float32).uniform_(0.75, 1.0)
        self._disturb_pos_limit_scale[env_ids] = pos_limit_scale
        self._disturb_rot_limit_scale[env_ids] = torch.empty(num, 2, dtype=torch.float32).uniform_(0.75, 1.0)
        self._disturb_pos_noise_scale[env_ids] = torch.empty(num, 3, dtype=torch.float32).uniform_(0.35, 0.60)
        self._disturb_rot_noise_scale[env_ids] = torch.empty(num, 2, dtype=torch.float32).uniform_(0.35, 0.60)
        self._disturb_pos_jitter_scale[env_ids] = torch.empty(num, 3, dtype=torch.float32).uniform_(0.02, 0.06)
        self._disturb_rot_jitter_scale[env_ids] = torch.empty(num, 2, dtype=torch.float32).uniform_(0.03, 0.08)

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

        ramp = torch.ones((num,), dtype=torch.float32)
        if self._cfg.disturb_ramp_seconds > 1e-8:
            ramp = (self._disturb_time / self._cfg.disturb_ramp_seconds).clamp(0.0, 1.0)

        base_pos_limit = self._cfg.disturbance_scale * torch.tensor(
            [self._cfg.disturb_pos_xy_max, self._cfg.disturb_pos_xy_max, self._cfg.disturb_pos_z_max],
            dtype=torch.float32,
        )
        base_rot_limit = self._cfg.disturbance_scale * torch.tensor(
            [self._cfg.disturb_rot_max_deg, self._cfg.disturb_rot_max_deg],
            dtype=torch.float32,
        )
        pos_limit = ramp[:, None] * base_pos_limit[None, :] * self._disturb_pos_limit_scale
        rot_limit = ramp[:, None] * base_rot_limit[None, :] * self._disturb_rot_limit_scale
        pos_noise_std = ramp[:, None] * base_pos_limit[None, :] * self._disturb_pos_noise_scale
        rot_noise_std = ramp[:, None] * base_rot_limit[None, :] * self._disturb_rot_noise_scale
        pos_jitter_std = ramp[:, None] * base_pos_limit[None, :] * self._disturb_pos_jitter_scale
        rot_jitter_std = ramp[:, None] * base_rot_limit[None, :] * self._disturb_rot_jitter_scale

        prev_pos = self._disturb_pos.clone()
        prev_rot = self._disturb_rot_deg.clone()
        pos_noise = torch.randn(num, 3, dtype=torch.float32)
        rot_noise = torch.randn(num, 2, dtype=torch.float32)
        pos_jitter = torch.randn(num, 3, dtype=torch.float32)
        rot_jitter = torch.randn(num, 2, dtype=torch.float32)
        pos_blend = torch.sqrt((1.0 - self._disturb_pos_alpha**2).clamp(min=0.0))
        rot_blend = torch.sqrt((1.0 - self._disturb_rot_alpha**2).clamp(min=0.0))

        disturb_pos = (
            self._disturb_pos_alpha * prev_pos + pos_blend * pos_noise_std * pos_noise + pos_jitter_std * pos_jitter
        )
        disturb_rot_deg = (
            self._disturb_rot_alpha * prev_rot + rot_blend * rot_noise_std * rot_noise + rot_jitter_std * rot_jitter
        )
        disturb_pos = disturb_pos.clamp(-pos_limit, pos_limit)
        disturb_rot_deg = disturb_rot_deg.clamp(-rot_limit, rot_limit)

        self._disturb_pos[:] = disturb_pos
        self._disturb_rot_deg[:] = disturb_rot_deg
        self._disturb_lin_vel[:] = (disturb_pos - prev_pos) / max(self._cfg.ctrl_dt, 1e-8)
        self._disturb_ang_vel_deg[:] = (disturb_rot_deg - prev_rot) / max(self._cfg.ctrl_dt, 1e-8)

    def _apply_disturbance_to_stage(self, env_ids: torch.Tensor) -> None:
        num = env_ids.numel()
        disturb_rot_deg = torch.cat(
            [self._disturb_rot_deg[env_ids], torch.zeros((num, 1), dtype=torch.float32)],
            dim=-1,
        )
        disturb_rot_rad = disturb_rot_deg * torch.pi / 180.0
        disturb_rot_quat = _from_euler(disturb_rot_rad[..., 0], disturb_rot_rad[..., 1], disturb_rot_rad[..., 2])
        stage_quat = _quat_mul(disturb_rot_quat, self._stage_quat_init.expand(disturb_rot_quat.shape))
        roll_rad, pitch_rad, yaw_rad = _euler_xyz(_normalize_quat(stage_quat))
        roll_deg = roll_rad * 180.0 / torch.pi
        pitch_deg = pitch_rad * 180.0 / torch.pi
        yaw_deg = yaw_rad * 180.0 / torch.pi
        roll_deg = roll_deg.clamp(-self._cfg.disturb_rot_limit_deg, self._cfg.disturb_rot_limit_deg)
        pitch_deg = pitch_deg.clamp(-self._cfg.disturb_rot_limit_deg, self._cfg.disturb_rot_limit_deg)
        stage_euler_deg = torch.stack([roll_deg, pitch_deg, yaw_deg], dim=-1)
        stage_euler_rad = stage_euler_deg * torch.pi / 180.0
        stage_quat = _from_euler(stage_euler_rad[..., 0], stage_euler_rad[..., 1], stage_euler_rad[..., 2])
        stage_pos = self._stage_pos_init[None, :] + self._disturb_pos[env_ids]
        stage_ang_vel = (
            torch.cat(
                [self._disturb_ang_vel_deg[env_ids], torch.zeros((num, 1), dtype=torch.float32)],
                dim=-1,
            )
            * torch.pi
            / 180.0
        )
        self._write_body_state(
            "disturb_stage",
            env_ids,
            stage_pos,
            stage_quat,
            self._disturb_lin_vel[env_ids],
            stage_ang_vel,
        )

    def _get_disturbance_obs(self) -> torch.Tensor | None:
        if not (self._cfg.disturbance_enabled and self._cfg.disturbance_include_obs):
            return None
        rot_obs_scale = max(float(self._cfg.disturb_rot_limit_deg), 1e-6)
        ang_vel_obs_scale = max(float(self._cfg.disturb_ang_vel_obs_scale_deg_per_s), 1e-6)
        return torch.cat(
            [
                self._disturb_pos,
                self._disturb_lin_vel,
                self._disturb_rot_deg / rot_obs_scale,
                self._disturb_ang_vel_deg / ang_vel_obs_scale,
            ],
            dim=-1,
        )

    def _update_ball_kinematics(self) -> dict[str, torch.Tensor]:
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

        rel = _rotate_inverse(top_quat, ball_pos - top_pos)
        rel_vel = (rel - self._prev_rel) / self._cfg.ctrl_dt
        self._prev_rel[:] = rel
        filtered_rel_vel = self._cfg.vel_smooth * rel_vel + (1.0 - self._cfg.vel_smooth) * self._filtered_rel_vel
        self._filtered_rel_vel[:] = filtered_rel_vel
        self._last_rel_vel[:] = filtered_rel_vel

        quat_delta = _quat_mul(top_quat, _quat_conjugate(self._prev_top_quat))
        top_ang_vel = _quat_to_rotvec(quat_delta) / self._cfg.ctrl_dt
        filtered_top_ang_vel = (
            self._cfg.vel_smooth * top_ang_vel + (1.0 - self._cfg.vel_smooth) * self._filtered_top_ang_vel
        )
        self._filtered_top_ang_vel[:] = filtered_top_ang_vel
        self._last_top_ang_vel[:] = filtered_top_ang_vel
        self._prev_top_quat[:] = top_quat

        roll_rad, pitch_rad, _ = _euler_xyz(top_quat)
        roll_deg = roll_rad * 180.0 / torch.pi
        pitch_deg = pitch_rad * 180.0 / torch.pi
        return {
            "top_pos": top_pos,
            "ball_pos": ball_pos,
            "rel_xy": torch.norm(rel[:, :2], dim=-1),
            "tilt_deg": torch.maximum(torch.abs(roll_deg), torch.abs(pitch_deg)),
        }

    def _prepare_control_step(self) -> None:
        self._smooth_actions(self._policy_action)
        self._apply_pose_delta(self._action_exec)
        self._update_disturbance_state(advance=True)
        all_ids = torch.arange(self._num_envs, dtype=torch.int64)
        self._apply_disturbance_to_stage(all_ids)
        self.sim_data.execute(all_ids)
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl.copy_(self._compute_leg_ctrls(slice(None), self._target_pos, self._target_quat))
        self._ctrl_writes.execute()

    def _simulate_control_step(self) -> None:
        all_ids = torch.arange(self._num_envs, dtype=torch.int64)
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

    def _update_stillness(self, rel_xy: torch.Tensor, vel_xy: torch.Tensor) -> torch.Tensor:
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

        roll_rad, pitch_rad, _ = _euler_xyz(top_quat)
        roll_deg = roll_rad * 180.0 / torch.pi
        pitch_deg = pitch_rad * 180.0 / torch.pi
        top_ang_vel_local = _rotate_inverse(top_quat, self._filtered_top_ang_vel)
        obs_parts = [
            self._prev_rel,
            self._filtered_rel_vel,
            torch.stack(
                [
                    roll_deg / self._cfg.target_rotation_limit_deg,
                    pitch_deg / self._cfg.target_rotation_limit_deg,
                ],
                dim=-1,
            ),
            top_ang_vel_local,
            self._target_tilt_cmd / max(self._cfg.target_rotation_limit_deg, 1e-6),
            self._action_exec,
        ]
        disturb_obs = self._get_disturbance_obs()
        if disturb_obs is not None:
            obs_parts.append(disturb_obs)
        return state.replace(obs=torch.cat(obs_parts, dim=-1))

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        state_cache = self._update_ball_kinematics()

        top_pos = state_cache["top_pos"]
        ball_pos = state_cache["ball_pos"]
        rel_xy = state_cache["rel_xy"]
        tilt_deg = state_cache["tilt_deg"]

        rel_vel = self._last_rel_vel
        top_ang_vel = self._last_top_ang_vel
        vel_xy = torch.norm(rel_vel[:, :2], dim=-1)
        top_ang_mag = torch.norm(top_ang_vel, dim=-1)

        fall_z = top_pos[:, 2] - math.sin(math.radians(30.0)) * self._cfg.platform_radius
        fallen = (rel_xy > self._cfg.platform_radius) | (ball_pos[:, 2] < fall_z)

        center_score = (1.0 - rel_xy / max(self._cfg.platform_radius, 1e-6)).clamp(0.0, 1.0)
        term_center = self._cfg.k_center * center_score

        prev_zero_vel_rel_xy = self._prev_zero_vel_rel_xy
        zero_reference = torch.where(torch.isfinite(prev_zero_vel_rel_xy), prev_zero_vel_rel_xy, self._initial_rel_xy)
        zero_event = vel_xy <= self._cfg.zero_vel_thresh
        zero_closer_mask = zero_event & (rel_xy < zero_reference)
        zero_improve = (zero_reference - rel_xy).clamp(min=0.0)
        zero_improve_norm = (zero_improve / max(self._cfg.platform_radius, 1e-6)).clamp(0.0, 1.0)
        term_zero_vel_closer = torch.where(
            zero_closer_mask, self._cfg.k_progress * zero_improve_norm, torch.zeros_like(zero_improve_norm)
        )

        self._prev_zero_vel_rel_xy[zero_event] = rel_xy[zero_event]

        still_steps = self._update_stillness(rel_xy, vel_xy)
        success = still_steps >= self._cfg.still_steps_needed
        term_still_bonus = torch.where(
            success, torch.full_like(term_center, float(self._cfg.k_still)), torch.zeros_like(term_center)
        )

        reward = term_center + term_zero_vel_closer + term_still_bonus

        term_terminal = torch.zeros((self._num_envs,), dtype=torch.float32)
        reward = torch.where(fallen, torch.full_like(reward, float(self._cfg.fall_penalty)), reward)
        term_terminal[fallen] = float(self._cfg.fall_penalty)

        # Timeout flag for logging only; the framework owns truncation via its
        # terminated/truncated bookkeeping.
        timeout = torch.zeros((self._num_envs,), dtype=torch.bool)
        if self._cfg.max_episode_steps is not None and self._cfg.max_episode_steps > 0:
            timeout = (state.episode_steps + 1) >= self._cfg.max_episode_steps
            timeout &= ~(fallen | success)

        terminated = fallen | success

        state.metrics = {
            "rel_xy": rel_xy,
            "vel_xy": vel_xy,
            "tilt_deg": tilt_deg,
            "top_ang_mag": top_ang_mag,
            "still_steps": still_steps.to(torch.float32),
            "success": success.to(torch.float32),
            "fallen": fallen.to(torch.float32),
            "timeout": timeout.to(torch.float32),
            "disturb_pos_norm": torch.norm(self._disturb_pos, dim=-1),
            "disturb_rot_norm_deg": torch.norm(self._disturb_rot_deg, dim=-1),
        }
        state.reward_terms = {
            "center": term_center,
            "zero_vel_reference": zero_reference,
            "zero_vel_improve": zero_improve,
            "zero_vel_closer": term_zero_vel_closer,
            "still_bonus": term_still_bonus,
            "terminal": term_terminal,
        }

        state.reward[:] = reward
        state.terminated[:] = terminated
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        num = env_ids.numel()
        zeros3 = torch.zeros((num, 3), dtype=torch.float32)

        roll_deg = torch.empty(num, dtype=torch.float32).uniform_(
            float(self._cfg.min_init_tilt_deg), float(self._cfg.init_tilt_deg)
        )
        roll_deg *= torch.where(torch.rand(num) < 0.5, -1.0, 1.0)
        pitch_deg = torch.empty(num, dtype=torch.float32).uniform_(
            float(self._cfg.min_init_tilt_deg), float(self._cfg.init_tilt_deg)
        )
        pitch_deg *= torch.where(torch.rand(num) < 0.5, -1.0, 1.0)

        target_pos = self._top_pos_init.unsqueeze(0).expand(num, -1).contiguous()
        target_euler_deg = torch.stack([roll_deg, pitch_deg, torch.zeros((num,), dtype=torch.float32)], dim=-1)
        target_euler_rad = target_euler_deg * torch.pi / 180.0
        target_quat = _from_euler(target_euler_rad[..., 0], target_euler_rad[..., 1], target_euler_rad[..., 2])

        # Write episode-scoped state for the reset rows
        self._target_pos[env_ids] = target_pos
        self._target_quat[env_ids] = target_quat
        self._target_tilt_cmd[env_ids] = torch.stack([roll_deg, pitch_deg], dim=-1)
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

        stage_pose = torch.cat(
            [
                self._stage_pos_init.unsqueeze(0).expand(num, -1),
                _normalize_quat(self._stage_quat_init.unsqueeze(0).expand(num, -1)),
            ],
            dim=-1,
        )
        top_pose = torch.cat([target_pos, _normalize_quat(target_quat)], dim=-1)
        hidden_ball_pos = torch.tensor([0.0, 0.0, -10.0], dtype=torch.float32).unsqueeze(0).expand(num, -1)
        ball_pose = torch.cat([hidden_ball_pos, _identity_quat((num,))], dim=-1)
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
        self.sim_data.execute(env_ids)
        leg_lengths = self._compute_leg_ctrls(env_ids, target_pos, target_quat)
        self._reset_program.buffer("legs_position")[env_ids] = leg_lengths
        self._reset_program.execute(env_ids)

        radius = (
            self._cfg.platform_radius
            * self._cfg.init_ball_radius_ratio
            * torch.sqrt(torch.rand(num, dtype=torch.float32))
        )
        theta = torch.rand(num, dtype=torch.float32) * 2.0 * torch.pi
        local_xy = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)
        local_pos = torch.cat(
            [
                local_xy,
                torch.full((num, 1), self._cfg.top_surface_offset + self._cfg.ball_radius, dtype=torch.float32),
            ],
            dim=-1,
        )
        world_pos = target_pos + _rotate_vector(target_quat, local_pos)
        self._write_body_state("ball", env_ids, world_pos, _identity_quat((num,)), zeros3, zeros3)
        self._write_ctrl_rows(env_ids, torch.zeros((num, self._model_action_dim), dtype=torch.float32))
        self.sim_data.execute(env_ids)

        ball_pos = self.sim_data["ball_pos"][env_ids]
        top_pos = self.sim_data["top_pos"][env_ids]
        self._prev_top_quat[env_ids] = _normalize_quat(self.sim_data["top_quat"][env_ids])
        self._prev_rel[env_ids] = _rotate_inverse(self._prev_top_quat[env_ids], ball_pos - top_pos)
        initial_rel_xy = self._compute_rel_xy(env_ids)
        self._initial_rel_xy[env_ids] = initial_rel_xy
        self._prev_zero_vel_rel_xy[env_ids] = initial_rel_xy

        self._reset_episode_disturbance(env_ids)
        self._apply_disturbance_to_stage(env_ids)
        self.sim_data.execute(env_ids)
