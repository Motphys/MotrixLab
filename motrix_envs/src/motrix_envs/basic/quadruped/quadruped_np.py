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
    BodyJointPositionWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    DofPositionQuery,
    DofVelocityQuery,
    GeomPairCollidingQuery,
    GeomPositionQuery,
    GeomQuaternionQuery,
    GeomSpecsQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
    SensorValuesQuery,
    SitePositionQuery,
)
from motrix_env_core.sim.write import BodyJointVelocityWrite, CtrlTargetsWrite
from motrix_env_core.torch import reward
from motrix_envs.basic.quadruped.cfg import _LEG_BODY_GEOM_NAMES, QuadrupedBaseCfg

_RANGEFINDER_SENSORS = [f"rf_{row}{col}" for row in range(4) for col in range(5)]

# quadruped_fetch.xml declares the "target" site as a 0.4-radius cylinder; the
# legacy env read that site size at runtime. Site metadata is not part of the
# sim descriptor, so the verified asset value is pinned here.
_FETCH_TARGET_SITE_RADIUS = 0.4

_LEG_GEOM_NAMES = tuple(name for leg in _LEG_BODY_GEOM_NAMES for name in leg)
_WALK_GEOMS = ("floor", "eye_r", "eye_l", "torso", *_LEG_GEOM_NAMES)
_ESCAPE_GEOMS = ("floor", "terrain", "eye_r", "eye_l", "torso", *_LEG_GEOM_NAMES)
_FETCH_GEOMS = (
    "floor",
    "wall_px",
    "wall_py",
    "wall_nx",
    "wall_ny",
    "target_marker",
    "eye_r",
    "eye_l",
    "torso",
    "torso_belly",
    *_LEG_GEOM_NAMES,
    "ball",
)


def _collision_pairs(geoms):
    return tuple((first, second) for index, first in enumerate(geoms) for second in geoms[index + 1 :])


def _leg_geom_queries():
    return {
        key: query
        for name in _LEG_GEOM_NAMES
        for key, query in (
            (f"{name}__pos", GeomPositionQuery(geom=name)),
            (f"{name}__quat", GeomQuaternionQuery(geom=name)),
        )
    }


_BASE_SIM_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
    "actuator_ctrls": ActuatorCtrlQuery(),
    "torso_pos": LinkPositionQuery(link="torso"),
    "torso_quat": LinkQuaternionQuery(link="torso"),
    "velocimeter": SensorValuesQuery(sensors=("velocimeter",)),
    "imu": SensorValuesQuery(sensors=("imu_accel", "imu_gyro")),
}


_LOCOMOTION_DATA_QUERIES = {
    **_BASE_SIM_QUERIES,
    "colliding": GeomPairCollidingQuery(pairs=_collision_pairs(_WALK_GEOMS)),
}
_LOCOMOTION_MODEL_QUERIES = {"geoms": GeomSpecsQuery(names=("floor",))}

_ESCAPE_DATA_QUERIES = {
    **_BASE_SIM_QUERIES,
    "colliding": GeomPairCollidingQuery(pairs=_collision_pairs(_ESCAPE_GEOMS)),
    "workspace_pos": SitePositionQuery(site="workspace"),
}
_ESCAPE_MODEL_QUERIES = {"geoms": GeomSpecsQuery(names=("floor",))}

_FETCH_DATA_QUERIES = {
    **_BASE_SIM_QUERIES,
    "colliding": GeomPairCollidingQuery(pairs=_collision_pairs(_FETCH_GEOMS)),
    "target_pos": SitePositionQuery(site="target"),
    "ball_pos": LinkPositionQuery(link="ball"),
    "ball_geom_pos": GeomPositionQuery(geom="ball"),
    **_leg_geom_queries(),
}
_FETCH_MODEL_QUERIES = {"geoms": GeomSpecsQuery(names=("floor", "ball", *_LEG_GEOM_NAMES))}


def _quat_to_rotation_mats(quats: torch.Tensor) -> torch.Tensor:
    """Convert (x, y, z, w) quaternions, shape (..., 4), to rotation mats (..., 3, 3)."""
    x, y, z, w = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
    mats = quats.new_empty(quats.shape[:-1] + (3, 3))
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


def _rotate_z_axis(quat: torch.Tensor) -> torch.Tensor:
    """Forward-rotate the world Z axis by (x, y, z, w) quaternions, shape (..., 4)."""
    axis = quat.new_tensor([0.0, 0.0, 1.0])
    w = quat[..., 3:4]
    im = quat[..., :3]
    t = 2.0 * torch.cross(im, axis.expand_as(im), dim=-1)
    return axis + w * t + torch.cross(im, t, dim=-1)


class QuadrupedEnv(DirectEnv):
    _cfg: QuadrupedBaseCfg
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: QuadrupedBaseCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        if cfg.include_ball:
            data_queries, model_queries = _FETCH_DATA_QUERIES, _FETCH_MODEL_QUERIES
        elif cfg.include_origin:
            data_queries, model_queries = _ESCAPE_DATA_QUERIES, _ESCAPE_MODEL_QUERIES
        else:
            data_queries, model_queries = _LOCOMOTION_DATA_QUERIES, _LOCOMOTION_MODEL_QUERIES
        self.model = self.sim.compile_model(model_queries)
        self.sim_data = self.sim.compile_reads(data_queries)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        resets = {
            "torso_position": BodyPositionWrite(("torso",)),
            "torso_rotation": BodyRotationWrite(("torso",)),
            "torso_linear_velocity": BodyLinearVelocityWrite(("torso",)),
            "torso_angular_velocity": BodyAngularVelocityWrite(("torso",)),
            "joints_position": BodyJointPositionWrite("torso"),
            "joints_velocity": BodyJointVelocityWrite("torso"),
        }
        if cfg.include_ball:
            resets.update(
                {
                    "ball_position": BodyPositionWrite(("ball",)),
                    "ball_rotation": BodyRotationWrite(("ball",)),
                    "ball_linear_velocity": BodyLinearVelocityWrite(("ball",)),
                    "ball_angular_velocity": BodyAngularVelocityWrite(("ball",)),
                }
            )
        self._reset_program = self.sim.write_compiler.compile(resets, reset=True)
        self._torso_reset_position = self._reset_program.buffer("torso_position")[:, 0]
        self._torso_reset_rotation = self._reset_program.buffer("torso_rotation")[:, 0]
        self._torso_reset_linear_velocity = self._reset_program.buffer("torso_linear_velocity")[:, 0]
        self._torso_reset_angular_velocity = self._reset_program.buffer("torso_angular_velocity")[:, 0]
        self._joint_reset_position = self._reset_program.buffer("joints_position")
        self._joint_reset_velocity = self._reset_program.buffer("joints_velocity")
        if cfg.include_ball:
            self._ball_reset_position = self._reset_program.buffer("ball_position")[:, 0]
            self._ball_reset_rotation = self._reset_program.buffer("ball_rotation")[:, 0]
            self._ball_reset_linear_velocity = self._reset_program.buffer("ball_linear_velocity")[:, 0]
            self._ball_reset_angular_velocity = self._reset_program.buffer("ball_angular_velocity")[:, 0]
        self._cfg = cfg

        self._leg_geom_names: list[str] = []
        self._leg_geom_slices: list[slice] = []
        if cfg.include_ball:
            for leg_geoms in _LEG_BODY_GEOM_NAMES:
                start = len(self._leg_geom_names)
                for name in leg_geoms:
                    if name not in self.model.others["geoms"]:
                        continue
                    self._leg_geom_names.append(name)
                self._leg_geom_slices.append(slice(start, len(self._leg_geom_names)))
            if not self._leg_geom_names:
                self._leg_geom_slices = []

        self._body_dof_pos = self.num_dof_pos - 7 - (7 if cfg.include_ball else 0)
        self._body_dof_vel = self.num_dof_vel - 6 - (6 if cfg.include_ball else 0)
        self._dof_pos_slice = slice(7, 7 + self._body_dof_pos)
        self._dof_vel_slice = slice(6, 6 + self._body_dof_vel)
        self._ball_pos_slice = None
        self._ball_vel_slice = None
        self._floor_geom_size0 = float(self.model.others["geoms"]["floor"].size[0])
        if cfg.include_ball:
            self._ball_pos_slice = slice(self.num_dof_pos - 7, self.num_dof_pos)
            self._ball_vel_slice = slice(self.num_dof_vel - 6, self.num_dof_vel)
            self._ball_geom_size0 = float(self.model.others["geoms"]["ball"].size[0])
            # Leg geoms are capsules (two-entry size) or spheres (one-entry
            # size) in every quadruped asset, so the capsule branch of the
            # legacy shape check reduces to the size tuple length.
            self._leg_geom_sizes = {
                name: torch.as_tensor(self.model.others["geoms"][name].size, dtype=torch.float32).flatten()
                for name in self._leg_geom_names
            }

        self._init_dof_pos = self.model.init_dof_pos.clone()
        self._default_body_dof_pos = self._init_dof_pos[self._dof_pos_slice].clone()
        # The legacy constructor took max(cfg.terrain_size, |hfield bound[3]|)
        # when the scene defined height fields. Only quadruped_escape.xml has
        # an hfield and its bound[3] is 0.1 < cfg default 30.0, so the config
        # value already is the effective terrain size everywhere.
        self._terrain_size = float(cfg.terrain_size)

        self._init_obs_space()
        self._init_action_space()

    def _init_obs_space(self):
        num_obs = self._body_dof_pos + self._body_dof_vel + self.num_actuators
        num_obs += 3  # torso velocity
        num_obs += 1  # torso upright
        num_obs += 6  # imu accel + gyro

        if self._cfg.include_origin:
            num_obs += 3
        if self._cfg.include_rangefinder:
            num_obs += len(_RANGEFINDER_SENSORS)
        if self._cfg.include_ball:
            num_obs += 9
        if self._cfg.include_target:
            num_obs += 3

        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

    def _init_action_space(self):
        ctrl_ranges = np.asarray([spec.ctrl_range for spec in self.model.actuators], dtype=np.float32)
        self._action_space = gym.spaces.Box(
            ctrl_ranges[:, 0], ctrl_ranges[:, 1], (self.num_actuators,), dtype=np.float32
        )
        self._act_low = torch.from_numpy(ctrl_ranges[:, 0])
        self._act_high = torch.from_numpy(ctrl_ranges[:, 1])

        # Episode-scoped action history: full-batch buffers, reset writes the
        # done rows in place.
        self._actions = torch.zeros((self._num_envs, self.num_actuators), dtype=torch.float32)
        self._last_actions = torch.zeros((self._num_envs, self.num_actuators), dtype=torch.float32)

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        if self._cfg.clip_env_actions:
            actions = torch.clamp(actions, self._act_low, self._act_high)
        actions = actions.to(torch.float32)
        self._last_actions.copy_(self._actions)
        self._actions.copy_(actions)
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl.copy_(actions)
        self._ctrl_writes.execute()
        return state

    def _torso_frame(self, rows) -> torch.Tensor:
        return _quat_to_rotation_mats(self.sim_data["torso_quat"][rows])

    def _egocentric_state(self, rows) -> torch.Tensor:
        inputs = self.sim_data
        dof_pos = inputs["dof_pos"][rows][:, self._dof_pos_slice]
        dof_vel = inputs["dof_vel"][rows][:, self._dof_vel_slice]
        act = inputs["actuator_ctrls"][rows]
        return torch.cat([dof_pos, dof_vel, act], dim=-1)

    def _torso_upright(self, rows) -> torch.Tensor:
        return self._torso_frame(rows)[:, 2, 2]

    def _torso_velocity(self, rows) -> torch.Tensor:
        return self.sim_data["velocimeter"][rows]

    def _imu(self, rows) -> torch.Tensor:
        return self.sim_data["imu"][rows]

    def _rangefinder(self, rows) -> torch.Tensor:
        # Shipped scenes define no rf_* sensors, so enabling include_rangefinder
        # fails loudly here exactly like the legacy model-level sensor lookup.
        readings = torch.cat([self.sim_data[name][rows] for name in _RANGEFINDER_SENSORS], dim=-1)
        no_intersection = -1.0
        return torch.where(readings == no_intersection, torch.ones_like(readings), torch.tanh(readings))

    def _origin(self, rows) -> torch.Tensor:
        torso_pos = self.sim_data["torso_pos"][rows]
        torso_frame = self._torso_frame(rows)
        return -torch.einsum("ni,nij->nj", torso_pos, torso_frame)

    def _origin_distance(self, rows) -> torch.Tensor:
        workspace_pos = self.sim_data["workspace_pos"][rows]
        return torch.norm(workspace_pos, dim=-1)

    def _ball_state(self, rows) -> torch.Tensor:
        inputs = self.sim_data
        ball_pos = inputs["ball_pos"][rows]
        torso_pos = inputs["torso_pos"][rows]
        torso_frame = self._torso_frame(rows)

        ball_rel_pos = ball_pos - torso_pos
        root_linvel = inputs["dof_vel"][rows][:, :3]
        ball_vel = inputs["dof_vel"][rows][:, self._ball_vel_slice]
        ball_rel_vel = ball_vel[:, :3] - root_linvel
        ball_rot_vel = ball_vel[:, 3:]

        stacked = torch.stack([ball_rel_pos, ball_rel_vel, ball_rot_vel], dim=1)
        local = torch.einsum("nij,njk->nik", stacked, torso_frame)
        return local.reshape(-1, 9)

    def _target_position(self, rows) -> torch.Tensor:
        torso_pos = self.sim_data["torso_pos"][rows]
        torso_frame = self._torso_frame(rows)
        to_target = self.sim_data["target_pos"][rows] - torso_pos
        return torch.einsum("ni,nij->nj", to_target, torso_frame)

    def _ball_to_target_distance(self, rows) -> torch.Tensor:
        ball_pos = self.sim_data["ball_pos"][rows]
        target_pos = self.sim_data["target_pos"][rows]
        return torch.norm((target_pos - ball_pos)[:, :2], dim=-1)

    def _aggregate_leg_ball_proximity(self, geom_penalties: torch.Tensor) -> torch.Tensor:
        num_legs = len(self._leg_geom_slices)
        if num_legs == 0:
            return geom_penalties.new_zeros((geom_penalties.shape[0], 0))

        leg_penalties = []
        for geom_slice in self._leg_geom_slices:
            if geom_slice.start == geom_slice.stop:
                leg_penalties.append(geom_penalties.new_zeros((geom_penalties.shape[0],)))
            else:
                leg_penalties.append(geom_penalties[:, geom_slice].max(dim=-1).values)
        return torch.stack(leg_penalties, dim=-1)

    def _point_to_segment_distance(
        self, point: torch.Tensor, start: torch.Tensor, end: torch.Tensor
    ) -> torch.Tensor:
        segment = end - start
        segment_sq_norm = torch.sum(segment * segment, dim=-1)
        safe_norm = torch.where(segment_sq_norm > 1e-8, segment_sq_norm, torch.ones_like(segment_sq_norm))
        t = torch.sum((point - start) * segment, dim=-1) / safe_norm
        t = torch.where(segment_sq_norm > 1e-8, t.clamp(0.0, 1.0), torch.zeros_like(t))
        closest = start + t[:, None] * segment
        return torch.norm(point - closest, dim=-1)

    def _geom_ball_surface_clearance(
        self,
        geom_name: str,
        geom_pos: torch.Tensor,
        geom_quat: torch.Tensor,
        ball_pos: torch.Tensor,
        ball_radius: float,
    ) -> torch.Tensor:
        geom_size = self._leg_geom_sizes[geom_name]
        geom_radius = float(geom_size[0])

        if geom_size.shape[0] > 1 and float(geom_size[1]) > 0.0:
            half_length = float(geom_size[1])
            axis = _rotate_z_axis(geom_quat)
            start = geom_pos - axis * half_length
            end = geom_pos + axis * half_length
            center_distance = self._point_to_segment_distance(ball_pos, start, end)
        else:
            center_distance = torch.norm(ball_pos - geom_pos, dim=-1)

        return center_distance - (ball_radius + geom_radius)

    def _leg_body_ball_penalty(self, rows) -> torch.Tensor:
        if not self._leg_geom_names:
            return torch.zeros((self._num_envs,), dtype=torch.float32)

        inputs = self.sim_data
        ball_pos = inputs["ball_geom_pos"][rows]
        ball_radius = self._ball_geom_size0
        geom_penalties = []
        for geom_name in self._leg_geom_names:
            clearance = self._geom_ball_surface_clearance(
                geom_name,
                inputs[f"{geom_name}__pos"][rows],
                inputs[f"{geom_name}__quat"][rows],
                ball_pos,
                ball_radius,
            )
            geom_penalties.append(
                reward.tolerance(
                    -clearance,
                    bounds=(0.0, float("inf")),
                    margin=self._cfg.fetch_leg_ball_penalty_margin,
                    value_at_margin=0.0,
                    sigmoid="linear",
                )
            )

        proximity = torch.stack(geom_penalties, dim=-1)
        leg_penalties = self._aggregate_leg_ball_proximity(proximity)
        return leg_penalties.sum(dim=-1)

    def _fetch_stability_gate(self, torso_upright: torch.Tensor, torso_height: torch.Tensor) -> torch.Tensor:
        upright_gate = reward.tolerance(
            torso_upright,
            bounds=(self._cfg.fetch_stability_upright_min, float("inf")),
            margin=self._cfg.fetch_stability_upright_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        height_gate = reward.tolerance(
            torso_height,
            bounds=(self._cfg.fetch_stability_height_min, float("inf")),
            margin=self._cfg.fetch_stability_height_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        return upright_gate * height_gate

    def _fetch_fall_terminated(self, torso_upright: torch.Tensor, torso_height: torch.Tensor) -> torch.Tensor:
        return (torso_upright < self._cfg.fetch_fall_upright_min) | (torso_height < self._cfg.fetch_fall_height_min)

    def _upright_reward(self, torso_upright: torch.Tensor) -> torch.Tensor:
        deviation = math.cos(math.radians(self._cfg.deviation_angle))
        return reward.tolerance(
            torso_upright,
            bounds=(deviation, float("inf")),
            margin=1 + deviation,
            value_at_margin=0.0,
            sigmoid="linear",
        )

    def _move_reward(self, torso_vel: torch.Tensor) -> torch.Tensor:
        return reward.tolerance(
            torso_vel[:, 0],
            bounds=(self._cfg.desired_speed, float("inf")),
            margin=self._cfg.desired_speed,
            value_at_margin=0.5,
            sigmoid="linear",
        )

    def _backward_penalty(self, torso_vel: torch.Tensor) -> torch.Tensor:
        return (-torso_vel[:, 0]).clamp(min=0.0)

    def _escape_reward(self, rows) -> torch.Tensor:
        return reward.tolerance(
            self._origin_distance(rows),
            bounds=(self._terrain_size, float("inf")),
            margin=self._terrain_size,
            value_at_margin=0.0,
            sigmoid="linear",
        )

    def _radial_speed_reward(self) -> torch.Tensor:
        radial_speed_reward = torch.zeros((self._num_envs,), dtype=torch.float32)
        if not self._cfg.include_origin:
            return radial_speed_reward

        torso_pos = self.sim_data["torso_pos"]
        radial_vec = torso_pos[:, :2]
        radial_norm = torch.norm(radial_vec, dim=-1, keepdim=True)
        radial_dir = torch.where(radial_norm > 1e-6, radial_vec / radial_norm, torch.zeros_like(radial_vec))
        radial_speed = torch.sum(self.sim_data["dof_vel"][:, :2] * radial_dir, dim=-1)
        radial_speed = radial_speed.clamp(min=0.0)
        return reward.tolerance(
            radial_speed,
            bounds=(self._cfg.desired_speed, float("inf")),
            margin=self._cfg.desired_speed,
            value_at_margin=0.5,
            sigmoid="linear",
        )

    def _heading_reward(self) -> torch.Tensor:
        heading_reward = torch.zeros((self._num_envs,), dtype=torch.float32)
        if self._cfg.heading_reward_weight <= 0.0:
            return heading_reward

        torso_frame = self._torso_frame(slice(None))
        heading_xy = torso_frame[:, 0, :2]
        heading_norm = torch.norm(heading_xy, dim=-1, keepdim=True)
        heading_dir = torch.where(heading_norm > 1e-6, heading_xy / heading_norm, torch.zeros_like(heading_xy))
        heading_align = heading_dir[:, 0]
        return reward.tolerance(
            heading_align,
            bounds=(1.0, 1.0),
            margin=self._cfg.heading_reward_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )

    def _height_reward(self) -> torch.Tensor:
        torso_height = self.sim_data["torso_pos"][:, 2]
        return reward.tolerance(
            torso_height,
            bounds=(self._cfg.stand_height, float("inf")),
            margin=self._cfg.stand_height_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )

    def _lateral_reward(self, torso_vel: torch.Tensor) -> torch.Tensor:
        return reward.tolerance(
            torch.abs(torso_vel[:, 1]),
            bounds=(0.0, self._cfg.lateral_velocity_limit),
            margin=self._cfg.lateral_velocity_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )

    def _smooth_reward(self, state: ArrayEnvState) -> torch.Tensor:
        delta = self._actions - self._last_actions
        delta_norm = torch.norm(delta, dim=-1)
        return reward.tolerance(
            delta_norm,
            bounds=(0.0, 0.0),
            margin=self._cfg.action_smoothness_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )

    def _lin_vel_z_penalty(self, torso_vel: torch.Tensor) -> torch.Tensor:
        return torch.square(torso_vel[:, 2])

    def _ang_vel_xy_penalty(self) -> torch.Tensor:
        imu = self.sim_data["imu"]
        return torch.sum(torch.square(imu[:, 3:5]), dim=1)

    def _similar_to_default_penalty(self) -> torch.Tensor:
        body_dof_pos = self.sim_data["dof_pos"][:, self._dof_pos_slice]
        return torch.sum(torch.abs(body_dof_pos - self._default_body_dof_pos), dim=1)

    def _locomotion_reward_terms(
        self,
        upright_reward: torch.Tensor,
        move_reward: torch.Tensor,
        backward_penalty: torch.Tensor,
        height_reward: torch.Tensor,
        lateral_reward: torch.Tensor,
        heading_reward: torch.Tensor,
        smooth_reward: torch.Tensor,
        lin_vel_z_penalty: torch.Tensor,
        ang_vel_xy_penalty: torch.Tensor,
        similar_to_default_penalty: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "move": upright_reward * move_reward,
            "backward": backward_penalty,
            "height": height_reward,
            "lateral": lateral_reward,
            "heading": heading_reward,
            "smooth": smooth_reward,
            "lin_vel_z": lin_vel_z_penalty,
            "ang_vel_xy": ang_vel_xy_penalty,
            "similar_to_default": similar_to_default_penalty,
        }

    def _locomotion_reward_scales(self) -> dict[str, float]:
        return {
            "move": 1.0,
            "backward": -self._cfg.backward_penalty_weight,
            "height": self._cfg.height_reward_weight,
            "lateral": self._cfg.lateral_reward_weight,
            "heading": self._cfg.heading_reward_weight,
            "smooth": self._cfg.action_smoothness_weight,
            "lin_vel_z": -self._cfg.lin_vel_z_weight,
            "ang_vel_xy": -self._cfg.ang_vel_xy_weight,
            "similar_to_default": -self._cfg.similar_to_default_weight,
        }

    def _escape_reward_terms(
        self, upright_reward: torch.Tensor, escape_reward: torch.Tensor, radial_speed_reward: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            "escape": upright_reward * escape_reward,
            "radial": radial_speed_reward,
        }

    def _escape_reward_scales(self) -> dict[str, float]:
        return {
            "escape": 1.0,
            "radial": self._cfg.radial_velocity_weight,
        }

    def _sum_scaled_rewards(
        self, reward_terms: dict[str, torch.Tensor], reward_scales: dict[str, float]
    ) -> torch.Tensor:
        rewards = {name: value * reward_scales[name] for name, value in reward_terms.items()}
        return sum(rewards.values())

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        parts = [
            self._egocentric_state(slice(None)),
            self._torso_velocity(slice(None)),
            self._torso_upright(slice(None)).reshape(-1, 1),
            self._imu(slice(None)),
        ]

        if self._cfg.include_origin:
            parts.append(self._origin(slice(None)))
        if self._cfg.include_rangefinder:
            parts.append(self._rangefinder(slice(None)))
        if self._cfg.include_ball:
            parts.append(self._ball_state(slice(None)))
        if self._cfg.include_target:
            parts.append(self._target_position(slice(None)))

        return state.replace(obs=torch.cat(parts, dim=-1))

    def _base_locomotion_components(self, state: ArrayEnvState) -> dict[str, torch.Tensor]:
        torso_vel = self._torso_velocity(slice(None))
        return {
            "move": self._move_reward(torso_vel),
            "backward": self._backward_penalty(torso_vel),
            "height": self._height_reward(),
            "lateral": self._lateral_reward(torso_vel),
            "heading": self._heading_reward(),
            "smooth": self._smooth_reward(state),
            "lin_vel_z": self._lin_vel_z_penalty(torso_vel),
            "ang_vel_xy": self._ang_vel_xy_penalty(),
            "similar_to_default": self._similar_to_default_penalty(),
        }

    def _locomotion_reward(self, upright_reward: torch.Tensor, components: dict[str, torch.Tensor]) -> torch.Tensor:
        reward_terms = self._locomotion_reward_terms(
            upright_reward,
            components["move"],
            components["backward"],
            components["height"],
            components["lateral"],
            components["heading"],
            components["smooth"],
            components["lin_vel_z"],
            components["ang_vel_xy"],
            components["similar_to_default"],
        )
        return self._sum_scaled_rewards(reward_terms, self._locomotion_reward_scales())

    def _random_quaternion(self, num: int) -> torch.Tensor:
        q = torch.randn(num, 4, dtype=torch.float32)
        return q / torch.norm(q, dim=-1, keepdim=True)

    def _yaw_quaternion(self, yaw: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(yaw)
        half = yaw * 0.5
        return torch.stack([zeros, zeros, torch.sin(half), torch.cos(half)], dim=-1)

    def _execute_reset(self, env_ids: torch.Tensor, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> None:
        _torso_pose = dof_pos[:, :7]
        self._torso_reset_position[env_ids] = _torso_pose[:, :3]
        self._torso_reset_rotation[env_ids] = _torso_pose[:, 3:7]
        self._torso_reset_linear_velocity[env_ids] = dof_vel[:, :3]
        self._torso_reset_angular_velocity[env_ids] = dof_vel[:, 3:6]
        self._joint_reset_position[env_ids] = dof_pos[:, self._dof_pos_slice]
        self._joint_reset_velocity[env_ids] = dof_vel[:, self._dof_vel_slice]
        if self._cfg.include_ball:
            _ball_pose = dof_pos[:, self._ball_pos_slice]
            self._ball_reset_position[env_ids] = _ball_pose[:, :3]
            self._ball_reset_rotation[env_ids] = _ball_pose[:, 3:7]
            self._ball_reset_linear_velocity[env_ids] = dof_vel[:, self._ball_vel_slice][:, :3]
            self._ball_reset_angular_velocity[env_ids] = dof_vel[:, self._ball_vel_slice][:, 3:6]
        self._reset_program.execute(env_ids)

    def _lift_non_contacting(
        self, env_ids: torch.Tensor, dof_pos: torch.Tensor, dof_vel: torch.Tensor
    ) -> torch.Tensor:
        z = dof_pos[:, 2].clone()
        pending = torch.ones((env_ids.numel(),), dtype=torch.bool)
        for _ in range(1000):
            if not pending.any():
                break
            dof_pos[pending, 2] = z[pending]
            self._execute_reset(env_ids, dof_pos, dof_vel)
            self.sim_data.execute(env_ids)
            # Contact iff any declared collidable geom pair is colliding; this
            # reproduces the legacy global ``num_contacts > 0`` check exactly.
            pending = self.sim_data["colliding"][env_ids].max(dim=-1).values > 0
            z[pending] += 0.01
        return dof_pos

    def _finish_reset(self, env_ids: torch.Tensor, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> None:
        dof_pos = self._lift_non_contacting(env_ids, dof_pos, dof_vel)
        self._execute_reset(env_ids, dof_pos, dof_vel)
        self.sim_data.execute(env_ids)

        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0


@registry.env("dm-quadruped-walk")
@registry.env("dm-quadruped-run")
class QuadrupedLocomotionEnv(QuadrupedEnv):
    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        inputs = self.sim_data
        torso_upright = self._torso_upright(slice(None))
        upright_reward = self._upright_reward(torso_upright)
        locomotion_components = self._base_locomotion_components(state)
        rwd = self._locomotion_reward(upright_reward, locomotion_components)

        reward_components = {"upright": upright_reward}
        reward_components.update(locomotion_components)
        reward_components["total"] = rwd

        terminated = inputs["dof_pos"].isnan().any(dim=-1) | inputs["dof_vel"].isnan().any(dim=-1)
        rwd = torch.where(terminated, torch.zeros_like(rwd), rwd)
        state.reward_terms = reward_components

        state.reward[:] = rwd
        state.terminated[:] = terminated
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        num = env_ids.numel()
        dof_pos = self._init_dof_pos.unsqueeze(0).repeat(num, 1).contiguous()
        dof_vel = torch.zeros((num, self.num_dof_vel), dtype=torch.float32)

        if self._cfg.fix_heading:
            dof_pos[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        else:
            dof_pos[:, 3:7] = self._random_quaternion(num)

        self._finish_reset(env_ids, dof_pos, dof_vel)


@registry.env("dm-quadruped-escape")
class QuadrupedEscapeEnv(QuadrupedLocomotionEnv):
    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        inputs = self.sim_data
        torso_upright = self._torso_upright(slice(None))
        upright_reward = self._upright_reward(torso_upright)
        locomotion_components = self._base_locomotion_components(state)
        escape_reward = self._escape_reward(slice(None))
        radial_speed_reward = self._radial_speed_reward()

        reward_terms = self._locomotion_reward_terms(
            upright_reward,
            locomotion_components["move"],
            locomotion_components["backward"],
            locomotion_components["height"],
            locomotion_components["lateral"],
            locomotion_components["heading"],
            locomotion_components["smooth"],
            locomotion_components["lin_vel_z"],
            locomotion_components["ang_vel_xy"],
            locomotion_components["similar_to_default"],
        )
        reward_scales = self._locomotion_reward_scales()
        reward_terms.update(self._escape_reward_terms(upright_reward, escape_reward, radial_speed_reward))
        reward_scales.update(self._escape_reward_scales())
        rwd = self._sum_scaled_rewards(reward_terms, reward_scales)

        reward_components = {"upright": upright_reward}
        reward_components.update(locomotion_components)
        reward_components.update(
            {
                "escape": escape_reward,
                "radial": radial_speed_reward,
                "total": rwd,
            }
        )

        terminated = inputs["dof_pos"].isnan().any(dim=-1) | inputs["dof_vel"].isnan().any(dim=-1)
        rwd = torch.where(terminated, torch.zeros_like(rwd), rwd)
        state.reward_terms = reward_components

        state.reward[:] = rwd
        state.terminated[:] = terminated
        return state


@registry.env("dm-quadruped-fetch")
class QuadrupedFetchEnv(QuadrupedEnv):
    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        inputs = self.sim_data
        torso_upright = self._torso_upright(slice(None))
        upright_reward = self._upright_reward(torso_upright)
        torso_height = inputs["torso_pos"][:, 2]
        stability_gate = self._fetch_stability_gate(torso_upright, torso_height)
        target_radius = float(self._cfg.target_radius)
        if self._cfg.include_target:
            target_radius = _FETCH_TARGET_SITE_RADIUS

        ball_pos = inputs["ball_pos"]
        target_pos = inputs["target_pos"]
        torso_pos = inputs["torso_pos"]
        to_target = target_pos[:, :2] - ball_pos[:, :2]
        to_target_norm = torch.norm(to_target, dim=-1, keepdim=True)
        to_target_dir = torch.where(to_target_norm > 1e-6, to_target / to_target_norm, torch.zeros_like(to_target))
        torso_frame = self._torso_frame(slice(None))
        heading_xy = torso_frame[:, 0, :2]
        heading_norm = torch.norm(heading_xy, dim=-1, keepdim=True)
        heading_dir = torch.where(heading_norm > 1e-6, heading_xy / heading_norm, torch.zeros_like(heading_xy))
        to_ball = ball_pos[:, :2] - torso_pos[:, :2]
        to_ball_norm = torch.norm(to_ball, dim=-1, keepdim=True)
        to_ball_dir = torch.where(to_ball_norm > 1e-6, to_ball / to_ball_norm, torch.zeros_like(to_ball))

        behind_align = torch.sum(to_ball_dir * to_target_dir, dim=-1)
        behind_align_reward = reward.tolerance(
            behind_align,
            bounds=(1.0, 1.0),
            margin=self._cfg.fetch_behind_align_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        heading_align = torch.sum(heading_dir * to_ball_dir, dim=-1)
        face_ball_reward = reward.tolerance(
            heading_align,
            bounds=(1.0, 1.0),
            margin=self._cfg.fetch_heading_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        ball_to_robot = torso_pos[:, :2] - ball_pos[:, :2]
        back_dir = -to_target_dir
        corridor_lat = torch.norm(
            ball_to_robot - torch.sum(ball_to_robot * back_dir, dim=-1, keepdim=True) * back_dir,
            dim=-1,
        )
        corridor_reward = reward.tolerance(
            corridor_lat,
            bounds=(0.0, self._cfg.fetch_corridor_width),
            margin=self._cfg.fetch_corridor_width,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        ball_dist = torch.norm(to_ball, dim=-1)
        near_ball_reward = reward.tolerance(
            ball_dist,
            bounds=(0.0, self._cfg.fetch_ready_ball_distance),
            margin=self._cfg.fetch_ready_ball_distance,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        ready = behind_align_reward * face_ball_reward * corridor_reward * near_ball_reward
        ready_gate = reward.tolerance(
            ready,
            bounds=(self._cfg.fetch_ready_threshold, 1.0),
            margin=1.0 - self._cfg.fetch_ready_threshold,
            value_at_margin=0.0,
            sigmoid="linear",
        )

        behind_pos = ball_pos[:, :2] + back_dir * self._cfg.fetch_behind_distance
        ahead_pos = ball_pos[:, :2] + to_target_dir * self._cfg.fetch_ahead_distance
        stage_pos = (1.0 - ready_gate)[:, None] * behind_pos + ready_gate[:, None] * ahead_pos
        if self._cfg.fetch_side_stage_offset > 0.0:
            side_dir = torch.stack([-back_dir[:, 1], back_dir[:, 0]], dim=-1)
            ball_to_robot_side = torch.sum(ball_to_robot * side_dir, dim=-1)
            side_sign = torch.where(ball_to_robot_side >= 0.0, 1.0, -1.0)
            side_pos = behind_pos + (side_sign[:, None] * side_dir * self._cfg.fetch_side_stage_offset)
            use_side_stage = (
                (ready_gate < self._cfg.fetch_side_stage_gate_threshold)
                & (ball_dist < self._cfg.fetch_side_stage_ball_distance)
                & (behind_align < self._cfg.fetch_side_stage_align_threshold)
            )
            stage_pos = torch.where(use_side_stage[:, None], side_pos, stage_pos)

        to_stage = stage_pos - torso_pos[:, :2]
        stage_dist = torch.norm(to_stage, dim=-1)
        stage_dir = torch.where(stage_dist[:, None] > 1e-6, to_stage / stage_dist[:, None], torch.zeros_like(to_stage))

        speed_to_stage = torch.sum(inputs["dof_vel"][:, :2] * stage_dir, dim=-1)
        stage_move = reward.tolerance(
            speed_to_stage,
            bounds=(self._cfg.fetch_stage_speed, float("inf")),
            margin=self._cfg.fetch_stage_speed,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        backward_penalty = (-speed_to_stage).clamp(min=0.0)
        stage_reach = reward.tolerance(
            stage_dist,
            bounds=(0.0, self._cfg.fetch_stage_radius),
            margin=self._cfg.fetch_stage_radius,
            value_at_margin=0.0,
            sigmoid="linear",
        )

        fetch_reward = reward.tolerance(
            self._ball_to_target_distance(slice(None)),
            bounds=(0.0, target_radius),
            margin=self._cfg.fetch_reward_margin,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        ball_vel = inputs["dof_vel"][:, self._ball_vel_slice][:, :2]
        ball_speed_to_target = torch.sum(ball_vel * to_target_dir, dim=-1)
        push_reward = reward.tolerance(
            ball_speed_to_target.clamp(min=0.0),
            bounds=(self._cfg.fetch_push_speed, float("inf")),
            margin=self._cfg.fetch_push_speed,
            value_at_margin=0.0,
            sigmoid="linear",
        )
        away_penalty = (1.0 - ready_gate) * (-ball_speed_to_target).clamp(min=0.0)
        leg_ball_penalty = self._leg_body_ball_penalty(slice(None))

        rwd = stability_gate * upright_reward * stage_move
        rwd -= self._cfg.fetch_backward_penalty_weight * backward_penalty
        rwd += self._cfg.fetch_stage_reward_weight * (stability_gate * stage_reach)
        rwd += self._cfg.fetch_heading_weight * (stability_gate * face_ball_reward)
        rwd += self._cfg.fetch_ready_weight * (stability_gate * ready)
        rwd += self._cfg.fetch_reward_weight * (stability_gate * ready_gate * fetch_reward)
        rwd += self._cfg.fetch_push_reward_weight * (stability_gate * ready_gate * push_reward)
        rwd -= self._cfg.fetch_away_penalty_weight * away_penalty
        rwd -= self._cfg.fetch_leg_ball_penalty_weight * leg_ball_penalty

        reward_components = {
            "upright": upright_reward,
            "stage_move": stage_move,
            "stage_reach": stage_reach,
            "stability": stability_gate,
            "behind_align": behind_align_reward,
            "face_ball": face_ball_reward,
            "near_ball": near_ball_reward,
            "ready": ready,
            "ready_gate": ready_gate,
            "fetch": fetch_reward,
            "push": push_reward,
            "away": away_penalty,
            "leg_ball": leg_ball_penalty,
            "backward": backward_penalty,
            "total": rwd,
        }

        terminated = inputs["dof_pos"].isnan().any(dim=-1) | inputs["dof_vel"].isnan().any(dim=-1)
        terminated |= self._fetch_fall_terminated(torso_upright, torso_height)
        rwd = torch.where(terminated, torch.zeros_like(rwd), rwd)
        for key, value in reward_components.items():
            reward_components[key] = torch.where(terminated, torch.zeros_like(value), value)
        state.reward_terms = reward_components

        state.reward[:] = rwd
        state.terminated[:] = terminated
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        num = env_ids.numel()
        dof_pos = self._init_dof_pos.unsqueeze(0).repeat(num, 1).contiguous()
        dof_vel = torch.zeros((num, self.num_dof_vel), dtype=torch.float32)

        floor_radius = self._floor_geom_size0
        if floor_radius <= 0.0:
            floor_radius = self._terrain_size
        spawn_radius = 0.12 * floor_radius
        yaw = torch.rand(num, dtype=torch.float32) * 2.0 * torch.pi
        dof_pos[:, 0] = (torch.rand(num, dtype=torch.float32) * 2.0 - 1.0) * spawn_radius
        dof_pos[:, 1] = (torch.rand(num, dtype=torch.float32) * 2.0 - 1.0) * spawn_radius
        dof_pos[:, 3:7] = self._yaw_quaternion(yaw)

        ball_xy = (torch.rand(num, 2, dtype=torch.float32) * 2.0 - 1.0) * spawn_radius
        ball_qpos = self._ball_pos_slice
        dof_pos[:, ball_qpos.start : ball_qpos.start + 2] = ball_xy
        dof_pos[:, ball_qpos.start + 2] = self._ball_geom_size0
        dof_pos[:, ball_qpos.start + 3 : ball_qpos.stop] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=torch.float32
        )

        ball_qvel = self._ball_vel_slice
        dof_vel[:, ball_qvel.start : ball_qvel.stop] = 0.0

        self._finish_reset(env_ids, dof_pos, dof_vel)
