# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Robot-agnostic command-conditioned humanoid velocity tracking environment.

The shared implementation assumes a floating-base biped driven by one-DoF
position actuators. Robot-specific model names, default pose, pose weights, and
scene construction are supplied through :class:`HumanoidVelocityTrackingEnvCfg` configs.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from motrix_env_core.array.env import NpObs
from motrix_env_core.base import ObsSpace
from motrix_env_core.direct.env import ArrayEnvState, DirectEnv
from motrix_env_core.math import quaternion
from motrix_env_core.sim import (
    BodyAngularVelocityWrite,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    GeomSpecsQuery,
    JointPositionWrite,
)
from motrix_env_core.sim.backend import ActuatorType
from motrix_env_core.sim.write import CtrlTargetsWrite, JointVelocityWrite
from motrix_envs.locomotion.action_space import joint_position_action_space
from motrix_envs.locomotion.humanoid.cfg import HumanoidVelocityTrackingEnvCfg, humanoid_sim_queries
from motrix_envs.robot import HumanoidRobotCfg


@dataclass
class StateQuantities:
    """Batched physical quantities used by observations and rewards."""

    base_quat: np.ndarray
    base_lin_vel: np.ndarray
    base_ang_vel: np.ndarray
    projected_gravity: np.ndarray
    foot_pos: np.ndarray
    foot_quat: np.ndarray
    foot_clearance: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray


def _expected_foot_height(phi: np.ndarray, swing_height: float) -> np.ndarray:
    """Expected biped foot height from gait phase using a cubic Bezier profile."""

    def bezier(y_start, y_end, x):
        return y_start + (y_end - y_start) * (x**3 + 3 * (x**2 * (1 - x)))

    x = (phi + np.pi) / (2 * np.pi)
    stance = bezier(np.zeros_like(x), np.full_like(x, swing_height), 2 * x)
    swing = bezier(np.full_like(x, swing_height), np.zeros_like(x), 2 * x - 1)
    return np.where(x <= 0.5, stance, swing)


class HumanoidVelocityTrackingEnv(DirectEnv[HumanoidVelocityTrackingEnvCfg]):
    """Shared humanoid velocity-tracking environment configured entirely through model names."""

    def __init__(self, cfg: HumanoidVelocityTrackingEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        robot = cfg.scene.objs.robot
        if not isinstance(robot, HumanoidRobotCfg):
            raise TypeError(f"humanoid walk scene robot must be HumanoidRobotCfg, got {type(robot).__name__}")
        self._robot_cfg = robot
        self._base_link_name = robot.resolved_base_link_name
        asset = cfg.asset
        ground_geom = asset.ground_geom_name
        termination_geoms = tuple(name for name in asset.terminate_contact_geom_names if name != ground_geom)
        self.model = self.sim.compile_model({"geoms": GeomSpecsQuery(names=termination_geoms + (ground_geom,))})
        self._joint_names = tuple(spec.target_name for spec in self.model.actuators)
        queries = humanoid_sim_queries(
            base_link=self._base_link_name,
            foot_links=robot.resolved_foot_link_names,
            sole_sites=tuple(robot.resolve_name(name) for name in asset.foot_height_site_names),
            termination_geoms=termination_geoms,
            ground_geom=ground_geom,
            joints=self._joint_names,
        )
        self._termination_query = queries["termination_colliding"]
        self.sim_data = self.sim.compile_reads(queries)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._validate_asset_cfg()
        self._validate_model_contract()
        self._reset_program = self.sim.write_compiler.compile(
            {
                "base_position": BodyPositionWrite((self._base_link_name,)),
                "base_rotation": BodyRotationWrite((self._base_link_name,)),
                "base_linear_velocity": BodyLinearVelocityWrite((self._base_link_name,)),
                "base_angular_velocity": BodyAngularVelocityWrite((self._base_link_name,)),
                "joints_position": JointPositionWrite(self._joint_names),
                "joints_velocity": JointVelocityWrite(self._joint_names),
            },
            reset=True,
        )
        self._reset_position = self._reset_program.buffer("base_position")[:, 0]
        self._reset_rotation = self._reset_program.buffer("base_rotation")[:, 0]
        self._reset_linear_velocity = self._reset_program.buffer("base_linear_velocity")[:, 0]
        self._reset_angular_velocity = self._reset_program.buffer("base_angular_velocity")[:, 0]
        self._reset_joint_position = self._reset_program.buffer("joints_position")
        self._reset_joint_velocity = self._reset_program.buffer("joints_velocity")
        self._feet_link_names = tuple(robot.resolved_foot_link_names)
        self._num_action = self.num_actuators

        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)
        self._init_base_pose = self.model.init_dof_pos[:7].copy()
        self._init_joint_position = np.empty((len(self._joint_names),), dtype=np.float32)
        self._init_buffers()
        self._init_obs_space()
        self._init_action_space()
        self._resample_steps = max(int(round(cfg.commands.resampling_time / cfg.ctrl_dt)), 1)

    def _validate_asset_cfg(self) -> None:
        cfg = self.cfg.asset
        required_names = {
            "ground_geom_name": cfg.ground_geom_name,
        }
        missing = sorted(name for name, value in required_names.items() if not value)
        if missing:
            raise ValueError(f"humanoid walk asset config requires non-empty fields: {missing}")
        if len(cfg.foot_height_site_names) != 2 or not all(cfg.foot_height_site_names):
            raise ValueError("asset.foot_height_site_names must contain left and right sole-height site names")

    def _validate_model_contract(self) -> None:
        if not self._joint_names or len(set(self._joint_names)) != len(self._joint_names):
            raise ValueError("humanoid walk requires unique actuator target joints")
        for actuator in self.model.actuators:
            if actuator.actuator_type is not ActuatorType.POSITION:
                raise TypeError(f"humanoid walk actuator {actuator.name!r} must be a position actuator")

    def _resolve_joint_values(self, mapping: dict[str, float], label: str) -> np.ndarray:
        """Validate a joint-value mapping and order it to match the body's joint layout."""
        joint_names = self._joint_names
        expected = set(joint_names)
        provided = set(mapping)
        missing = sorted(expected.difference(provided))
        unknown = sorted(provided.difference(expected))
        if missing or unknown:
            raise KeyError(f"{label} must match robot joints exactly; missing={missing}, unknown={unknown}")
        values = np.asarray([mapping[name] for name in joint_names], dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} values must be finite")
        return values

    def _init_buffers(self) -> None:
        cfg = self.cfg
        robot = self._robot_cfg
        if "default" not in robot.key_pose.poses:
            raise ValueError("humanoid walk robot must define key pose 'default'")
        default_by_joint = {
            robot.resolve_name(name): value
            for name, value in zip(robot.key_pose.joint_names, robot.key_pose.poses["default"], strict=True)
        }
        self.default_joint_angles = self._resolve_joint_values(default_by_joint, "robot key pose 'default'")
        self.pose_weights = self._resolve_joint_values(cfg.reward_config.pose_weights, "reward_config.pose_weights")
        if np.any(self.pose_weights < 0.0):
            raise ValueError("reward_config.pose_weights must be non-negative")

        self.default_angles = np.asarray(
            [default_by_joint[actuator.target_name] for actuator in self.model.actuators],
            dtype=np.float32,
        )
        self._init_joint_position[:] = self.default_joint_angles

        self._gait_freq = 1.0 / cfg.gait.period
        self._phase_dt = 2.0 * np.pi * self._gait_freq * cfg.ctrl_dt
        self._termination_geoms = self._resolve_termination_geoms()
        self._num_termination_pairs = len(self._termination_geoms)
        # Foot-link gravity direction at the default pose, captured on first
        # reset. Foot orientation penalties are measured against this
        # reference so robots whose ankle-link frames are not world-aligned
        # (e.g. onshape-to-robot exports) are scored correctly.
        self._default_foot_gravity: np.ndarray | None = None

        cur = cfg.curriculum
        self._penalty_terms = set(cur.penalty_terms)
        self._penalty_scale = cur.initial_scale if cur.enabled else 1.0
        self._avg_ep_len = 0.0
        self._max_steps = cfg.max_episode_steps

    def _resolve_termination_geoms(self) -> tuple[str, ...]:
        """Validate the task's explicitly configured termination geoms."""
        ground = self.cfg.asset.ground_geom_name
        if ground not in self.model.others["geoms"]:
            raise KeyError(f"unknown humanoid ground geom: {ground!r}")
        query = self._termination_query
        pinned = tuple(a for a, b in query.pairs)
        for a, b in query.pairs:
            if b != ground:
                raise ValueError(f"termination pair {a!r}x{b!r} must reference the configured ground geom {ground!r}")
        return pinned

    def _init_action_space(self) -> None:
        self._action_space = joint_position_action_space(
            self.model.actuators,
            self.default_angles,
            self.cfg.control_config.action_scale,
        )

    def _init_obs_space(self) -> None:
        num_joint_pos = self.default_joint_angles.shape[0]
        num_joint_vel = len(self._joint_names)
        actor_dim = 3 + 3 + 2 + 1 + num_joint_pos + num_joint_vel + self._num_action + 2 + 2
        critic_dim = 3 + actor_dim
        self._observation_space = ObsSpace(
            policy=gym.spaces.Box(-np.inf, np.inf, (actor_dim,), dtype=np.float32),
            value=gym.spaces.Box(-np.inf, np.inf, (critic_dim,), dtype=np.float32),
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> ObsSpace:
        return self._observation_space

    def _ground_height(self, env_ids: np.ndarray, xy: np.ndarray) -> np.ndarray:
        return self.sim.sample_terrain_height(self.cfg.asset.ground_geom_name, env_ids, xy)

    def _state_quantities(self, rows, env_ids: np.ndarray) -> StateQuantities:
        inputs = self.sim_data
        base_quat = inputs["base_quat"][rows]
        lin = inputs["base_lin_vel"][rows]
        ang = inputs["base_ang_vel"][rows]
        foot_pos = inputs["foot_pos"][rows]
        sole_pos = np.stack([inputs["sole_l_pos"][rows], inputs["sole_r_pos"][rows]], axis=1)
        ground_z = self._ground_height(env_ids, sole_pos[:, :, :2])
        return StateQuantities(
            base_quat=base_quat,
            base_lin_vel=quaternion.rotate_inverse(base_quat, lin),
            base_ang_vel=quaternion.rotate_inverse(base_quat, ang),
            projected_gravity=quaternion.rotate_inverse(base_quat, self.gravity_vec),
            foot_pos=foot_pos,
            foot_quat=inputs["foot_quat"][rows],
            foot_clearance=(sole_pos[:, :, 2] - ground_z).astype(np.float32),
            dof_pos=inputs["robot_joint_pos"][rows],
            dof_vel=inputs["robot_joint_vel"][rows],
        )

    def _phase(self, episode_steps: np.ndarray, info: dict) -> np.ndarray:
        steps = episode_steps.astype(np.float32).reshape(-1, 1)
        phase = np.fmod(steps * self._phase_dt + info["phase_offset"] + np.pi, 2 * np.pi) - np.pi
        cmd = info["commands"]
        stand = (np.linalg.norm(cmd[:, :2], axis=1) < 0.01) & (np.abs(cmd[:, 2]) < 0.01)
        phase[stand] = np.pi
        return phase.astype(np.float32)

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState) -> ArrayEnvState:
        steps = state.episode_steps
        due = (steps % self._resample_steps == 0) & (steps > 0)
        if np.any(due):
            state.info["commands"][due] = self.resample_commands(int(due.sum()))

        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        return state

    def physics_step(self) -> None:
        actions = self._state.info["current_actions"]
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = np.asarray(self._compute_target(actions), dtype=np.float32)
        self._ctrl_writes.execute()
        self.sim.step(self._cfg.sim_substeps)

    def _compute_target(self, actions: np.ndarray) -> np.ndarray:
        return actions * self.cfg.control_config.action_scale + self.default_angles

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        """Build the full observation from cached sim reads and info.

        Reads the transition/reset-owned ``info["phase"]`` cache; observation
        never refreshes the simulator or mutates reward/termination info.
        """
        all_ids = np.arange(self._num_envs, dtype=np.int64)
        q = self._state_quantities(slice(None), all_ids)
        info = state.info
        nrm = self.cfg.normalization
        ang_vel = q.base_ang_vel * nrm.base_ang_vel
        lin_vel = q.base_lin_vel * nrm.base_lin_vel
        gravity = q.projected_gravity
        cmd = info["commands"]
        dof_pos = (q.dof_pos - self.default_joint_angles) * nrm.dof_pos
        dof_vel = q.dof_vel * nrm.dof_vel
        actions = info["current_actions"]
        phase = info["phase"]
        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)

        noisy_dof_pos = dof_pos + np.random.uniform(-1, 1, dof_pos.shape).astype(np.float32) * nrm.noise_dof_pos
        noisy_dof_vel = dof_vel + np.random.uniform(-1, 1, dof_vel.shape).astype(np.float32) * nrm.noise_dof_vel
        actor = np.hstack(
            [
                ang_vel,
                gravity,
                cmd[:, :2],
                cmd[:, 2:3],
                noisy_dof_pos,
                noisy_dof_vel,
                actions,
                sin_phase,
                cos_phase,
            ]
        ).astype(np.float32)
        critic = np.hstack(
            [
                lin_vel,
                ang_vel,
                gravity,
                cmd[:, :2],
                cmd[:, 2:3],
                dof_pos,
                dof_vel,
                actions,
                sin_phase,
                cos_phase,
            ]
        ).astype(np.float32)
        return state.replace(obs=NpObs(policy=actor, value=critic))

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        all_ids = np.arange(self._num_envs, dtype=np.int64)
        q = self._state_quantities(slice(None), all_ids)
        state.info["phase"] = self._phase(state.episode_steps, state.info)
        state = self.update_terminated(state)
        self._update_curriculum(state)
        state = self.update_reward(state, q)
        return state

    def _update_curriculum(self, state: ArrayEnvState) -> None:
        cur = self.cfg.curriculum
        if not cur.enabled:
            return
        ep_len = state.episode_steps.astype(np.float64) + 1.0
        truncated = ep_len >= self._max_steps if self._max_steps else np.zeros_like(ep_len, dtype=bool)
        done = state.terminated | truncated
        if not np.any(done):
            return
        self._avg_ep_len = 0.99 * self._avg_ep_len + 0.01 * float(ep_len[done].mean())
        if self._avg_ep_len < cur.level_down_threshold:
            self._penalty_scale *= 1.0 - cur.degree
        elif self._avg_ep_len > cur.level_up_threshold:
            self._penalty_scale *= 1.0 + cur.degree
        self._penalty_scale = float(np.clip(self._penalty_scale, cur.min_scale, cur.max_scale))

    def update_terminated(self, state: ArrayEnvState) -> ArrayEnvState:
        if self._num_termination_pairs == 0:
            return state.replace(terminated=np.zeros((self._num_envs,), dtype=bool))
        colliding = self.sim_data["termination_colliding"]
        return state.replace(terminated=colliding.any(axis=1))

    def resample_commands(self, num_envs: int) -> np.ndarray:
        limits = np.asarray(self.cfg.commands.vel_limit, dtype=np.float32)
        if limits.shape != (2, 3):
            raise ValueError(f"commands.vel_limit must have shape (2, 3), got {limits.shape}")
        commands = np.random.uniform(low=limits[0], high=limits[1], size=(num_envs, 3)).astype(np.float32)
        stand = np.random.uniform(size=(num_envs,)) < self.cfg.commands.stand_prob
        commands[stand] = 0.0
        return commands

    def _sample_phase_offset(self, num_envs: int) -> np.ndarray:
        offset = np.zeros((num_envs, 2), dtype=np.float32)
        offset[:, 0] = np.random.uniform(-np.pi, np.pi, size=(num_envs,))
        offset[:, 1] = np.fmod(offset[:, 0] + 2 * np.pi, 2 * np.pi) - np.pi
        return offset

    def _sample_init_base_pose(self, env_ids: np.ndarray, num_reset: int) -> np.ndarray:
        pose = np.broadcast_to(self._init_base_pose, (num_reset, self._init_base_pose.shape[0])).copy()
        spawn_range = self.cfg.spawn_xy_range
        if spawn_range > 0.0:
            xy = np.random.uniform(-spawn_range, spawn_range, size=(num_reset, 2)).astype(np.float32)
            grid = np.array(
                [[dx, dy] for dx in (-0.15, 0.0, 0.15) for dy in (-0.15, 0.0, 0.15)],
                dtype=np.float32,
            )
            patch = xy[:, None, :] + grid[None, :, :]
            ground = self._ground_height(env_ids, patch).max(axis=1)
            pose[:, :2] = xy
            pose[:, 2] = self._init_base_pose[2] + ground
        return pose

    def reset(self, env_ids: np.ndarray) -> dict:
        num_reset = len(env_ids)
        row_ids = np.asarray(env_ids, dtype=np.int64)
        _reset_pose = self._sample_init_base_pose(row_ids, num_reset)
        self._reset_position[env_ids] = _reset_pose[:, :3]
        self._reset_rotation[env_ids] = _reset_pose[:, 3:7]
        self._reset_linear_velocity[env_ids] = 0.0
        self._reset_angular_velocity[env_ids] = 0.0
        self._reset_joint_position[env_ids] = self._init_joint_position
        self._reset_joint_velocity[env_ids] = 0.0
        self._reset_program.execute(env_ids)
        self.sim_data.execute(row_ids)
        if self._default_foot_gravity is None:
            foot_quat = self.sim_data["foot_quat"][row_ids[0]]
            self._default_foot_gravity = quaternion.rotate_inverse(foot_quat, self.gravity_vec)

        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "commands": self.resample_commands(num_reset),
            "phase_offset": self._sample_phase_offset(num_reset),
            "phase": np.zeros((num_reset, 2), dtype=np.float32),
        }
        info["phase"] = self._phase(np.zeros((num_reset,), dtype=np.uint64), info)
        return info

    def update_reward(self, state: ArrayEnvState, q: StateQuantities) -> ArrayEnvState:
        scales = self.cfg.reward_config.scales
        terms = self._get_reward(q, state.info)
        missing_scales = sorted(name for name in terms if not hasattr(scales, name))
        if missing_scales:
            raise KeyError(f"reward_config.scales is missing terms: {missing_scales}")
        weighted = {
            name: value
            * getattr(scales, name)
            * (self._penalty_scale if name in self._penalty_terms else 1.0)
            * self.cfg.ctrl_dt
            for name, value in terms.items()
        }
        state.info["Reward"] = dict(weighted)
        state.metrics = {"penalty_scale": self._penalty_scale}
        reward = sum(weighted.values())
        return state.replace(reward=reward.astype(np.float32))

    def _get_reward(self, q: StateQuantities, info: dict) -> dict[str, np.ndarray]:
        cfg = self.cfg.reward_config
        cmd = info["commands"]
        return {
            "tracking_lin_vel": self._r_tracking_lin_vel(q, cmd, cfg.tracking_sigma),
            "tracking_ang_vel": self._r_tracking_ang_vel(q, cmd, cfg.tracking_sigma),
            "penalty_ang_vel_xy": np.sum(np.square(q.base_ang_vel[:, :2]), axis=1),
            "penalty_orientation": np.sum(np.square(q.projected_gravity[:, :2]), axis=1),
            "penalty_action_rate": np.sum(np.square(info["current_actions"] - info["last_actions"]), axis=1),
            "feet_phase": self._r_feet_phase(q, info),
            "pose": np.sum(self.pose_weights * np.square(q.dof_pos - self.default_joint_angles), axis=1),
            "penalty_close_feet_xy": self._r_close_feet(q, cfg.close_feet_threshold),
            "penalty_feet_ori": self._r_feet_ori(q),
            "alive": np.ones((q.dof_pos.shape[0],), dtype=np.float32),
        }

    def _r_tracking_lin_vel(self, q: StateQuantities, cmd: np.ndarray, sigma: float) -> np.ndarray:
        error = np.sum(np.square(cmd[:, :2] - q.base_lin_vel[:, :2]), axis=1)
        return np.exp(-error / sigma)

    def _r_tracking_ang_vel(self, q: StateQuantities, cmd: np.ndarray, sigma: float) -> np.ndarray:
        error = np.square(cmd[:, 2] - q.base_ang_vel[:, 2])
        return np.exp(-error / sigma)

    def _r_feet_phase(self, q: StateQuantities, info: dict) -> np.ndarray:
        gait = self.cfg.gait
        reference_height = _expected_foot_height(info["phase"], gait.swing_height)
        error = np.sum(np.square(q.foot_clearance - reference_height), axis=1)
        return np.exp(-error / gait.feet_phase_sigma)

    def _r_close_feet(self, q: StateQuantities, threshold: float) -> np.ndarray:
        left_xy = q.foot_pos[:, 0, :2]
        right_xy = q.foot_pos[:, 1, :2]
        forward = quaternion.rotate_vector(q.base_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        yaw = np.arctan2(forward[:, 1], forward[:, 0])
        distance = np.abs(
            np.cos(yaw) * (left_xy[:, 1] - right_xy[:, 1]) - np.sin(yaw) * (left_xy[:, 0] - right_xy[:, 0])
        )
        return (distance < threshold).astype(np.float32)

    def _r_feet_ori(self, q: StateQuantities) -> np.ndarray:
        # Magnitude of the cross product between the current foot-frame gravity
        # and its default-pose reference. With a world-aligned default
        # (gravity = (0, 0, -1)) this reduces exactly to sqrt(gx^2 + gy^2).
        total = np.zeros((q.foot_quat.shape[0],), dtype=np.float32)
        for foot_index in range(2):
            foot_gravity = quaternion.rotate_inverse(q.foot_quat[:, foot_index], self.gravity_vec)
            reference = self._default_foot_gravity[foot_index]
            total = total + np.linalg.norm(np.cross(foot_gravity, reference), axis=1)
        return total
