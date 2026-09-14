# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generic quadruped walk task with joystick velocity tracking."""

import gymnasium as gym
import numpy as np

from motrix_env_core.array.env import ArrayEnvState, NpObs
from motrix_env_core.base import ObsSpace
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.sim import (
    ActuatorCtrlQuery,
    ActuatorKdQuery,
    ActuatorKpQuery,
    BodyAngularVelocityWrite,
    BodyCenterOfMassQuery,
    BodyJointPositionQuery,
    BodyJointPositionWrite,
    BodyJointVelocityQuery,
    BodyLinearVelocityWrite,
    BodyMassQuery,
    BodyPositionWrite,
    BodyRotationWrite,
    DofVelocityQuery,
    GeomFrictionQuery,
    LinkPositionQuery,
    SensorValuesQuery,
)
from motrix_env_core.sim.model import ActuatorType
from motrix_env_core.sim.write import (
    ActuatorDampingWrite,
    ActuatorKpWrite,
    BodyComWrite,
    BodyJointVelocityWrite,
    BodyMassWrite,
    CtrlTargetsWrite,
    GeomFrictionWrite,
)
from motrix_envs.locomotion.action_space import asymmetric_residual_action_space
from motrix_envs.locomotion.quadruped.cfg import QuadrupedWalkEnvCfg
from motrix_envs.locomotion.quadruped.velocity_command import RandomPlanarVelocityBinding
from motrix_envs.robot import QuadrupedRobotCfg


def _sim_data_queries(cfg: QuadrupedWalkEnvCfg):
    base_link_name = cfg.scene.objs.robot.resolved_base_link_name
    return {
        "joint_dof_pos": BodyJointPositionQuery(body=base_link_name),
        "joint_dof_vel": BodyJointVelocityQuery(body=base_link_name),
        "dof_vel": DofVelocityQuery(),
        "actuator_ctrls": ActuatorCtrlQuery(),
        "local_linvel": SensorValuesQuery(sensors=(cfg.sensor.local_linvel,)),
        "gyro": SensorValuesQuery(sensors=(cfg.sensor.gyro,)),
        "upvector": SensorValuesQuery(sensors=(cfg.sensor.upvector,)),
        "base_pos": LinkPositionQuery(link=base_link_name),
        "front_left_contact": SensorValuesQuery(sensors=("front_left_contact",)),
        "front_right_contact": SensorValuesQuery(sensors=("front_right_contact",)),
        "rear_left_contact": SensorValuesQuery(sensors=("rear_left_contact",)),
        "rear_right_contact": SensorValuesQuery(sensors=("rear_right_contact",)),
        "FL_pos": SensorValuesQuery(sensors=(cfg.sensor.foot_positions[0],)),
        "FR_pos": SensorValuesQuery(sensors=(cfg.sensor.foot_positions[1],)),
        "RL_pos": SensorValuesQuery(sensors=(cfg.sensor.foot_positions[2],)),
        "RR_pos": SensorValuesQuery(sensors=(cfg.sensor.foot_positions[3],)),
    }


def _sim_model_queries(cfg: QuadrupedWalkEnvCfg):
    base_link_name = cfg.scene.objs.robot.resolved_base_link_name
    return {
        "actuator_kp": ActuatorKpQuery(),
        "actuator_kd": ActuatorKdQuery(),
        "base_mass": BodyMassQuery(name=base_link_name),
        "base_com": BodyCenterOfMassQuery(name=base_link_name),
        "ground_friction": GeomFrictionQuery(name=cfg.ground_geom_name),
    }


_FOOTPRINT = np.array(
    [[dx, dy] for dx in (-0.25, 0.0, 0.25) for dy in (-0.15, 0.0, 0.15)],
    dtype=np.float32,
)


def _resolve_key_pose(
    robot: QuadrupedRobotCfg,
    actuators,
    pose_name: str,
) -> np.ndarray:
    actuator_joint_names: list[str] = []
    for spec in actuators:
        actuator_joint_names.append(spec.target_name)

    if pose_name not in robot.key_pose.poses:
        raise ValueError(f"Quadruped robot does not define key pose {pose_name!r}")
    positions = {
        robot.resolve_name(name): value
        for name, value in zip(robot.key_pose.joint_names, robot.key_pose.poses[pose_name])
    }
    missing = sorted(set(actuator_joint_names).difference(positions))
    extra = sorted(set(positions).difference(actuator_joint_names))
    if missing or extra:
        raise ValueError(
            f"Quadruped robot key pose {pose_name!r} must match actuator joint targets exactly: "
            f"missing={missing}, extra={extra}"
        )
    return np.asarray([positions[name] for name in actuator_joint_names], dtype=np.float32)


class QuadrupedWalkTask(DirectEnv[QuadrupedWalkEnvCfg]):
    """Base quadruped walk task driven by a :class:`QuadrupedRobotCfg`."""

    def __init__(self, cfg: QuadrupedWalkEnvCfg, num_envs=1, backend: str | None = None):
        robot = cfg.scene.objs.robot
        if not isinstance(robot, QuadrupedRobotCfg):
            raise TypeError(f"Quadruped walk scene robot must be QuadrupedRobotCfg, got {type(robot).__name__}")
        legs = (
            robot.legs.front_left,
            robot.legs.front_right,
            robot.legs.rear_left,
            robot.legs.rear_right,
        )
        if any(leg.contact_geom_name is None for leg in legs):
            raise ValueError("Quadruped walk requires a contact geom name for every leg")
        foot_position_sensors = cfg.sensor.foot_positions
        if len(foot_position_sensors) != len(legs) or any(not name for name in foot_position_sensors):
            raise ValueError("Quadruped walk requires one non-empty foot position sensor name per leg")
        if len(set(foot_position_sensors)) != len(foot_position_sensors):
            raise ValueError("Quadruped walk foot position sensor names must be unique")

        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_sim_model_queries(cfg))
        self.sim_data = self.sim.compile_reads(_sim_data_queries(cfg))
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._base_link_name = robot.resolved_base_link_name
        self._reset_program = self.sim.write_compiler.compile(
            {
                "base_position": BodyPositionWrite((self._base_link_name,)),
                "base_rotation": BodyRotationWrite((self._base_link_name,)),
                "base_linear_velocity": BodyLinearVelocityWrite((self._base_link_name,)),
                "base_angular_velocity": BodyAngularVelocityWrite((self._base_link_name,)),
                "joints_position": BodyJointPositionWrite(self._base_link_name),
                "joints_velocity": BodyJointVelocityWrite(self._base_link_name),
            },
            reset=True,
        )
        self._reset_position = self._reset_program.buffer("base_position")[:, 0]
        self._reset_rotation = self._reset_program.buffer("base_rotation")[:, 0]
        self._reset_linear_velocity = self._reset_program.buffer("base_linear_velocity")[:, 0]
        self._reset_angular_velocity = self._reset_program.buffer("base_angular_velocity")[:, 0]
        self._reset_joint_position = self._reset_program.buffer("joints_position")
        self._reset_joint_velocity = self._reset_program.buffer("joints_velocity")
        self._num_action = self.num_actuators
        self._num_feet = len(legs)
        self._feet_contact_sensors = (
            "front_left_contact",
            "front_right_contact",
            "rear_left_contact",
            "rear_right_contact",
        )
        self._feet_position_sensors = foot_position_sensors
        velocity_cfg = cfg.commands.velocity
        root_seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        command_seed, randomization_seed = np.random.SeedSequence(root_seed).spawn(2)
        self._command_rng = np.random.default_rng(command_seed)
        self._randomization_rng = np.random.default_rng(randomization_seed)
        self._velocity_command_binding = RandomPlanarVelocityBinding(
            velocity_cfg.lower,
            velocity_cfg.upper,
            rng=self._command_rng,
            standing_probability=velocity_cfg.standing_probability,
        )
        if not np.isfinite(velocity_cfg.standing_threshold) or velocity_cfg.standing_threshold < 0.0:
            raise ValueError(
                f"standing_threshold must be finite and non-negative, got {velocity_cfg.standing_threshold}"
            )

        self.default_angles = _resolve_key_pose(robot, self.model.actuators, cfg.key_pose_name)
        if self.default_angles.shape != (self._num_action,):
            raise ValueError(
                "Quadruped default joint positions must match actuator count: "
                f"got {self.default_angles.shape}, expected {(self._num_action,)}"
            )
        ctrl_ranges = np.asarray([spec.ctrl_range for spec in self.model.actuators], dtype=np.float32)
        self._joint_position_lower = ctrl_ranges[:, 0]
        self._joint_position_upper = ctrl_ranges[:, 1]
        self._init_action_space()
        self._init_obs_space()

        self._init_base_pose = self.model.init_dof_pos[:7].copy()
        self._init_base_pose[:3] = np.asarray(cfg.initial_base_position, dtype=np.float32)
        self._init_joint_position = self.default_angles.copy()

        has_non_position_actuator = any(
            spec.actuator_type is not ActuatorType.POSITION for spec in self.model.actuators
        )
        if cfg.randomization.enabled and has_non_position_actuator:
            raise TypeError("Quadruped PD randomization requires position actuators")
        self._randomize_action_delay = cfg.randomization.enabled and any(cfg.randomization.action_delay_steps)
        # Declare the enabled override writes once and batch every reset's
        # randomization into a single program execution.
        randomize_writes = {}
        self._randomize_kp = cfg.randomization.enabled and cfg.randomization.kp_scale_range != (1.0, 1.0)
        self._randomize_damping = cfg.randomization.enabled and cfg.randomization.damping_scale_range != (1.0, 1.0)
        self._randomize_friction = cfg.randomization.enabled and cfg.randomization.sliding_friction_range is not None
        self._randomize_base_mass = cfg.randomization.enabled and cfg.randomization.base_mass_scale_range != (1.0, 1.0)
        self._randomize_base_com = cfg.randomization.enabled and any(cfg.randomization.base_com_offset_noise)
        actuator_names = tuple(spec.name for spec in self.model.actuators)
        if self._randomize_kp:
            randomize_writes["kp"] = ActuatorKpWrite(actuator_names)
        if self._randomize_damping:
            randomize_writes["damping"] = ActuatorDampingWrite(actuator_names)
        if self._randomize_friction:
            randomize_writes["friction"] = GeomFrictionWrite((cfg.ground_geom_name,))
        if self._randomize_base_mass:
            randomize_writes["mass"] = BodyMassWrite((self._base_link_name,))
        if self._randomize_base_com:
            randomize_writes["com"] = BodyComWrite((self._base_link_name,))
        self._randomize_writes = self.sim.write_compiler.compile(randomize_writes) if randomize_writes else None

        self.feet_contact = np.zeros((num_envs, self._num_feet), dtype=bool)
        self.feet_pos = np.zeros((num_envs, self._num_feet, 3), dtype=np.float32)

    # ---- obs/action space plumbing ----------------------------------------

    def _policy_obs_dim(self) -> int:
        # gyro(3) + gravity(3) + dof_diff(A) + dof_vel(A) + last_actions(A)
        # + command(3) + feet_phase(num_feet)
        return 3 + 3 + 3 * self._num_action + 3 + self._num_feet

    def _value_obs_dim(self) -> int:
        return self._policy_obs_dim() + 3  # + local linvel(3)

    def _init_action_space(self):
        self._action_space = asymmetric_residual_action_space(
            np.asarray([self._joint_position_lower, self._joint_position_upper], dtype=np.float32),
            self.default_angles,
            self.cfg.control_config.action_scale,
        )

    def _init_obs_space(self):
        self._observation_space = ObsSpace(
            policy=gym.spaces.Box(-np.inf, np.inf, (self._policy_obs_dim(),), dtype=np.float32),
            value=gym.spaces.Box(-np.inf, np.inf, (self._value_obs_dim(),), dtype=np.float32),
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> ObsSpace:
        return self._observation_space

    # ---- state accessors --------------------------------------------------

    def get_dof_pos(self) -> np.ndarray:
        return self.sim_data["joint_dof_pos"]

    def get_dof_vel(self) -> np.ndarray:
        return self.sim_data["joint_dof_vel"]

    def get_local_linvel(self) -> np.ndarray:
        return self.sim_data["local_linvel"]

    def get_gyro(self) -> np.ndarray:
        return self.sim_data["gyro"]

    def get_gravity(self) -> np.ndarray:
        return self.sim_data["upvector"]

    # ---- contact / phase helpers -----------------------------------------

    def _obs_noise(self, data: np.ndarray, scale: float) -> np.ndarray:
        level = float(self.cfg.noise_config.level)
        if level <= 0.0:
            return data
        noise = np.random.uniform(-1.0, 1.0, data.shape).astype(data.dtype)
        return data + level * scale * noise

    def _read_feet_contact(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        rows = slice(None) if env_ids is None else env_ids
        inputs = self.sim_data
        contacts = np.zeros((self._num_envs, self._num_feet), dtype=bool)
        for i, name in enumerate(self._feet_contact_sensors):
            value = inputs[name][rows]
            contacts[rows, i] = value[:, 0] > 0.0
        return contacts

    def _update_feet_buffers(self, env_ids: np.ndarray | None = None) -> np.ndarray:
        contacts = self._read_feet_contact(env_ids)
        rows = slice(None) if env_ids is None else env_ids
        self.feet_contact[rows, :] = contacts[rows, :]
        for i, name in enumerate(self._feet_position_sensors):
            self.feet_pos[rows, i, :] = self.sim_data[name][rows]
        return contacts if env_ids is None else contacts[env_ids]

    def _advance_phase(self, info: dict):
        commands = info["commands"]
        standing = np.linalg.norm(commands, axis=1) < self.cfg.commands.velocity.standing_threshold
        phase = info["phase"]
        phase = np.fmod(phase + self.cfg.ctrl_dt * self.cfg.gait_frequency, 1.0).astype(np.float32, copy=False)
        phase[standing] = 0.0
        feet_phase = info["feet_phase"]
        # Each trot pair swings together; pair ``n`` is offset by half a cycle
        # relative to pair ``n-1``.
        for pair_idx, (i, j) in enumerate(self.cfg.trot_pairs):
            offset = 0.5 * pair_idx
            value = (phase + offset) % 1.0
            feet_phase[:, i] = value
            feet_phase[:, j] = value
        feet_phase[standing] = 0.0
        info["phase"] = phase
        info["feet_phase"] = feet_phase

    # ---- env loop ---------------------------------------------------------

    def apply_action(self, actions: np.ndarray, state: ArrayEnvState) -> ArrayEnvState:
        actions = np.asarray(actions, dtype=np.float32)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        if self._randomize_action_delay:
            use_last = state.info["action_delay_steps"] == 1
            exec_actions = np.where(use_last[:, None], state.info["last_actions"], actions)
        else:
            exec_actions = state.info["last_actions"] if self.cfg.control_config.simulate_action_latency else actions
        targets = exec_actions * self.cfg.control_config.action_scale + self.default_angles
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = targets.astype(np.float32, copy=False)
        self._ctrl_writes.execute()
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        noise_cfg = self.cfg.noise_config
        diff = self.get_dof_pos() - self.default_angles
        noisy_gyro = self._obs_noise(self.get_gyro(), noise_cfg.scale_gyro)
        noisy_gravity = self._obs_noise(self.get_gravity(), noise_cfg.scale_gravity)
        noisy_diff = self._obs_noise(diff, noise_cfg.scale_joint_angle)
        noisy_dof_vel = self._obs_noise(self.get_dof_vel(), noise_cfg.scale_joint_vel)
        noisy_linvel = self._obs_noise(self.get_local_linvel(), noise_cfg.scale_linvel)
        command = state.info["commands"]
        last_actions = state.info["current_actions"]
        feet_phase = state.info["feet_phase"]

        policy = np.concatenate(
            [
                noisy_gyro,
                -noisy_gravity,
                noisy_diff,
                noisy_dof_vel,
                last_actions,
                command,
                feet_phase,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        value = np.concatenate([policy, noisy_linvel], axis=1).astype(np.float32, copy=False)
        expected_policy = self._policy_obs_dim()
        expected_value = self._value_obs_dim()
        if policy.shape[1] != expected_policy or value.shape[1] != expected_value:
            raise ValueError(
                f"Unexpected quadruped obs shapes: policy={policy.shape}, "
                f"expected policy[{expected_policy}]; value={value.shape}, "
                f"expected value[{expected_value}]"
            )
        return state.replace(obs=NpObs(policy=policy, value=value))

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        self._update_commands(state.info)
        self._advance_phase(state.info)
        state.info["contacts"] = self._update_feet_buffers()

        terminated = self.get_gravity()[:, 2] <= 0.5
        reward = self._compute_reward(state.info, self.get_local_linvel(), self.get_gyro(), self.get_dof_pos())
        return state.replace(reward=reward, terminated=terminated)

    def _compute_reward(self, info: dict, linvel: np.ndarray, gyro: np.ndarray, dof_pos: np.ndarray) -> np.ndarray:
        reward = np.zeros((self._num_envs,), dtype=np.float32)
        reward_cfg = self.cfg.reward_config
        reward_fns = {
            "tracking_lin_vel": self._reward_tracking_lin_vel,
            "tracking_ang_vel": self._reward_tracking_ang_vel,
            "lin_vel_z": self._reward_lin_vel_z,
            "ang_vel_xy": self._reward_ang_vel_xy,
            "base_height": self._reward_base_height,
            "action_rate": self._reward_action_rate,
            "similar_to_default": self._reward_similar_to_default,
            "contact": self._reward_contact,
            "swing_feet_z": self._reward_swing_feet_z,
            "swing_contact": self._reward_swing_contact,
        }
        reward_items = {}
        for name, reward_fn in reward_fns.items():
            scale = getattr(reward_cfg.scales, name)
            if scale == 0:
                continue
            rew = reward_fn(info, linvel, gyro, dof_pos)
            weighted = (rew * scale).astype(np.float32, copy=False)
            reward += weighted
            reward_items[name] = weighted

        info["Reward"] = reward_items
        return reward * self.cfg.ctrl_dt

    def reset(self, env_ids: np.ndarray):
        num_reset = len(env_ids)
        base_pose = np.tile(self._init_base_pose, (num_reset, 1))
        base_linear_velocity = np.zeros((num_reset, 3), dtype=np.float32)
        base_angular_velocity = np.zeros((num_reset, 3), dtype=np.float32)
        joint_position = np.tile(self._init_joint_position, (num_reset, 1))
        joint_velocity = np.zeros((num_reset, self._reset_joint_velocity.shape[1]), dtype=np.float32)
        self._randomize_dof_noise(env_ids, base_linear_velocity, base_angular_velocity, joint_position, joint_velocity)

        spawn_range = self.cfg.spawn_xy_range
        if spawn_range > 0.0:
            xy = np.random.uniform(-spawn_range, spawn_range, size=(num_reset, 2)).astype(np.float32)
            ground_height = self.sim.sample_terrain_height(
                self.cfg.ground_geom_name, env_ids, xy[:, None, :] + _FOOTPRINT[None, :, :]
            ).max(axis=1)
            base_pose[:, :2] = xy
            base_pose[:, 2] = self._init_base_pose[2] + ground_height

        _reset_pose = base_pose
        self._reset_position[env_ids] = _reset_pose[:, :3]
        self._reset_rotation[env_ids] = _reset_pose[:, 3:7]
        self._reset_linear_velocity[env_ids] = base_linear_velocity
        self._reset_angular_velocity[env_ids] = base_angular_velocity
        self._reset_joint_position[env_ids] = joint_position
        self._reset_joint_velocity[env_ids] = joint_velocity
        self._reset_program.execute(env_ids)
        action_delay_steps = self._randomize_params(env_ids)

        self.sim_data.execute(np.asarray(env_ids, dtype=np.int64))
        contacts = self._update_feet_buffers(env_ids)

        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "commands": self.resample_commands(num_reset),
            "command_resampling_time": self._sample_command_resampling_time(num_reset),
            "phase": np.zeros((num_reset,), dtype=np.float32),
            "feet_phase": np.zeros((num_reset, self._num_feet), dtype=np.float32),
            "contacts": contacts,
        }
        if action_delay_steps is not None:
            info["action_delay_steps"] = action_delay_steps
        return info

    def resample_commands(self, num_envs: int) -> np.ndarray:
        return self._velocity_command_binding.read_command(batch_size=num_envs).values.copy()

    def _sample_command_resampling_time(self, num_envs: int) -> np.ndarray:
        interval = self.cfg.commands.velocity.resampling_seconds_range
        if interval is None:
            return np.full((num_envs,), np.inf, dtype=np.float32)
        return self._command_rng.uniform(*interval, size=num_envs).astype(np.float32)

    def _update_commands(self, info: dict) -> None:
        if self.cfg.commands.velocity.resampling_seconds_range is None:
            return
        remaining = info["command_resampling_time"] - self.cfg.ctrl_dt
        due = remaining <= 0.0
        num_due = int(np.count_nonzero(due))
        if num_due:
            info["commands"][due] = self.resample_commands(num_due)
            remaining[due] = self._sample_command_resampling_time(num_due)
        info["command_resampling_time"] = remaining.astype(np.float32, copy=False)

    def _randomize_dof_noise(
        self,
        env_ids: np.ndarray,
        base_linear_velocity: np.ndarray,
        base_angular_velocity: np.ndarray,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
    ) -> None:
        num_reset = len(env_ids)
        randomization = self.cfg.randomization
        if not randomization.enabled:
            return
        rng = self._randomization_rng
        num_action = self._num_action

        if randomization.joint_pos_noise > 0.0:
            joint_position[:, -num_action:] += rng.uniform(
                -randomization.joint_pos_noise,
                randomization.joint_pos_noise,
                size=(num_reset, num_action),
            ).astype(np.float32)
            np.clip(
                joint_position[:, -num_action:],
                self._joint_position_lower,
                self._joint_position_upper,
                out=joint_position[:, -num_action:],
            )
        if randomization.joint_vel_noise > 0.0:
            joint_velocity[:, -num_action:] += rng.uniform(
                -randomization.joint_vel_noise,
                randomization.joint_vel_noise,
                size=(num_reset, num_action),
            ).astype(np.float32)

        if any(randomization.base_lin_vel_noise):
            base_lin_vel_noise = np.asarray(randomization.base_lin_vel_noise, dtype=np.float32)
            base_linear_velocity[:] += rng.uniform(-base_lin_vel_noise, base_lin_vel_noise, size=(num_reset, 3)).astype(
                np.float32
            )
        if any(randomization.base_ang_vel_noise):
            base_ang_vel_noise = np.asarray(randomization.base_ang_vel_noise, dtype=np.float32)
            base_angular_velocity[:] += rng.uniform(
                -base_ang_vel_noise, base_ang_vel_noise, size=(num_reset, 3)
            ).astype(np.float32)

    def _randomize_params(self, env_ids: np.ndarray) -> np.ndarray | None:
        num_reset = len(env_ids)
        randomization = self.cfg.randomization
        if not randomization.enabled:
            return None
        rng = self._randomization_rng
        num_action = self._num_action

        action_delay_steps = None
        if self._randomize_action_delay:
            delay_low, delay_high = randomization.action_delay_steps
            if delay_low == delay_high:
                action_delay_steps = np.full((num_reset,), delay_low, dtype=np.int32)
            else:
                action_delay_steps = rng.integers(delay_low, delay_high + 1, size=num_reset, dtype=np.int32)

        # All override writes for this reset batch into one program execution.
        program = self._randomize_writes
        if program is None:
            return action_delay_steps

        if self._randomize_kp:
            kp_scale = rng.uniform(*self.cfg.randomization.kp_scale_range, size=(num_reset, num_action)).astype(
                np.float32
            )
            program.buffer("kp")[env_ids] = self.model.others["actuator_kp"][None, :] * kp_scale
        if self._randomize_damping:
            damping_scale = rng.uniform(
                *self.cfg.randomization.damping_scale_range, size=(num_reset, num_action)
            ).astype(np.float32)
            program.buffer("damping")[env_ids] = self.model.others["actuator_kd"][None, :] * damping_scale
        if self._randomize_friction:
            friction = np.tile(self.model.others["ground_friction"], (num_reset, 1))
            friction[:, 0] = rng.uniform(*self.cfg.randomization.sliding_friction_range, size=num_reset).astype(
                np.float32
            )
            program.buffer("friction")[env_ids, 0] = friction
        if self._randomize_base_mass:
            base_mass_scale = rng.uniform(*self.cfg.randomization.base_mass_scale_range, size=num_reset).astype(
                np.float32
            )
            program.buffer("mass")[env_ids, 0] = np.full((num_reset,), self.model.others["base_mass"]) * base_mass_scale
        if self._randomize_base_com:
            base_com_noise = np.asarray(self.cfg.randomization.base_com_offset_noise, dtype=np.float32)
            base_com_offset = rng.uniform(-base_com_noise, base_com_noise, size=(num_reset, 3)).astype(np.float32)
            program.buffer("com")[env_ids, 0] = np.tile(self.model.others["base_com"], (num_reset, 1)) + base_com_offset
        program.execute(env_ids)

        return action_delay_steps

    # ---- reward functions -------------------------------------------------

    def _reward_tracking_lin_vel(self, info, linvel, gyro, dof_pos) -> np.ndarray:
        del gyro, dof_pos
        commands = info["commands"]
        error = np.sum(np.square(commands[:, :2] - linvel[:, :2]), axis=1)
        return np.exp(-error / self.cfg.reward_config.tracking_lin_vel_sigma)

    def _reward_tracking_ang_vel(self, info, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, dof_pos
        commands = info["commands"]
        error = np.square(commands[:, 2] - gyro[:, 2])
        return np.exp(-error / self.cfg.reward_config.tracking_ang_vel_sigma)

    def _reward_lin_vel_z(self, info, linvel, gyro, dof_pos) -> np.ndarray:
        del info, gyro, dof_pos
        return np.square(linvel[:, 2])

    def _reward_ang_vel_xy(self, info, linvel, gyro, dof_pos) -> np.ndarray:
        del info, linvel, dof_pos
        return np.sum(np.square(gyro[:, :2]), axis=1)

    def _reward_base_height(self, info, linvel, gyro, dof_pos) -> np.ndarray:
        del info, linvel, gyro, dof_pos
        base_pos = self.sim_data["base_pos"]
        env_ids = np.arange(self._num_envs, dtype=np.int64)
        ground_height = self.sim.sample_terrain_height(self.cfg.ground_geom_name, env_ids, base_pos[:, None, :2])[:, 0]
        base_height = base_pos[:, 2].astype(np.float32) - ground_height
        return np.square(base_height - self.cfg.reward_config.base_height_target)

    def _reward_action_rate(self, info, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, gyro, dof_pos
        current = info["current_actions"]
        last = info["last_actions"]
        return np.sum(np.square(current - last), axis=1)

    def _reward_similar_to_default(self, info, linvel, gyro, dof_pos) -> np.ndarray:
        del info, linvel, gyro
        return np.sum(np.abs(dof_pos - self.default_angles), axis=1)

    def _reward_swing_feet_z(self, info, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, gyro, dof_pos
        feet_phase = info["feet_phase"]
        contacts = info["contacts"]
        valid_swing = (feet_phase >= 0.6) & ~contacts
        reward_cfg = self.cfg.reward_config
        # feet_pos is body-relative (framepos with ref=imu). The foot lifts
        # `target_height` above its stance position, so the body-frame target is
        # target_height - base_height_target (e.g. 0.1 - 0.3 = -0.2). This keeps
        # the reward invariant to the body's world-frame vertical motion.
        target_z = reward_cfg.target_foot_height - reward_cfg.base_height_target
        height_error = np.square(self.feet_pos[:, :, 2] - target_z)
        sigma_sq = reward_cfg.swing_feet_height_sigma**2
        swing_rew = np.exp(-height_error / sigma_sq) * valid_swing
        return np.sum(swing_rew, axis=1) / self._num_feet

    def _reward_contact(self, info, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, gyro, dof_pos
        res = np.zeros((self._num_envs,), dtype=np.float32)
        feet_phase = info["feet_phase"]
        contacts = info["contacts"]
        for i in range(self._num_feet):
            target_contact = feet_phase[:, i] < 0.6
            res += (contacts[:, i] == target_contact).astype(np.float32)
        return res / self._num_feet

    def _reward_swing_contact(self, info, linvel, gyro, dof_pos) -> np.ndarray:
        del linvel, gyro, dof_pos
        is_swing = info["feet_phase"] >= 0.6
        swing_contacts = info["contacts"] & is_swing
        return np.sum(swing_contacts, axis=1) / self._num_feet


__all__ = ["QuadrupedWalkTask"]
