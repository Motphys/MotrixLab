# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Generic quadruped walk task with joystick velocity tracking."""

import math

import gymnasium as gym
import numpy as np
import torch
from motrix_robots.quadruped import QuadrupedRobotCfg

from motrix_env_core.array.env import ArrayEnvState, EnvObs
from motrix_env_core.base import ObsSpace
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.mdp.action_space import asymmetric_residual_action_space
from motrix_env_core.sim import (
    ActuatorCtrlQuery,
    ActuatorKdQuery,
    ActuatorKpQuery,
    BodyAngularVelocityWrite,
    BodyJointPositionQuery,
    BodyJointPositionWrite,
    BodyJointVelocityQuery,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    DofVelocityQuery,
    GeomFrictionQuery,
    LinkCenterOfMassQuery,
    LinkMassQuery,
    LinkPositionQuery,
    SensorValuesQuery,
)
from motrix_env_core.sim.model import ActuatorType
from motrix_env_core.sim.write import (
    ActuatorDampingWrite,
    ActuatorKpWrite,
    BodyJointVelocityWrite,
    CtrlTargetsWrite,
    GeomFrictionWrite,
    LinkComWrite,
    LinkMassWrite,
)
from motrix_envs.locomotion.quadruped.cfg import QuadrupedWalkEnvCfg
from motrix_envs.locomotion.quadruped.velocity_command import RandomPlanarVelocityBinding


def _torch_uniform(rng: torch.Generator, low, high, size) -> torch.Tensor:
    """Sample float32 uniform values in ``[low, high)`` from one torch stream."""
    low = torch.as_tensor(low, dtype=torch.float32)
    high = torch.as_tensor(high, dtype=torch.float32)
    return low + (high - low) * torch.rand(size, generator=rng)


def _sim_data_queries(cfg: QuadrupedWalkEnvCfg):
    robot = cfg.scene.objs.robot
    base_link_name = robot.resolved_base_link_name
    scene_sensor_names = {name for name, _ in cfg.scene.iter_sensors()}

    def resolve_sensor_name(name: str) -> str:
        return name if name in scene_sensor_names else robot.resolve_name(name)

    return {
        "joint_dof_pos": BodyJointPositionQuery(body=base_link_name),
        "joint_dof_vel": BodyJointVelocityQuery(body=base_link_name),
        "dof_vel": DofVelocityQuery(),
        "actuator_ctrls": ActuatorCtrlQuery(),
        "local_linvel": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.local_linvel),)),
        "gyro": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.gyro),)),
        "upvector": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.upvector),)),
        "base_pos": LinkPositionQuery(link=base_link_name),
        "front_left_contact": SensorValuesQuery(sensors=("front_left_contact",)),
        "front_right_contact": SensorValuesQuery(sensors=("front_right_contact",)),
        "rear_left_contact": SensorValuesQuery(sensors=("rear_left_contact",)),
        "rear_right_contact": SensorValuesQuery(sensors=("rear_right_contact",)),
        "FL_pos": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.foot_positions[0]),)),
        "FR_pos": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.foot_positions[1]),)),
        "RL_pos": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.foot_positions[2]),)),
        "RR_pos": SensorValuesQuery(sensors=(resolve_sensor_name(cfg.sensor.foot_positions[3]),)),
    }


def _sim_model_queries(cfg: QuadrupedWalkEnvCfg):
    base_link_name = cfg.scene.objs.robot.resolved_base_link_name
    return {
        "actuator_kp": ActuatorKpQuery(),
        "actuator_kd": ActuatorKdQuery(),
        "base_mass": LinkMassQuery(name=base_link_name),
        "base_com": LinkCenterOfMassQuery(name=base_link_name),
        "ground_friction": GeomFrictionQuery(name=cfg.ground_geom_name),
    }


_FOOTPRINT = torch.tensor(
    [[dx, dy] for dx in (-0.25, 0.0, 0.25) for dy in (-0.15, 0.0, 0.15)],
    dtype=torch.float32,
)


def _resolve_key_pose(
    robot: QuadrupedRobotCfg,
    actuators,
    pose_name: str,
) -> torch.Tensor:
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
    return torch.tensor([positions[name] for name in actuator_joint_names], dtype=torch.float32)


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
        # Two independent torch streams derived from one root seed drawn from
        # the global torch RNG (training frameworks seed torch, so runs stay
        # reproducible under their seeding).
        root_seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        self._command_rng = torch.Generator()
        self._command_rng.manual_seed(root_seed)
        self._randomization_rng = torch.Generator()
        self._randomization_rng.manual_seed(root_seed + 1)
        self._velocity_command_binding = RandomPlanarVelocityBinding(
            torch.as_tensor(velocity_cfg.lower, dtype=torch.float32),
            torch.as_tensor(velocity_cfg.upper, dtype=torch.float32),
            rng=self._command_rng,
            standing_probability=velocity_cfg.standing_probability,
        )
        if not math.isfinite(velocity_cfg.standing_threshold) or velocity_cfg.standing_threshold < 0.0:
            raise ValueError(
                f"standing_threshold must be finite and non-negative, got {velocity_cfg.standing_threshold}"
            )

        self.default_angles = _resolve_key_pose(robot, self.model.actuators, cfg.key_pose_name)
        if self.default_angles.shape != (self._num_action,):
            raise ValueError(
                "Quadruped default joint positions must match actuator count: "
                f"got {tuple(self.default_angles.shape)}, expected {(self._num_action,)}"
            )
        ctrl_ranges = torch.tensor([spec.ctrl_range for spec in self.model.actuators], dtype=torch.float32)
        self._joint_position_lower = ctrl_ranges[:, 0]
        self._joint_position_upper = ctrl_ranges[:, 1]
        self._init_action_space()
        self._init_obs_space()

        self._init_base_pose = self.model.init_dof_pos[:7].clone()
        self._init_base_pose[:3] = torch.tensor(cfg.initial_base_position, dtype=torch.float32)
        self._init_joint_position = self.default_angles.clone()

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
            randomize_writes["mass"] = LinkMassWrite((self._base_link_name,))
        if self._randomize_base_com:
            randomize_writes["com"] = LinkComWrite((self._base_link_name,))
        self._randomize_writes = self.sim.write_compiler.compile(randomize_writes) if randomize_writes else None

        self.feet_contact = torch.zeros((num_envs, self._num_feet), dtype=torch.bool)
        self.feet_pos = torch.zeros((num_envs, self._num_feet, 3), dtype=torch.float32)

        # Episode-scoped task state: full-batch buffers, reset writes the done
        # rows in place.
        self._commands = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._command_resampling_time = torch.full((num_envs,), torch.inf, dtype=torch.float32)
        self._phase = torch.zeros(num_envs, dtype=torch.float32)
        self._feet_phase = torch.zeros((num_envs, self._num_feet), dtype=torch.float32)
        self._current_actions = torch.zeros((num_envs, self._num_action), dtype=torch.float32)
        self._last_actions = torch.zeros((num_envs, self._num_action), dtype=torch.float32)
        self._action_delay_steps = torch.zeros(num_envs, dtype=torch.int64)

    # ---- obs/action space plumbing ----------------------------------------

    def _policy_obs_dim(self) -> int:
        # gyro(3) + gravity(3) + dof_diff(A) + dof_vel(A) + last_actions(A)
        # + command(3) + feet_phase(num_feet)
        return 3 + 3 + 3 * self._num_action + 3 + self._num_feet

    def _value_obs_dim(self) -> int:
        return self._policy_obs_dim() + 3  # + local linvel(3)

    def _init_action_space(self):
        self._action_space = asymmetric_residual_action_space(
            torch.stack((self._joint_position_lower, self._joint_position_upper)),
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

    def get_dof_pos(self) -> torch.Tensor:
        return self.sim_data["joint_dof_pos"]

    def get_dof_vel(self) -> torch.Tensor:
        return self.sim_data["joint_dof_vel"]

    def get_local_linvel(self) -> torch.Tensor:
        return self.sim_data["local_linvel"]

    def get_gyro(self) -> torch.Tensor:
        return self.sim_data["gyro"]

    def get_gravity(self) -> torch.Tensor:
        return self.sim_data["upvector"]

    # ---- contact / phase helpers -----------------------------------------

    def _obs_noise(self, data: torch.Tensor, scale: float) -> torch.Tensor:
        level = float(self.cfg.noise_config.level)
        if level <= 0.0:
            return data
        noise = torch.rand(data.shape, dtype=torch.float32) * 2.0 - 1.0
        return data + level * scale * noise

    def _read_feet_contact(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        rows = slice(None) if env_ids is None else env_ids
        inputs = self.sim_data
        contacts = torch.zeros((self._num_envs, self._num_feet), dtype=torch.bool)
        for i, name in enumerate(self._feet_contact_sensors):
            contacts[rows, i] = inputs[name][rows, 0] > 0.0
        return contacts

    def _update_feet_buffers(self, env_ids: torch.Tensor | None = None) -> None:
        contacts = self._read_feet_contact(env_ids)
        rows = slice(None) if env_ids is None else env_ids
        self.feet_contact[rows, :] = contacts[rows, :]
        for i, name in enumerate(self._feet_position_sensors):
            self.feet_pos[rows, i, :] = self.sim_data[name][rows]

    def _advance_phase(self):
        commands = self._commands
        standing = torch.linalg.vector_norm(commands, dim=1) < self.cfg.commands.velocity.standing_threshold
        phase = self._phase
        phase[:] = torch.remainder(phase + self.cfg.ctrl_dt * self.cfg.gait_frequency, 1.0)
        phase[standing] = 0.0
        feet_phase = self._feet_phase
        # Each trot pair swings together; pair ``n`` is offset by half a cycle
        # relative to pair ``n-1``.
        for pair_idx, (i, j) in enumerate(self.cfg.trot_pairs):
            offset = 0.5 * pair_idx
            value = torch.remainder(phase + offset, 1.0)
            feet_phase[:, i] = value
            feet_phase[:, j] = value
        feet_phase[standing] = 0.0

    # ---- env loop ---------------------------------------------------------

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        actions = torch.as_tensor(actions, dtype=torch.float32)
        self._last_actions.copy_(self._current_actions)
        self._current_actions.copy_(actions)
        if self._randomize_action_delay:
            use_last = self._action_delay_steps == 1
            exec_actions = torch.where(use_last[:, None], self._last_actions, actions)
        else:
            exec_actions = self._last_actions if self.cfg.control_config.simulate_action_latency else actions
        targets = exec_actions * self.cfg.control_config.action_scale + self.default_angles
        self._ctrl_writes.buffer("ctrl").copy_(targets)
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
        command = self._commands
        last_actions = self._current_actions
        feet_phase = self._feet_phase

        policy = torch.cat(
            [
                noisy_gyro,
                -noisy_gravity,
                noisy_diff,
                noisy_dof_vel,
                last_actions,
                command,
                feet_phase,
            ],
            dim=1,
        )
        value = torch.cat([policy, noisy_linvel], dim=1)
        expected_policy = self._policy_obs_dim()
        expected_value = self._value_obs_dim()
        if policy.shape[1] != expected_policy or value.shape[1] != expected_value:
            raise ValueError(
                f"Unexpected quadruped obs shapes: policy={tuple(policy.shape)}, "
                f"expected policy[{expected_policy}]; value={tuple(value.shape)}, "
                f"expected value[{expected_value}]"
            )
        return state.replace(obs=EnvObs(policy=policy, value=value))

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        self._update_commands()
        self._advance_phase()
        self._update_feet_buffers()

        terminated = self.get_gravity()[:, 2] <= self.cfg.termination_min_up_z
        reward = self._compute_reward(state, self.get_local_linvel(), self.get_gyro(), self.get_dof_pos())
        return state.replace(reward=reward, terminated=terminated)

    def _compute_reward(
        self, state: ArrayEnvState, linvel: torch.Tensor, gyro: torch.Tensor, dof_pos: torch.Tensor
    ) -> torch.Tensor:
        reward = torch.zeros((self._num_envs,), dtype=torch.float32)
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
            rew = reward_fn(linvel, gyro, dof_pos)
            weighted = rew * scale
            reward += weighted
            reward_items[name] = weighted

        state.reward_terms = reward_items
        return reward * self.cfg.ctrl_dt

    def reset(self, env_ids: torch.Tensor) -> None:
        num_reset = env_ids.numel()
        base_pose = self._init_base_pose.repeat(num_reset, 1)
        base_linear_velocity = torch.zeros((num_reset, 3), dtype=torch.float32)
        base_angular_velocity = torch.zeros((num_reset, 3), dtype=torch.float32)
        joint_position = self._init_joint_position.repeat(num_reset, 1)
        joint_velocity = torch.zeros((num_reset, self._reset_joint_velocity.shape[1]), dtype=torch.float32)
        self._randomize_dof_noise(env_ids, base_linear_velocity, base_angular_velocity, joint_position, joint_velocity)

        spawn_range = self.cfg.spawn_xy_range
        if spawn_range > 0.0:
            xy = (torch.rand(num_reset, 2, dtype=torch.float32) * 2.0 - 1.0) * spawn_range
            ground_height = (
                self.sim.sample_terrain_height(self.cfg.ground_geom_name, env_ids, xy[:, None, :] + _FOOTPRINT)
                .max(dim=1)
                .values
            )
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
        self._randomize_params(env_ids)

        self.sim_data.execute(env_ids)
        self._update_feet_buffers(env_ids)

        self._commands[env_ids] = torch.as_tensor(self.resample_commands(num_reset))
        self._command_resampling_time[env_ids] = self._sample_command_resampling_time(num_reset)
        self._phase[env_ids] = 0.0
        self._feet_phase[env_ids] = 0.0
        self._current_actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0

    def resample_commands(self, num_envs: int) -> torch.Tensor:
        return self._velocity_command_binding.read_command(batch_size=num_envs).values

    def _sample_command_resampling_time(self, num_envs: int) -> torch.Tensor:
        interval = self.cfg.commands.velocity.resampling_seconds_range
        if interval is None:
            return torch.full((num_envs,), torch.inf, dtype=torch.float32)
        low, high = interval
        return low + torch.rand(num_envs, dtype=torch.float32) * (high - low)

    def _update_commands(self) -> None:
        if self.cfg.commands.velocity.resampling_seconds_range is None:
            return
        remaining = self._command_resampling_time
        remaining -= self.cfg.ctrl_dt
        due = remaining <= 0.0
        num_due = int(due.sum())
        if num_due:
            self._commands[due] = torch.as_tensor(self.resample_commands(num_due))
            remaining[due] = self._sample_command_resampling_time(num_due)

    def _randomize_dof_noise(
        self,
        env_ids: torch.Tensor,
        base_linear_velocity: torch.Tensor,
        base_angular_velocity: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
    ) -> None:
        num_reset = env_ids.numel()
        randomization = self.cfg.randomization
        if not randomization.enabled:
            return
        rng = self._randomization_rng
        num_action = self._num_action

        if randomization.joint_pos_noise > 0.0:
            joint_position[:, -num_action:] += _torch_uniform(
                rng, -randomization.joint_pos_noise, randomization.joint_pos_noise, (num_reset, num_action)
            )
            joint_position[:, -num_action:] = torch.clamp(
                joint_position[:, -num_action:],
                min=self._joint_position_lower,
                max=self._joint_position_upper,
            )
        if randomization.joint_vel_noise > 0.0:
            joint_velocity[:, -num_action:] += _torch_uniform(
                rng, -randomization.joint_vel_noise, randomization.joint_vel_noise, (num_reset, num_action)
            )

        if any(randomization.base_lin_vel_noise):
            noise = torch.as_tensor(randomization.base_lin_vel_noise, dtype=torch.float32)
            base_linear_velocity += _torch_uniform(rng, -noise, noise, (num_reset, 3))
        if any(randomization.base_ang_vel_noise):
            noise = torch.as_tensor(randomization.base_ang_vel_noise, dtype=torch.float32)
            base_angular_velocity += _torch_uniform(rng, -noise, noise, (num_reset, 3))

    def _randomize_params(self, env_ids: torch.Tensor) -> None:
        num_reset = env_ids.numel()
        randomization = self.cfg.randomization
        if not randomization.enabled:
            return
        rng = self._randomization_rng
        num_action = self._num_action

        if self._randomize_action_delay:
            delay_low, delay_high = randomization.action_delay_steps
            if delay_low == delay_high:
                self._action_delay_steps[env_ids] = delay_low
            else:
                self._action_delay_steps[env_ids] = torch.randint(
                    delay_low, delay_high + 1, (num_reset,), generator=rng
                )

        # All override writes for this reset batch into one program execution.
        program = self._randomize_writes
        if program is None:
            return

        if self._randomize_kp:
            kp_scale = _torch_uniform(rng, *self.cfg.randomization.kp_scale_range, (num_reset, num_action))
            program.buffer("kp")[env_ids] = self.model.others["actuator_kp"][None, :] * kp_scale
        if self._randomize_damping:
            damping_scale = _torch_uniform(rng, *self.cfg.randomization.damping_scale_range, (num_reset, num_action))
            program.buffer("damping")[env_ids] = self.model.others["actuator_kd"][None, :] * damping_scale
        if self._randomize_friction:
            friction = self.model.others["ground_friction"].repeat(num_reset, 1)
            friction[:, 0] = _torch_uniform(rng, *self.cfg.randomization.sliding_friction_range, (num_reset,))
            program.buffer("friction")[env_ids, 0] = friction
        if self._randomize_base_mass:
            base_mass_scale = _torch_uniform(rng, *self.cfg.randomization.base_mass_scale_range, (num_reset,))
            program.buffer("mass")[env_ids, 0] = self.model.others["base_mass"] * base_mass_scale
        if self._randomize_base_com:
            noise = torch.as_tensor(self.cfg.randomization.base_com_offset_noise, dtype=torch.float32)
            base_com_offset = _torch_uniform(rng, -noise, noise, (num_reset, 3))
            program.buffer("com")[env_ids, 0] = self.model.others["base_com"] + base_com_offset
        program.execute(env_ids)

    # ---- reward functions -------------------------------------------------

    def _reward_tracking_lin_vel(self, linvel, gyro, dof_pos) -> torch.Tensor:
        del gyro, dof_pos
        commands = self._commands
        error = torch.sum(torch.square(commands[:, :2] - linvel[:, :2]), dim=1)
        return torch.exp(-error / self.cfg.reward_config.tracking_lin_vel_sigma)

    def _reward_tracking_ang_vel(self, linvel, gyro, dof_pos) -> torch.Tensor:
        del linvel, dof_pos
        commands = self._commands
        error = torch.square(commands[:, 2] - gyro[:, 2])
        return torch.exp(-error / self.cfg.reward_config.tracking_ang_vel_sigma)

    def _reward_lin_vel_z(self, linvel, gyro, dof_pos) -> torch.Tensor:
        del gyro, dof_pos
        return torch.square(linvel[:, 2])

    def _reward_ang_vel_xy(self, linvel, gyro, dof_pos) -> torch.Tensor:
        del linvel, dof_pos
        return torch.sum(torch.square(gyro[:, :2]), dim=1)

    def _reward_base_height(self, linvel, gyro, dof_pos) -> torch.Tensor:
        del linvel, gyro, dof_pos
        base_pos = self.sim_data["base_pos"]
        env_ids = torch.arange(self._num_envs, dtype=torch.int64)
        ground_height = self.sim.sample_terrain_height(self.cfg.ground_geom_name, env_ids, base_pos[:, None, :2])[:, 0]
        base_height = base_pos[:, 2] - ground_height
        return torch.square(base_height - self.cfg.reward_config.base_height_target)

    def _reward_action_rate(self, linvel, gyro, dof_pos) -> torch.Tensor:
        del linvel, gyro, dof_pos
        current = self._current_actions
        last = self._last_actions
        return torch.sum(torch.square(current - last), dim=1)

    def _reward_similar_to_default(self, linvel, gyro, dof_pos) -> torch.Tensor:
        del linvel, gyro
        return torch.sum(torch.abs(dof_pos - self.default_angles), dim=1)

    def _reward_swing_feet_z(self, linvel, gyro, dof_pos) -> torch.Tensor:
        del linvel, gyro, dof_pos
        feet_phase = self._feet_phase
        contacts = self.feet_contact
        valid_swing = (feet_phase >= 0.6) & ~contacts
        reward_cfg = self.cfg.reward_config
        # feet_pos is body-relative (framepos with ref=imu). The foot lifts
        # `target_height` above its stance position, so the body-frame target is
        # target_height - base_height_target (e.g. 0.1 - 0.3 = -0.2). This keeps
        # the reward invariant to the body's world-frame vertical motion.
        target_z = reward_cfg.target_foot_height - reward_cfg.base_height_target
        height_error = torch.square(self.feet_pos[:, :, 2] - target_z)
        sigma_sq = reward_cfg.swing_feet_height_sigma**2
        swing_rew = torch.exp(-height_error / sigma_sq) * valid_swing
        return torch.sum(swing_rew, dim=1) / self._num_feet

    def _reward_contact(self, linvel, gyro, dof_pos) -> torch.Tensor:
        del linvel, gyro, dof_pos
        res = torch.zeros((self._num_envs,), dtype=torch.float32)
        feet_phase = self._feet_phase
        contacts = self.feet_contact
        for i in range(self._num_feet):
            target_contact = feet_phase[:, i] < 0.6
            res += (contacts[:, i] == target_contact).to(torch.float32)
        return res / self._num_feet

    def _reward_swing_contact(self, linvel, gyro, dof_pos) -> torch.Tensor:
        del linvel, gyro, dof_pos
        is_swing = self._feet_phase >= 0.6
        swing_contacts = self.feet_contact & is_swing
        return torch.sum(swing_contacts, dim=1) / self._num_feet


__all__ = ["QuadrupedWalkTask"]
