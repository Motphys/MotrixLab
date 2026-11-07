# Copyright Motphys Technology Co., Ltd. 2025, 2026

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.math.quaternion_array import rotate_inverse
from motrix_env_core.sim import (
    ActuatorCtrlQuery,
    BodyAngularVelocityWrite,
    BodyJointPositionQuery,
    BodyJointPositionWrite,
    BodyJointVelocityQuery,
    BodyLinearVelocityWrite,
    BodyPositionWrite,
    BodyRotationWrite,
    GeomPairCollidingQuery,
    LinkPositionQuery,
    LinkQuaternionQuery,
    SensorValuesQuery,
)
from motrix_env_core.sim.write import BodyJointVelocityWrite, CtrlTargetsWrite
from motrix_envs.locomotion.go1.cfg import Go1WalkDirectStairsEnvCfg

from .common import generate_repeating_array

_GRAVITY = torch.tensor([0, 0, -1], dtype=torch.float32)


def _sim_data_queries(cfg: Go1WalkDirectStairsEnvCfg):
    return {
        "robot_joint_pos": BodyJointPositionQuery(body=cfg.asset.body_name),
        "robot_joint_vel": BodyJointVelocityQuery(body=cfg.asset.body_name),
        "actuator_ctrls": ActuatorCtrlQuery(),
        "root_pos": LinkPositionQuery(link="trunk"),
        "root_quat": LinkQuaternionQuery(link="trunk"),
        "local_linvel": SensorValuesQuery(sensors=("local_linvel",)),
        "gyro": SensorValuesQuery(sensors=("gyro",)),
        "foot_contact_forces": SensorValuesQuery(
            sensors=("FR_foot_contact", "FL_foot_contact", "RR_foot_contact", "RL_foot_contact")
        ),
        "termination_colliding": GeomPairCollidingQuery(pairs=(("trunk", "floor"),)),
        "foot_colliding": GeomPairCollidingQuery(
            pairs=(("FR_foot", "floor"), ("FL_foot", "floor"), ("RR_foot", "floor"), ("RL_foot", "floor"))
        ),
    }


@registry.env("go1-stairs-terrain-walk")
class Go1WalkStairsTask(DirectEnv):
    def __init__(self, cfg: Go1WalkDirectStairsEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model({})
        self.sim_data = self.sim.compile_reads(_sim_data_queries(cfg))
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "base_position": BodyPositionWrite((cfg.asset.body_name,)),
                "base_rotation": BodyRotationWrite((cfg.asset.body_name,)),
                "base_linear_velocity": BodyLinearVelocityWrite((cfg.asset.body_name,)),
                "base_angular_velocity": BodyAngularVelocityWrite((cfg.asset.body_name,)),
                "joints_position": BodyJointPositionWrite(cfg.asset.body_name),
                "joints_velocity": BodyJointVelocityWrite(cfg.asset.body_name),
            },
            reset=True,
        )
        self._reset_position = self._reset_program.buffer("base_position")[:, 0]
        self._reset_rotation = self._reset_program.buffer("base_rotation")[:, 0]
        self._reset_linear_velocity = self._reset_program.buffer("base_linear_velocity")[:, 0]
        self._reset_angular_velocity = self._reset_program.buffer("base_angular_velocity")[:, 0]
        self._reset_joint_position = self._reset_program.buffer("joints_position")
        self._reset_joint_velocity = self._reset_program.buffer("joints_velocity")
        self._init_action_space()
        self._init_obs_space()
        self._num_action = self._action_space.shape[0]
        self._num_observation = self._observation_space.shape[0]
        height_list = [-1, 0.5, 1.5]
        offset_h = [[2, 0, 2, 1, 1], [2, 2, 1, 0, 0], [1, 1, 2, 1, 2], [0, 1, 0, 2, 0], [0, 1, 1, 0, 2]]
        offset = []
        for i in range(5):
            for j in range(5):
                h_index = offset_h[j][i]
                offset.append([(i - 2) * 8.0, (j - 2) * 8.0, height_list[h_index]])
        self.offset_list = torch.tensor(offset, dtype=torch.float32)
        self._init_base_pose = self.model.init_dof_pos[:7].clone()
        self._init_buffer()
        self.period_counter = 0

    def _init_obs_space(self):
        num_dof_vel = self.num_dof_vel  # linvel + gyro + joint_vel
        num_joint_angle = self.num_dof_pos - 7
        num_gravity = 3
        num_actions = self.num_actuators
        num_command = 3
        num_contact_force = 12

        num_obs = num_dof_vel + num_joint_angle + num_gravity + num_actions + num_command + num_contact_force
        assert num_obs == 60

        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

    def _init_action_space(self):
        lows = np.asarray([spec.ctrl_range[0] for spec in self.model.actuators], dtype=np.float32)
        highs = np.asarray([spec.ctrl_range[1] for spec in self.model.actuators], dtype=np.float32)
        self._action_space = gym.spaces.Box(
            lows,
            highs,
            (self.num_actuators,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def get_dof_pos(self):
        return self.sim_data["robot_joint_pos"]

    def get_dof_vel(self):
        return self.sim_data["robot_joint_vel"]

    def _init_buffer(self):
        cfg = self._cfg
        assert isinstance(cfg, Go1WalkDirectStairsEnvCfg)
        # init buffers

        self.reset_buf = torch.ones(self._num_envs, dtype=torch.bool)
        self.kps = torch.full((self._num_action,), cfg.control_config.stiffness, dtype=torch.float32)
        self.kds = torch.full((self._num_action,), cfg.control_config.damping, dtype=torch.float32)
        self.commands_scale = torch.tensor(
            (
                [
                    cfg.normalization.lin_vel,
                    cfg.normalization.lin_vel,
                    cfg.normalization.ang_vel,
                ]
            ),
            dtype=torch.float32,
        )

        self.default_angles = torch.zeros(self._num_action, dtype=torch.float32)
        self.hip_indices = []
        self.calf_indices = []
        actuator_names = [spec.name for spec in self.model.actuators]
        for i in range(self.num_actuators):
            for name in cfg.init_state.default_joint_angles.keys():
                if name in actuator_names[i]:
                    self.default_angles[i] = cfg.init_state.default_joint_angles[name]
            if "hip" in actuator_names[i]:
                self.hip_indices.append(i)
            if "calf" in actuator_names[i]:
                self.calf_indices.append(i)

        self._init_joint_position = self.default_angles.clone()

        # Contact-pair counts follow the cfg-pinned collision queries: the
        # legacy code derived them by substring-matching the scene's single
        # "floor" ground geom against the trunk and the four foot geoms.
        self.num_check = self.sim_data["termination_colliding"].shape[-1]
        self.foot_check_num = self.sim_data["foot_colliding"].shape[-1]

        # Episode-scoped task state: full-batch buffers, reset writes the done
        # rows in place.
        num_envs = self._num_envs
        self._commands = torch.zeros((num_envs, 3), dtype=torch.float32)
        self._contacts = torch.zeros((num_envs, self.foot_check_num), dtype=torch.bool)
        self._contact_force = torch.zeros((num_envs, 12), dtype=torch.float32)
        self._feet_air_time = torch.zeros((num_envs, self.foot_check_num), dtype=torch.float32)
        self._last_dof_vel = torch.zeros((num_envs, self._num_action), dtype=torch.float32)
        self._current_actions = torch.zeros((num_envs, self._num_action), dtype=torch.float32)
        self._last_actions = torch.zeros((num_envs, self._num_action), dtype=torch.float32)

        spacing = 2.0
        cols = int(np.ceil(np.sqrt(self._num_envs)))
        offsets = []
        for i in range(self._num_envs):
            row = i // cols
            col = i % cols
            x = col * spacing
            y = row * spacing
            z = 0.0
            offsets.append([x, y, z])
        self.offsets = torch.tensor(offsets, dtype=torch.float32)

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState):
        actions = torch.as_tensor(actions, dtype=torch.float32)
        # Copy values: the read buffer is a view that the upcoming physics-step
        # read overwrites in place.
        self._last_dof_vel.copy_(self.get_dof_vel())
        self._last_actions.copy_(self._current_actions)
        self._current_actions.copy_(actions)
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl.copy_(self._compute_torques(actions))
        self._ctrl_writes.execute()
        return state

    def _compute_torques(self, actions):
        # Compute torques from actions.
        # pd controller
        actions_scaled = actions * self.cfg.control_config.action_scale
        torques = self.kps * (actions_scaled + self.default_angles - self.get_dof_pos()) - self.kds * self.get_dof_vel()
        return torques

    def get_local_linvel(self) -> torch.Tensor:
        return self.sim_data["local_linvel"]

    def get_gyro(self) -> torch.Tensor:
        return self.sim_data["gyro"]

    def compute_observation(self, state: ArrayEnvState):
        """Build the full observation from cached sim reads and episode state."""
        inputs = self.sim_data
        linear_vel = inputs["local_linvel"]
        gyro = inputs["gyro"]
        base_quat = inputs["root_quat"]
        local_gravity = rotate_inverse(base_quat, _GRAVITY)
        diff = inputs["robot_joint_pos"] - self.default_angles
        noisy_linvel = linear_vel * self.cfg.normalization.lin_vel
        noisy_gyro = gyro * self.cfg.normalization.ang_vel
        noisy_joint_angle = diff * self.cfg.normalization.dof_pos
        noisy_joint_vel = inputs["robot_joint_vel"] * self.cfg.normalization.dof_vel
        command = self._commands * self.commands_scale
        last_actions = self._current_actions
        contact_force = self._contact_force

        obs = torch.cat(
            [
                noisy_linvel,
                noisy_gyro,
                local_gravity,
                noisy_joint_angle,
                noisy_joint_vel,
                last_actions,
                command,
                contact_force,
            ],
            dim=1,
        )
        return state.replace(obs=obs)

    def compute_transition(self, state):
        self.sim_data.execute()
        # Contact bookkeeping is reward state derived from the refreshed cache.
        self._contacts.copy_(self.sim_data["foot_colliding"])
        self.update_feet_air_time()
        self.update_contact_force()
        state = self.update_terminated(state)
        state = self.update_reward(state)
        return state

    def update_terminated(self, state: ArrayEnvState) -> ArrayEnvState:
        termination_check = self.sim_data["termination_colliding"]
        terminated = termination_check.any(dim=1)

        over_speed = torch.sum(torch.square(self.get_local_linvel()[:, :2]), dim=1) > 1e8
        terminated = terminated | over_speed
        return state.replace(
            terminated=terminated,
        )

    def update_feet_air_time(self):
        self._feet_air_time += self.cfg.ctrl_dt
        self._feet_air_time *= ~self._contacts

    def update_contact_force(self):
        base_quat = self.sim_data["root_quat"]
        foot_forces = self.sim_data["foot_contact_forces"]
        force = []
        for k in range(len(self.cfg.sensor.feet)):
            contact_force = foot_forces[:, 3 * k : 3 * k + 3]
            contact_force = rotate_inverse(base_quat, contact_force)
            force.append(contact_force)
        self._contact_force.copy_(torch.cat(force, dim=1))

    def resample_commands(self, num_envs: int):
        low = torch.as_tensor(np.asarray(self.cfg.commands.vel_limit[0]), dtype=torch.float32)
        high = torch.as_tensor(np.asarray(self.cfg.commands.vel_limit[1]), dtype=torch.float32)
        return low + (high - low) * torch.rand((num_envs, 3), dtype=torch.float32)

    def update_reward(self, state: ArrayEnvState) -> ArrayEnvState:
        terminated = state.terminated

        reward_dict = self._get_reward()

        rewards = {k: v * self.cfg.reward_config.scales[k] for k, v in reward_dict.items()}
        rwd = sum(rewards.values())
        rwd = torch.clamp(rwd, 0.0, 10000.0)
        if "termination" in self.cfg.reward_config.scales:
            termination = self._reward_termination(terminated) * self.cfg.reward_config.scales["termination"]
            rwd = rwd + termination

        rwd = torch.where(terminated, torch.zeros((), dtype=torch.float32), rwd)

        return state.replace(reward=rwd)

    def reset(self, env_ids: torch.Tensor) -> None:
        num_reset = env_ids.numel()

        base_pose = self._init_base_pose.repeat(num_reset, 1)

        num_period = 25
        idx = generate_repeating_array(num_period, num_reset, self.period_counter)
        self.period_counter = (self.period_counter + num_reset) % num_period
        base_pose[:, :3] = self.offset_list[torch.as_tensor(idx, dtype=torch.int64)]

        _reset_pose = base_pose
        self._reset_position[env_ids] = _reset_pose[:, :3]
        self._reset_rotation[env_ids] = _reset_pose[:, 3:7]
        self._reset_linear_velocity[env_ids] = 0.0
        self._reset_angular_velocity[env_ids] = 0.0
        self._reset_joint_position[env_ids] = self._init_joint_position
        self._reset_joint_velocity[env_ids] = 0.0
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)

        self._commands[env_ids] = self.resample_commands(num_reset)
        self._current_actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._last_dof_vel[env_ids] = 0.0
        self._feet_air_time[env_ids] = 0.0
        self._contacts[env_ids] = False
        self._contact_force[env_ids] = 0.0

    def _get_reward(self) -> dict[str, torch.Tensor]:
        commands = self._commands
        return {
            "lin_vel_z": self._reward_lin_vel_z(),
            "ang_vel_xy": self._reward_ang_vel_xy(),
            "orientation": self._reward_orientation(),
            "torques": self._reward_torques(),
            "dof_vel": self._reward_dof_vel(),
            "dof_acc": self._reward_dof_acc(),
            "action_rate": self._reward_action_rate(),
            "tracking_lin_vel": self._reward_tracking_lin_vel(commands),
            "tracking_ang_vel": self._reward_tracking_ang_vel(commands),
            "stand_still": self._reward_stand_still(commands),
            "hip_pos": self._reward_hip_pos(commands),
            "calf_pos": self._reward_calf_pos(commands),
            "feet_air_time": self._reward_feet_air_time(commands),
            "feet_stumble": self._reward_feet_stumble(),
        }

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.get_local_linvel()[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.get_gyro()[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        base_quat = self.sim_data["root_quat"]
        gravity = rotate_inverse(base_quat, _GRAVITY)
        return torch.sum(torch.square(gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.sim_data["actuator_ctrls"]), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.get_dof_vel()), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(
            torch.square((self._last_dof_vel - self.get_dof_vel()) / self.cfg.ctrl_dt),
            dim=1,
        )

    def _reward_action_rate(self):
        # Penalize changes in actions
        action_diff = self._current_actions - self._last_actions
        return torch.sum(torch.square(action_diff), dim=1)

    def _reward_termination(self, done):
        # Terminal reward / penalty
        return done

    def _reward_feet_air_time(self, commands: torch.Tensor):
        # Reward long steps
        feet_air_time = self._feet_air_time
        first_contact = (feet_air_time > 0.0) * self._contacts
        # reward only on first contact with the ground
        rew_airTime = torch.sum((feet_air_time - 0.5) * first_contact, dim=1)
        # no reward for zero command
        rew_airTime *= torch.linalg.vector_norm(commands[:, :2], dim=1) > 0.1
        return rew_airTime

    def _reward_tracking_lin_vel(self, commands: torch.Tensor):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(commands[:, :2] - self.get_local_linvel()[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, commands: torch.Tensor):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(commands[:, 2] - self.get_gyro()[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_stand_still(self, commands: torch.Tensor):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.get_dof_pos() - self.default_angles), dim=1) * (
            torch.linalg.vector_norm(commands, dim=1) < 0.1
        )

    def _reward_hip_pos(self, commands: torch.Tensor):
        return (0.8 - torch.abs(commands[:, 1])) * torch.sum(
            torch.square(self.get_dof_pos()[:, self.hip_indices] - self.default_angles[self.hip_indices]),
            dim=1,
        )

    def _reward_calf_pos(self, commands: torch.Tensor):
        return (0.8 - torch.abs(commands[:, 1])) * torch.sum(
            torch.square(self.get_dof_pos()[:, self.calf_indices] - self.default_angles[self.calf_indices]),
            dim=1,
        )

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        foot_forces = self.sim_data["foot_contact_forces"]
        is_stumble = torch.zeros((self._num_envs,), dtype=torch.float32)
        for k in range(len(self.cfg.sensor.feet)):
            contact_force = foot_forces[:, 3 * k : 3 * k + 3]
            is_stumble += (
                torch.linalg.vector_norm(contact_force, dim=1) > 5 * torch.abs(contact_force[:, 2])
            ).to(torch.float32)
        return is_stumble
