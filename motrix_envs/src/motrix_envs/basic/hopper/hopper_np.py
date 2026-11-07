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
    BodyJointPositionLimitsQuery,
    BodyJointPositionWrite,
    DofPositionQuery,
    DofVelocityQuery,
    LinkPositionQuery,
    SensorValuesQuery,
)
from motrix_env_core.sim.write import BodyJointVelocityWrite, CtrlTargetsWrite
from motrix_envs.basic.hopper.cfg import HopperStandCfg

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
    "actuator_ctrls": ActuatorCtrlQuery(),
    "torso_pos": LinkPositionQuery(link="torso"),
    "foot_pos": LinkPositionQuery(link="foot"),
    "torso_subtreelinvel": SensorValuesQuery(sensors=("torso_subtreelinvel",)),
    "touch_toe": SensorValuesQuery(sensors=("touch_toe",)),
    "touch_heel": SensorValuesQuery(sensors=("touch_heel",)),
}
_SIM_MODEL_QUERIES = {"joint_position_limits": BodyJointPositionLimitsQuery(body="torso")}


@registry.env("dm-hopper-stand")
@registry.env("dm-hopper-hop")
class HopperEnv(DirectEnv):
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: HopperStandCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {"hopper_position": BodyJointPositionWrite("torso"), "hopper_velocity": BodyJointVelocityWrite("torso")},
            reset=True,
        )
        self._reset_position = self._reset_program.buffer("hopper_position")
        self._reset_velocity = self._reset_program.buffer("hopper_velocity")
        self._init_obs_space()
        self._init_action_space()

        self._stand_height = cfg.stand_height
        self._hop_speed = cfg.hop_speed
        self._joint_pos_lower, self._joint_pos_upper = self.model.others["joint_position_limits"]

    def _init_obs_space(self):
        num = 0
        num += self.num_dof_pos - 1
        num += self.num_dof_vel
        num += 2
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num,), dtype=np.float32)

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

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState):
        self._ctrl_writes.buffer("ctrl").copy_(actions)
        self._ctrl_writes.execute()
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        qpos = inputs["dof_pos"][:, 1:]
        qvel = inputs["dof_vel"]
        num_env = qpos.shape[0]

        toe = torch.log1p(inputs["touch_toe"].reshape(num_env, -1)[:, 0])
        heel = torch.log1p(inputs["touch_heel"].reshape(num_env, -1)[:, 0])

        touch = torch.stack([toe, heel], dim=-1)  # shape -> (num_env, 2)
        obs = torch.cat([qpos, qvel, touch], dim=-1)
        return state.replace(obs=obs)

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        inputs = self.sim_data

        toe = torch.log1p(inputs["touch_toe"][:, 0])
        heel = torch.log1p(inputs["touch_heel"][:, 0])

        # === physical values ===
        torso_pos = inputs["torso_pos"]
        foot_pos = inputs["foot_pos"]
        torso_height = torso_pos[:, 2] - foot_pos[:, 2]

        torso_vel = inputs["torso_subtreelinvel"]
        speed = torso_vel[:, 0]

        dof_pos = inputs["dof_pos"]
        dof_vel = inputs["dof_vel"]

        # === terminated ===
        over_speed = torch.sum(torch.square(dof_vel[:, 4:7]), dim=-1) > 1e8
        terminated = (
            dof_pos.isnan().any(dim=-1)
            | dof_vel.isnan().any(dim=-1)
            | inputs["touch_toe"][:, 0].isnan()
            | inputs["touch_heel"][:, 0].isnan()
        )
        terminated |= over_speed

        standing = torch.as_tensor(
            reward.tolerance(
                torso_height,
                bounds=(self._stand_height, 2.0),
                margin=self._stand_height * 0.5,
            ),
            dtype=torch.float32,
        )

        if self._hop_speed > 0.0:
            hopping = torch.as_tensor(
                reward.tolerance(
                    speed,
                    bounds=(self._hop_speed * 0.3, float("inf")),
                    margin=self._hop_speed * 0.3,
                    value_at_margin=0.0,
                    sigmoid="linear",
                ),
                dtype=torch.float32,
            )

            leg_vel = torch.linalg.norm(dof_vel[:, 4:7], dim=-1)
            leg_bonus = torch.tanh(leg_vel * 0.3) * 0.2 * standing

            knee_vel = dof_vel[:, 5]
            extend_reward = torch.clamp(knee_vel, min=0.0) * 0.2 * standing

            stand_condition = (torso_height > self._stand_height * 0.8).float()
            effective_hop_reward = hopping * stand_condition

            contact_strength = toe + heel
            contact_reward = torch.clamp(contact_strength, 0.0, 1.0) * 0.1 * standing

            rwd = standing * 0.8 + effective_hop_reward * 0.8 + leg_bonus * 0.5 + extend_reward + contact_reward
            state.reward_terms = {"stand": standing, "hop": effective_hop_reward, "total": rwd}
            if rwd.mean() > 1000:
                print(
                    "standing",
                    torch.sum(standing),
                    "effective_hop_reward",
                    torch.sum(effective_hop_reward),
                    "leg_bonus",
                    torch.sum(leg_bonus),
                    "extend_reward",
                    torch.sum(extend_reward),
                    "contact_reward",
                    torch.sum(contact_reward),
                )

        else:
            control_magnitude = torch.linalg.norm(inputs["actuator_ctrls"], dim=-1)
            small_control = torch.as_tensor(
                reward.tolerance(
                    control_magnitude,
                    bounds=(0, 1),
                    margin=1,
                    value_at_margin=0,
                    sigmoid="quadratic",
                ),
                dtype=torch.float32,
            )
            small_control = (small_control + 4) / 5

            rwd = standing * small_control
            state.reward_terms = {"stand": standing, "control": small_control, "total": rwd}

        rwd[terminated] = 0.0

        state.reward[:] = rwd
        state.terminated[:] = terminated
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        num_reset = env_ids.numel()

        dof_pos = torch.zeros((num_reset, self._reset_position.shape[1]), dtype=torch.float32)

        dof_pos[:, 2] = 0

        if self._reset_position.shape[1] > 3:
            lower = self._joint_pos_lower[3:]
            upper = self._joint_pos_upper[3:]
            dof_pos[:, 3:] = torch.rand((num_reset, self._reset_position.shape[1] - 3), dtype=torch.float32) * (
                upper - lower
            ) + lower

        dof_vel = torch.zeros((num_reset, self._reset_velocity.shape[1]), dtype=torch.float32)

        self._reset_position[env_ids] = dof_pos
        self._reset_velocity[env_ids] = dof_vel
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)
