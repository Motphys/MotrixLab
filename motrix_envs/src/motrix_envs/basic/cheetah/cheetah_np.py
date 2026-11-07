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
    BodyJointPositionWrite,
    DofPositionQuery,
    DofVelocityQuery,
    LinkPositionQuery,
    SensorValuesQuery,
)
from motrix_env_core.sim.write import BodyJointVelocityWrite, CtrlTargetsWrite
from motrix_envs.basic.cheetah.cfg import CheetahEnvCfg

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
    "torso_pos": LinkPositionQuery(link="torso"),
    "torso_subtreelinvel": SensorValuesQuery(sensors=("torso_subtreelinvel",)),
}
_SIM_MODEL_QUERIES = {}


@registry.env("dm-cheetah")
class CheetahEnv(DirectEnv):
    _observation_space: gym.spaces.Box
    _action_space: gym.spaces.Box

    def __init__(self, cfg: CheetahEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model({})
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {"cheetah_position": BodyJointPositionWrite("torso"), "cheetah_velocity": BodyJointVelocityWrite("torso")},
            reset=True,
        )
        self._reset_position = self._reset_program.buffer("cheetah_position")
        self._reset_velocity = self._reset_program.buffer("cheetah_velocity")
        self._init_obs_space()
        self._init_action_space()
        self._run_speed = cfg.run_speed

    def _init_obs_space(self):
        obs_dim = (self.num_dof_pos - 1) + self.num_dof_vel
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)

    def _init_action_space(self):
        self._action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actuators,),
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        self._ctrl_writes.buffer("ctrl").copy_(actions)
        self._ctrl_writes.execute()
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        pos = inputs["dof_pos"][:, 1:]  # exclude x position
        vel = inputs["dof_vel"]
        obs = torch.cat([pos, vel], dim=-1)
        return state.replace(obs=obs)

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        inputs = self.sim_data
        # === Terminated ===
        terminated = inputs["dof_pos"].isnan().any(dim=-1)
        terminated |= inputs["dof_vel"].isnan().any(dim=-1)

        # ==== compute reward ====
        vel = inputs["torso_subtreelinvel"]
        rwd_speed = torch.as_tensor(
            reward.tolerance(
                vel[:, 0],
                bounds=(self._run_speed, float("inf")),
                margin=self._run_speed,
                value_at_margin=0.0,
                sigmoid="linear",
            ),
            dtype=torch.float32,
        )

        torso_height = inputs["torso_pos"][:, 2]
        rwd_posture = -1.5 * (torso_height - 0.75) ** 2
        rwd_posture = torch.clamp(rwd_posture, -1.0, 1.0)

        rwd = rwd_speed + rwd_posture

        state.reward[:] = rwd
        state.terminated[:] = terminated
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        num = env_ids.numel()

        qpos = torch.zeros((num, self._reset_position.shape[1]), dtype=torch.float32)
        dof_vel = torch.zeros((num, self._reset_velocity.shape[1]), dtype=torch.float32)
        self._reset_position[env_ids] = qpos
        self._reset_velocity[env_ids] = dof_vel
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)
