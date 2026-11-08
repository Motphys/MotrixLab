# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.sim import (
    DofPositionQuery,
    DofVelocityQuery,
    JointPositionWrite,
)
from motrix_env_core.sim.write import CtrlTargetsWrite, JointVelocityWrite

from .cfg import CartPoleEnvCfg

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
}
_SIM_MODEL_QUERIES = {}


@registry.env("cartpole")
class CartPoleEnv(DirectEnv):
    _cfg: CartPoleEnvCfg

    def __init__(self, cfg: CartPoleEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "cartpole_position": JointPositionWrite(("slider", "hinge")),
                "cartpole_velocity": JointVelocityWrite(("slider", "hinge")),
            },
            reset=True,
        )
        self._reset_position = self._reset_program.buffer("cartpole_position")
        self._reset_velocity = self._reset_program.buffer("cartpole_velocity")
        self._action_space = gym.spaces.Box(-3.0, 3.0, (1,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState):
        self._ctrl_writes.buffer("ctrl").copy_(actions)
        self._ctrl_writes.execute()
        return state

    def compute_transition(self, state: ArrayEnvState):
        self.sim_data.execute()
        inputs = self.sim_data
        dof_pos = inputs["dof_pos"]

        cart_pos = dof_pos[:, 0]
        angle = dof_pos[:, 1]
        terminated = angle.isnan() | (angle.abs() > 0.2)
        terminated = terminated | (cart_pos < -0.8)
        terminated = terminated | (cart_pos > 0.8)

        state.reward[:] = 1.0
        state.terminated[:] = terminated
        return state

    def compute_observation(self, state: ArrayEnvState):
        obs = torch.cat([self.sim_data["dof_pos"], self.sim_data["dof_vel"]], dim=-1)
        assert obs.shape == (self._num_envs, 4)
        return state.replace(obs=obs)

    def reset(self, env_ids: torch.Tensor) -> None:
        cfg: CartPoleEnvCfg = self._cfg
        rows = env_ids.numel()
        scale = cfg.reset_noise_scale
        dof_pos = (torch.rand(rows, 2, dtype=torch.float32) * 2.0 - 1.0) * scale
        dof_vel = (torch.rand(rows, 2, dtype=torch.float32) * 2.0 - 1.0) * scale

        self._reset_position[env_ids] = dof_pos
        self._reset_velocity[env_ids] = dof_vel
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)
