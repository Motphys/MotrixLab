# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core import registry
from motrix_env_core.array.env import ArrayEnvState
from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.sim import (
    ActuatorCtrlQuery,
    DofPositionQuery,
    DofVelocityQuery,
    JointPositionWrite,
)
from motrix_env_core.sim.write import CtrlTargetsWrite, JointVelocityWrite
from motrix_env_core.torch import reward as reward_utils

from .cfg import PendulumEnvCfg

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
    "actuator_ctrls": ActuatorCtrlQuery(),
}
_SIM_MODEL_QUERIES = {}


@registry.env("pendulum")
class PendulumEnv(DirectEnv):
    _cfg: PendulumEnvCfg

    def __init__(self, cfg: PendulumEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {"pendulum_position": JointPositionWrite(("hinge",)), "pendulum_velocity": JointVelocityWrite(("hinge",))},
            reset=True,
        )
        self._reset_position = self._reset_program.buffer("pendulum_position")
        self._reset_velocity = self._reset_program.buffer("pendulum_velocity")
        ctrl_limits = np.asarray([spec.ctrl_range for spec in self.model.actuators], dtype=np.float32).T
        self._action_low = float(ctrl_limits[0, 0])
        self._action_high = float(ctrl_limits[1, 0])
        self._action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)
        # Episode-scoped control history for the control-delta penalty.
        self._prev_ctrl = torch.zeros(self._num_envs, dtype=torch.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState):
        actions = torch.clamp(actions, -1.0, 1.0)
        scaled = self._action_low + (actions + 1.0) * 0.5 * (self._action_high - self._action_low)
        self._ctrl_writes.buffer("ctrl").copy_(scaled)
        self._ctrl_writes.execute()
        return state

    def compute_observation(self, state: ArrayEnvState):
        angle = self.sim_data["dof_pos"][:, 0]
        ang_vel = self.sim_data["dof_vel"][:, 0]
        obs = torch.stack([torch.cos(angle), torch.sin(angle), ang_vel], dim=-1)
        assert obs.shape == (self._num_envs, 3)
        return state.replace(obs=obs)

    def compute_transition(self, state: ArrayEnvState):
        self.sim_data.execute()
        inputs = self.sim_data
        dof_pos = inputs["dof_pos"]
        dof_vel = inputs["dof_vel"]
        angle = dof_pos[:, 0]
        ang_vel = dof_vel[:, 0]

        # compute reward
        angle_wrapped = (angle + np.pi) % (2 * np.pi) - np.pi
        ctrl = inputs["actuator_ctrls"][:, 0]
        # In this model, zero angle corresponds to the hanging-down position.
        # Shift the target by pi to encourage the upright (inverted) posture.
        upright = (1.0 + torch.cos(angle_wrapped)) * 0.5
        prev_ctrl = self._prev_ctrl
        ctrl_delta = ctrl - prev_ctrl
        vel_penalty = 0.2 * (ang_vel**2)
        energy = 0.5 * ang_vel**2 + (1.0 - torch.cos(angle_wrapped))
        energy_target = 2.0
        energy_reward = torch.as_tensor(
            reward_utils.tolerance(
                energy,
                bounds=(energy_target, energy_target),
                margin=2.0,
                value_at_margin=0.1,
                sigmoid="gaussian",
            ),
            dtype=torch.float32,
        )
        reward = 3.0 * upright + energy_reward - vel_penalty - 0.001 * ctrl**2 - 0.001 * ctrl_delta**2

        # compute terminated
        terminated = dof_pos.isnan().any(dim=-1) | dof_vel.isnan().any(dim=-1)

        state.reward[:] = reward
        state.terminated[:] = terminated
        self._prev_ctrl[:] = ctrl
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        cfg: PendulumEnvCfg = self._cfg
        reset_noise_scale = getattr(cfg, "reset_noise_scale", 0.0)
        num_reset = env_ids.numel()
        dof_pos = (torch.rand(num_reset, 1, dtype=torch.float32) * 2.0 - 1.0) * np.pi
        dof_vel = torch.zeros(num_reset, 1, dtype=torch.float32)
        if reset_noise_scale > 0.0:
            dof_vel = (torch.rand(num_reset, 1, dtype=torch.float32) * 2.0 - 1.0) * reset_noise_scale

        self._reset_position[env_ids] = dof_pos
        self._reset_velocity[env_ids] = dof_vel
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)
        self._prev_ctrl[env_ids] = 0.0
