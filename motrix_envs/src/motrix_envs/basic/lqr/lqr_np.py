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

from .cfg import LqrBaseCfg

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
    "actuator_ctrls": ActuatorCtrlQuery(),
}
_SIM_MODEL_QUERIES = {}


def _normalize_actions(actions: torch.Tensor, num_envs: int, num_actuators: int) -> torch.Tensor:
    if actions.ndim == 1:
        if num_envs != 1 or actions.shape[0] != num_actuators:
            raise ValueError(f"Expected action shape ({num_envs}, {num_actuators}) or ({num_actuators},).")
        actions = actions.reshape(1, num_actuators)
    if actions.shape != (num_envs, num_actuators):
        raise ValueError(f"Expected action shape ({num_envs}, {num_actuators}), got {tuple(actions.shape)}.")
    return actions.contiguous()


@registry.env("dm-lqr-2-1")
@registry.env("dm-lqr-6-2")
class LqrEnv(DirectEnv):
    _cfg: LqrBaseCfg

    def __init__(self, cfg: LqrBaseCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        joints = tuple(f"q{i}" for i in range(cfg.expected_nq))
        self._reset_program = self.sim.write_compiler.compile(
            {"state_position": JointPositionWrite(joints), "state_velocity": JointVelocityWrite(joints)}, reset=True
        )
        self._reset_position = self._reset_program.buffer("state_position")
        self._reset_velocity = self._reset_program.buffer("state_velocity")

        self._nq = int(self.num_dof_pos)
        self._nv = int(self.num_dof_vel)
        self._nu = int(self.num_actuators)

        if self._nq != cfg.expected_nq or self._nv != cfg.expected_nq:
            raise ValueError(f"LQR model mismatch: expected nq=nv={cfg.expected_nq}, got nq={self._nq}, nv={self._nv}.")
        if self._nu != cfg.expected_nu:
            raise ValueError(f"LQR model mismatch: expected nu={cfg.expected_nu}, got nu={self._nu}.")

        obs_dim = self._nq + self._nv
        ctrl_ranges = np.asarray([spec.ctrl_range for spec in self.model.actuators], dtype=np.float32)
        self._action_low = torch.from_numpy(ctrl_ranges[:, 0].copy())
        self._action_high = torch.from_numpy(ctrl_ranges[:, 1].copy())
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self._action_space = gym.spaces.Box(ctrl_ranges[:, 0], ctrl_ranges[:, 1], (self._nu,), dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        actions = _normalize_actions(actions, self._num_envs, self._nu)
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl.copy_(torch.clamp(actions, min=self._action_low, max=self._action_high))
        self._ctrl_writes.execute()
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        qpos = inputs["dof_pos"]
        qvel = inputs["dof_vel"]
        return state.replace(obs=torch.cat([qpos, qvel], dim=-1))

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        inputs = self.sim_data
        qpos = inputs["dof_pos"]
        qvel = inputs["dof_vel"]
        ctrl = inputs["actuator_ctrls"]

        out_of_bounds = (qpos.abs() > self._cfg.boundary_position_limit).any(dim=-1)
        out_of_bounds |= (qvel.abs() > self._cfg.boundary_velocity_limit).any(dim=-1)
        position_norm = torch.linalg.norm(qpos, dim=-1)
        velocity_norm = torch.linalg.norm(qvel, dim=-1)
        state_cost = 0.5 * torch.sum(torch.square(qpos), dim=-1)
        velocity_cost = 0.5 * self._cfg.velocity_cost_coef * torch.sum(torch.square(qvel), dim=-1)
        control_cost = 0.5 * self._cfg.control_cost_coef * torch.sum(torch.square(ctrl), dim=-1)
        success = (position_norm <= self._cfg.success_position_tol) & (velocity_norm <= self._cfg.success_velocity_tol)
        success &= ~out_of_bounds
        success_reward = self._cfg.success_bonus * success.float()
        boundary_penalty = self._cfg.out_of_bounds_penalty * out_of_bounds.float()
        reward = 1.0 - (state_cost + velocity_cost + control_cost) + success_reward - boundary_penalty

        terminated = success | out_of_bounds
        terminated |= qpos.isnan().any(dim=-1) | qvel.isnan().any(dim=-1)
        terminated |= ctrl.isnan().any(dim=-1)

        state.metrics = {
            "position_norm": position_norm,
            "velocity_norm": velocity_norm,
            "success": success.float(),
            "out_of_bounds": out_of_bounds.float(),
        }
        state.reward_terms = {
            "state_cost": -state_cost,
            "velocity_cost": -velocity_cost,
            "control_cost": -control_cost,
            "success_bonus": success_reward,
            "out_of_bounds_penalty": -boundary_penalty,
        }

        state.reward[:] = reward
        state.terminated[:] = terminated
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        num_envs = env_ids.numel()

        qpos = torch.randn(num_envs, self._nq, dtype=torch.float32)
        norms = torch.linalg.norm(qpos, dim=-1, keepdim=True)
        zero_norm = norms[:, 0] < 1e-8
        if zero_norm.any():
            qpos[zero_norm, 0] = 1.0
            norms = torch.linalg.norm(qpos, dim=-1, keepdim=True)
        qpos *= self._cfg.reset_position_norm / norms.clamp_min(1e-8)

        qvel = torch.zeros((num_envs, self._nv), dtype=torch.float32)
        self._reset_position[env_ids] = qpos
        self._reset_velocity[env_ids] = qvel
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)
