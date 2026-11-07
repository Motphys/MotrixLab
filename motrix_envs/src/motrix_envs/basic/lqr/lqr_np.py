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


def _normalize_actions(actions: np.ndarray, num_envs: int, num_actuators: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        if num_envs != 1 or actions.shape[0] != num_actuators:
            raise ValueError(f"Expected action shape ({num_envs}, {num_actuators}) or ({num_actuators},).")
        actions = actions.reshape(1, num_actuators)
    if actions.shape != (num_envs, num_actuators):
        raise ValueError(f"Expected action shape ({num_envs}, {num_actuators}), got {actions.shape}.")
    return np.ascontiguousarray(actions)


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
        self._action_low = ctrl_ranges[:, 0]
        self._action_high = ctrl_ranges[:, 1]
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self._action_space = gym.spaces.Box(self._action_low, self._action_high, (self._nu,), dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        actions = actions.detach().cpu().numpy()
        actions = _normalize_actions(actions, self._num_envs, self._nu)
        ctrl = self._ctrl_writes.buffer("ctrl")
        ctrl[:] = np.clip(actions, self._action_low, self._action_high)
        self._ctrl_writes.execute()
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        inputs = self.sim_data
        qpos = np.asarray(inputs["dof_pos"], dtype=np.float32)
        qvel = np.asarray(inputs["dof_vel"], dtype=np.float32)
        return state.replace(obs=np.concatenate([qpos, qvel], axis=-1))

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        inputs = self.sim_data
        qpos = np.asarray(inputs["dof_pos"], dtype=np.float32)
        qvel = np.asarray(inputs["dof_vel"], dtype=np.float32)
        ctrl = np.asarray(inputs["actuator_ctrls"], dtype=np.float32)

        out_of_bounds = np.any(np.abs(qpos) > self._cfg.boundary_position_limit, axis=-1)
        out_of_bounds |= np.any(np.abs(qvel) > self._cfg.boundary_velocity_limit, axis=-1)
        position_norm = np.linalg.norm(qpos, axis=-1)
        velocity_norm = np.linalg.norm(qvel, axis=-1)
        state_cost = 0.5 * np.sum(np.square(qpos), axis=-1)
        velocity_cost = 0.5 * self._cfg.velocity_cost_coef * np.sum(np.square(qvel), axis=-1)
        control_cost = 0.5 * self._cfg.control_cost_coef * np.sum(np.square(ctrl), axis=-1)
        success = (position_norm <= self._cfg.success_position_tol) & (velocity_norm <= self._cfg.success_velocity_tol)
        success &= ~out_of_bounds
        success_reward = self._cfg.success_bonus * success.astype(np.float32)
        boundary_penalty = self._cfg.out_of_bounds_penalty * out_of_bounds.astype(np.float32)
        reward = 1.0 - (state_cost + velocity_cost + control_cost) + success_reward - boundary_penalty

        terminated = success | out_of_bounds
        terminated |= np.isnan(qpos).any(axis=-1) | np.isnan(qvel).any(axis=-1)
        terminated |= np.isnan(ctrl).any(axis=-1)

        state.metrics = {
            "position_norm": position_norm.astype(np.float32),
            "velocity_norm": velocity_norm.astype(np.float32),
            "success": success.astype(np.float32),
            "out_of_bounds": out_of_bounds.astype(np.float32),
        }
        state.reward_terms = {
            "state_cost": (-state_cost).astype(np.float32),
            "velocity_cost": (-velocity_cost).astype(np.float32),
            "control_cost": (-control_cost).astype(np.float32),
            "success_bonus": success_reward.astype(np.float32),
            "out_of_bounds_penalty": (-boundary_penalty).astype(np.float32),
        }

        return state.replace(
            reward=reward.astype(np.float32),
            terminated=terminated,
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.detach().cpu().numpy()
        num_envs = len(env_ids)

        qpos = np.random.standard_normal((num_envs, self._nq)).astype(np.float32)
        norms = np.linalg.norm(qpos, axis=-1, keepdims=True)
        zero_norm = norms[:, 0] < 1e-8
        if np.any(zero_norm):
            qpos[zero_norm, 0] = 1.0
            norms = np.linalg.norm(qpos, axis=-1, keepdims=True)
        qpos *= self._cfg.reset_position_norm / np.clip(norms, 1e-8, None)

        qvel = np.zeros((num_envs, self._nv), dtype=np.float32)
        self._reset_position[env_ids] = qpos
        self._reset_velocity[env_ids] = qvel
        self._reset_program.execute(env_ids)
        self.sim_data.execute(np.asarray(env_ids, np.int64))
