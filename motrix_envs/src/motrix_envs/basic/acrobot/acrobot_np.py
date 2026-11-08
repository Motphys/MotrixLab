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
    SitePositionQuery,
)
from motrix_env_core.sim.write import CtrlTargetsWrite, JointVelocityWrite
from motrix_env_core.torch import reward

from .cfg import AcrobotEnvCfg

_SIM_DATA_QUERIES = {
    "dof_pos": DofPositionQuery(),
    "dof_vel": DofVelocityQuery(),
    "tip_pos": SitePositionQuery(site="tip"),
    "target_pos": SitePositionQuery(site="target"),
}
_SIM_MODEL_QUERIES = {}


@registry.env("acrobot")
class AcrobotEnv(DirectEnv):
    _cfg: AcrobotEnvCfg

    def __init__(self, cfg: AcrobotEnvCfg, num_envs=1, backend: str | None = None):
        super().__init__(cfg, num_envs, backend=backend)
        self.model = self.sim.compile_model(_SIM_MODEL_QUERIES)
        self.sim_data = self.sim.compile_reads(_SIM_DATA_QUERIES)
        self._ctrl_writes = self.sim.write_compiler.compile({"ctrl": CtrlTargetsWrite()})
        self._reset_program = self.sim.write_compiler.compile(
            {
                "arm_position": JointPositionWrite(("shoulder", "elbow")),
                "arm_velocity": JointVelocityWrite(("shoulder", "elbow")),
            },
            reset=True,
        )
        self._reset_position = self._reset_program.buffer("arm_position")
        self._reset_velocity = self._reset_program.buffer("arm_velocity")
        self._action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
        self._target_radius = 0.2

        self._step_count = torch.zeros(self._num_envs, dtype=torch.int64)
        self._max_steps = int(cfg.max_episode_seconds / cfg.ctrl_dt)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState):
        actions = torch.clamp(actions, -1.0, 1.0)
        self._ctrl_writes.buffer("ctrl").copy_(actions)
        self._ctrl_writes.execute()
        return state

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        dof_pos = self.sim_data["dof_pos"]
        dof_vel = self.sim_data["dof_vel"]
        shoulder_angle = dof_pos[:, 0]
        elbow_angle = dof_pos[:, 1]

        upper_arm_horizontal = torch.cos(shoulder_angle)
        upper_arm_vertical = torch.sin(shoulder_angle)

        total_angle = shoulder_angle + elbow_angle
        lower_arm_horizontal = torch.cos(total_angle)
        lower_arm_vertical = torch.sin(total_angle)

        obs = torch.cat(
            [
                upper_arm_horizontal.reshape(-1, 1),
                lower_arm_horizontal.reshape(-1, 1),
                upper_arm_vertical.reshape(-1, 1),
                lower_arm_vertical.reshape(-1, 1),
                dof_vel,
            ],
            dim=-1,
        )
        return state.replace(obs=obs)

    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        self.sim_data.execute()
        inputs = self.sim_data
        dof_pos = inputs["dof_pos"]
        dof_vel = inputs["dof_vel"]
        tip_pos = inputs["tip_pos"]
        target_pos = inputs["target_pos"]
        dist_to_target = torch.linalg.norm(tip_pos - target_pos, dim=-1)

        base_rwd = torch.as_tensor(
            reward.tolerance(
                dist_to_target,
                bounds=(0, self._target_radius),
                margin=0,
                value_at_margin=0.0,
                sigmoid="linear",
            ),
            dtype=torch.float32,
        )

        in_target = dist_to_target < self._target_radius
        continuous_reward = 0.1 * in_target.float()

        distance_reward = 0.3 * (1.0 - torch.clamp(dist_to_target / 2.0, 0.0, 1.0))

        vel_magnitude = dof_vel.abs().mean(dim=-1)
        velocity_penalty = 0.01 * torch.clamp(vel_magnitude - 2.0, min=0.0)

        rwd = base_rwd + continuous_reward + distance_reward - velocity_penalty

        rwd = rwd * self._cfg.reward_scale

        self._step_count += 1

        terminated = self._step_count >= self._max_steps
        terminated |= dof_pos.isnan().any(dim=-1)
        terminated |= dof_vel.isnan().any(dim=-1)
        terminated |= rwd.isnan()

        state.reward[:] = rwd
        state.terminated[:] = terminated
        return state

    def reset(self, env_ids: torch.Tensor) -> None:
        num_reset = env_ids.numel()

        shoulder_angle = (torch.rand(num_reset, dtype=torch.float32) * 2.0 - 1.0) * np.pi
        elbow_angle = (torch.rand(num_reset, dtype=torch.float32) * 2.0 - 1.0) * np.pi

        dof_pos = torch.stack([shoulder_angle, elbow_angle], dim=-1)
        dof_vel = torch.zeros((num_reset, 2), dtype=torch.float32)

        self._reset_position[env_ids] = dof_pos
        self._reset_velocity[env_ids] = dof_vel
        self._reset_program.execute(env_ids)
        self.sim_data.execute(env_ids)

    def _reset_done_envs(self):
        """
        Reset the environments that are done
        """
        super()._reset_done_envs()
        done = self._state.done
        if bool(done.any()):
            self._step_count[done] = 0
