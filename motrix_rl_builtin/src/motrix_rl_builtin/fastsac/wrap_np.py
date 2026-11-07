# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""NumPy-frontend env wrapper for FastSAC.

Wraps a Motrix ``ArrayEnv`` as the torch-tensor vectorized environment consumed
by FastSAC. Manager environments already expose torch state and consume torch
actions; both pass through zero-copy (only a device move when training
off-CPU). NumPy-frontend (Direct) environments convert at this boundary.

The actor maps its tanh output to the action range declared by the environment;
the wrapper only clips to that standard contract before stepping. Done
environments are auto-reset by the environment; the returned observation is therefore
the post-reset observation for those environments.
"""

import numpy as np
import torch

from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.renderer import RenderConfig, create_renderer
from motrix_rl.utils import env_infos
from motrix_rl_builtin.fastsac.wrap import FastSacEnvWrap, FastSacStepResult


class FastSacNpEnvWrap(FastSacEnvWrap):
    """Wrap a Motrix ``ArrayEnv`` as a torch-tensor vectorized env for FastSAC."""

    def __init__(self, env: ArrayEnv, device: torch.device, render: RenderConfig | None = None):
        super().__init__(env, device)
        self._env: ArrayEnv = env
        self._low_t = torch.as_tensor(self._low, dtype=torch.float32)
        self._high_t = torch.as_tensor(self._high, dtype=torch.float32)
        self._renderer = create_renderer(env, render)

    def _to_torch(self, arr, dtype=torch.float32) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return arr.to(device=self.device, dtype=dtype)
        return torch.as_tensor(np.asarray(arr), dtype=dtype, device=self.device)

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._env.init_state()
        return self._to_torch(state.obs.policy), self._to_torch(state.obs.value_or_policy)

    def step(self, actions: torch.Tensor) -> FastSacStepResult:
        env_actions = torch.clamp(actions.detach().cpu(), self._low_t, self._high_t)
        state = self._env.step(env_actions)
        return FastSacStepResult(
            obs=self._to_torch(state.obs.policy),
            critic_obs=self._to_torch(state.obs.value_or_policy),
            reward=self._to_torch(state.reward),
            terminated=self._to_torch(state.terminated, dtype=torch.bool),
            truncated=self._to_torch(state.truncated, dtype=torch.bool),
            final_obs=state.final_obs,
            info=env_infos(state),
        )
