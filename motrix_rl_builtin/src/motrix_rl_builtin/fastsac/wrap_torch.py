# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Torch-backend env wrapper for FastSAC.

Wraps a Motrix ``TorchEnv`` as the torch-tensor vectorized environment consumed
by FastSAC.

The actor maps its tanh output to the action range declared by the environment;
the wrapper only clips to that standard contract before stepping. Done
environments are auto-reset by ``TorchEnv``; the returned observation is therefore
the post-reset observation for those environments.
"""

import torch

from motrix_env_core.renderer import RenderConfig, create_renderer
from motrix_env_motrixsim.torch_env import TorchEnv
from motrix_rl.utils import env_infos
from motrix_rl_builtin.fastsac.wrap import FastSacEnvWrap, FastSacStepResult


class FastSacTorchEnvWrap(FastSacEnvWrap):
    """Wrap a Motrix ``TorchEnv`` as a torch-tensor vectorized env for FastSAC."""

    def __init__(self, env: TorchEnv, device: torch.device, render: RenderConfig | None = None):
        super().__init__(env, device)
        self._env: TorchEnv = env
        self._low = torch.as_tensor(self._low, dtype=torch.float32, device=env.device)
        self._high = torch.as_tensor(self._high, dtype=torch.float32, device=env.device)
        self._renderer = create_renderer(env, render)

    def _to_torch(self, tensor: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=dtype)

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._env.init_state()
        return self._to_torch(state.obs.policy), self._to_torch(state.obs.value_or_policy)

    def step(self, actions: torch.Tensor) -> FastSacStepResult:
        env_actions = actions.detach().to(device=self._env.device, dtype=torch.float32)
        env_actions = torch.clamp(env_actions, self._low, self._high)
        state = self._env.step(env_actions)
        return FastSacStepResult(
            obs=self._to_torch(state.obs.policy),
            critic_obs=self._to_torch(state.obs.value_or_policy),
            reward=self._to_torch(state.reward),
            terminated=self._to_torch(state.terminated, dtype=torch.bool),
            truncated=self._to_torch(state.truncated, dtype=torch.bool),
            final_obs=None,
            info=env_infos(state),
        )
