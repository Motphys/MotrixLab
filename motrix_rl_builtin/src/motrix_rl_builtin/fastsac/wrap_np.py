# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Torch-frontend env wrapper for FastSAC.

Wraps a Motrix ``ArrayEnv`` (Direct or Manager frontend) as the torch-tensor
vectorized environment consumed by FastSAC. The environment already exposes
torch state and consumes torch actions; state passes through with only a
dtype/device move.

The actor maps its tanh output to the action range declared by the environment;
the wrapper only clips to that standard contract before stepping. Done
environments are auto-reset by the environment; the returned observation is therefore
the post-reset observation for those environments.
"""

import torch

from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.renderer import RenderConfig, create_renderer
from motrix_rl.utils import env_infos
from motrix_rl_builtin.fastsac.wrap import FastSacEnvWrap, FastSacStepResult


class FastSacNpEnvWrap(FastSacEnvWrap):
    """Wrap a Motrix ``ArrayEnv`` as a torch-tensor vectorized env for FastSAC."""

    def __init__(self, env: ArrayEnv, device: torch.device, render: RenderConfig | None = None):
        super().__init__(env, device)
        self._env: ArrayEnv = env
        self._low_t = self.action_low
        self._high_t = self.action_high
        self._renderer = create_renderer(env, render)

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._env.init_state()
        return state.obs.policy.to(self.device), state.obs.value_or_policy.to(self.device)

    def step(self, actions: torch.Tensor) -> FastSacStepResult:
        env_actions = torch.clamp(actions.detach().cpu(), self._low_t, self._high_t)
        state = self._env.step(env_actions)
        return FastSacStepResult(
            obs=state.obs.policy.to(self.device),
            critic_obs=state.obs.value_or_policy.to(self.device),
            reward=state.reward.to(self.device),
            terminated=state.terminated.to(device=self.device, dtype=torch.bool),
            truncated=state.truncated.to(device=self.device, dtype=torch.bool),
            final_obs=state.final_obs,
            info=env_infos(state),
        )
