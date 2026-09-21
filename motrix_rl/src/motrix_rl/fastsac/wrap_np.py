# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Numpy-backend env wrapper for FastSAC.

Wraps a Motrix ``DirectEnv`` as the torch-tensor vectorized environment consumed
by FastSAC.

The actor maps its tanh output to the action range declared by the environment;
the wrapper only clips to that standard contract before stepping. Done
environments are auto-reset by the environment; the returned observation is therefore
the post-reset observation for those environments.
"""

import numpy as np
import torch

from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.renderer import RenderConfig, create_renderer
from motrix_rl.fastsac.wrap import FastSacEnvWrap
from motrix_rl.utils import env_infos


class FastSacNpEnvWrap(FastSacEnvWrap):
    """Wrap a Motrix ``DirectEnv`` as a torch-tensor vectorized env for FastSAC."""

    def __init__(self, env: DirectEnv, device: torch.device, render: RenderConfig | None = None):
        super().__init__(env, device)
        self._env: DirectEnv = env
        self._renderer = create_renderer(env, render)

    def _to_torch(self, arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.asarray(arr), dtype=dtype, device=self.device)

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._env.init_state()
        self.last_info = env_infos(state)
        return self._to_torch(state.obs.policy), self._to_torch(state.obs.value_or_policy)

    def step(self, actions: torch.Tensor):
        np_actions = actions.detach().cpu().numpy().astype(np.float32)
        np_actions = np.clip(np_actions, self._low, self._high)
        state = self._env.step(np_actions)
        self.last_info = env_infos(state)
        return (
            self._to_torch(state.obs.policy),
            self._to_torch(state.obs.value_or_policy),
            self._to_torch(state.reward),
            self._to_torch(state.terminated, dtype=torch.bool),
            self._to_torch(state.truncated, dtype=torch.bool),
        )
