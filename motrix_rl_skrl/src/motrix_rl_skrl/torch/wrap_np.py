# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import gymnasium
import numpy as np
import torch
from skrl.envs.wrappers.torch import Wrapper as SkrlWrapper

from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.renderer import RenderConfig, VideoRecorder, create_renderer
from motrix_env_core.sim.backend import SimRenderer
from motrix_rl.device import resolve_default_device
from motrix_rl.utils import env_infos


def _as_torch(value, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Zero-copy tensor view for both numpy and torch env state."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


class SkrlNpWrapper(SkrlWrapper):
    """
    Wrap the numpy-based environment to be compatible with skrl (PyTorch)

    Manager environments already expose torch state and pass through
    zero-copy; NumPy-frontend (Direct) environments convert here.
    """

    _env: ArrayEnv
    _renderer: SimRenderer | VideoRecorder | None = None

    def __init__(self, env: ArrayEnv, render: RenderConfig | None = None):
        super().__init__(env)
        # skrl's fallback for device-less envs is cuda-else-cpu; resolve with
        # the shared cuda -> mps -> cpu policy so Apple Silicon trains on MPS.
        # CUDA is normalized to an explicit index: skrl realizes tensors on
        # cuda:<current>, and torch.device("cuda") != torch.device("cuda:0").
        if not hasattr(env, "device"):
            device = resolve_default_device()
            self._device = torch.device("cuda", torch.cuda.current_device()) if device.type == "cuda" else device
        self._renderer = create_renderer(env, render)

    def reset(self) -> tuple[torch.Tensor, Any]:
        state = self._env.init_state()
        return _as_torch(state.obs.policy, torch.float32, self.device), env_infos(state)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Any,
    ]:
        actions = actions.cpu().numpy()
        state = self._env.step(actions)
        return (
            _as_torch(state.obs.policy, torch.float32, self.device),
            _as_torch(state.reward.reshape(-1, 1), torch.float32, self.device),
            _as_torch(state.terminated.reshape(-1, 1), torch.bool, self.device),
            _as_torch(state.truncated.reshape(-1, 1), torch.bool, self.device),
            env_infos(state),
        )

    def state(self) -> torch.Tensor:
        return _as_torch(self._env.state.obs.value_or_policy, torch.float32, self.device)

    def render(self, *args, **kwargs) -> bool | None:
        if self._renderer is None:
            return None
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def observation_space(self) -> gymnasium.Space:
        return self._env.policy_observation_space

    @property
    def state_space(self) -> gymnasium.Space:
        return self._env.value_observation_space

    @property
    def action_space(self) -> gymnasium.Space:
        return self._env.action_space
