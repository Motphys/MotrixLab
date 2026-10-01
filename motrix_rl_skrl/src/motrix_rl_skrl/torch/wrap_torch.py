# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import gymnasium
import torch
from skrl.envs.wrappers.torch import Wrapper as SkrlWrapper

from motrix_env_core.renderer import RenderConfig, VideoRecorder, create_renderer
from motrix_env_core.sim.backend import SimRenderer
from motrix_env_motrixsim.torch_env import TorchEnv
from motrix_rl.utils import env_infos


class SkrlTorchWrapper(SkrlWrapper):
    """
    Wrap the torch-based environment to be compatible with skrl (PyTorch)
    """

    _env: TorchEnv
    _renderer: SimRenderer | VideoRecorder | None = None

    def __init__(self, env: TorchEnv, render: RenderConfig | None = None):
        super().__init__(env)
        self._renderer = create_renderer(env, render)

    def _to_train_device(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=dtype)

    def reset(self) -> tuple[torch.Tensor, Any]:
        state = self._env.init_state()
        return self._to_train_device(state.obs.policy, torch.float32), env_infos(state)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Any,
    ]:
        state = self._env.step(actions.detach().to(device=self._env.device, dtype=torch.float32))
        return (
            self._to_train_device(state.obs.policy, torch.float32),
            self._to_train_device(state.reward.reshape(-1, 1), torch.float32),
            self._to_train_device(state.terminated.reshape(-1, 1), torch.bool),
            self._to_train_device(state.truncated.reshape(-1, 1), torch.bool),
            env_infos(state),
        )

    def state(self) -> torch.Tensor:
        return self._to_train_device(self._env.state.obs.value_or_policy, torch.float32)

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
