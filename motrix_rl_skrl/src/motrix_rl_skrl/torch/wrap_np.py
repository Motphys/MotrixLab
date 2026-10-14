# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import gymnasium
import torch
from skrl.envs.wrappers.torch import Wrapper as SkrlWrapper

from motrix_env_core.direct.env import DirectEnv
from motrix_env_core.renderer import RenderConfig, VideoRecorder, create_renderer
from motrix_env_core.sim.backend import SimRenderer
from motrix_rl.device import resolve_default_device
from motrix_rl.utils import env_infos


class SkrlNpWrapper(SkrlWrapper):
    """
    Wrap the numpy-based environment to be compatible with skrl (PyTorch)
    """

    _env: DirectEnv
    _renderer: SimRenderer | VideoRecorder | None = None

    def __init__(self, env: DirectEnv, render: RenderConfig | None = None):
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
        return torch.tensor(state.obs.policy, dtype=torch.float32, device=self.device), env_infos(state)

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
            torch.tensor(state.obs.policy, dtype=torch.float32, device=self.device),
            torch.tensor(state.reward.reshape(-1, 1), dtype=torch.float32, device=self.device),
            torch.tensor(state.terminated.reshape(-1, 1), dtype=torch.bool, device=self.device),
            torch.tensor(state.truncated.reshape(-1, 1), dtype=torch.bool, device=self.device),
            env_infos(state),
        )

    def state(self) -> torch.Tensor:
        return torch.tensor(self._env.state.obs.value_or_policy, dtype=torch.float32, device=self.device)

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
