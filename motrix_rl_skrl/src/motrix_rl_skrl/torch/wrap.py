# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import gymnasium
import torch
from skrl.envs.wrappers.torch import Wrapper as SkrlWrapper

from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.renderer import RenderConfig, VideoRecorder, create_renderer
from motrix_env_core.sim.backend import SimRenderer
from motrix_rl.device import resolve_default_device
from motrix_rl.utils import env_infos


class SkrlTorchEnvWrap(SkrlWrapper):
    """
    Wrap a Motrix ArrayEnv (Direct or Manager frontend) for skrl (PyTorch)

    The environment exposes torch state and consumes torch actions; the
    wrapper only moves tensors onto the training device.
    """

    _env: ArrayEnv
    _renderer: SimRenderer | VideoRecorder | None = None

    def __init__(self, env: ArrayEnv, render: RenderConfig | None = None):
        self._sim_device = env.device
        super().__init__(env)
        # Keep CUDA rollouts on the simulator GPU. CPU simulation uses the
        # project's default learning device (CUDA, MPS, or CPU).
        if self._sim_device.type == "cpu":
            device = resolve_default_device()
            self._device = torch.device("cuda", torch.cuda.current_device()) if device.type == "cuda" else device
        self._renderer = create_renderer(env, render)

    def reset(self) -> tuple[torch.Tensor, Any]:
        state = self._env.initialize()
        return state.obs.policy.to(device=self.device, dtype=torch.float32), env_infos(state)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Any,
    ]:
        state = self._env.step(torch.as_tensor(actions, dtype=torch.float32, device=self._sim_device).detach())
        return (
            state.obs.policy.to(device=self.device, dtype=torch.float32),
            state.reward.reshape(-1, 1).to(device=self.device, dtype=torch.float32),
            state.terminated.reshape(-1, 1).to(device=self.device, dtype=torch.bool),
            state.truncated.reshape(-1, 1).to(device=self.device, dtype=torch.bool),
            env_infos(state),
        )

    def state(self) -> torch.Tensor:
        return self._env.state.obs.value_or_policy.to(device=self.device, dtype=torch.float32)

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
        return self._env.observation_spec.policy.space

    @property
    def state_space(self) -> gymnasium.Space:
        return self._env.observation_spec.value_or_policy.space

    @property
    def action_space(self) -> gymnasium.Space:
        return self._env.action_space
