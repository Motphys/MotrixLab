# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import abc

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core.base import ABEnv


class FastSacEnvWrap(abc.ABC):
    """Base environment adapter consumed by FastSAC."""

    def __init__(
        self,
        env: ABEnv,
        device: torch.device,
    ) -> None:
        action_space = env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("FastSAC supports only Box action spaces.")

        self._env = env
        self.device = device
        self.num_envs = env.num_envs
        self._low = np.asarray(action_space.low, dtype=np.float32)
        self._high = np.asarray(action_space.high, dtype=np.float32)
        self.last_info: dict = {}
        self._renderer = None

    @property
    def env(self) -> ABEnv:
        return self._env

    @property
    def action_low(self) -> torch.Tensor:
        return torch.as_tensor(self._low, dtype=torch.float32, device=self.device)

    @property
    def action_high(self) -> torch.Tensor:
        return torch.as_tensor(self._high, dtype=torch.float32, device=self.device)

    @abc.abstractmethod
    def reset(self) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abc.abstractmethod
    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def render(self, *args, **kwargs) -> bool | None:
        if self._renderer is None:
            return None
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
