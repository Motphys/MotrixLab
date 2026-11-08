# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import gymnasium as gym
import torch

from motrix_env_core.array.env import FinalObs
from motrix_env_core.base import ABEnv
from motrix_env_core.renderer import RenderConfig, create_renderer
from motrix_rl.utils import env_infos


@dataclass
class FastSacStepResult:
    """One step's tensors, diagnostics and optional owned pre-reset snapshot.

    Observations follow the frontend's auto-reset semantics. ``final_obs``
    carries truncated rows before reset, or is absent when no snapshot exists.
    """

    obs: torch.Tensor
    critic_obs: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    final_obs: FinalObs | None
    info: dict


class FastSacEnvWrap:
    """Environment adapter consumed by FastSAC.

    Wraps a Motrix environment (``ArrayEnv`` frontends) as the torch-tensor
    vectorized environment consumed by FastSAC. The frontend already exposes
    torch state and consumes torch actions; state passes through with only a
    device move to the trainer device. Environments simulate on CPU, so the
    actor maps its tanh output to the action range declared by the
    environment and the wrapper clips on CPU before stepping. Done
    environments are auto-reset by the environment; the returned observation
    is therefore the post-reset observation for those environments.
    """

    def __init__(self, env: ABEnv, device: torch.device, render: RenderConfig | None = None) -> None:
        action_space = env.action_space
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("FastSAC supports only Box action spaces.")

        self._env = env
        self.device = device
        self.num_envs = env.num_envs
        # Bounds live on CPU alongside the env-bound actions clamped in step.
        self._low_t = torch.as_tensor(action_space.low, dtype=torch.float32)
        self._high_t = torch.as_tensor(action_space.high, dtype=torch.float32)
        self._renderer = create_renderer(env, render)

    @property
    def env(self) -> ABEnv:
        return self._env

    @property
    def action_low(self) -> torch.Tensor:
        return self._low_t.to(self.device)

    @property
    def action_high(self) -> torch.Tensor:
        return self._high_t.to(self.device)

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
            # None when no row truncated or the implementation does not
            # capture finals.
            final_obs=state.final_obs,
            info=env_infos(state),
        )

    @property
    def max_episode_steps(self) -> int | None:
        """The environment's time-limit in control steps, or ``None``."""
        resolver = getattr(self._env, "_max_episode_steps", None)
        return resolver() if resolver is not None else None

    def render(self, *args, **kwargs) -> bool | None:
        if self._renderer is None:
            return None
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
