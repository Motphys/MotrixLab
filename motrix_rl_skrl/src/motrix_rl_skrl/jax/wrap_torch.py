# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import gymnasium
import jax
import jax.numpy as jnp
import torch
from skrl.envs.wrappers.jax import Wrapper as SkrlWrapper

from motrix_env_core.renderer import RenderConfig, VideoRecorder, create_renderer
from motrix_env_core.sim.backend import SimRenderer
from motrix_env_motrixsim.torch_env import TorchEnv
from motrix_rl.utils import env_infos


class SkrlTorchWrapper(SkrlWrapper):
    """
    Wrap the torch-based environment to be compatible with skrl
    """

    _env: TorchEnv
    _renderer: SimRenderer | VideoRecorder | None = None

    def __init__(self, env: TorchEnv, render: RenderConfig | None = None):
        super().__init__(env)
        self._renderer = create_renderer(env, render)

    def _to_jax(self, tensor: torch.Tensor, dtype) -> jax.Array:
        platform = self.device.platform
        target = torch.device("cuda", self.device.id) if platform in {"cuda", "gpu"} else torch.device("cpu")
        array = jax.dlpack.from_dlpack(tensor.detach().to(device=target))
        if array.dtype != dtype:
            array = array.astype(dtype)
        return jax.device_put(array, self.device)

    def reset(self) -> tuple[jax.Array, Any]:
        state = self._env.init_state()
        return self._to_jax(state.obs.policy, jnp.float32), env_infos(state)

    def step(
        self, actions: jax.Array
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        Any,
    ]:
        env_actions = torch.utils.dlpack.from_dlpack(actions).to(device=self._env.device, dtype=torch.float32)
        state = self._env.step(env_actions)
        return (
            self._to_jax(state.obs.policy, jnp.float32),
            self._to_jax(state.reward.reshape(-1, 1), jnp.float32),
            self._to_jax(state.terminated.reshape(-1, 1), jnp.bool_),
            self._to_jax(state.truncated.reshape(-1, 1), jnp.bool_),
            env_infos(state),
        )

    def state(self) -> jax.Array:
        return self._to_jax(self._env.state.obs.value_or_policy, jnp.float32)

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
