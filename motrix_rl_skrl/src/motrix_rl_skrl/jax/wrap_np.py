# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from skrl.envs.wrappers.jax import Wrapper as SkrlWrapper

from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.renderer import RenderConfig, VideoRecorder, create_renderer
from motrix_env_core.sim.backend import SimRenderer
from motrix_rl.utils import env_infos


def _as_jnp(value, dtype) -> jax.Array:
    """Zero-copy JAX view over both numpy and torch env state."""
    return jnp.asarray(np.asarray(value), dtype=dtype)


class SkrlNpWrapper(SkrlWrapper):
    """
    Wrap the numpy-based environment to be compatible with skrl

    Manager environments expose torch state; ``np.asarray`` yields the
    zero-copy numpy views the JAX path consumes.
    """

    _env: ArrayEnv
    _renderer: SimRenderer | VideoRecorder | None = None

    def __init__(self, env: ArrayEnv, render: RenderConfig | None = None):
        super().__init__(env)
        self._renderer = create_renderer(env, render)

    def reset(self) -> tuple[jax.Array, Any]:
        state = self._env.init_state()
        return _as_jnp(state.obs.policy, jnp.float32), env_infos(state)

    def step(
        self, actions: jax.Array
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        Any,
    ]:
        actions = np.asarray(actions)
        state = self._env.step(actions)
        return (
            _as_jnp(state.obs.policy, jnp.float32),
            _as_jnp(state.reward.reshape(-1, 1), jnp.float32),
            _as_jnp(state.terminated.reshape(-1, 1), jnp.bool_),
            _as_jnp(state.truncated.reshape(-1, 1), jnp.bool_),
            env_infos(state),
        )

    def state(self) -> jax.Array:
        return _as_jnp(self._env.state.obs.value_or_policy, jnp.float32)

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
