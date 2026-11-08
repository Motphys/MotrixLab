# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import torch
from skrl.envs.wrappers.jax import Wrapper as SkrlWrapper

from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.renderer import RenderConfig, VideoRecorder, create_renderer
from motrix_env_core.sim.backend import SimRenderer
from motrix_rl.utils import env_infos


def _as_jnp(value, dtype) -> jax.Array:
    """Convert torch env state to a JAX array: host view plus device transfer."""
    return jnp.asarray(np.asarray(value), dtype=dtype)


class SkrlJaxEnvWrap(SkrlWrapper):
    """
    Wrap the torch-state ArrayEnv (DirectEnv/ManagerEnv) for skrl's JAX path.

    Manager environments expose torch state; ``np.asarray`` yields zero-copy
    host views that JAX then transfers to its device.
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
        # ``from_dlpack`` is the direct JAX-to-torch path; ``.cpu()`` lands
        # GPU-resident batches on the host where the CPU frontend steps.
        state = self._env.step(torch.from_dlpack(actions).cpu())
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
