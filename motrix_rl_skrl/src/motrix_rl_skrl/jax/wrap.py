# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import gymnasium
import jax
import jax.numpy as jnp
import torch
from skrl import config
from skrl.envs.wrappers.jax import Wrapper as SkrlWrapper

from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.renderer import RenderConfig, VideoRecorder, create_renderer
from motrix_env_core.sim.backend import SimRenderer
from motrix_rl.utils import env_infos


def _as_jnp(value, dtype, device: jax.Device) -> jax.Array:
    """Convert values on the requested JAX device, sharing storage on the same GPU."""
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.is_cuda and device.platform == "gpu" and value.device.index == device.local_hardware_id:
            return jax.dlpack.from_dlpack(value.contiguous(), device=device).astype(dtype)
        value = value.cpu().numpy()
    return jnp.asarray(value, dtype=dtype, device=device)


class SkrlJaxEnvWrap(SkrlWrapper):
    """
    Wrap the torch-state ArrayEnv (DirectEnv/ManagerEnv) for skrl's JAX path.

    State is placed on the wrapper's JAX device for both rollout and training.
    """

    _env: ArrayEnv
    _renderer: SimRenderer | VideoRecorder | None = None

    def __init__(self, env: ArrayEnv, render: RenderConfig | None = None):
        self._sim_device = env.device
        super().__init__(env)
        if self._sim_device.type == "cuda":
            self._device = config.jax.parse_device(str(self._sim_device))
        self._renderer = create_renderer(env, render)

    def reset(self) -> tuple[jax.Array, Any]:
        state = self._env.initialize()
        return _as_jnp(state.obs.policy, jnp.float32, self.device), env_infos(state)

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
        state = self._env.step(torch.from_dlpack(actions).to(self._sim_device))
        return (
            _as_jnp(state.obs.policy, jnp.float32, self.device),
            _as_jnp(state.reward.reshape(-1, 1), jnp.float32, self.device),
            _as_jnp(state.terminated.reshape(-1, 1), jnp.bool_, self.device),
            _as_jnp(state.truncated.reshape(-1, 1), jnp.bool_, self.device),
            env_infos(state),
        )

    def state(self) -> jax.Array:
        return _as_jnp(self._env.state.obs.value_or_policy, jnp.float32, self.device)

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
