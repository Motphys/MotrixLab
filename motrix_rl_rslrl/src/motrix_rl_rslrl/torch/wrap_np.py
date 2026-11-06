# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""VecEnv wrapper for adapting ArrayEnv environments to RSLRL's VecEnv interface."""

import numpy as np
import torch
from rsl_rl.env.vec_env import VecEnv
from tensordict import TensorDict

from motrix_env_core.array.env import ArrayEnv, NpObs
from motrix_env_core.renderer import RenderConfig, create_renderer


def _as_torch(value, device: torch.device) -> torch.Tensor:
    """Zero-copy tensor view for both numpy and torch env state."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.from_numpy(np.asarray(value)).to(device)


class RslrlNpEnvWrap(VecEnv):
    """Adapter class that wraps ArrayEnv environments to RSLRL's VecEnv interface.

    RSLRL expects a VecEnv interface with specific methods for stepping,
    resetting, and accessing observations. This adapter converts between the
    ``ArrayEnvState`` format and RSLRL's expected format; manager environments
    already expose torch state and pass through zero-copy.
    """

    def __init__(self, env: ArrayEnv, device: torch.device, render: RenderConfig | None = None):
        """Initialize the VecEnv adapter.

        Args:
            env: The DirectEnv instance to wrap
            device: PyTorch device for tensors
        """
        self._env = env
        self._device = device
        self._state = None
        self._num_envs = env.num_envs

        # Set max_episode_length from env config
        self.max_episode_length = self._env.cfg.max_episode_steps if self._env.cfg.max_episode_steps else 10000

        # Episode length buffer for tracking
        self.episode_length_buf = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

        # Configuration dict for RSLRL logger
        self.cfg = {
            "env_name": self._env.cfg.__class__.__name__,
        }

        # Initialize the environment state
        self.reset()
        self._renderer = create_renderer(env, render)

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs

    @property
    def num_obs(self) -> int:
        """Size of observation space."""
        return self._env.policy_observation_space.shape[0]

    @property
    def num_privileged_obs(self) -> int | None:
        """Size of privileged critic observations, if provided."""
        if not self._env.has_value_observation:
            return None
        return self._env.value_observation_space.shape[0]

    @property
    def num_actions(self) -> int:
        """Size of action space."""
        return self._env.action_space.shape[0]

    @property
    def device(self) -> torch.device:
        """PyTorch device for tensors."""
        return self._device

    @property
    def unwrapped(self) -> "RslrlNpEnvWrap":
        """Return the unwrapped environment (self for this wrapper)."""
        return self

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # Convert torch actions to numpy
        actions_np = actions.cpu().numpy()

        # Step the environment
        state = self._env.step(actions_np)
        self._state = state

        # Update episode length buffer
        self.episode_length_buf += 1
        # Reset episode length for done environments
        done_np = np.asarray(state.done, dtype=bool)
        self.episode_length_buf[done_np] = 0

        rewards = _as_torch(state.reward, self._device)

        # Merge terminated and truncated into dones
        dones = _as_torch(np.asarray(state.done, dtype=np.float32), self._device)

        obs = self._to_tensordict(state.obs)

        # Build extras dict (RSLRL calls it "extras" not "infos")
        # time_outs: rows truncated without failing, for value bootstrapping.
        extras = {"time_outs": _as_torch(np.asarray(state.truncated & ~state.terminated, dtype=bool), self._device)}

        return obs, rewards, dones, extras

    def reset(self) -> tuple[TensorDict, dict]:
        """Reset all environments.

        Returns:
            Tuple of (observations, extras)
            - observations: TensorDict with observation groups
            - extras: dict with episode information
        """
        state = self._env.init_state()
        self._state = state

        # Reset episode length buffer
        self.episode_length_buf.zero_()

        obs = self._to_tensordict(state.obs)

        # Build extras dict
        extras = {}

        return obs, extras

    def get_observations(self) -> TensorDict:
        """Get current observations without stepping the environment.

        Returns:
            Current observations as TensorDict
        """
        if self._state is None:
            obs, _ = self.reset()
            return obs

        return self._to_tensordict(self._state.obs)

    def _to_tensordict(self, obs: NpObs) -> TensorDict:
        tensors = {"policy": _as_torch(obs.policy, self._device)}
        if obs.value is not None:
            tensors["value"] = _as_torch(obs.value, self._device)
        return TensorDict(tensors, batch_size=[self._num_envs], device=self._device)

    def render(self, *args, **kwargs) -> bool | None:
        if self._renderer is None:
            return None
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
