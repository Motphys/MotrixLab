# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import abc
import dataclasses
from dataclasses import dataclass
from typing import Generic, TypeVar

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core.base import ABEnv, EnvCfg, ObsSpace
from motrix_env_core.perf import Perf, perf_root
from motrix_env_core.sim.backend import RenderConfig, SimRenderer

EnvCfgType = TypeVar("EnvCfgType", bound=EnvCfg)


@dataclass
class EnvObs:
    """Policy/value observations in the public torch state contract."""

    policy: torch.Tensor
    value: torch.Tensor | None = None

    @property
    def value_or_policy(self) -> torch.Tensor:
        return self.policy if self.value is None else self.value


@dataclass
class FinalObs:
    """Owned pre-reset observations aligned with torch ``env_ids``."""

    env_ids: torch.Tensor
    obs: EnvObs


@dataclass
class ArrayEnvState:
    """Public transition state; all lifecycle values are torch tensors."""

    obs: EnvObs
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_steps: torch.Tensor
    # Per-term reward breakdown for the latest transition. Freshly written
    # every transition; empty when the environment does not decompose reward.
    reward_terms: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    # Live diagnostics view. Per-environment quantities stay unreduced as
    # ``(num_envs,)`` tensors; batch-level gauges are scalars. Values reflect
    # the latest step and are reduced by :meth:`process_metrics`.
    metrics: dict[str, float | torch.Tensor] = dataclasses.field(default_factory=dict)
    # None when no row truncated or the implementation does not capture finals.
    final_obs: FinalObs | None = None

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated

    def replace(self, **updates) -> "ArrayEnvState":
        return dataclasses.replace(self, **updates)

    def process_metrics(self) -> dict[str, float]:
        """Reduce per-environment metrics into batch-level scalar floats.

        Per-environment tensors are reduced with their mean (bool tensors
        become trigger rates); scalars pass through. The result is a fresh,
        small dict that is safe to retain across steps.
        """
        return {
            name: float(value.float().mean().item()) if isinstance(value, torch.Tensor) else float(value)
            for name, value in self.metrics.items()
        }

    def validate(self):
        """Validate the public transition state shapes and tensor types."""
        num_envs = self.obs.policy.shape[0]
        assert self.obs.policy.shape[0] == num_envs, self.obs.policy.shape
        if self.obs.value is not None:
            assert self.obs.value.shape[0] == num_envs, self.obs.value.shape
        assert self.reward.shape == (num_envs,), self.reward.shape
        assert self.terminated.shape == (num_envs,), self.terminated.shape
        assert self.truncated.shape == (num_envs,), self.truncated.shape
        assert self.episode_steps.shape == (num_envs,), self.episode_steps.shape
        for value in (self.obs.policy, self.obs.value, self.reward, self.terminated, self.truncated, self.episode_steps):
            if value is not None:
                assert isinstance(value, torch.Tensor), type(value)


def _as_obs(obs: EnvObs | torch.Tensor) -> EnvObs:
    """Normalize an already torch-native observation value."""
    if isinstance(obs, EnvObs):
        assert isinstance(obs.policy, torch.Tensor)
        if obs.value is not None:
            assert isinstance(obs.value, torch.Tensor)
        return obs
    assert isinstance(obs, torch.Tensor)
    return EnvObs(policy=obs)


def _coerce_obs(obs: EnvObs | torch.Tensor | np.ndarray) -> EnvObs:
    """Convert legacy backend/task return values before strict normalization."""
    if isinstance(obs, EnvObs):
        if not isinstance(obs.policy, torch.Tensor):
            obs.policy = torch.as_tensor(obs.policy)
        if obs.value is not None and not isinstance(obs.value, torch.Tensor):
            obs.value = torch.as_tensor(obs.value)
        return obs
    return EnvObs(policy=torch.as_tensor(obs))


class ArrayEnv(ABEnv, Generic[EnvCfgType]):
    """Simulator-agnostic environment lifecycle with a torch-only public state."""

    _cfg: EnvCfgType
    _state: ArrayEnvState | None = None
    _render_spacing: float

    def __init__(self, cfg: EnvCfgType, num_envs: int = 1):
        self._cfg = cfg
        self._num_envs = num_envs
        self.perf = Perf()
        if cfg.scene is None:
            raise ValueError("EnvCfg.scene must be configured")
        self._render_spacing = cfg.render_spacing

    @property
    def state(self) -> ArrayEnvState:
        """Get the current environment state."""
        return self._state

    @property
    def cfg(self) -> EnvCfgType:
        """Get the environment configuration."""
        return self._cfg

    @property
    def render_spacing(self) -> float:
        """Get the render spacing for multi-environment layouts."""
        return self._render_spacing

    @property
    def num_envs(self) -> int:
        """Get the number of parallel environment rows."""
        return self._num_envs

    def init_state(self) -> ArrayEnvState:
        """Create the initial environment transition state."""
        obs = self._zeros_obs(self.observation_space)
        self._state = self._new_state(
            obs,
            self._zeros_reward(),
            self._zeros_terminated(),
            self._zeros_truncated(),
            torch.zeros((self._num_envs,), dtype=torch.long),
        )
        self._reset_done_envs()
        with self.perf.scope("observation"):
            self._state = self.compute_observation(self._state)
        self._state = self._state.replace(obs=_coerce_obs(self._state.obs))
        self._coerce_state(self._state)
        self._state.validate()
        return self._state

    def _new_state(self, obs, reward, terminated, truncated, episode_steps) -> ArrayEnvState:
        """Assemble the framework-owned transition state."""
        return ArrayEnvState(obs=obs, reward=reward, terminated=terminated, truncated=truncated, episode_steps=episode_steps)

    @property
    def policy_observation_space(self) -> gym.spaces.Box:
        return self.observation_space.policy if isinstance(self.observation_space, ObsSpace) else self.observation_space

    @property
    def value_observation_space(self) -> gym.spaces.Box:
        space = self.observation_space
        return space.value_or_policy if isinstance(space, ObsSpace) else space

    @property
    def has_value_observation(self) -> bool:
        return isinstance(self.observation_space, ObsSpace) and self.observation_space.value is not None

    def _zeros_box(self, space: gym.spaces.Box) -> torch.Tensor:
        return torch.zeros((self._num_envs, *space.shape), dtype=torch.from_numpy(np.empty((), dtype=space.dtype)).dtype)

    def _zeros_reward(self) -> torch.Tensor:
        return torch.zeros((self._num_envs,), dtype=torch.float32)

    def _zeros_terminated(self) -> torch.Tensor:
        return torch.ones((self._num_envs,), dtype=torch.bool)

    def _zeros_truncated(self) -> torch.Tensor:
        return torch.zeros((self._num_envs,), dtype=torch.bool)

    def _zeros_obs(self, space) -> EnvObs:
        obs_space = space if isinstance(space, ObsSpace) else ObsSpace(policy=space)
        value = None if obs_space.value is None else self._zeros_box(obs_space.value)
        return EnvObs(policy=self._zeros_box(obs_space.policy), value=value)

    def _assign_obs(self, dst: EnvObs, mask: torch.Tensor, src):
        src = _coerce_obs(src)
        dst.policy[mask] = src.policy
        if dst.value is not None:
            assert src.value is not None
            dst.value[mask] = src.value

    def _coerce_state(self, state: ArrayEnvState) -> ArrayEnvState:
        state.obs = _as_obs(state.obs)
        for name in ("reward", "terminated", "truncated", "episode_steps"):
            value = getattr(state, name)
            setattr(state, name, value if isinstance(value, torch.Tensor) else torch.as_tensor(value))
        state.reward_terms = {
            name: value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            for name, value in state.reward_terms.items()
        }
        state.metrics = {
            name: value if isinstance(value, (int, float, torch.Tensor)) else torch.as_tensor(value)
            for name, value in state.metrics.items()
        }
        if state.final_obs is not None:
            assert state.final_obs.env_ids.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
            state.final_obs.obs = _as_obs(state.final_obs.obs)
        return state

    def _reset_done_envs(self) -> None:
        """Reset all environment rows marked terminated or truncated."""
        state = self._state
        done = state.done
        if not bool(done.any()):
            return
        state.episode_steps[done] = 0
        self.reset(torch.nonzero(done, as_tuple=False).flatten())

    @abc.abstractmethod
    def reset(self, env_ids: torch.Tensor) -> None: ...

    def _capture_final_obs(self, env_ids: torch.Tensor) -> FinalObs | None:
        """Capture truncated rows before reset; unsupported by default."""
        return None

    def _max_episode_steps(self) -> int | None:
        return self._cfg.max_episode_steps

    def _update_truncate(self):
        """Mark rows that reached the configured episode limit."""
        max_episode_steps = self._max_episode_steps()
        if max_episode_steps:
            self._state.truncated[:] = self._state.episode_steps >= max_episode_steps

    @abc.abstractmethod
    def create_renderer(self, config: RenderConfig) -> SimRenderer: ...

    @abc.abstractmethod
    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState: ...

    @abc.abstractmethod
    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState: ...

    @abc.abstractmethod
    def physics_step(self): ...

    def _prev_physics_step(self):
        """Clear per-step transition outputs before advancing physics."""
        if self._state.final_obs is not None:
            self._state = self._state.replace(final_obs=None)
        self._state.reward[:] = 0.0
        self._state.terminated[:] = False
        self._state.truncated[:] = False

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
        """Compute observations from simulator data after transition/reset."""
        return state

    @perf_root("step")
    def step(self, actions: torch.Tensor) -> ArrayEnvState:
        """Advance one control step through action, physics, reset, and observation."""
        if self._state is None:
            self.init_state()
        self._prev_physics_step()
        with self.perf.scope("apply_action"):
            self._state = self.apply_action(actions, self._state)
            self._coerce_state(self._state)
        with self.perf.scope("physics"):
            self.physics_step()
        with self.perf.scope("transition"):
            self._state = self.compute_transition(self._state)
            self._coerce_state(self._state)
        self._state.episode_steps += 1
        self._update_truncate()
        trunc_ids = torch.nonzero(self._state.truncated, as_tuple=False).flatten()
        if trunc_ids.numel():
            with self.perf.scope("final_obs"):
                self._state = self._state.replace(final_obs=self._capture_final_obs(trunc_ids))
        with self.perf.scope("reset"):
            self._reset_done_envs()
        with self.perf.scope("observation"):
            self._state = self.compute_observation(self._state)
            self._state.obs = _coerce_obs(self._state.obs)
            self._coerce_state(self._state)
        return self._state


__all__ = ["ArrayEnv", "ArrayEnvState", "EnvCfgType", "FinalObs", "EnvObs"]
