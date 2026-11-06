# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import abc
import dataclasses
from dataclasses import dataclass
from typing import Generic, TypeVar

import array_api_compat.numpy as array_namespace_numpy
import gymnasium as gym
import numpy as np
import torch

from motrix_env_core.base import ABEnv, EnvCfg, ObsSpace
from motrix_env_core.perf import Perf, perf_root
from motrix_env_core.sim.backend import (
    RenderConfig,
    SimRenderer,
)

# The concrete config type an ArrayEnv subclass consumes. Subclasses parametrize
# via ``ArrayEnv[MyCfg]`` so ``self.cfg`` is typed as ``MyCfg`` (IDE hint +
# pyright narrowing). Unparametrized subclasses fall back to ``EnvCfg``
# (back-compat).
EnvCfgType = TypeVar("EnvCfgType", bound=EnvCfg)

# The array flavor an environment's state carries end to end: numpy arrays for
# NumPy-frontend environments, torch tensors for the manager pipeline. Each
# environment consistently produces one of the two; the lifecycle is
# duck-typed on it and never branches per element type.
T = TypeVar("T", np.ndarray, torch.Tensor)


@dataclass
class EnvObs(Generic[T]):
    """One observation slot; numpy arrays for NumPy-frontend environments,
    torch tensors for the manager pipeline."""

    policy: T
    value: T | None = None

    @property
    def value_or_policy(self) -> T:
        return self.policy if self.value is None else self.value


@dataclass
class FinalObs(Generic[T]):
    """Owned pre-reset observations of the current transition's truncated rows.

    Rows in ``obs`` align with ``env_ids``. Arrays remain valid across steps;
    consumers must treat them as read-only. ``obs.value`` stays optional.
    ``env_ids`` follows the frontend's currency: numpy for NumPy-frontend
    environments, a torch integer tensor for the manager pipeline.
    """

    env_ids: np.ndarray | torch.Tensor
    obs: EnvObs[T]


@dataclass
class ArrayEnvState(Generic[T]):
    """Environment transition state.

    ``obs``, ``reward``, ``terminated``, ``truncated`` and ``final_obs`` store
    numpy arrays for NumPy-frontend environments and torch tensors for the
    manager pipeline; each environment class consistently produces one of the
    two, so consumers never branch on the element type.
    """

    obs: EnvObs[T]
    reward: T
    terminated: T
    truncated: T
    episode_steps: np.ndarray
    # Per-term reward breakdown for the latest transition: term name to a
    # ``(num_envs,)`` array (numpy for the NumPy frontend, torch column views
    # for the manager pipeline). Freshly written every transition; empty when
    # the environment does not decompose its reward.
    reward_terms: dict[str, T] = dataclasses.field(default_factory=dict)
    # Live diagnostics view: per-environment quantities stay unreduced as
    # ``(num_envs,)`` arrays and are views into manager buffers that kernels
    # overwrite every step; batch-level gauges are scalars. Values always
    # reflect the latest step: retain values across steps through
    # :meth:`process_metrics`, which returns a fresh dict of reduced scalars.
    metrics: dict[str, float | np.ndarray] = dataclasses.field(default_factory=dict)
    # None when no row truncated or the implementation does not capture finals.
    final_obs: FinalObs[T] | None = None

    @property
    def done(self) -> T:
        """
        Check if the environment is done.
        """
        return self.terminated | self.truncated

    def replace(self, **updates) -> "ArrayEnvState[T]":
        return dataclasses.replace(self, **updates)

    def process_metrics(self) -> dict[str, float]:
        """Reduce per-environment metrics into batch-level scalar floats.

        Per-env arrays are reduced with their mean (bool arrays become trigger
        rates); scalars pass through. The result is a fresh, small dict that is
        safe to retain across steps, unlike the raw ``metrics`` arrays.
        """
        return {
            name: float(np.mean(value)) if isinstance(value, np.ndarray) else float(value)
            for name, value in self.metrics.items()
        }

    def validate(self):
        num_envs = self.obs.policy.shape[0]
        assert self.obs.policy.shape[0] == num_envs, self.obs.policy.shape
        if self.obs.value is not None:
            assert self.obs.value.shape[0] == num_envs, self.obs.value.shape
        assert self.reward.shape == (num_envs,), self.reward.shape
        assert self.terminated.shape == (num_envs,), self.terminated.shape
        assert self.truncated.shape == (num_envs,), self.truncated.shape
        assert self.episode_steps.shape == (num_envs,), self.episode_steps.shape


def _as_obs_space(space: gym.Space | ObsSpace) -> ObsSpace:
    if isinstance(space, ObsSpace):
        return space
    assert isinstance(space, gym.spaces.Box)
    return ObsSpace(policy=space)


def _as_obs(obs: "EnvObs[T] | np.ndarray") -> "EnvObs[T]":
    if isinstance(obs, EnvObs):
        return obs
    assert isinstance(obs, np.ndarray)
    return EnvObs(policy=obs)


class ArrayEnv(ABEnv, Generic[EnvCfgType, T]):
    """Simulator-agnostic NumPy frontend environment lifecycle.

    ``ArrayEnv`` owns the step / auto-reset / truncation lifecycle shared by
    NumPy-frontend environments. It is deliberately free of simulator-state
    concepts: subclasses address simulator rows themselves inside the reset
    hooks, so the frontend only ever speaks in environment indices.
    """

    _cfg: EnvCfgType
    _state: ArrayEnvState[T] = None
    _render_spacing: float

    def __init__(self, cfg: EnvCfgType, num_envs: int = 1):
        self._cfg = cfg
        self._num_envs = num_envs
        self.perf = Perf()
        if cfg.scene is None:
            raise ValueError("EnvCfg.scene must be configured")
        self._render_spacing = cfg.render_spacing

    @property
    def state(self) -> ArrayEnvState[T]:
        """
        Get the current environment state
        """
        return self._state

    @property
    def cfg(self) -> EnvCfgType:
        """
        Get the environment configuration
        """
        return self._cfg

    @property
    def render_spacing(self) -> float:
        """
        Get the render spacing, with which the multi-envs will be rendered seperately in grid
        """
        return self._render_spacing

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def init_state(self) -> ArrayEnvState[T]:
        """
        Create a new environment state
        """
        obs = self._zeros_obs(self.observation_space)
        reward = self._zeros_reward()
        terminated = self._zeros_terminated()
        truncated = self._zeros_truncated()
        episode_steps = np.zeros((self._num_envs,), dtype=np.uint64)
        self._state = self._new_state(obs, reward, terminated, truncated, episode_steps)
        self._reset_done_envs()
        with self.perf.scope("observation"):
            self._state = self.compute_observation(self._state)
        self._state = self._state.replace(obs=_as_obs(self._state.obs))
        self._state.validate()
        return self._state

    def _new_state(
        self,
        obs: EnvObs[T],
        reward: T,
        terminated: T,
        truncated: T,
        episode_steps: np.ndarray,
    ) -> ArrayEnvState[T]:
        """Assemble the environment state; subclasses add simulator-owned fields."""
        return ArrayEnvState(
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            episode_steps=episode_steps,
        )

    @property
    def policy_observation_space(self) -> gym.spaces.Box:
        return _as_obs_space(self.observation_space).policy

    @property
    def value_observation_space(self) -> gym.spaces.Box:
        return _as_obs_space(self.observation_space).value_or_policy

    @property
    def has_value_observation(self) -> bool:
        return _as_obs_space(self.observation_space).value is not None

    # Array module for the frontend's native currency (array_api_compat
    # namespaces expose zeros/ones/full with module dtypes); numpy frontends
    # inherit this default, torch frontends override the attribute only.
    _xp = array_namespace_numpy

    def _zeros_box(self, space: gym.spaces.Box) -> T:
        dtype = {np.dtype(np.float32): self._xp.float32, np.dtype(np.float64): self._xp.float64}[np.dtype(space.dtype)]
        return self._xp.zeros((self._num_envs, *space.shape), dtype=dtype)

    def _zeros_reward(self) -> T:
        return self._xp.zeros((self._num_envs,), dtype=self._xp.float32)

    def _zeros_terminated(self) -> T:
        return self._xp.ones((self._num_envs,), dtype=self._xp.bool)

    def _zeros_truncated(self) -> T:
        return self._xp.zeros((self._num_envs,), dtype=self._xp.bool)

    def _zeros_obs(self, space: gym.Space | ObsSpace) -> EnvObs[T]:
        obs_space = _as_obs_space(space)
        value = None if obs_space.value is None else self._zeros_box(obs_space.value)
        return EnvObs(policy=self._zeros_box(obs_space.policy), value=value)

    def _assign_obs(self, dst: EnvObs[T], mask: np.ndarray, src: "EnvObs[T] | np.ndarray"):
        src = _as_obs(src)
        dst.policy[mask] = src.policy
        if dst.value is not None:
            assert src.value is not None
            dst.value[mask] = src.value
        else:
            assert src.value is None

    def _reset_done_envs(self) -> None:
        """
        Reset the environments that are done
        """
        state = self._state
        with self.perf.scope("done_mask"):
            # np.asarray is a no-op for numpy state and a zero-copy view for
            # torch-backed manager state; host control flow stays numpy.
            done = np.asarray(state.done)
            assert done.shape == (self._num_envs,)
            has_done = done.any()
        if not has_done:
            return

        with self.perf.scope("select_done"):
            np.putmask(state.episode_steps, done, 0)
            env_ids = np.flatnonzero(done)
        with self.perf.scope("reset_envs"):
            self.reset(env_ids)

    @abc.abstractmethod
    def reset(self, env_ids: np.ndarray | torch.Tensor) -> None:
        """Write reset rows for the selected environments.

        ``env_ids`` follows the frontend's currency: numpy arrays for the
        NumPy frontend (DirectEnv), torch tensors for the manager pipeline.

        Subclasses address their simulator's rows themselves. Observation
        generation is deferred until :meth:`compute_observation`, after reset
        writes have completed.
        """

    def _capture_final_obs(self, env_ids: np.ndarray) -> "FinalObs[T] | None":
        """Capture truncated rows before reset; default implementation is unsupported.

        Called after transition and truncation updates, before any reset write.
        Implementations must leave normal observations and task state unchanged
        and return owned row arrays, without advancing observation history.
        """
        return None

    def _max_episode_steps(self) -> int | None:
        return self._cfg.max_episode_steps

    def _update_truncate(self):
        """
        Truncate the environments that have reached max episode length
        """
        max_episode_steps = self._max_episode_steps()
        if not max_episode_steps:
            return
        # Assignment through the slot works for both numpy arrays and torch
        # tensors (the latter receive the numpy mask via from_numpy).
        truncated = self._state.episode_steps >= max_episode_steps
        storage = self._state.truncated
        if isinstance(storage, np.ndarray):
            storage[:] = truncated
        else:
            storage[:] = torch.from_numpy(truncated)

    @abc.abstractmethod
    def create_renderer(self, config: RenderConfig) -> SimRenderer:
        """Create the backend-owned renderer bound to this environment."""

    @abc.abstractmethod
    def apply_action(self, actions: T, state: ArrayEnvState[T]) -> ArrayEnvState[T]:
        """
        Apply the action to the environment

        Args:
            actions (T): The actions to apply — the same array flavor the
                environment carries end to end (numpy arrays for the NumPy
                frontend, torch tensors for the manager pipeline)
            state (ArrayEnvState): The environment state to apply the actions.
        """

    @abc.abstractmethod
    def compute_transition(self, state: ArrayEnvState[T]) -> ArrayEnvState[T]:
        """
        Update the environment state after physics step

        This is the only full simulator-data refresh point of a step: concrete
        environments execute their whole read program here before deriving
        reward, termination, and metrics. It must not write ``state.obs`` —
        observations belong to :meth:`compute_observation`.

        Args:
            state (ArrayEnvState): The environment state to update
        """

    @abc.abstractmethod
    def physics_step(self):
        """Advance the simulator by the configured substeps."""

    def _prev_physics_step(self):
        if self._state.final_obs is not None:
            self._state = self._state.replace(final_obs=None)
        state = self._state
        state.reward[:] = 0.0
        state.terminated[:] = False
        state.truncated[:] = False

    def compute_observation(self, state: ArrayEnvState[T]) -> ArrayEnvState[T]:
        """Compute observations for all environments after reset.

        Called once after the first reset of :meth:`init_state` and once per
        :meth:`step` after episode bookkeeping and auto-reset. Implementations
        must purely construct the observation from the simulator data already
        refreshed by :meth:`compute_transition` (whole batch) and reset
        (selected rows) without executing further reads. The default preserves
        observations already present in ``state``.
        """
        return state

    @perf_root("step")
    def step(self, actions: T) -> ArrayEnvState[T]:
        """Advance one control step.

        ``actions`` flows the environment's array flavor: numpy arrays for
        NumPy-frontend environments, torch tensors for the manager pipeline.

        Timing: ``apply_action`` -> ``physics_step`` -> ``compute_transition``
        (the only full simulator-data refresh) -> episode bookkeeping ->
        optional pre-reset ``final_obs`` capture for truncated rows ->
        ``reset(env_ids)`` for done rows (may refresh selected rows) ->
        ``compute_observation`` (whole batch, pure construction).
        """
        if self._state is None:
            with self.perf.scope("init_state"):
                self.init_state()

        self._prev_physics_step()
        with self.perf.scope("apply_action"):
            self._state = self.apply_action(actions, self._state)
            assert self._state is not None, "apply_action must return a valid environment state"
        with self.perf.scope("physics"):
            self.physics_step()
        with self.perf.scope("transition"):
            self._state = self.compute_transition(self._state)
        self._state.episode_steps += 1
        self._update_truncate()
        trunc_ids = np.flatnonzero(np.asarray(self._state.truncated))
        if trunc_ids.size:
            with self.perf.scope("final_obs"):
                self._state = self._state.replace(final_obs=self._capture_final_obs(trunc_ids))
        with self.perf.scope("reset"):
            self._reset_done_envs()
        with self.perf.scope("observation"):
            self._state = self.compute_observation(self._state)
        self._state = self._state.replace(obs=_as_obs(self._state.obs))
        return self._state


__all__ = ["ArrayEnv", "ArrayEnvState", "EnvCfgType", "FinalObs", "EnvObs"]
