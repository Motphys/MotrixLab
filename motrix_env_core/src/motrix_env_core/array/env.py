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
    """Owned pre-reset observations of the current transition's truncated rows.

    Rows in ``obs`` align with ``env_ids``. Tensors remain valid across steps;
    consumers must treat them as read-only. ``obs.value`` stays optional.
    """

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

    def validate(self) -> None:
        """Validate that the transition state's fields share one batch width.

        Callers guarantee tensor types: task- and framework-produced values
        are torch tensors by the frontend contract.
        """
        num_envs = self.obs.policy.shape[0]
        if self.obs.value is not None:
            assert self.obs.value.shape[0] == num_envs, self.obs.value.shape
        assert self.reward.shape == (num_envs,), self.reward.shape
        assert self.terminated.shape == (num_envs,), self.terminated.shape
        assert self.truncated.shape == (num_envs,), self.truncated.shape
        assert self.episode_steps.shape == (num_envs,), self.episode_steps.shape


def _as_obs(obs: EnvObs | torch.Tensor) -> EnvObs:
    """Normalize an observation value to :class:`EnvObs`.

    A bare tensor is adopted as the policy observation; ``EnvObs`` values
    pass through unchanged.
    """
    if isinstance(obs, EnvObs):
        return obs
    return EnvObs(policy=obs)


def _as_obs_space(space: gym.spaces.Box | ObsSpace) -> ObsSpace:
    """Normalize an observation space; a bare policy Box is adopted as such."""
    if isinstance(space, ObsSpace):
        return space
    return ObsSpace(policy=space)


class ArrayEnv(ABEnv, Generic[EnvCfgType]):
    """Simulator-agnostic environment lifecycle with a torch-only public state.

    ``ArrayEnv`` owns the step / auto-reset / truncation lifecycle shared by
    torch-frontend environments. It is deliberately free of simulator-state
    concepts: subclasses address simulator rows themselves inside the reset
    hooks, so the frontend only ever speaks in environment indices.
    """

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
        """Create the initial environment transition state.

        Reward starts zeroed, truncation cleared, and termination set for
        every row so the first auto-reset seeds all lanes.
        """
        obs = self._zeros_obs(self.observation_space)
        num_envs = self._num_envs
        self._state = self._new_state(
            obs,
            torch.zeros((num_envs,), dtype=torch.float32),
            torch.ones((num_envs,), dtype=torch.bool),
            torch.zeros((num_envs,), dtype=torch.bool),
            torch.zeros((num_envs,), dtype=torch.long),
        )
        self._reset_done_envs()
        with self.perf.scope("observation"):
            self._state = self.compute_observation(self._state)
        self._state = self._state.replace(obs=_as_obs(self._state.obs))
        self._state.validate()
        return self._state

    def _new_state(
        self,
        obs: EnvObs,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        episode_steps: torch.Tensor,
    ) -> ArrayEnvState:
        """Assemble the framework-owned transition state; subclasses add simulator-owned fields."""
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

    def _zeros_box(self, space: gym.spaces.Box) -> torch.Tensor:
        dtype = torch.from_numpy(np.empty((), dtype=space.dtype)).dtype
        return torch.zeros((self._num_envs, *space.shape), dtype=dtype)

    def _zeros_obs(self, space: gym.spaces.Box | ObsSpace) -> EnvObs:
        obs_space = _as_obs_space(space)
        value = None if obs_space.value is None else self._zeros_box(obs_space.value)
        return EnvObs(policy=self._zeros_box(obs_space.policy), value=value)

    def _assign_obs(self, dst: EnvObs, mask: torch.Tensor, src: EnvObs | torch.Tensor) -> None:
        src = _as_obs(src)
        dst.policy[mask] = src.policy
        if dst.value is not None:
            assert src.value is not None
            dst.value[mask] = src.value

    def _reset_done_envs(self) -> None:
        """Reset all environment rows marked terminated or truncated."""
        state = self._state
        with self.perf.scope("done_mask"):
            done = state.done
            has_done = bool(done.any())
        if not has_done:
            return

        with self.perf.scope("select_done"):
            state.episode_steps[done] = 0
            env_ids = torch.nonzero(done, as_tuple=False).flatten()
        with self.perf.scope("reset_envs"):
            self.reset(env_ids)

    @abc.abstractmethod
    def reset(self, env_ids: torch.Tensor) -> None:
        """Write reset rows for the selected environments.

        Subclasses address their simulator's rows themselves. Observation
        generation is deferred until :meth:`compute_observation`, after reset
        writes have completed.
        """

    def _capture_final_obs(self, env_ids: torch.Tensor) -> FinalObs | None:
        """Capture truncated rows before reset; default implementation is unsupported.

        Called after transition and truncation updates, before any reset write.
        Implementations must leave normal observations and task state unchanged
        and return owned row tensors, without advancing observation history.
        """
        return None

    def _max_episode_steps(self) -> int | None:
        return self._cfg.max_episode_steps

    def _update_truncate(self) -> None:
        """Mark rows that reached the configured episode limit."""
        max_episode_steps = self._max_episode_steps()
        if max_episode_steps:
            self._state.truncated[:] = self._state.episode_steps >= max_episode_steps

    @abc.abstractmethod
    def create_renderer(self, config: RenderConfig) -> SimRenderer:
        """Create the backend-owned renderer bound to this environment."""

    @abc.abstractmethod
    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> ArrayEnvState:
        """Apply the action to the environment.

        Args:
            actions (torch.Tensor): The actions to apply
            state (ArrayEnvState): The environment state to apply the actions.
        """

    @abc.abstractmethod
    def compute_transition(self, state: ArrayEnvState) -> ArrayEnvState:
        """Update the environment state after physics step

        This is the only full simulator-data refresh point of a step: concrete
        environments execute their whole read program here before deriving
        reward, termination, and metrics. It must not write ``state.obs`` —
        observations belong to :meth:`compute_observation`.

        Args:
            state (ArrayEnvState): The environment state to update
        """

    @abc.abstractmethod
    def physics_step(self) -> None:
        """Advance the simulator by the configured substeps."""

    def _prev_physics_step(self) -> None:
        """Clear per-step transition outputs before advancing physics."""
        if self._state.final_obs is not None:
            self._state = self._state.replace(final_obs=None)
        self._state.reward[:] = 0.0
        self._state.terminated[:] = False
        self._state.truncated[:] = False

    def compute_observation(self, state: ArrayEnvState) -> ArrayEnvState:
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
    def step(self, actions: torch.Tensor) -> ArrayEnvState:
        """Advance one control step.

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
        with self.perf.scope("physics"):
            self.physics_step()
        with self.perf.scope("transition"):
            self._state = self.compute_transition(self._state)
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
            self._state.obs = _as_obs(self._state.obs)
        return self._state


__all__ = ["ArrayEnv", "ArrayEnvState", "EnvCfgType", "FinalObs", "EnvObs"]
