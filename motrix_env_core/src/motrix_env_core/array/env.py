# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import abc
import dataclasses
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Generic, TypeVar

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core.base import ABEnv, EnvCfg
from motrix_env_core.observations import ObservationSpec
from motrix_env_core.perf import Perf, perf_root
from motrix_env_core.sim.backend import RenderConfig, SimRenderer

EnvCfgType = TypeVar("EnvCfgType", bound=EnvCfg)


@dataclass
class EnvObs:
    """Named observations in the public torch state contract; policy is required."""

    groups: dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        if "policy" not in self.groups:
            raise ValueError("EnvObs requires a 'policy' observation group.")

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.groups[name]

    @property
    def policy(self) -> torch.Tensor:
        return self.groups["policy"]

    @property
    def value(self) -> torch.Tensor | None:
        return self.groups.get("value")

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


@dataclass(frozen=True)
class ArrayEnvState:
    """Public transition state; all lifecycle values are torch tensors.

    Frozen on purpose: the state is a projection over frontend-owned buffers.
    Supported updates are in-place tensor writes through the views, owner
    rebinding on the concrete frontend, and :meth:`replace` for derived
    snapshots; field rebinding on a projected state would be silently lost,
    so it raises instead.
    """

    obs: EnvObs
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    episode_steps: torch.Tensor
    # Per-term reward breakdown for the latest transition, purely for
    # logging and diagnostics (console panel, TensorBoard). The framework
    # never derives ``reward`` by suming these entries: the manager frontend
    # sums its weighted terms into the total itself, direct tasks report the
    # total independently, and the entries need not sum to it. Freshly
    # written every transition; empty when the environment does not
    # decompose reward.
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
        for observation in self.obs.groups.values():
            assert observation.shape[0] == num_envs, observation.shape
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
    return EnvObs({"policy": obs})


class ArrayEnv(ABEnv, Generic[EnvCfgType]):
    """Simulator-agnostic environment lifecycle with a torch-only public state.

    ``ArrayEnv`` owns the step / auto-reset / truncation lifecycle shared by
    torch-frontend environments. It is deliberately free of simulator-state
    concepts: subclasses address simulator rows themselves inside the reset
    hooks, so the frontend only ever speaks in environment indices.

    The public ``state`` is an abstract projection owned by each frontend:
    it owns nothing itself, and every tensor it exposes belongs to an owning
    component — the episode-lifecycle tensors below, the frontend's
    reward/termination storage, and the frontend's published observation.
    """

    _cfg: EnvCfgType
    _render_spacing: float

    def __init__(self, cfg: EnvCfgType, num_envs: int = 1):
        self._cfg = cfg
        self._num_envs = num_envs
        self.device = torch.device(cfg.sim.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._step_context = nullcontext
        self.perf = Perf()
        self.engine_error_codes = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self.engine_error_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._truncated = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._episode_steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._initialized = False
        self._final_obs: FinalObs | None = None
        self._metrics: dict[str, float | torch.Tensor] = {}
        if cfg.scene is None:
            raise ValueError("EnvCfg.scene must be configured")
        self._render_spacing = cfg.render_spacing

    @property
    @abc.abstractmethod
    def state(self) -> ArrayEnvState:
        """Current public transition state, projected by the concrete frontend."""

    @property
    def is_initialized(self) -> bool:
        """Whether the live state exists and the environment can step."""
        return self._initialized

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

    def initialize(self) -> "ArrayEnvState":
        """Create and return the initial environment transition state.

        Episode bookkeeping is cleared, the frontend seeds its owned state,
        and every lane is reset explicitly (no done-mask seeding trick).
        The returned state is the live state also exposed through ``self.state``.
        """
        self.engine_error_codes.zero_()
        self.engine_error_mask.zero_()
        self._final_obs = None
        self._truncated.zero_()
        self._episode_steps.zero_()
        self._init_owned_state()
        self.reset(torch.arange(self._num_envs, device=self.device))
        state = self.state
        with self.perf.scope("observation"):
            self.compute_observation(state)
        state.validate()
        self._initialized = True
        return state

    def _init_owned_state(self) -> None:
        """Initialize frontend-owned runtime state before the first reset.

        Frontends seed their owned reward/termination outputs and any
        additional owned state (compiled programs, kernel buffers) here;
        the reset pass over all lanes runs afterwards.
        """

    def _zeros_box(self, space: gym.spaces.Box) -> torch.Tensor:
        dtype = torch.from_numpy(np.empty((), dtype=space.dtype)).dtype
        return torch.zeros((self._num_envs, *space.shape), dtype=dtype, device=self.device)

    def _zeros_obs(self, spec: ObservationSpec) -> EnvObs:
        return EnvObs({name: self._zeros_box(group.space) for name, group in spec.groups.items()})

    def _assign_obs(self, dst: EnvObs, mask: torch.Tensor, src: EnvObs | torch.Tensor) -> None:
        src = _as_obs(src)
        for name, destination in dst.groups.items():
            destination[mask] = src[name]

    def _reset_done_envs(self) -> None:
        """Reset all environment rows marked terminated or truncated."""
        with self.perf.scope("done_mask"):
            done = self.state.terminated | self._truncated
            has_done = bool(done.any())
        if not has_done:
            return

        self._clear_engine_error_rewards()
        with self.perf.scope("select_done"):
            self._episode_steps[done] = 0
            env_ids = torch.nonzero(done, as_tuple=False).flatten()
        with self.perf.scope("reset_envs"):
            self.reset(env_ids)

    def _clear_engine_error_rewards(self) -> None:
        """Publish a finite failure reward, only from the done/reset slow path."""
        self.state.reward.masked_fill_(self.engine_error_mask, 0.0)

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
            torch.ge(self._episode_steps, max_episode_steps, out=self._truncated)

    @abc.abstractmethod
    def create_renderer(self, config: RenderConfig) -> SimRenderer:
        """Create the backend-owned renderer bound to this environment."""

    @abc.abstractmethod
    def apply_action(self, actions: torch.Tensor, state: ArrayEnvState) -> None:
        """Apply the action to the environment.

        Args:
            actions (torch.Tensor): The actions to apply
            state (ArrayEnvState): The environment state to apply the actions.
        """

    @abc.abstractmethod
    def compute_transition(self, state: ArrayEnvState) -> None:
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
        """Advance physics and overwrite ``engine_error_codes`` for every row.

        Zero means success; nonzero backend codes trigger termination and reset.
        Codes and ``engine_error_mask`` remain available after reset until the
        next step. Exceptions unrelated to row-level numerical errors propagate.
        """

    def _prev_physics_step(self) -> None:
        """Clear per-step transition outputs before advancing physics."""
        self._final_obs = None
        state = self.state
        state.reward[:] = 0.0
        state.terminated[:] = False
        state.truncated[:] = False

    def _pre_compute_observation(self) -> None:
        """Prepare the step endpoint's observation write target after auto-reset.

        Called only by ``step()``, immediately before obtaining the state
        projection passed to ``compute_observation``. Frontends may select an
        alternate buffer here without overwriting the published starting obs.
        Initialization and explicit observation calls do not invoke this hook.
        """

    def compute_observation(self, state: ArrayEnvState) -> None:
        """Publish whole-batch observations from already-refreshed inputs.

        Called after initialization and each control step. No additional
        simulator reads belong in this method. The default preserves
        observations already present in ``state``.
        """
        return None

    @perf_root("step")
    def step(self, actions: torch.Tensor) -> ArrayEnvState:
        """Advance one control step.

        Timing: ``apply_action`` -> ``physics_step`` -> ``compute_transition``
        (the only full simulator-data refresh) -> episode bookkeeping ->
        optional pre-reset ``final_obs`` capture for truncated rows ->
        ``reset(env_ids)`` for done rows (may refresh selected rows) ->
        ``_pre_compute_observation`` (select endpoint buffers) ->
        ``compute_observation`` (whole-batch endpoint publication).

        Buffer lifetime (DirectEnv and ManagerEnv):
            The returned state is a frozen projection, not an owned snapshot.
            Its tensors and diagnostic dictionaries belong to the environment;
            callers must treat them as read-only. Freezing the state prevents
            field rebinding, not mutation of the underlying buffers.

            ``state.obs`` uses alternating buffers. Observations returned by
            this call remain valid through the next ``step()`` and its return,
            allowing collectors to record both endpoints afterwards. The step
            after that may overwrite them. This guarantee covers ordinary
            stepping only: explicit initialization, reset, or observation
            publication may invalidate borrowed observations earlier.

            ``reward``, ``terminated``, ``truncated``, ``episode_steps``,
            ``reward_terms``, and ``metrics`` are live views, not double-buffered
            snapshots. Consume or copy them before the next step or another
            mutating environment call; retaining the state object does not
            retain their values. Same-device ``Tensor.to()`` need not copy.

            ``final_obs``, when present, is an owned pre-reset snapshot and
            remains valid across later steps even though the environment's
            current ``final_obs`` is replaced or cleared. Clone any other
            tensors (and copy dictionary contents) that must outlive the
            borrowed-buffer lifetime; finish asynchronous copies before reuse.
        """
        with self._step_context():
            return self._step(actions)

    def _step(self, actions: torch.Tensor) -> ArrayEnvState:
        if not self.is_initialized:
            with self.perf.scope("initialize"):
                self.initialize()

        self._prev_physics_step()
        # Cache the projection for the transition phases: its tensor fields
        # are views over the owner tensors, so in-place writes pass through.
        state = self.state
        with self.perf.scope("apply_action"):
            self.apply_action(actions, state)
        with self.perf.scope("physics"):
            self.physics_step()
            self.engine_error_mask.copy_(self.engine_error_codes != 0)
        # TODO: Failed physics rows still participate in the batch transition.
        # Consider excluding them if needed, without a separate reset path.
        with self.perf.scope("transition"):
            self.compute_transition(state)
        state.terminated.logical_or_(self.engine_error_mask)
        self._episode_steps += 1
        self._update_truncate()
        self._process_episode_boundaries()
        with self.perf.scope("observation"):
            self._pre_compute_observation()
            # The hook may have selected another observation buffer; do not
            # pass the cached starting projection as the endpoint write target.
            self.compute_observation(self.state)
        # Publish a fresh projection: step bookkeeping above rebound owner
        # fields (the final-observation snapshot, the reward-term breakdown)
        # that the cached view still holds stale.
        return self.state

    def _process_episode_boundaries(self) -> None:
        trunc_ids = torch.nonzero(self._truncated, as_tuple=False).flatten()
        if trunc_ids.numel():
            with self.perf.scope("final_obs"):
                self._final_obs = self._capture_final_obs(trunc_ids)
        with self.perf.scope("reset"):
            self._reset_done_envs()


__all__ = ["ArrayEnv", "ArrayEnvState", "EnvCfgType", "EnvObs", "FinalObs"]
