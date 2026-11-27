# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import numbers
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch

from motrix_env_core.array.env import EnvObs, FinalObs
from motrix_env_core.config import configclass
from motrix_env_core.manager.context import BuildContext
from motrix_env_core.manager.terms import BaseTerm, canonicalize_term_args
from motrix_env_core.observations import ObservationGroupSpec, ObservationSpec, append_observation, repeat_observation

if TYPE_CHECKING:
    from motrix_env_core.manager.env import ManagerBasedEnvCfg


@configclass(kw_only=True)
class ObservationTermCfg(abc.ABC):
    """Configuration that creates one environment-local observation term."""

    @abc.abstractmethod
    def __call__(self, ctx: BuildContext) -> ObsTerm:
        """Assemble the runtime term (dispatch plus static arguments)."""


@dataclass(frozen=True, slots=True, init=False)
class ObsTerm(BaseTerm):
    """Pure observation dispatch with a fixed output width.

    Read context and write only the supplied output, except observation noise
    may consume the default Torch stream through ctx.rand. Do not advance
    history or commands: truncated steps may observe again before reset.
    """

    size: int

    def __init__(self, size: int, dispatch: Callable[..., Any], *args: Any) -> None:
        object.__setattr__(self, "size", size)
        BaseTerm.__init__(self, dispatch, *args)

    def __post_init__(self) -> None:
        BaseTerm.__post_init__(self)
        if not isinstance(self.size, numbers.Integral) or isinstance(self.size, (bool, np.bool_)):
            raise TypeError("Observation term size must be an integer.")
        if self.size <= 0:
            raise ValueError("Observation term size must be positive.")


@configclass
class ManagerObservationGroupCfg:
    """Typed declaration group for one named observation group's terms."""

    history_length: int = 1

    def to_dict(self) -> dict[str, ObservationTermCfg]:
        """Return observation term configs keyed by their declaration names."""
        term_cfgs: dict[str, ObservationTermCfg] = {}
        for term_field in fields(self):
            term_cfg = getattr(self, term_field.name)
            if isinstance(term_cfg, ObservationTermCfg):
                term_cfgs[term_field.name] = term_cfg
        return term_cfgs


@configclass
class ManagerObservationsCfg:
    """Typed declaration groups for one environment's observation terms."""

    def to_dict(self) -> dict[str, ManagerObservationGroupCfg]:
        """Return observation groups keyed by their declaration names."""
        group_cfgs: dict[str, ManagerObservationGroupCfg] = {}
        for group_field in fields(self):
            group_name = group_field.name
            group_cfg = getattr(self, group_name)
            if not isinstance(group_cfg, ManagerObservationGroupCfg):
                raise TypeError(
                    f"Observation group {group_name!r} must be a ManagerObservationGroupCfg, "
                    f"got {type(group_cfg).__name__}."
                )
            group_cfgs[group_name] = group_cfg
        return group_cfgs


@dataclass(frozen=True)
class ObservationTermEntry:
    name: str
    term: ObsTerm
    size: int


@dataclass(frozen=True)
class ObservationGroupEntry:
    name: str
    terms: tuple[ObservationTermEntry, ...]
    size: int
    history_length: int = 1


def create_observation_groups(
    cfg: ManagerBasedEnvCfg,
    ctx: BuildContext,
) -> dict[str, ObservationGroupEntry]:
    """Create and validate runtime observation terms grouped by output buffer."""
    groups: dict[str, ObservationGroupEntry] = {}
    for group_name, term_cfgs in cfg.observation_cfgs().items():
        entries = []
        group_size = 0
        for term_name, term_cfg in term_cfgs.items():
            created = term_cfg(ctx)
            if not isinstance(created, ObsTerm):
                raise TypeError(
                    f"Observation term {group_name}.{term_name} __call__() must return ObsTerm, "
                    f"got {type(created).__name__}."
                )
            term = ObsTerm(
                int(created.size),
                created.dispatch,
                *canonicalize_term_args(created.args, context=f"Observation term {group_name}.{term_name}"),
            )
            resolved_size = int(term.size)
            entries.append(ObservationTermEntry(term_name, term, resolved_size))
            group_size += resolved_size
        if entries:
            group_cfg = cfg.observation_group_cfgs()[group_name]
            history_length = group_cfg.history_length if isinstance(group_cfg, ManagerObservationGroupCfg) else 1
            if not isinstance(history_length, numbers.Integral) or isinstance(history_length, (bool, np.bool_)):
                raise TypeError("Observation history_length must be an integer.")
            if history_length < 1:
                raise ValueError("Observation history_length must be at least 1.")
            groups[group_name] = ObservationGroupEntry(group_name, tuple(entries), group_size, history_length)
    if "policy" not in groups:
        raise ValueError("Manager environment config requires a 'policy' observation group.")
    return groups


__all__ = [
    "ObservationManager",
    "ManagerObservationGroupCfg",
    "ManagerObservationsCfg",
    "ObsTerm",
    "ObservationGroupEntry",
    "ObservationTermCfg",
    "ObservationTermEntry",
    "create_observation_groups",
]


@dataclass
class _SwapObsBuffer:
    """Alternate borrowed outputs so a transition can retain its starting obs."""

    slots: tuple[EnvObs, EnvObs]
    index: int = 0

    @property
    def current(self) -> EnvObs:
        return self.slots[self.index]

    def swap(self) -> None:
        self.index = 1 - self.index


class ObservationManager:
    """Own static groups, observation buffers, and episode-local history.

    Construction allocates all output storage and history from resolved groups
    and batch width. The assembly caller binds the numerical observation
    kernel after program construction, before any publication.
    Public environment results remain stacked; no separate frame API is exposed.
    """

    def __init__(self, groups: dict[str, ObservationGroupEntry], num_envs: int) -> None:
        self.groups = groups
        self.num_envs = num_envs
        specs = {
            name: ObservationGroupSpec(
                gym.spaces.Box(-np.inf, np.inf, (group.size,), dtype=np.float32), group.history_length
            )
            for name, group in groups.items()
        }
        self.spec = ObservationSpec(specs)
        self._frame_spec = ObservationSpec(
            {name: ObservationGroupSpec(group.frame_space) for name, group in self.spec.groups.items()}
        )
        self._history_reset = torch.ones(num_envs, dtype=torch.bool)

        self._stack_buffer = _SwapObsBuffer((self._zeros(self.spec), self._zeros(self.spec)))
        has_history = any(group.history_length > 1 for group in self.groups.values())
        self._frame_buffer = (
            _SwapObsBuffer(
                (
                    self._zeros(self._frame_spec),
                    self._zeros(self._frame_spec),
                )
            )
            if has_history
            else self._stack_buffer
        )
        self._scratch = self._zeros(self._frame_spec)
        self._history = {
            name: torch.zeros((self.num_envs, group.history_length, group.size), dtype=torch.float32)
            for name, group in self.groups.items()
            if group.history_length > 1
        }

    def bind_observe_kernel(self, observe_kernel: Callable[[EnvObs], None]) -> None:
        """Bind the compiled program's executor before the first publication."""
        self._observe_kernel = observe_kernel

    def _zeros(self, spec: ObservationSpec) -> EnvObs:
        return EnvObs(
            {
                name: torch.zeros((self.num_envs, *group.space.shape), dtype=torch.float32)
                for name, group in spec.groups.items()
            }
        )

    @property
    def observation(self) -> EnvObs:
        return self._stack_buffer.current

    @property
    def frame_output(self) -> EnvObs:
        """Current single-frame kernel write target, not a published read view."""
        return self._frame_buffer.current

    def begin_step(self) -> EnvObs:
        """Select the next borrowed output without advancing any history."""
        self._frame_buffer.swap()
        if self._stack_buffer is not self._frame_buffer:
            self._stack_buffer.swap()
        return self._stack_buffer.current

    def reset(self, env_ids: torch.Tensor) -> None:
        """Invalidate only episode-reset histories, not simulator-only resets."""
        self._history_reset.index_fill_(0, env_ids.long(), True)

    def reset_masked(self, world_mask: torch.Tensor) -> None:
        """Invalidate selected histories for the next observation publication."""
        self._history_reset.logical_or_(world_mask)

    def publish_reset(self, env_ids: torch.Tensor) -> None:
        """Repeat-fill explicit episode-reset lanes, leaving other rows untouched."""
        reset = env_ids
        self._observe_kernel(self._scratch)
        frame = self._scratch
        output = self._stack_buffer.current
        for name, group in self.groups.items():
            values = frame[name][reset]
            destination = output[name]
            if group.history_length == 1:
                destination[reset] = values
            else:
                initialized = repeat_observation(values, group.history_length)
                self._history[name][reset] = initialized
                destination[reset] = initialized.flatten(1)
        self._history_reset.index_fill_(0, env_ids.long(), False)

    def commit(self) -> None:
        """Publish exactly one control-step endpoint, repeat-filling new episodes."""
        self._observe_kernel(self.frame_output)
        frame = self._frame_buffer.current
        output = self._stack_buffer.current
        for name, group in self.groups.items():
            values = frame[name]
            destination = output[name]
            if group.history_length == 1:
                if destination is not values:
                    destination.copy_(values)
            else:
                history = self._history[name]
                history.copy_(append_observation(history, values, self._history_reset))
                destination.copy_(history.flatten(1))
        self._history_reset.fill_(False)

    def preview(self, env_ids: torch.Tensor) -> FinalObs:
        """Return owned sparse endpoint windows without committing history."""
        self._observe_kernel(self._scratch)
        frame = self._scratch
        stacked = {}
        for name, group in self.groups.items():
            values = frame[name][env_ids]
            stacked[name] = (
                values
                if group.history_length == 1
                else append_observation(self._history[name][env_ids], values).flatten(1)
            )
        return FinalObs(env_ids.clone(), EnvObs(stacked))
