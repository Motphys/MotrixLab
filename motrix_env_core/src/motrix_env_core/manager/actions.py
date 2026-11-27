# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np
import torch

import motrix_env_core.manager.context as context
from motrix_env_core.config import configclass
from motrix_env_core.manager.context import BuildContext
from motrix_env_core.manager.kernel_data import validate_kernel_data
from motrix_env_core.sim.write import CtrlTargetsWrite

if TYPE_CHECKING:
    from motrix_env_core.sim.backend import SimBackend
    from motrix_env_core.sim.model import ActuatorSpec


@dataclass(frozen=True, slots=True)
class ActionState:
    """Manager-owned raw-action history shared by every action type.

    Array fields are zero-copy torch tensors over host-owned numpy arenas, so
    host code may write through ``tensor.numpy()`` views while kernels write
    through the tensors — both see one backing buffer.

    Attributes:
        action_queue: Raw policy actions, host ``(N, W, A)`` and kernel
            lane ``(W, A)``; ``W >= 2`` retains current and previous actions.
        action_ptr: Shared one-element integer cursor holding the current
            slot. Only the manager advances it; partial reset leaves it intact.
    """

    action_queue: torch.Tensor
    action_ptr: torch.Tensor

    def current(self) -> torch.Tensor:
        """Return current raw actions: host ``(N, A)``, kernel lane ``(A,)``.

        The returned view shares history storage and does not advance the cursor.
        The gather form keeps the index on tensors so compiled terms trace it
        without a data-dependent scalar extraction.
        """
        queue = self.action_queue
        index = self.action_ptr.view(1, 1, 1).expand(queue.shape[0], 1, queue.shape[-1])
        return queue.gather(1, index).squeeze(1)

    def previous(self) -> torch.Tensor:
        """Return previous raw actions: host ``(N, A)``, kernel lane ``(A,)``.

        The returned view wraps around the ring without copying data.
        """
        queue = self.action_queue
        prev = (self.action_ptr.view(1, 1) - 1) % queue.shape[1]
        index = prev.expand(queue.shape[0], 1, queue.shape[-1])
        return queue.gather(1, index).squeeze(1)


class ActionTerm(abc.ABC):
    """Extensible host-side action pipeline with kernel-bound state."""

    def __init__(self, action_space: gym.spaces.Box, state: ActionState) -> None:
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("ActionTerm action_space must be a gym.spaces.Box.")
        if not isinstance(state, ActionState):
            raise TypeError("ActionTerm state must be an ActionState.")
        self.action_space = action_space
        self.state = validate_kernel_data(state, context="ActionTerm state")

    def process(self, actions: torch.Tensor) -> torch.Tensor | None:
        """Record raw tensor actions and process one batched action slice."""
        state = self.state
        ptr = state.action_ptr
        ptr.add_(1).remainder_(state.action_queue.shape[1])
        state.action_queue.index_copy_(1, ptr, actions.unsqueeze(1))
        return self._process(actions)

    @abc.abstractmethod
    def _process(self, actions: torch.Tensor) -> torch.Tensor | None:
        """Process one batched tensor action slice into route-local controls."""

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear raw history and reset action-specific state for selected rows."""
        self.state.action_queue.index_fill_(0, env_ids.long(), 0)
        self._reset(env_ids)

    def reset_masked(self, world_mask: torch.Tensor) -> None:
        """Clear selected action histories and reset action-specific state."""
        self.state.action_queue.masked_fill_(world_mask[:, None, None], 0)
        self._reset_masked(world_mask)

    def _reset_masked(self, world_mask: torch.Tensor) -> None:
        """Optional device-side counterpart of the action-specific reset."""
        raise NotImplementedError

    @abc.abstractmethod
    def _reset(self, env_ids: torch.Tensor) -> None:
        """Reset action-specific state for selected environment rows."""


@configclass(kw_only=True)
class ActionCfg(abc.ABC):
    """Configuration that creates one environment-local action term."""

    # Actuator route: ``None`` selects all actuators; ``()`` selects none.
    actuator_names: tuple[str, ...] | None = None

    @abc.abstractmethod
    def __call__(self, ctx: context.BuildContext, actuators: tuple[ActuatorSpec, ...] | None) -> ActionTerm:
        """Assemble the action space, persistent state, and host callbacks."""


@configclass
class ManagerActionsCfg:
    """Typed declaration group for one environment's action terms."""

    def to_dict(self) -> dict[str, ActionCfg]:
        """Return action configs keyed by their declaration names."""
        term_cfgs: dict[str, ActionCfg] = {}
        for term_field in fields(self):
            term_cfg = getattr(self, term_field.name)
            if not isinstance(term_cfg, ActionCfg):
                raise TypeError(
                    f"Manager action {term_field.name!r} must be an ActionCfg, got {type(term_cfg).__name__}."
                )
            term_cfgs[term_field.name] = term_cfg
        return term_cfgs


class ActionManager:
    """Build and execute the configured manager action terms."""

    def __init__(
        self,
        cfgs: dict[str, ActionCfg],
        build_context: BuildContext,
        sim: SimBackend,
        actuators: tuple[ActuatorSpec, ...],
    ) -> None:
        self.cfgs = cfgs
        self.actuators = self._resolve_actuators(cfgs, actuators)
        self.writes = sim.compile_action_writes(
            {name: CtrlTargetsWrite(cfg.actuator_names) for name, cfg in cfgs.items() if cfg.actuator_names != ()}
        )
        self.terms: dict[str, ActionTerm] = {}
        for name, cfg in cfgs.items():
            term = cfg(build_context, self.actuators[name])
            if not isinstance(term, ActionTerm):
                raise TypeError(f"Manager action {name!r} must return an ActionTerm.")
            queue = term.state.action_queue
            if queue.ndim != 3 or queue.shape[1] < 2:
                raise ValueError(f"Manager action {name!r} state.action_queue must have shape (N, W, A), W >= 2.")
            if queue.shape[2] != term.action_space.shape[0]:
                raise ValueError(f"Manager action {name!r} action dimension must match action space.")
            self.terms[name] = term
        self.space, self.slices = self._build_action_space()

    @staticmethod
    def _resolve_actuators(cfgs, model_actuators):
        routes, owners = {}, {}
        by_name = {spec.name: spec for spec in model_actuators}
        for term_name, cfg in cfgs.items():
            if cfg.actuator_names == ():
                routes[term_name] = None
                continue
            names = cfg.actuator_names or tuple(by_name)
            if not names:
                raise ValueError(f"Manager action {term_name!r} actuator route must not be empty.")
            if len(set(names)) != len(names):
                raise ValueError(f"Manager action {term_name!r} actuator names must be unique.")
            actuators = []
            for actuator_name in names:
                try:
                    actuator = by_name[actuator_name]
                except KeyError:
                    raise ValueError(
                        f"Manager action {term_name!r} references unknown actuator {actuator_name!r}."
                    ) from None
                if actuator_name in owners:
                    raise ValueError(
                        f"Manager actions {owners[actuator_name]!r} and {term_name!r} both control "
                        f"actuator {actuator_name!r}."
                    )
                owners[actuator_name] = term_name
                actuators.append(actuator)
            routes[term_name] = tuple(actuators)
        return routes

    def _build_action_space(self):
        lows, highs, slices = [], [], {}
        offset = 0
        for name, term in self.terms.items():
            space = term.action_space
            if not isinstance(space, gym.spaces.Box):
                raise TypeError(f"Manager action {name!r} must produce a gym.spaces.Box.")
            if len(space.shape) != 1 or space.shape[0] <= 0:
                raise ValueError(f"Manager action {name!r} space must have a non-empty one-dimensional shape.")
            if space.dtype != np.dtype(np.float32):
                raise TypeError(f"Manager action {name!r} space dtype must be float32, got {space.dtype}.")
            size = space.shape[0]
            slices[name] = slice(offset, offset + size)
            lows.append(space.low)
            highs.append(space.high)
            offset += size
        return gym.spaces.Box(np.concatenate(lows), np.concatenate(highs), dtype=np.float32), slices

    def apply(self, actions: torch.Tensor) -> None:
        expected_shape = (actions.shape[0], self.space.shape[0])
        if actions.shape != expected_shape:
            raise ValueError(f"Expected action shape {expected_shape}, got {tuple(actions.shape)}.")
        for name, term in self.terms.items():
            controls = term.process(actions[:, self.slices[name]])
            actuators = self.actuators[name]
            if actuators is None:
                assert controls is None
                continue
            assert controls is not None
            self.writes.buffer(name).copy_(controls)
        self.writes.execute()

    def reset(self, env_ids: torch.Tensor) -> None:
        for term in self.terms.values():
            term.reset(env_ids)

    @property
    def supports_masked_reset(self) -> bool:
        return all(type(term)._reset_masked is not ActionTerm._reset_masked for term in self.terms.values())

    def reset_masked(self, world_mask: torch.Tensor) -> None:
        for term in self.terms.values():
            term.reset_masked(world_mask)

    def clear_targets_masked(self, world_mask: torch.Tensor) -> None:
        for name, actuators in self.actuators.items():
            if actuators is not None:
                self.writes.buffer(name).masked_fill_(world_mask[:, None], 0)

    def clear_targets(self, env_ids: torch.Tensor) -> None:
        for name, actuators in self.actuators.items():
            if actuators is not None:
                self.writes.buffer(name).index_fill_(0, env_ids, 0.0)


__all__ = ["ActionManager"]
