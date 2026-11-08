# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import gymnasium as gym
import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager.kernel_data import validate_kernel_data

if TYPE_CHECKING:
    from motrix_env_core.manager.env import ManagerEnv
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
        ptr = (int(state.action_ptr[0]) + 1) % state.action_queue.shape[1]
        state.action_ptr[0] = ptr
        state.action_queue[:, ptr] = actions
        return self._process(actions)

    @abc.abstractmethod
    def _process(self, actions: torch.Tensor) -> torch.Tensor | None:
        """Process one batched tensor action slice into route-local controls."""

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear raw history and reset action-specific state for selected rows."""
        self.state.action_queue[env_ids] = 0.0
        self._reset(env_ids)

    @abc.abstractmethod
    def _reset(self, env_ids: torch.Tensor) -> None:
        """Reset action-specific state for selected environment rows."""


@configclass(kw_only=True)
class ActionCfg(abc.ABC):
    """Configuration that creates one environment-local action term."""

    # Actuator route: ``None`` selects all actuators; ``()`` selects none.
    actuator_names: tuple[str, ...] | None = None

    @abc.abstractmethod
    def __call__(self, env: ManagerEnv, actuators: tuple[ActuatorSpec, ...] | None) -> ActionTerm:
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
