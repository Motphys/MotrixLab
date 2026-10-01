# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from dataclasses import fields
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.numba.kernel_data import SharedArray, canonicalize_kernel_data, kernel_data
from motrix_env_core.numba.manager.dispatch import dispatch

if TYPE_CHECKING:
    from motrix_env_core.numba.manager.env import ManagerEnv
    from motrix_env_core.sim.model import ActuatorSpec


@kernel_data
class ActionState:
    """Manager-owned raw-action history shared by every action type.

    Attributes:
        action_queue: Raw policy actions, host ``(N, W, A)`` and kernel
            lane ``(W, A)``; ``W >= 2`` retains current and previous actions.
        action_ptr: Shared one-element integer cursor holding the current
            slot. Only the manager advances it; partial reset leaves it intact.
    """

    action_queue: np.ndarray
    action_ptr: SharedArray

    @dispatch
    def current(self) -> np.ndarray:
        """Return current raw actions: host ``(N, A)``, kernel lane ``(A,)``.

        The returned view shares history storage and does not advance the cursor.
        """
        return self.action_queue[..., self.action_ptr[0], :]

    @dispatch
    def previous(self) -> np.ndarray:
        """Return previous raw actions: host ``(N, A)``, kernel lane ``(A,)``.

        The returned view wraps around the ring without copying data.
        """
        ptr = (self.action_ptr[0] - 1) % self.action_queue.shape[-2]
        return self.action_queue[..., ptr, :]


class ActionTerm(abc.ABC):
    """Extensible host-side action pipeline with kernel-bound state."""

    def __init__(self, action_space: gym.spaces.Box, state: ActionState) -> None:
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("ActionTerm action_space must be a gym.spaces.Box.")
        if not isinstance(state, ActionState):
            raise TypeError("ActionTerm state must be an ActionState.")
        self.action_space = action_space
        self.state = canonicalize_kernel_data(state, context="ActionTerm state")

    def process(self, actions: np.ndarray) -> np.ndarray | None:
        """Record raw actions and process one batched action slice."""
        state = self.state
        ptr = (int(state.action_ptr[0]) + 1) % state.action_queue.shape[1]
        state.action_ptr[0] = ptr
        state.action_queue[:, ptr] = actions
        return self._process(actions)

    @abc.abstractmethod
    def _process(self, actions: np.ndarray) -> np.ndarray | None:
        """Process one batched action slice into route-local controls."""

    def reset(self, env_ids: np.ndarray) -> None:
        """Clear raw history and reset action-specific state for selected rows."""
        self.state.action_queue[env_ids] = 0.0
        self._reset(env_ids)

    @abc.abstractmethod
    def _reset(self, env_ids: np.ndarray) -> None:
        """Reset action-specific state for selected environment rows."""


@configclass(kw_only=True)
class ActionCfg(abc.ABC):
    """Configuration that creates one environment-local action term."""

    actuator_names: tuple[str, ...] | None = None
    """Actuator route: ``None`` selects all actuators; ``()`` selects none."""

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
