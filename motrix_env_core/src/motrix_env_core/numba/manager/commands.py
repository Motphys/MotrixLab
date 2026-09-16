# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from motrix_env_core.config import configclass
from motrix_env_core.numba.kernel_data import kernel_data
from motrix_env_core.numba.manager.dispatch import dispatch

if TYPE_CHECKING:
    from motrix_env_core.numba.manager.context import ManagerContext
    from motrix_env_core.numba.manager.env import ManagerEnv


@dataclass
class ResetContext:
    """Host-side inputs exposed to a command during selected-environment reset."""

    env_ids: np.ndarray
    terminated: np.ndarray
    metrics: dict[str, Any]


@kernel_data
class CommandTerm(abc.ABC):
    """Environment-local KernelData command pipeline with persistent runtime state.

    Contract: every command term carries a ``command`` field — one
    ``(num_envs, command_dim)`` float array holding the lane's goal vector.
    The kernel lowering hands each lane a writable row view, and generic
    consumers (for example ``mdp.observations.CommandObsCfg``)
    read it by field name.
    """

    command: np.ndarray

    @dispatch
    @abc.abstractmethod
    def update(self, ctx: ManagerContext) -> None:
        """Update one lane's derived command data in the fused kernel."""

    @dispatch
    @abc.abstractmethod
    def advance(self, ctx: ManagerContext) -> None:
        """Advance one lane's persistent command state in the transition kernel.

        Setting ``ctx.sim_reset_requested`` for the lane requests simulator-state
        rematerialization after the transition: the lane joins the step's
        reset-kernel run (:meth:`reset_env` followed by the configured sim
        reset terms) together with any episode resets, and kernel inputs are
        reread for it. This is not an episode reset: episode bookkeeping, host
        lifecycle resets, and action-term state are untouched.
        """

    @abc.abstractmethod
    def reset(self, ctx: ResetContext) -> None:
        """Prepare host-side data for selected environment resets."""

    @dispatch
    @abc.abstractmethod
    def reset_env(self, ctx: ManagerContext) -> None:
        """Reset one lane's persistent command state in the reset kernel."""

    def on_transition(self) -> None:
        """Update host-side command state after a transition kernel."""


@configclass(kw_only=True)
class CommandCfg(abc.ABC):
    """Configuration that creates one environment-local command term."""

    @abc.abstractmethod
    def __call__(self, env: ManagerEnv) -> CommandTerm:
        """Create the concrete KernelData command term."""


@configclass
class ManagerCommandsCfg:
    """Typed declaration group for one environment's command terms."""

    def to_dict(self) -> dict[str, CommandCfg]:
        """Return command configs keyed by their declaration names."""
        term_cfgs: dict[str, CommandCfg] = {}
        for term_field in fields(self):
            term_cfg = getattr(self, term_field.name)
            if not isinstance(term_cfg, CommandCfg):
                raise TypeError(
                    f"Manager command {term_field.name!r} must be a CommandCfg, got {type(term_cfg).__name__}."
                )
            term_cfgs[term_field.name] = term_cfg
        return term_cfgs
