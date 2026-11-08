# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

from motrix_env_core.config import configclass

if TYPE_CHECKING:
    from motrix_env_core.manager.context import ManagerContext
    from motrix_env_core.manager.env import ManagerEnv


@dataclass(frozen=True, slots=True)
class CommandTerm(abc.ABC):
    """Environment-local kernel-data command pipeline with persistent runtime state.

    Contract: every command term carries a ``command`` field — one
    ``(num_envs, command_dim)`` float tensor holding the lane's goal vector.
    Torch executor hooks operate on the whole-batch tensors of the live term
    instance, and generic consumers (for example
    ``mdp.observations.CommandObsCfg``) read the goal by field name.

    All three hooks are instance methods dispatched by the executor against
    the live term (the compiled graphs guard on its identity). Inside
    ``reset``, ``ctx.reset_ids`` carries this pass's compact reset rows,
    ``ctx.reset_episode`` marks the episode resets among them, and
    ``ctx.terminated`` mirrors the latest transition's termination union —
    episode-boundary statistics (adaptive sampling, curricula) are expressed
    as gathered tensor math and published through ``metric()`` fields,
    including 0-d scalar metrics for batch-level gauges.
    """

    command: torch.Tensor

    @abc.abstractmethod
    def update(self, ctx: ManagerContext) -> None:
        """Update the derived command data in the fused kernel."""

    @abc.abstractmethod
    def advance(self, ctx: ManagerContext) -> None:
        """Advance persistent command state in the transition kernel.

        Setting ``ctx.sim_reset_requested`` for the lane requests simulator-state
        rematerialization after the transition: the lane joins the step's
        reset-kernel run (:meth:`reset` on the pass's compact rows followed by
        the configured sim reset terms) together with any episode resets, and
        kernel inputs are reread for it. This is not an episode reset: episode
        bookkeeping, host lifecycle resets, and action-term state are
        untouched.
        """

    @abc.abstractmethod
    def reset(self, ctx: ManagerContext) -> None:
        """Reset persistent command state in the reset kernel.

        Runs for episode-reset and mid-episode sim-reset rows alike (gathered
        via ``ctx.reset_ids``); gate episode-boundary behavior on
        ``ctx.reset_episode``.
        """


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
