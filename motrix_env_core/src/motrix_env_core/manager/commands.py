# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from dataclasses import dataclass, fields

import torch

import motrix_env_core.manager.context as context
from motrix_env_core.config import configclass
from motrix_env_core.manager.kernel_data import validate_kernel_data
from motrix_env_core.manager.metrics import collect_metrics, materialize_metrics


# Keep inherited fields out of base-class slots: Torch 2.7 Dynamo only resolves
# slots listed on the concrete class, while Python 3.11+ omits inherited slots.
@dataclass(frozen=True)
class CommandTerm(abc.ABC):
    """Environment-local kernel-data command pipeline with persistent runtime state.

    Contract: every command term carries a ``command`` field — one
    ``(num_envs, command_dim)`` float tensor holding the lane's goal vector.
    Torch executor hooks operate on the whole-batch tensors of the live term
    instance, and generic consumers (for example
    ``mdp.observations.CommandObsCfg``) read the goal by field name.

    All three hooks are instance methods dispatched by the executor against
    the live term (the compiled graphs guard on its identity). Inside
    ``reset``, explicit ``env_ids`` selects this pass's rows and the aligned
    ``episode_reset`` marks episode resets among them;
    ``ctx.terminated`` mirrors the latest transition's termination union —
    episode-boundary statistics (adaptive sampling, curricula) are expressed
    as gathered tensor math and published through ``metric()`` fields,
    including 0-d scalar metrics for batch-level gauges.
    """

    command: torch.Tensor

    @abc.abstractmethod
    def update(self, ctx: context.ManagerContext) -> None:
        """Update the derived command data in the fused kernel."""

    @abc.abstractmethod
    def advance(self, ctx: context.ManagerContext) -> None:
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
    def reset(self, ctx: context.ManagerContext, env_ids: torch.Tensor, episode_reset: torch.Tensor) -> None:
        """Reset persistent command state in the reset kernel.

        Args:
            ctx: Whole-batch (N rows) task data; gather selected state with env_ids.
            env_ids: Original environment indices for this reset pass, shape (K,).
            episode_reset: Bool (K,) aligned with env_ids. True means a full
                episode reset, including initialization and explicit restarts;
                False means simulator-only reset within the same episode.
                This is not a termination flag or proof a previous episode ended.

        Both kinds reset command state. Gate episode-specific statistics on
        episode_reset; simulator reset terms subsequently process all env_ids.
        """


@configclass(kw_only=True)
class CommandCfg(abc.ABC):
    @abc.abstractmethod
    def __call__(self, ctx: context.BuildContext) -> CommandTerm:
        """Construct one independently owned command term without an environment."""
        raise NotImplementedError


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


class CommandManager:
    """Build and own the configured command terms and their metrics."""

    def __init__(
        self,
        cfgs: dict[str, CommandCfg],
        build_context: context.BuildContext,
        num_envs: int,
    ) -> None:
        # Command factories must be independent: the context they receive
        # carries an empty commands mapping (and no actions) by contract.
        self.cfgs = cfgs
        self.terms: dict[str, CommandTerm] = {}
        self.metrics: dict[str, torch.Tensor] = {}
        for name, cfg in cfgs.items():
            term = cfg(build_context)
            term = validate_kernel_data(
                materialize_metrics(term, num_envs, context=f"Manager command {name!r}"),
                context=f"Manager command {name!r}",
            )
            self.terms[name] = term
            for metric_name, value in collect_metrics(
                term, num_envs, context=f"Manager command {name!r}", context_path=("commands", name)
            ):
                if metric_name in self.metrics:
                    raise ValueError(f"Duplicate per-environment metric name: {metric_name!r}")
                self.metrics[metric_name] = value


__all__ = ["CommandCfg", "CommandManager", "CommandTerm", "ManagerCommandsCfg"]
