# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Logical context contract and runtime/build contexts for manager MDP functions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.manager.random import RandomSource

if TYPE_CHECKING:
    from motrix_env_core.base import EnvCfg
    from motrix_env_core.manager.actions import ActionTerm
    from motrix_env_core.manager.commands import CommandTerm
    from motrix_env_core.sim import SimModel


@dataclass(slots=True)
class ManagerContext:
    """Framework-owned manager data available to every compiled term.

    Terms receive this host instance directly: attribute access, ``Map``
    string lookups, and term-state methods trace under Dynamo with guards on
    object identity. Every entry is the live host object — the actual
    ``CommandTerm``/``ActionState`` dataclass instances, zero-copy tensor
    views over the simulator read arena, and writable torch views over the
    per-environment metrics arrays — so kernels and host lifecycle observe one
    state. The class is deliberately not frozen: terms write flags with
    augmented assignment (``ctx.sim_reset_requested |= mask``), whose in-place
    tensor ``__ior__`` rebinds the attribute slot to the same tensor.
    """

    env_id: torch.Tensor
    actions: Map
    commands: Map
    metrics: Map
    sim: Map
    dt: float
    # Writable per-lane flag: a command term sets it inside a fused kernel to
    # request simulator-state rematerialization for that lane after the
    # transition. It is not an episode reset: episode bookkeeping and
    # action-term state are untouched (see CommandTerm.advance).
    sim_reset_requested: torch.Tensor
    # Read-only mirror of the latest transition's termination union, backing
    # episode-boundary statistics inside reset hooks.
    terminated: torch.Tensor
    # Stateless capability, not tensor kernel data. All consumers on a device
    # share Torch's default stream; separate contexts do not imply RNG isolation.
    rand: RandomSource


@dataclass(frozen=True, slots=True)
class BuildContext:
    """Explicit construction inputs and independently owned terms, without an environment.

    ManagerEnv constructs terms in this order: commands (declaration order),
    command metric materialization/validation, simulator resets, actions,
    rewards, terminations, then observations.

    Availability during each factory call:

    * Command: ``actions`` is empty; ``commands`` contains only preceding
      commands, whose metrics have not yet been materialized. Later commands
      and the command currently being constructed are unavailable.
    * Simulator reset and action: ``commands`` is complete and validated;
      ``actions`` is empty throughout all action factories. Action construction
      publishes the collection only after every action factory returns.
    * Reward, termination, and observation: both collections are complete.

    Treat dependencies as read-only. These mappings are not a lazy resolver:
    missing entries are not constructed on access. A factory must not retain
    an early mapping expecting it to become the final action collection.
    Contexts constructed outside ManagerEnv must receive their dependencies
    explicitly; this ordering only describes ManagerEnv's construction flow.

    ``num_envs`` is the row count of the instance being constructed, not the
    batch size of a later rollout. ``seed`` initializes that owner's
    construction/reset generators; numerical terms access the process-default
    Torch RNG through the explicit ``ManagerContext.rand`` capability.
    """

    cfg: EnvCfg
    actions: Mapping[str, ActionTerm]
    commands: Mapping[str, CommandTerm]
    # Compiled scene model when one exists; model-free task builds construct
    # terms without it.
    model: SimModel | None = None
    num_envs: int = field(kw_only=True)
    seed: int = field(kw_only=True)


__all__ = ["BuildContext", "ManagerContext"]
