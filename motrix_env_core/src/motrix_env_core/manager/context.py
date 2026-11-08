# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Logical context contract and runtime/build contexts for manager MDP functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.manager.rand import RandValue

if TYPE_CHECKING:
    from motrix_env_core.base import EnvCfg
    from motrix_env_core.manager.env import ManagerEnv
    from motrix_env_core.sim import SimModel

T = TypeVar("T")


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
    rand: RandValue
    sim: Map
    dt: float
    # Writable per-lane flag: a command term sets it inside a fused kernel to
    # request simulator-state rematerialization for that lane after the
    # transition. It is not an episode reset: episode bookkeeping and
    # action-term state are untouched (see CommandTerm.advance).
    sim_reset_requested: torch.Tensor
    # Read-only flags valid inside reset hooks. ``reset_ids`` are the compact
    # rows of the current reset pass (dynamic length K, aligned with every
    # gathered tensor and compact staging buffer); ``reset_episode`` marks
    # which of those rows are episode resets (False for mid-episode
    # sim-reset rows), so a ``reset`` hook can apply episode-boundary
    # statistics selectively.
    reset_ids: torch.Tensor
    reset_episode: torch.Tensor
    # Read-only mirror of the latest transition's termination union, backing
    # episode-boundary statistics inside reset hooks.
    terminated: torch.Tensor


class BuildContext:
    """NumPy build context retained for the current runtime builder.

    Handed to ``__call__`` after the model and read program are compiled.
    Exposes a curated surface of the environment — the term's address, the
    compiled model (including ``bodies``), and the already-created
    action/command term registries. Terms must not reach past
    this surface into environment internals.

    Note: reset terms are built before the read program and the action and
    command registries exist; their build contexts must not touch
    ``action_terms``/``command_terms``.
    """

    def __init__(self, env: ManagerEnv, address: str) -> None:
        self._env = env
        self.address = address

    @property
    def cfg(self) -> EnvCfg:
        return self._env.cfg

    @property
    def model(self) -> SimModel:
        return self._env.model

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def num_actuators(self) -> int:
        return self._env.num_actuators

    @property
    def action_terms(self) -> Any:
        return self._env.action_terms

    @property
    def command_terms(self) -> Any:
        return self._env.command_terms


__all__ = ["BuildContext", "ManagerContext"]
