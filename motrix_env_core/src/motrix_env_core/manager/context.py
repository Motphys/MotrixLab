# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Logical context contract and runtime/build contexts for manager MDP functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from motrix_env_core.manager.kernel_data import Map, kernel_data
from motrix_env_core.manager.rand import RandValue

if TYPE_CHECKING:
    from motrix_env_core.base import EnvCfg
    from motrix_env_core.manager.env import ManagerEnv
    from motrix_env_core.sim import SimModel

T = TypeVar("T")


@kernel_data
class ManagerContext:
    """Framework-owned manager data available to every compiled term."""

    env_id: np.ndarray
    actions: Map
    commands: Map
    metrics: Map[np.ndarray]
    rand: RandValue
    sim: Map
    dt: np.float32
    # Writable per-lane flag: a command term sets it inside a fused kernel to
    # request simulator-state rematerialization for that lane after the
    # transition. It is not an episode reset: episode bookkeeping and
    # action-term state are untouched (see CommandTerm.advance).
    sim_reset_requested: np.ndarray


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
