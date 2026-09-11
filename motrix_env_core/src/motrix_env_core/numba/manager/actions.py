# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from dataclasses import fields
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from motrix_env_core.config import configclass

if TYPE_CHECKING:
    from motrix_env_core.numba.manager.env import ManagerEnv
    from motrix_env_core.sim.backend import ActuatorSpec
    from motrix_env_core.sim.read import PhysicsReadProgram


class ActionTerm(abc.ABC):
    """Environment-local KernelData action pipeline with persistent runtime state."""

    @abc.abstractmethod
    def action_space(self, env: ManagerEnv, actuators: tuple[ActuatorSpec, ...] | None) -> gym.spaces.Box:
        """Return the unbatched action space."""

    @abc.abstractmethod
    def process(self, actions: np.ndarray) -> np.ndarray | None:
        """Process one action batch and return route-local actuator controls."""

    @abc.abstractmethod
    def reset(self, env_ids: np.ndarray) -> None:
        """Reset persistent action state for selected environments."""

    def prepare(self, sim_data: PhysicsReadProgram) -> None:
        """Snapshot simulator reads immediately before processing an action."""
        del sim_data


@configclass(kw_only=True)
class ActionCfg(abc.ABC):
    """Configuration that creates one environment-local action term."""

    actuator_names: tuple[str, ...] | None = None

    @abc.abstractmethod
    def __call__(self, env: ManagerEnv, actuators: tuple[ActuatorSpec, ...] | None) -> ActionTerm:
        """Create the concrete KernelData action term."""


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
