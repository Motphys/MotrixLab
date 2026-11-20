# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-owned analytic bindings over framework-owned imagination inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager.actions import ActionState
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.kernel_data import Map
from motrix_env_core.manager.random import RandomSource
from motrix_rl_supermimic.task_config import TaskMirror
from motrix_rl_supermimic.world_context import WorldContext
from motrix_rl_supermimic.world_state import WorldState


@dataclass(frozen=True)
class ImaginationInputs:
    """Flat, named views of the caller's explicit predicted physical inputs."""

    frame: torch.Tensor
    state: Map[torch.Tensor]
    context: Map[torch.Tensor]
    actions: Map[ActionState]


@dataclass(frozen=True)
class ImaginationBindings:
    """Task-owned reference objects and analytic simulator views."""

    commands: Map
    sim: Map


@dataclass(kw_only=True)
class ImaginationTerm(ABC):
    """Simulator-free task behavior and evaluation data owned by one build."""

    mirror: TaskMirror | None = None
    evaluation_metadata: dict[str, str | int | float] = field(default_factory=dict)
    return_normalizer: float | None = None

    @abstractmethod
    def bind(self, inputs: ImaginationInputs) -> ImaginationBindings:
        """Derive task reference and analytic views without command lifecycle calls."""
        raise NotImplementedError

    @abstractmethod
    def bind_reference(self, frame: torch.Tensor) -> ImaginationBindings:
        """Bind reference-only goals without requiring predicted physical inputs."""
        raise NotImplementedError

    @abstractmethod
    def valid_starts(self, frame: torch.Tensor, horizon: int) -> torch.Tensor:
        """Return which reference frames support the requested imagined horizon."""
        raise NotImplementedError


class ImaginationRuntime:
    """Build one flat Manager context shared by all numerical rollout heads.

    State and context terms receive named slices of the original flat tensors;
    action history construction preserves gradients through both action samples.
    No physical inputs are synthesized and no mutable contexts are cached.
    """

    def __init__(
        self,
        term: ImaginationTerm,
        world_state: WorldState,
        world_context: WorldContext,
        action_slices: Mapping[str, slice],
        dt: float,
    ) -> None:
        self.term = term
        self.world_state = world_state
        self.world_context = world_context
        self.action_slices = dict(action_slices)
        self.action_dim = sum(part.stop - part.start for part in self.action_slices.values())
        self.dt = dt

    def bind(
        self,
        frame: torch.Tensor,
        *,
        state: torch.Tensor,
        action: torch.Tensor,
        previous: torch.Tensor,
        context: torch.Tensor,
    ) -> ManagerContext:
        frame = frame.reshape(-1)
        n = frame.numel()
        state = state.reshape(n, self.world_state.size)
        context = context.reshape(n, self.world_context.size)
        action = action.reshape(n, self.action_dim)
        previous = previous.reshape(n, self.action_dim)
        actions = Map(
            {
                name: ActionState(
                    torch.stack((previous[..., part], action[..., part]), 1),
                    torch.ones(1, dtype=torch.long, device=action.device),
                )
                for name, part in self.action_slices.items()
            }
        )
        inputs = ImaginationInputs(
            frame=frame,
            state=Map({name: state[..., part] for name, part in self.world_state.slices.items()}),
            context=Map({name: context[..., part] for name, part in self.world_context.slices.items()}),
            actions=actions,
        )
        return self._context(frame, actions, self.term.bind(inputs))

    def bind_reference(self, frame: torch.Tensor) -> ManagerContext:
        frame = frame.reshape(-1)
        return self._context(frame, Map({}), self.term.bind_reference(frame))

    def _context(self, frame: torch.Tensor, actions: Map[ActionState], bindings: ImaginationBindings) -> ManagerContext:
        n = frame.numel()
        device = frame.device
        return ManagerContext(
            env_id=torch.arange(n, device=device),
            actions=actions,
            commands=bindings.commands,
            metrics=Map({}),
            sim=bindings.sim,
            dt=self.dt,
            sim_reset_requested=torch.zeros(n, 1, dtype=torch.bool, device=device),
            terminated=torch.zeros(n, dtype=torch.bool, device=device),
            rand=RandomSource(),
        )


@configclass
class ImaginationCfg(ABC):
    @abstractmethod
    def __call__(self, ctx: BuildContext) -> ImaginationTerm:
        raise NotImplementedError
