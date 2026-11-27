# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Framework-owned imagination context assembly over task-owned derivation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import torch

from motrix_env_core.config import configclass
from motrix_env_core.manager.actions import ActionState
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.random import RandomSource
from motrix_rl_supermimic.world_context import WorldContext
from motrix_rl_supermimic.world_state import WorldStateManager


class TaskMirror(Protocol):
    """Task symmetry used together for world-model window augmentation."""

    def state(self, state: torch.Tensor) -> torch.Tensor: ...

    def action(self, action: torch.Tensor) -> torch.Tensor: ...

    def context(self, context: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class ReferenceView:
    """The reference contract numerical terms may read from ``ctx.commands["reference"]``.

    Live reference commands expose the same fields with the same semantics;
    imagination synthesizes this view so the identical term functions run in
    physics and imagination without task-authored mirrors or bindings.

    ``physical_state`` is the flat canonical world state, ``context`` the flat
    world context, and ``command`` the goal vector derived from ``frame``.
    """

    frame: torch.Tensor
    context: torch.Tensor
    command: torch.Tensor
    physical_state: torch.Tensor


@dataclass(kw_only=True)
class ImaginationTerm(ABC):
    """Simulator-free task derivation owned by one build."""

    mirror: TaskMirror | None = None
    evaluation_metadata: dict[str, str | int | float] = field(default_factory=dict)
    return_normalizer: float | None = None

    @abstractmethod
    def goals(self, frame: torch.Tensor) -> torch.Tensor:
        """Derive the reference goal vector for each frame index."""
        raise NotImplementedError

    def reference_view(self, frame: torch.Tensor, context: torch.Tensor, state: torch.Tensor) -> ReferenceView:
        """Bind one independent numerical reference; tasks may add typed derived fields."""
        return ReferenceView(frame, context, self.goals(frame), state)

    @abstractmethod
    def valid_starts(self, frame: torch.Tensor, horizon: int) -> torch.Tensor:
        """Return which reference frames support the requested imagined horizon."""
        raise NotImplementedError


class ImaginationContextFactory:
    """Build one flat Manager context shared by all numerical rollout heads.

    The reference follows the :class:`ReferenceView` contract; simulator views
    come from the world state's :meth:`WorldStateManager.views` (the inverse of
    capture). Action history construction preserves gradients through both
    action samples. No physical inputs are synthesized and no mutable contexts
    are cached.

    Besides imagination rollouts, :meth:`context` is the framework's general
    standalone-context builder: any consumer outside the manager env (batch
    evaluators, exported inference) that can supply the flat world state,
    context, and action tensors runs the identical numerical terms through it.
    """

    def __init__(
        self,
        term: ImaginationTerm,
        world_state: WorldStateManager,
        world_context: WorldContext,
        action_slices: Mapping[str, slice],
        dt: float,
        reference_command: str,
    ) -> None:
        self.reference_command = reference_command
        self.term = term
        self.world_state = world_state
        self.world_context = world_context
        self.action_slices = dict(action_slices)
        self.action_dim = sum(part.stop - part.start for part in self.action_slices.values())
        self.dt = dt

    def context(
        self,
        frame: torch.Tensor,
        *,
        state: torch.Tensor,
        action: torch.Tensor,
        previous: torch.Tensor,
        xi: torch.Tensor,
    ) -> ManagerContext:
        """Assemble raw tensors into a runnable Manager context.

        ``action`` lands in the slot ``ActionState.current()`` returns — it is
        the action the observation terms should see, typically the last applied
        one — while ``previous`` fills the history slot behind it.
        """
        frame = frame.reshape(-1)
        n = frame.numel()
        state = state.reshape(n, self.world_state.size)
        xi = xi.reshape(n, self.world_context.size)
        action = action.reshape(n, self.action_dim)
        previous = previous.reshape(n, self.action_dim)
        actions = {
            name: ActionState(
                torch.stack((previous[..., part], action[..., part]), 1),
                torch.ones(1, dtype=torch.long, device=action.device),
            )
            for name, part in self.action_slices.items()
        }
        reference = self.term.reference_view(frame, xi, state)
        return self._assemble(frame, actions, {self.reference_command: reference}, self.world_state.views(state))

    def reference_context(self, frame: torch.Tensor) -> ManagerContext:
        frame = frame.reshape(-1)
        empty = frame.new_empty((frame.numel(), 0), dtype=torch.float32)
        reference = ReferenceView(frame, empty, self.term.goals(frame), empty)
        return self._assemble(frame, {}, {self.reference_command: reference}, {})

    def _assemble(
        self,
        frame: torch.Tensor,
        actions: Mapping[str, ActionState],
        commands: Mapping[str, ReferenceView],
        sim: Mapping[str, torch.Tensor],
    ) -> ManagerContext:
        n = frame.numel()
        device = frame.device
        return ManagerContext(
            env_id=torch.arange(n, device=device),
            actions=actions,
            commands=commands,
            metrics={},
            sim=sim,
            dt=self.dt,
            sim_reset_requested=torch.zeros(n, 1, dtype=torch.bool, device=device),
            terminated=torch.zeros(n, dtype=torch.bool, device=device),
            # The standalone context has no reward total to mirror; the
            # imagined reward is returned by the task heads, not written
            # into the context.
            reward=torch.zeros(n, dtype=torch.float32, device=device),
            rand=RandomSource(),
        )


@configclass
class ImaginationCfg(ABC):
    @abstractmethod
    def __call__(self, ctx: BuildContext) -> ImaginationTerm:
        raise NotImplementedError
