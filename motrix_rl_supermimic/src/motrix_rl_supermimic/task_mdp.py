# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""TaskMDP: the stable ABI between the MBRL learner and a task.

A task contributes two things to the world-model MBRL loop: a physics
environment that answers *how to run* the MDP (transitions, reset, hard
termination, rendering) and a TaskMDP that answers *how to evaluate any
state — real or imagined*. Deprived of transitions, the same TaskMDP serves
both the real path (simulator states) and the imagination path (world-model
rollouts), which is the whole point of the contract.

Design discipline (structural, enforced by review and the toy tests): pure
functions, no env/simulator handles, no hidden state, device-agnostic,
``torch.compile``-friendly. Construction inputs must be serializable so the
collector and learner processes can each instantiate one from the same
config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

__all__ = ["StateSpec", "TaskMDP", "optional_task_hook"]

_OPTIONAL_HOOKS = (
    "reference_action",
    "mirror_state",
    "mirror_action",
    "mirror_context",
    "imagined_reward",
    "transport_fields",
    "curriculum_state",
    "load_curriculum",
    "evaluation_metadata",
    "return_normalizer",
)


def optional_task_hook(mdp: "TaskMDP", name: str):
    """Resolve an optional TaskMDP extension, or None when not implemented.

    Protocol method stubs must not be mistaken for implementations: a hook is
    present only when the concrete task class defines it itself.
    """
    if name not in _OPTIONAL_HOOKS:
        raise ValueError(f"{name} is not an optional TaskMDP hook")
    feature = getattr(type(mdp), name, None)
    if feature is None or feature is getattr(TaskMDP, name, None):
        return None
    return getattr(mdp, name)


@dataclass(frozen=True)
class StateSpec:
    """Static shape contract of one task's canonical state tuple.

    The canonical state is the flat physical tuple ``(p, quat, v, w, q, qd)``
    (root position ``(..., 3)``, root rotation as ``xyzw`` quaternion
    ``(..., 4)``, root linear/angular velocity ``(..., 3)`` each, joint
    positions/velocities ``(..., num_joints)`` each) flattened on the last
    axis. ``state_groups`` partitions ``state_dim`` into these component
    widths and gives each an error floor — the scale below which a world-model
    prediction error counts as solved — defaulting to 1.0 per group.

    Attributes:
        state_dim: Flat canonical state dimension.
        action_dim: Action dimension.
        context_dim: Task context dimension (motion-goal state); 0 for
            context-free tasks.
        observation_dim: Policy observation dimension.
        critic_observation_dim: Privileged critic input dimension.
        command_dim: Per-step commanded-goal dimension.
        state_groups: ``(width, err_floor)`` entries partitioning
            ``state_dim``; empty means one group covering the whole state.
    """

    state_dim: int
    action_dim: int
    context_dim: int
    observation_dim: int
    critic_observation_dim: int
    command_dim: int
    state_groups: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        if self.state_groups and sum(w for w, _ in self.state_groups) != self.state_dim:
            raise ValueError(
                f"state_groups widths {tuple(w for w, _ in self.state_groups)} do not sum "
                f"to state_dim {self.state_dim}."
            )

    def groups(self) -> tuple[tuple[int, float], ...]:
        """State groups with the default single-group fallback resolved."""
        return self.state_groups if self.state_groups else ((self.state_dim, 1.0),)


@runtime_checkable
class TaskMDP(Protocol):
    """Analytic MDP heads of a task, shared by real and imagined states.

    Every member maps onto one imagination-loop consumption point; anything
    imagination does not need (transitions, reset, hard termination,
    rendering) stays in the environment. There is deliberately no ``step``:
    transitions belong to the simulator on the real path and to the world
    model on the imagination path.

    ``state`` is always the flat canonical tensor ``(..., state_dim)``;
    ``frame`` is the task-owned per-environment reference-frame index
    ``(...,)`` integer tensor; ``context`` is ``(..., context_dim)``.
    """

    spec: StateSpec

    def observation(
        self, state: torch.Tensor, prev_action: torch.Tensor, context: torch.Tensor, frame: torch.Tensor
    ) -> torch.Tensor:
        """Noise-free proprioceptive policy observation; differentiable."""
        ...

    def add_observation_noise(self, observation: torch.Tensor) -> torch.Tensor:
        """Explicit sensor noise; randomness must be process-isolated."""
        ...

    def reward(
        self, state: torch.Tensor, frame: torch.Tensor, action: torch.Tensor, prev_action: torch.Tensor
    ) -> torch.Tensor:
        """Per-step reward of ``state`` against reference ``frame``; differentiable (BPTT main channel)."""
        ...

    def soft_termination(self, state: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        """Continuous death probability in ``[0, 1]`` (hard termination stays in the env)."""
        ...

    def critic_observation(self, state: torch.Tensor, frame: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Privileged critic input; differentiable."""
        ...

    def command(self, frame: torch.Tensor, horizon: int) -> torch.Tensor:
        """Commanded goal over the next ``horizon`` steps; deterministic in ``frame``, never touches the env."""
        ...

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        """Normalize raw context to unit-ish scale for network consumption."""
        ...

    # Optional task extensions the provider consumes only when present
    # (``getattr(mdp, name)``); every one of them is task-owned so robot
    # layout, motion organization and scoring semantics never leak into the
    # generic learner. They are deliberately not part of the required
    # Protocol surface.

    def reference_action(self) -> torch.nn.Module:
        """Base action implied by the commanded goal, as a frozen module.

        Returned module maps flattened goals ``(..., command_dim *
        goal_frames)`` to base actions ``(..., action_dim)``; the provider
        adds the residual policy output. All constants must be registered
        buffers so ``.to(device/dtype)`` follows the policy. This enables
        the agent's ``pi_ref_skip``; without it the policy is pure residual.
        """
        ...

    def mirror_state(self, state: torch.Tensor) -> torch.Tensor:
        """Mirror image of a canonical state tuple (left/right swap)."""
        ...

    def mirror_action(self, action: torch.Tensor) -> torch.Tensor:
        """Mirror image of an action."""
        ...

    def mirror_context(self, context: torch.Tensor) -> torch.Tensor:
        """Mirror image of a raw context."""
        ...

    def imagined_reward(self, state, frame, action, prev_action) -> torch.Tensor:
        """Training-specific reward for imagined states when it differs from
        the real ``reward``; omit when they are identical."""
        ...

    def transport_fields(self) -> dict:
        """Task-owned buffers that travel through the async transport (e.g.
        curriculum state). The collector clones them into every transition;
        the learner copies received values back in. Also used by checkpoint
        save/load through the same buffers."""
        ...

    def curriculum_state(self) -> dict:
        """Checkpoint snapshot of the curriculum state (tensors keyed by name)."""
        ...

    def load_curriculum(self, state: dict) -> None:
        """Restore curriculum state from a checkpoint snapshot."""
        ...

    def evaluation_metadata(self) -> dict:
        """Task metadata the task's official evaluator expects in checkpoint
        ``args`` (e.g. paths and plant timestep); merged by the learner."""
        ...

    def return_normalizer(self) -> float:
        """Divisor turning a raw per-step reward into the task's normalized
        return scale (e.g. the reference action interval). The LR-decay
        trigger threshold is defined on this scale."""
        ...
