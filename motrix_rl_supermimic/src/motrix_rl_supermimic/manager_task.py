# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Simulator-free views over the existing Manager numerical term pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from gymnasium.spaces import Box

from motrix_env_core.manager.context import ManagerContext
from motrix_env_core.manager.observations import ObsTerm
from motrix_env_core.manager.resolver import ObservationTermCall, RewardTermCall, TerminationTermCall
from motrix_env_core.manager.rewards import RewardTerm
from motrix_env_core.manager.terminations import TerminationTerm
from motrix_env_core.observations import ObservationSpec
from motrix_rl_supermimic.imagination import ImaginationTerm
from motrix_rl_supermimic.task_mdp import StateSpec
from motrix_rl_supermimic.world_state import WorldState


def resolve_manager_task(  # noqa: PLR0913
    spec: StateSpec,
    dt: float,
    reward_weight: float,
    imagination: ImaginationTerm,
    policy: ObsTerm,
    value: ObsTerm,
    reward: RewardTerm,
    soft_termination: TerminationTerm,
    *,
    goal: ObsTerm,
    goal_frames: int,
    model_context_center: torch.Tensor,
    model_context_scale: torch.Tensor,
    observation_spec: ObservationSpec,
    action_space: Box,
    world_state: WorldState,
) -> ManagerTaskMDP:
    """Fuse the selected numerical terms into the simulator-free task MDP.

    Only numerical calls are selected. No command update/advance, reset, reads
    or simulator writes are executed by imagination.
    """
    from motrix_env_core.manager.resolver import ManagerTermResolver

    resolver = ManagerTermResolver()
    return ManagerTaskMDP(
        spec=spec,
        world_state=world_state,
        dt=dt,
        reward_weight=reward_weight,
        imagination=imagination,
        policy_call=resolver.resolve_observation("policy", "task", policy),
        value_call=resolver.resolve_observation("value", "task", value),
        goal_call=resolver.resolve_observation("goal", "task", goal),
        reward_call=resolver.resolve_reward("task", 0, reward),
        soft_call=resolver.resolve_termination("soft", 0, soft_termination),
        observation_spec=observation_spec,
        action_space=action_space,
        goal_frames=goal_frames,
        model_context_center=model_context_center,
        model_context_scale=model_context_scale,
    )


@dataclass
class ManagerTaskMDP:
    """Simulator-free MDP over fused Manager numerical calls.

    Callers bind frame-only or predicted inputs once through ``imagination`` and
    share the resulting Context across numerical heads. Heads return flat N-row
    tensors; callers restore their original leading batch dimensions. Affine
    parameters normalize world-model context independently of that binding.
    """

    spec: StateSpec
    world_state: WorldState
    dt: float
    reward_weight: float
    imagination: ImaginationTerm
    policy_call: ObservationTermCall
    value_call: ObservationTermCall
    goal_call: ObservationTermCall
    reward_call: RewardTermCall
    soft_call: TerminationTermCall
    observation_spec: ObservationSpec
    action_space: Box
    goal_frames: int
    model_context_center: torch.Tensor
    model_context_scale: torch.Tensor

    @property
    def action_dim(self) -> int:
        return self.action_space.shape[0]

    @property
    def observation_dim(self) -> int:
        """Width of one policy observation frame, before history stacking."""
        return self.observation_spec.policy.frame_space.shape[0]

    @property
    def critic_observation_dim(self) -> int:
        """Width of one critic observation frame, before history stacking."""
        return self.observation_spec.value_or_policy.frame_space.shape[0]

    @property
    def goal_dim(self) -> int:
        """Total width of the configured multi-frame goal."""
        return self.observation_spec["goal"].space.shape[0]

    def observation(self, ctx: ManagerContext) -> torch.Tensor:
        return self._observation(self.policy_call, ctx, self.observation_dim)

    def reward(self, ctx: ManagerContext) -> torch.Tensor:
        return self.reward_call.invoke(ctx).to(torch.float32) * self.reward_weight * self.dt

    def soft_termination(self, ctx: ManagerContext) -> torch.Tensor:
        return self.soft_call.invoke(ctx)

    def critic_observation(self, ctx: ManagerContext) -> torch.Tensor:
        return self._observation(self.value_call, ctx, self.critic_observation_dim)

    def goal(self, ctx: ManagerContext) -> torch.Tensor:
        """Evaluate the configured goal term using only reference frames/static data.

        Goal terms must not depend on physical state or mutate command lifecycle
        state: imagination may pre-encode all horizon goals in one batch.
        """
        return self._observation(self.goal_call, ctx, self.goal_dim)

    def _observation(self, call: ObservationTermCall, ctx: ManagerContext, width: int) -> torch.Tensor:
        out = torch.empty((ctx.env_id.numel(), width), device=ctx.env_id.device, dtype=torch.float32)
        call.invoke(ctx, out)
        return out

    def normalize_context(self, context: torch.Tensor) -> torch.Tensor:
        return (context - self.model_context_center) / self.model_context_scale
