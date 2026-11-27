# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""The simulator-free task: explicit binding and evaluation layers.

:class:`ImaginedTask` composes two sibling layers over the shared
:class:`~motrix_env_core.manager.context.ManagerContext` contract:

* :class:`TaskHeads` — the pure evaluation layer: context in, task quantities
  out. It never builds contexts and owns no binding machinery.
* :class:`~motrix_rl_supermimic.imagination.ImaginationContextFactory` — the pure
  binding layer: raw flat tensors in, a runnable context out. It never
  consumes the heads.

The facade owns both plus the world-model wiring artifacts, so no object
both produces and consumes the context.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from gymnasium.spaces import Box

from motrix_env_core.manager.actions import ActionTerm
from motrix_env_core.manager.commands import CommandTerm
from motrix_env_core.manager.context import BuildContext, ManagerContext
from motrix_env_core.manager.observations import ObsTerm, create_observation_groups
from motrix_env_core.manager.resolver import ManagerTermResolver, ObservationTermCall
from motrix_env_core.manager.rewards import RewardTerm, create_reward_terms
from motrix_env_core.observations import ObservationGroupSpec, ObservationSpec
from motrix_rl_supermimic.imagination import ImaginationContextFactory, ImaginationTerm
from motrix_rl_supermimic.world_context import WorldContext
from motrix_rl_supermimic.world_state import WorldStateManager

if TYPE_CHECKING:
    from motrix_rl_supermimic.env import SuperMimicEnvCfg


class TaskHeads:
    """Pure evaluation of the assembled Manager numerical calls.

    Heads consume a :class:`ManagerContext` and return flat N-row tensors;
    callers restore their original leading batch dimensions. The resolved
    calls are the same terms the physical environment runs — the task is
    their simulator-free evaluation. ``compile_heads`` compiles the
    tensor-only methods in place, once per owner.
    """

    def __init__(  # noqa: PLR0913
        self,
        policy_call: ObservationTermCall,
        value_call: ObservationTermCall,
        goal_call: ObservationTermCall,
        reward_call,
        soft_call,
        observation_spec: ObservationSpec,
        action_space: Box,
        goal_frames: int,
        dt: float,
        reward_weight: float,
        world_context: WorldContext,
    ) -> None:
        self.policy_call = policy_call
        self.value_call = value_call
        self.goal_call = goal_call
        self.reward_call = reward_call
        self.soft_call = soft_call
        self.observation_spec = observation_spec
        self.action_space = action_space
        self.goal_frames = goal_frames
        self.dt = dt
        self.reward_weight = reward_weight
        self.world_context = world_context

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
        return self.world_context.normalize(context)

    def compile_heads(self) -> None:
        """Compile the tensor-only heads in place, once per owner.

        The environment constructor and learner each own a separately
        constructed task and call this once when compilation is enabled.
        Simulator I/O and lifecycle bookkeeping stay eager across the backend
        boundary; optional task hooks remain untouched.
        """
        self.observation = torch.compile(self.observation, dynamic=False)
        self.reward = torch.compile(self.reward, dynamic=False)
        self.soft_termination = torch.compile(self.soft_termination, dynamic=False)
        self.critic_observation = torch.compile(self.critic_observation, dynamic=False)
        self.goal = torch.compile(self.goal, dynamic=False)
        self.normalize_context = torch.compile(self.normalize_context, dynamic=False)


class ImaginedTask:
    """The simulator-free task: :class:`TaskHeads` plus its input binding.

    Callers bind inputs through ``contexts`` and evaluate quantities or read
    metadata through ``heads``. The task owns both and the world-model
    wiring artifacts (world state/context, imagination term); the caller
    chooses the variant (``cfg.for_imagined()`` for the learner's imagination
    program) before construction — assembly itself never applies one.
    """

    def __init__(self, cfg: SuperMimicEnvCfg, device: str) -> None:
        with torch.device(device):
            cfg.validate_goal_frames()
            if not cfg.world_model_state or cfg.imagination is None or cfg.soft_termination is None:
                raise ValueError("MBRL configuration requires world_model_state, imagination and soft_termination")
            commands: dict[str, CommandTerm] = {}
            build = BuildContext(cfg=cfg, actions={}, commands=commands, model=None, num_envs=1, seed=0)
            for name, command_cfg in cfg.command_cfgs().items():
                commands[name] = command_cfg(build)
            actions: dict[str, ActionTerm] = {}
            action_slices: dict[str, slice] = {}
            action_offset = 0
            for name, action in cfg.action_cfgs().items():
                build = BuildContext(cfg=cfg, actions=actions, commands=commands, model=None, num_envs=1, seed=0)
                # The analytic build has no compiled scene model: model-independent
                # factories receive no actuators, named actuator routes are not
                # supported.
                actuators = () if action.actuator_names == () else None
                term = action(build, actuators)
                actions[name] = term
                action_width = term.action_space.shape[0]
                action_slices[name] = slice(action_offset, action_offset + action_width)
                action_offset += action_width

            context = BuildContext(cfg=cfg, actions=actions, commands=commands, model=None, num_envs=1, seed=0)
            world_state = WorldStateManager({name: term(context) for name, term in cfg.world_model_state.items()})
            world_context = WorldContext({name: term(context) for name, term in cfg.world_model_context.items()})
            imagination: ImaginationTerm = cfg.imagination(context)
            groups = create_observation_groups(cfg, context)
            for name in ("policy", "value", "goal"):
                if name not in groups:
                    raise ValueError(f"MBRL requires observation group {name!r}")
            if groups["goal"].history_length != 1:
                raise ValueError("MBRL goal observation must have history_length=1")
            action_space = Box(
                np.concatenate([term.action_space.low for term in actions.values()]),
                np.concatenate([term.action_space.high for term in actions.values()]),
                dtype=np.float32,
            )
            observation_spec = ObservationSpec(
                groups={
                    name: ObservationGroupSpec(Box(-float("inf"), float("inf"), (group.size,)), group.history_length)
                    for name, group in groups.items()
                }
            )
            resolver = ManagerTermResolver()

            def observation(group_name: str) -> ObsTerm:
                group = groups[group_name]
                calls = tuple(
                    (resolver.resolve_observation(group_name, entry.name, entry.term), entry.size)
                    for entry in group.terms
                )

                def dispatch(ctx: ManagerContext, out: torch.Tensor) -> None:
                    offset = 0
                    for call, size in calls:
                        call.invoke(ctx, out[:, offset : offset + size])
                        offset += size

                return ObsTerm(group.size, dispatch)

            rewards = create_reward_terms(cfg.reward_cfgs(), context)
            reward_cfgs = cfg.reward_cfgs()
            calls = tuple(
                (resolver.resolve_reward(name, index, term), reward_cfgs[name].weight)
                for index, (name, term) in enumerate(rewards.items())
            )

            def reward(ctx: ManagerContext) -> torch.Tensor:
                total = ctx.env_id.new_zeros(ctx.env_id.shape, dtype=torch.float32)
                for call, weight in calls:
                    total = total + call.invoke(ctx).to(torch.float32) * weight
                return total

            self.heads = TaskHeads(
                policy_call=resolver.resolve_observation("policy", "task", observation("policy")),
                value_call=resolver.resolve_observation("value", "task", observation("value")),
                goal_call=resolver.resolve_observation("goal", "task", observation("goal")),
                reward_call=resolver.resolve_reward("task", 0, RewardTerm(reward)),
                soft_call=resolver.resolve_termination("soft", 0, cfg.soft_termination(context)),
                observation_spec=observation_spec,
                action_space=action_space,
                goal_frames=cfg.goal_frames,
                dt=cfg.ctrl_dt,
                reward_weight=1.0,
                world_context=world_context,
            )
            self.contexts = ImaginationContextFactory(
                imagination, world_state, world_context, action_slices, cfg.ctrl_dt, cfg.reference_command
            )
            self.world_state = world_state
            self.world_context = world_context
            self.imagination = imagination
            # Construct modules later inside the learner's initialization seed scope.
            self.reference_action = cfg.reference_action
