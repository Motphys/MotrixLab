# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Opt-in actor imagination/loss CUDA graph; optimizers and critic training stay in the Learner.

Only the actual policy parameters are registered with the graphed module. WM
and critic views share their live weights, frozen by policy_update's
try/finally, while all running normalizers use address-stable private buffer
snapshots refreshed before every call.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch
from torch import nn

from motrix_env_core.observations import append_observation
from motrix_rl_supermimic.config import TrainerCfg
from motrix_rl_supermimic.manager_task import ManagerTaskMDP
from motrix_rl_supermimic.networks import Critic, Policy, RunningNorm, WorldModel
from motrix_rl_supermimic.survival import survival_cumprod

__all__ = ["ActorLoss", "ActorLossGraph"]


GraphInputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
GraphOutputs = tuple[torch.Tensor, ...]
InputSignature = tuple[tuple[tuple[int, ...], torch.dtype, torch.device, bool], ...]
ModuleT = TypeVar("ModuleT", bound=nn.Module)


@dataclass
class ActorModels:
    """Borrowed models; Learner owns device, freezing, updates and checkpoints."""

    wm: WorldModel
    critic: Critic


@dataclass
class ActorUpdate:
    """Decoded graph output; only loss retains the policy gradient path."""

    loss: torch.Tensor
    states: list[torch.Tensor]
    frames: list[torch.Tensor]
    goals: list[torch.Tensor]
    rewards: torch.Tensor
    terminations: torch.Tensor
    survival_before: torch.Tensor


class NormalizerSnapshots:
    """Private, address-stable buffers with explicitly declared model norms."""

    def __init__(self) -> None:
        self.pairs: list[tuple[RunningNorm, RunningNorm]] = []

    def model_view(self, module: ModuleT, norms: tuple[RunningNorm, ...]) -> ModuleT:
        # Deepcopy a private model shell, sharing child modules and parameters
        # except the explicitly supplied normalizers. No private module-tree writes.
        memo = {id(child): child for child in module.children()}
        memo.update((id(parameter), parameter) for parameter in module.parameters())
        for actual in norms:
            snapshot = copy.deepcopy(actual)
            memo[id(actual)] = snapshot
            self.pairs.append((actual, snapshot))
        return copy.deepcopy(module, memo)

    @torch.no_grad()
    def sync(self) -> None:
        for actual, snapshot in self.pairs:
            for name, value in actual.named_buffers():
                snapshot.get_buffer(name).copy_(value)


class ActorLoss(nn.Module):
    """Tensor-only actor math; registers policy but borrows the frozen models."""

    def __init__(self, pi: Policy, models: ActorModels, mdp: ManagerTaskMDP, cfg: TrainerCfg) -> None:
        super().__init__()
        self.norms = NormalizerSnapshots()
        self.pi = self.norms.model_view(pi, (pi.obs_norm, pi.goal_norm))
        self.models = ActorModels(
            wm=self.norms.model_view(models.wm, (models.wm.in_norm, models.wm.out_norm, models.wm.inc_norm)),
            critic=self.norms.model_view(models.critic, (models.critic.norm,)),
        )
        self.mdp = mdp
        self.cfg = cfg
        if cfg.compile_actor_kernel:
            self.imagination_step = torch.compile(self.imagination_step, dynamic=False)
            self.reward_termination = torch.compile(self.reward_termination, dynamic=False)

    def sync_norms(self) -> None:
        self.norms.sync()

    def imagination_step(
        self,
        state: torch.Tensor,
        obs: torch.Tensor,
        xi: torch.Tensor,
        context: torch.Tensor,
        frame: torch.Tensor,
        goal: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action = self.pi(obs, goal, z)
        state = self.models.wm.step(state, action, context)
        frame = frame + 1
        ctx = self.mdp.imagination.bind_context(frame, state=state, action=action, previous=action, context=xi)
        next_obs = self.mdp.observation(ctx)
        obs = append_observation(obs, next_obs)
        return state, action, obs, frame

    def reward_termination(
        self,
        states: torch.Tensor,
        frames: torch.Tensor,
        actions: torch.Tensor,
        previous: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx = self.mdp.imagination.bind_context(
            frames, state=states, action=actions, previous=previous, context=context
        )
        return self.mdp.reward(ctx), self.mdp.soft_termination(ctx)

    def critic_input(
        self,
        state: torch.Tensor,
        frame: torch.Tensor,
        context: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        # Match the eager critic path: zero actions and already-normalized context.
        zero_action = context.new_zeros((*frame.shape, self.mdp.action_dim))
        ctx = self.mdp.imagination.bind_context(
            frame, state=state, action=zero_action, previous=zero_action, context=context
        )
        observation = self.mdp.critic_observation(ctx).reshape(*frame.shape, self.mdp.critic_observation_dim)
        return torch.cat((observation, goal), -1)

    def forward(
        self,
        initial_flat: torch.Tensor,
        xi: torch.Tensor,
        frame: torch.Tensor,
        previous: torch.Tensor,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> GraphOutputs:
        cfg = self.cfg
        state = initial_flat
        context = self.mdp.normalize_context(xi)
        # Goals depend only on reference frames, so encode the whole
        # sequence before predicting any states.
        step_offsets = torch.arange(cfg.horizon + 1, device=frame.device)
        reference_frames = frame[None, :] + step_offsets[:, None]
        goal_sequence = self.mdp.goal(self.mdp.imagination.bind_context(reference_frames)).reshape(
            *reference_frames.shape, goal.shape[-1]
        )
        encoded_goals = self.pi.encode_goal(goal_sequence)
        goal = goal_sequence[0]
        states, frames, goals, actions, previous_actions = [], [], [], [], []
        for t in range(cfg.horizon):
            previous_actions.append(previous)
            state, action, obs, frame = self.imagination_step(state, obs, xi, context, frame, goal, encoded_goals[t])
            goal = goal_sequence[t + 1]
            states.append(state)
            frames.append(frame)
            goals.append(goal)
            actions.append(action)
            previous = action

        def flat(xs: Sequence[torch.Tensor]) -> torch.Tensor:
            return torch.stack(xs).flatten(0, 1)

        all_states = flat(states)
        r, d = self.reward_termination(
            all_states,
            flat(frames),
            flat(actions),
            flat(previous_actions),
            xi[None].expand(cfg.horizon, -1, -1).flatten(0, 1),
        )
        r, d = r.view(cfg.horizon, -1), d.view(cfg.horizon, -1)
        # Graph-safe survival product: ATen's cumprod backward is not
        # CUDA-graph capturable; this formulation is bit-equal on values and
        # VJP (see tests/test_recipe_contracts.py).
        survival = survival_cumprod(1 - d)
        before = torch.cat((torch.ones_like(survival[:1]), survival[:-1]), 0)
        discounts = (cfg.gamma ** torch.arange(cfg.horizon, device=frame.device))[:, None]
        objective = (discounts * before * r).sum(0)
        objective = objective + cfg.gamma**cfg.horizon * survival[-1] * self.models.critic(
            self.critic_input(state, frame, context, goal)
        )
        loss = -objective.mean()
        return (
            loss,
            all_states.detach(),
            torch.stack(frames).detach(),
            torch.stack(goals).detach(),
            r.detach(),
            d.detach(),
            before.detach(),
        )


class ActorLossGraph:
    """Per-learner lazy capture cache, deliberately not an nn.Module.

    Capture warmup must not consume training randomness. Input signatures are
    fixed after capture; changing them is an error, never an eager fallback.
    This manager is an ordinary learner attribute, not checkpoint payload.
    """

    def __init__(self, pi: Policy, models: ActorModels, mdp: ManagerTaskMDP, cfg: TrainerCfg) -> None:
        self.pi = pi
        self.models = models
        self.mdp = mdp
        self.cfg = cfg
        self.capture: CapturedActorLoss | None = None

    def __call__(
        self,
        initial_flat: torch.Tensor,
        xi: torch.Tensor,
        frame: torch.Tensor,
        previous: torch.Tensor,
        obs: torch.Tensor,
        goal: torch.Tensor,
    ) -> ActorUpdate:
        args = tuple(x.contiguous() for x in (initial_flat, xi, frame, previous, obs, goal))
        key = tuple((tuple(x.shape), x.dtype, x.device, x.requires_grad) for x in args)
        if self.capture is None:
            self.capture = self._capture(args, key)
        capture = self.capture
        if key != capture.signature:
            raise RuntimeError("Actor input signature changed; recapture disabled")
        capture.module.sync_norms()
        loss, states, frames, goals, r, d, before = capture.module(*args)
        shaped = states.reshape(self.cfg.horizon, self.cfg.pi_batch, self.models.wm.world_state.size)
        return ActorUpdate(loss, list(shaped.unbind(0)), list(frames.unbind(0)), list(goals.unbind(0)), r, d, before)

    def _capture(self, args: GraphInputs, signature: InputSignature) -> CapturedActorLoss:
        module = ActorLoss(self.pi, self.models, self.mdp, self.cfg)
        module.sync_norms()
        rng_cpu = torch.get_rng_state().clone()
        rng_cuda = torch.cuda.get_rng_state_all()
        # Third-party capability detection: supported Torch versions do not all
        # expose the compiled-autograd capture-stream override.
        override = getattr(torch.autograd.graph, "set_override_stale_capture_stream", None)
        try:
            if override is not None:
                override(True)
            torch.cuda.make_graphed_callables(module, sample_args=args, num_warmup_iters=1, allow_unused_input=True)
            torch.cuda.synchronize()
        finally:
            if override is not None:
                override(False)
            torch.set_rng_state(rng_cpu)
            torch.cuda.set_rng_state_all(rng_cuda)
        # make_graphed_callables replaces this module's forward in place.
        return CapturedActorLoss(module, signature)


@dataclass
class CapturedActorLoss:
    """Publish the captured module and its fixed signature together."""

    module: ActorLoss
    signature: InputSignature
