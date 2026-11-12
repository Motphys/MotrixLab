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

import torch
from torch import nn

from motrix_rl_supermimic import state as st
from motrix_rl_supermimic.networks import RunningNorm
from motrix_rl_supermimic.survival import survival_cumprod
from motrix_rl_supermimic.task_mdp import optional_task_hook

__all__ = ["ActorLoss", "ActorLossGraph"]


class ActorLoss(nn.Module):
    """Tensor-only actor math, in the same operation order as policy_update."""

    def __init__(self, learner):
        super().__init__()
        self.norm_pairs = []

        def view(module):
            result = copy.copy(module)
            result._modules = module._modules.copy()
            for name, child in module._modules.items():
                if isinstance(child, RunningNorm):
                    static = copy.deepcopy(child)
                    result._modules[name] = static
                    self.norm_pairs.append((child, static))
            return result

        self.pi = view(learner.pi)
        if tuple(id(p) for p in self.pi.parameters()) != tuple(id(p) for p in learner.pi.parameters()):
            raise RuntimeError("ActorLoss policy view lost live parameter aliases")
        # With overlap_wm the actor imagines with the frozen iteration-start
        # world model; its parameter storage is updated in place each round.
        object.__setattr__(self, "wm", view(getattr(learner, "wm_imag", learner.wm)))
        object.__setattr__(self, "critic", view(learner.critic))
        for name in ("mdp", "cfg", "dev"):
            object.__setattr__(self, name, getattr(learner, name))
        self.imagined_reward = optional_task_hook(learner.mdp, "imagined_reward")
        self.batch_goal_enc = learner.cfg.trainer.batch_goal_enc
        if learner.cfg.trainer.compile_actor_kernel:
            self.imagination_step = torch.compile(self.imagination_step, dynamic=False)
            self.reward_termination = torch.compile(self.reward_termination, dynamic=False)
        if len(self.norm_pairs) != 6:
            raise RuntimeError("Expected pi obs/goal, WM in/out/inc and critic norms")

    @torch.no_grad()
    def sync_norms(self) -> None:
        for actual, static in self.norm_pairs:
            for name, value in actual.named_buffers():
                target = static.get_buffer(name)
                target.copy_(value)

    def _goal(self, frame, horizon):
        return self.mdp.command(frame, horizon).flatten(-2)

    def imagination_step(self, state, obs, xi, context, frame, goal, z=None):
        action = self.pi(obs, goal, z)
        state = self.wm.step(state, action, context)
        frame = frame + 1
        next_obs = self.mdp.add_observation_noise(
            self.mdp.observation(st.flatten_state(state), action, xi, frame)
        )
        obs = torch.cat((obs[:, 1:], next_obs[:, None]), 1)
        return state, action, obs, frame

    def reward_termination(self, states, frames, actions, previous):
        reward_fn = self.imagined_reward if self.imagined_reward is not None else self.mdp.reward
        return reward_fn(states, frames, actions, previous), self.mdp.soft_termination(states, frames)

    def critic_input(self, state, frame, context, goal):
        return torch.cat((self.mdp.critic_observation(state, frame, context), goal), -1)

    def forward(self, initial_flat, xi, frame, previous, obs, goal):
        cfg = self.cfg.trainer
        state = st.unflatten_state(self.wm.spec, initial_flat)
        initial = state
        context = self.mdp.normalize_context(xi)
        z_all = None
        if self.batch_goal_enc:
            # All imagined goals depend only on the reference, so encode the
            # whole H+1 horizon in one batch (upstream-validated perf change).
            frames_all = (frame[None] + torch.arange(cfg.horizon + 1, device=frame.device)[:, None]).reshape(-1)
            goals_all = self.mdp.command(frames_all, cfg.goal_frames).reshape(cfg.horizon + 1, frame.shape[0], -1)
            z_all = self.pi.encode_goal(goals_all)
            goal = goals_all[0]
        states, frames, goals, actions, previous_actions = [], [], [], [], []
        for t in range(cfg.horizon):
            previous_actions.append(previous)
            state, action, obs, frame = self.imagination_step(
                state, obs, xi, context, frame, goal, z_all[t] if z_all is not None else None
            )
            goal = goals_all[t + 1] if z_all is not None else self._goal(frame, cfg.goal_frames)
            states.append(state)
            frames.append(frame)
            goals.append(goal)
            actions.append(action)
            previous = action
        flat = lambda xs: torch.stack(xs).flatten(0, 1)
        all_states = tuple(flat([s[i] for s in states]) for i in range(6))
        r, d = self.reward_termination(
            st.flatten_state(all_states), flat(frames), flat(actions), flat(previous_actions)
        )
        r, d = r.view(cfg.horizon, -1), d.view(cfg.horizon, -1)
        # Graph-safe survival product: ATen's cumprod backward is not
        # CUDA-graph capturable; this formulation is bit-equal on values and
        # VJP (see tests/test_recipe_contracts.py).
        survival = survival_cumprod(1 - d)
        before = torch.cat((torch.ones_like(survival[:1]), survival[:-1]), 0)
        discounts = (cfg.gamma ** torch.arange(cfg.horizon, device=self.dev))[:, None]
        objective = (discounts * before * r).sum(0)
        objective = objective + cfg.gamma**cfg.horizon * survival[-1] * self.critic(
            self.critic_input(st.flatten_state(state), frame, context, goal)
        )
        loss = -objective.mean()
        return (loss,) + tuple(x.detach() for x in all_states) + (
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

    def __init__(self, learner):
        self.learner = learner
        self.module = None
        self.callable = None
        self.key = None

    def __call__(self, initial_flat, xi, frame, previous, obs, goal):
        args = tuple(x.contiguous() for x in (initial_flat, xi, frame, previous, obs, goal))
        key = tuple((tuple(x.shape), x.dtype, x.device, x.requires_grad) for x in args)
        if self.callable is None:
            module = ActorLoss(self.learner)
            module.sync_norms()
            rng_cpu = torch.get_rng_state().clone()
            rng_cuda = torch.cuda.get_rng_state_all()
            try:
                # Compiled subfunctions leave AccumulateGrad nodes bound to the
                # default stream from warmup; redirect them to the capture
                # stream (same override the reference implementation uses).
                override = getattr(torch.autograd.graph, "set_override_stale_capture_stream", None)
                if override is not None:
                    override(True)
                graph = torch.cuda.make_graphed_callables(
                    module, sample_args=args, num_warmup_iters=1, allow_unused_input=True
                )
                torch.cuda.synchronize()
            finally:
                if override is not None:
                    override(False)
                torch.set_rng_state(rng_cpu)
                torch.cuda.set_rng_state_all(rng_cuda)
            self.module, self.callable, self.key = module, graph, key
        if key != self.key:
            raise RuntimeError("Actor input signature changed; recapture disabled")
        self.module.sync_norms()
        output = self.callable(*args)
        loss = output[0]
        cfg = self.learner.cfg.trainer
        shaped = [x.reshape(cfg.horizon, cfg.pi_batch, *x.shape[1:]).unbind(0) for x in output[1:7]]
        states = list(zip(*shaped))
        frames, goals, r, d, before = output[7:]
        return loss, states, list(frames.unbind(0)), list(goals.unbind(0)), r, d, before
