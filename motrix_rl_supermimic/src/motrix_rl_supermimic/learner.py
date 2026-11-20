# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
# Adapted from SuperMimic (supermimic_vibed), Copyright (c) 2026 ETH Zurich,
# Chong Zhang. MIT License.

"""World-model fit / imagination learner consuming only the TaskMDP ABI.

Imagination is a differentiable loop inside ``policy_update``: the frozen
world model acts as the dynamics while the TaskMDP's analytic heads provide
observations, rewards, and soft termination. Nothing here knows a robot.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from motrix_env_core.observations import append_observation
from motrix_rl_supermimic.actor_graph import ActorLossGraph, ActorModels
from motrix_rl_supermimic.config import SuperMimicCfg
from motrix_rl_supermimic.imagination import ImaginationRuntime
from motrix_rl_supermimic.manager_task import ManagerTaskMDP
from motrix_rl_supermimic.networks import Critic, WorldModel, build_policy, resolve_amp_dtype, run_mlp
from motrix_rl_supermimic.optim import MuonAdam, policy_multiplier
from motrix_rl_supermimic.randomness import derive_seed, model_initialization, resolve_seed
from motrix_rl_supermimic.replay import ReplayBuffer
from motrix_rl_supermimic.survival import SurvivalCumprod
from motrix_rl_supermimic.task_mdp import compile_task_heads
from motrix_rl_supermimic.transitions import Transitions
from motrix_rl_supermimic.wm_graph import WMLossGraph

__all__ = ["Learner"]


def _requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _make_optimizer(
    opt: str,
    model: nn.Module,
    mlps: list[nn.Module],
    lr: float,
    betas: tuple[float, float] = (0.9, 0.95),
    device: str = "cpu",
) -> torch.optim.Adam | MuonAdam:
    on_cuda = torch.device(device).type == "cuda"
    if opt == "muon":
        return MuonAdam(model, mlps, lr=lr, betas=betas, fused=on_cuda)
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, fused=on_cuda)
    raise ValueError(f"Unknown optimizer {opt!r}; expected 'adam' or 'muon'")


class Learner:
    def __init__(
        self,
        cfg: SuperMimicCfg,
        task: ManagerTaskMDP,
        binding: ImaginationRuntime,
        num_envs: int,
        action_dt: float,
        device: str,
        output_dir: Path,
        *,
        seed: int | None = None,
    ) -> None:
        self.cfg, self.mdp, self.dev = cfg, task, device
        self.binding = binding
        self.root_seed = resolve_seed(seed)
        self.observation_spec = task.observation_spec
        if self.observation_spec.value_or_policy.history_length != 1:
            raise ValueError("SuperMimic critic requires value history_length=1; value history is not supported")
        a, t = cfg.agent, cfg.trainer
        on_cuda = torch.device(device).type == "cuda"
        if (t.cuda_graph_wm_loss or t.cuda_graph_actor_loss) and not on_cuda:
            raise ValueError("cuda_graph_wm_loss/cuda_graph_actor_loss require a CUDA device")
        if (t.cuda_graph_wm_loss or t.cuda_graph_actor_loss) and (t.compile or t.compile_wm):
            raise ValueError("cuda_graph_wm_loss/cuda_graph_actor_loss are mutually exclusive with compile/compile_wm")
        amp_dtype = resolve_amp_dtype(t.amp)
        if t.overlap_wm and not on_cuda:
            raise ValueError("overlap_wm requires a CUDA device")
        if t.compile_actor_kernel and not t.cuda_graph_actor_loss:
            raise ValueError("compile_actor_kernel requires cuda_graph_actor_loss")
        if t.compile_wm_kernel and not t.cuda_graph_wm_loss:
            raise ValueError("compile_wm_kernel requires cuda_graph_wm_loss")
        self.goal_dim = task.goal_dim
        wm_width = a.wm_hidden or a.hidden
        wm_kw = {"hidden": (wm_width,) * 3} if wm_width else {}
        with model_initialization(derive_seed(self.root_seed, "model/world_model")):
            self.wm = WorldModel(
                task.world_state,
                task.world_context.size,
                structured=a.wm_structured,
                dt=task.dt,
                amp_dtype=amp_dtype,
                action_dim=task.action_dim,
                **wm_kw,
            )
        self.wm = self.wm.to(self.dev)
        with model_initialization(derive_seed(self.root_seed, "model/policy")):
            reference_action = self.mdp.reference_action() if a.pi_ref_skip else None
            self.pi = build_policy(
                a,
                task.observation_dim,
                self.goal_dim,
                task.action_dim,
                "cpu",
                observation_spec=self.observation_spec.policy,
                reference_action=reference_action,
                amp_dtype=amp_dtype,
            )
        self.pi = self.pi.to(self.dev)
        with model_initialization(derive_seed(self.root_seed, "model/critic")):
            self.critic = Critic(
                task.critic_observation_dim + self.goal_dim,
                amp_dtype=amp_dtype,
                **({"hidden": (a.hidden,) * 2} if a.hidden else {}),
            )
        self.critic = self.critic.to(self.dev)
        self.opt_wm = _make_optimizer(t.wm_opt, self.wm, [self.wm.net], lr=t.lr, betas=(0.9, 0.999), device=device)
        self.opt_pi = _make_optimizer(
            t.pi_opt, self.pi, [self.pi.enc, self.pi.dec], lr=t.lr * t.pi_lr_mult, device=device
        )
        self.pi_ema = a.pi_ema
        self.pi_avg = None
        if self.pi_ema > 0.0:
            self.pi_avg = torch.optim.swa_utils.AveragedModel(
                self.pi, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.pi_ema), use_buffers=True
            )
            # Start averaging from the initial weights so the first actor
            # update lerps instead of copying the live policy.
            self.pi_avg.update_parameters(self.pi)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=t.lr)
        # overlap_wm: frozen iteration-start WM for the actor; live WM updates
        # run on a side stream driven by the async loop.
        self.wm_imag = copy.deepcopy(self.wm).to(self.dev) if t.overlap_wm else self.wm
        if self.wm_imag is not self.wm:
            for p in self.wm_imag.parameters():
                p.requires_grad_(False)
        self.wm_side = torch.cuda.Stream() if self.wm_imag is not self.wm else None
        self.buffer = ReplayBuffer(
            t.buffer_steps,
            num_envs,
            self.dev,
            state_dim=task.world_state.size,
            context_dim=task.world_context.size,
            action_dim=task.action_dim,
        )
        valid_starts = self.mdp.imagination.valid_starts
        horizon = t.horizon
        self._valid_start_frames = lambda frames: valid_starts(frames, horizon)
        self.err_floor = task.world_state.error_floor(self.dev)
        self.mirror = self.mdp.imagination.mirror
        normalizer = self.mdp.imagination.return_normalizer
        self.return_normalizer = action_dt if normalizer is None else normalizer
        if a.wm_mirror > 0 and self.mirror is None:
            raise ValueError("agent.wm_mirror requires the task's mirror extension")
        self._wm_loss_graph = None
        if t.cuda_graph_wm_loss:
            self._wm_loss_graph = WMLossGraph(
                self.wm,
                self.err_floor,
                self.mdp.normalize_context,
                t.wm_window,
                t.wm_batch,
                compile_kernel=t.compile_wm_kernel,
            )
        self._actor_loss_graph = None
        if t.cuda_graph_actor_loss:
            self._actor_loss_graph = ActorLossGraph(
                self.pi, ActorModels(self.wm_imag, self.critic), self.mdp, self.binding, t
            )
        self.iteration = self.sim_steps = 0
        self.decay_at: int | None = None
        self._nonfinite_actor = torch.zeros((), device=device)
        self.output = Path(output_dir)
        self._last_imag_survival = torch.zeros(())
        if t.compile or t.compile_wm:
            self.wm.step = torch.compile(self.wm.step)
        if t.compile:
            self.pi.forward = torch.compile(self.pi.forward)
            # Keep TaskMDP heads compiled as one shared tensor-only boundary;
            # this covers collection, normalization, and imagination instead
            # of compiling only reward here.
            compile_task_heads(self.mdp)

    @torch.no_grad()
    def observe(self, transitions: Transitions) -> torch.Tensor:
        """Consume a time-major batch of real transitions and return scaled mean reward.

        Inputs have leading (steps, environments) dimensions and canonical transition
        dtypes. Transport ownership and decoding belong to the caller; replay owns
        its copies after this method returns. Normalizers advance in sampling order.
        """
        if self.wm_side is not None:
            # Queued WM updates still read replay and live normalizer buffers.
            torch.cuda.current_stream(self.dev).wait_stream(self.wm_side)
        on_device = transitions.to(self.dev)
        steps = on_device.reward.shape[0]
        for step in range(steps):
            row = on_device.at_step(step)
            self.buffer.add(row)
            # Reconstruct policy inputs rather than transporting them. Observation
            # noise is freshly sampled, matching the imagination path.
            ctx = self.binding.bind(
                row.frame,
                state=row.state,
                action=row.previous_action,
                previous=row.previous_action,
                context=row.context,
            )
            self.pi.obs_norm.update(self.mdp.observation(ctx))
            self.pi.goal_norm.update(self.mdp.goal(ctx))
            self.wm.in_norm.update(self.wm.inputs(row.state, row.action, self.mdp.normalize_context(row.context)))
            self.wm.out_norm.update(self.wm.targets(row.state, row.next_state))
            self.wm.inc_norm.update(self.wm.increments(row.state, row.next_state))
            self.sim_steps += row.state.shape[0]
        return on_device.reward.mean() / self.return_normalizer

    def sync_imag_wm(self) -> None:
        """Wait for the side-stream WM updates, then copy live WM -> wm_imag.

        Called at iteration start by the async loop; the actor (and its CUDA
        graph, which holds wm_imag parameter storage) then imagines with the
        iteration-start world model."""
        if self.wm_imag is self.wm:
            return
        torch.cuda.current_stream().wait_stream(self.wm_side)
        with torch.no_grad():
            torch._foreach_copy_(
                list(self.wm_imag.parameters()) + list(self.wm_imag.buffers()),
                [x.detach() for x in list(self.wm.parameters()) + list(self.wm.buffers())],
            )

    # ------------------------------------------------------------------ world model
    def _mirror_windows(self, batch: dict[str, torch.Tensor], count: int) -> dict[str, torch.Tensor]:
        """Replace the first ``count`` windows with their task-mirror image."""
        result = dict(batch)
        for key in ("s", "s1"):
            mirrored = self.mirror.state(batch[key][:count])
            result[key] = torch.cat((mirrored, batch[key][count:]))
        result["a"] = torch.cat((self.mirror.action(batch["a"][:count]), batch["a"][count:]))
        result["xi"] = torch.cat((self.mirror.context(batch["xi"][:count]), batch["xi"][count:]))
        return result

    def wm_update(self) -> torch.Tensor:
        cfg = self.cfg.trainer
        batch = self.buffer.sample_windows(cfg.wm_batch, cfg.wm_window)
        if self.cfg.agent.wm_mirror > 0:
            batch = self._mirror_windows(batch, int(cfg.wm_batch * self.cfg.agent.wm_mirror))
        if self._wm_loss_graph is not None:
            loss = self._wm_loss_graph(batch["s"][:, 0], batch["a"], batch["xi"], batch["s1"], batch["mask"])
        else:
            state = batch["s"][:, 0]
            predictions = []
            for j in range(cfg.wm_window):
                state = self.wm.step(state, batch["a"][:, j], self.mdp.normalize_context(batch["xi"][:, j]))
                predictions.append(state)
            states = torch.stack(predictions, 1)
            std = torch.maximum(self.wm.inc_norm.std(), self.err_floor)
            error = (self.wm.state_error(states, batch["s1"]) / std).square().mean(-1)
            loss = (error * batch["mask"]).sum() / cfg.wm_batch
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        return loss.detach() / cfg.wm_window

    # ------------------------------------------------------------------ imagination
    def starts(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        cfg = self.cfg.trainer
        return self.buffer.sample_starts(
            cfg.pi_batch,
            self.observation_spec.policy.history_length,
            valid_frames=self._valid_start_frames,
        )

    def observation_history(self, history: dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct noisy frames, then repeat the actual episode first frame."""
        ctx = self.binding.bind(
            history["f"],
            state=history["s"],
            action=history["a_prev"],
            previous=history["a_prev"],
            context=history["xi"],
        )
        obs = self.mdp.observation(ctx).reshape(*history["f"].shape, self.mdp.observation_dim)
        # Gather after noise so repeated padding shares one noisy realization.
        return obs.gather(1, history["index"][..., None].expand_as(obs))

    def critic_input(
        self, state: torch.Tensor, frame: torch.Tensor, context: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        # Preserve the critic's action-independent inputs and normalized context.
        zero_action = context.new_zeros((*frame.shape, self.mdp.action_dim))
        ctx = self.binding.bind(frame, state=state, action=zero_action, previous=zero_action, context=context)
        observation = self.mdp.critic_observation(ctx).reshape(*frame.shape, self.mdp.critic_observation_dim)
        return torch.cat((observation, goal), -1)

    def _apply_lr(self) -> float:
        """Base cosine on every optimizer; the policy additionally carries the
        multiplier/decay/anneal schedule. Returns the effective policy lr."""
        t = self.cfg.trainer
        lr = t.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * min(self.iteration / t.lr_decay_iters, 1))))
        for opt in (self.opt_wm, self.opt_v):
            for group in opt.param_groups:
                group["lr"] = lr
        pi_lr = lr * policy_multiplier(t, self.iteration, self.decay_at)
        for group in self.opt_pi.param_groups:
            group["lr"] = pi_lr
        return pi_lr

    def policy_update(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg.trainer
        start, history = self.starts()
        state = start["s"]
        initial = state
        xi, frame, previous = start["xi"], start["f"], start["a_prev"]
        context = self.mdp.normalize_context(xi)
        with torch.no_grad():
            obs = self.observation_history(history)
        # Goals depend only on reference frames; query the complete sequence once
        # and share it with both eager and graphed actor execution.
        step_offsets = torch.arange(cfg.horizon + 1, device=frame.device)
        reference_frames = frame[None, :] + step_offsets[:, None]
        goal_sequence = self.mdp.goal(self.binding.bind_reference(reference_frames)).reshape(
            *reference_frames.shape, self.goal_dim
        )
        goal = initial_goal = goal_sequence[0]
        states, frames, goals, actions, previous_actions = [], [], [], [], []
        _requires_grad(self.wm, False)
        _requires_grad(self.critic, False)
        try:
            if self._actor_loss_graph is not None:
                result = self._actor_loss_graph(start["s"], xi, frame, previous, obs, goal_sequence)
                loss, states = result.loss, result.states
                frames, goals = result.frames, result.goals
                r, d, before = result.rewards, result.terminations, result.survival_before
            else:
                encoded_goals = self.pi.encode_goal(goal_sequence)
                goal = goal_sequence[0]
                wm_actor = self.wm_imag if self.wm_imag is not self.wm else self.wm
                for t in range(cfg.horizon):
                    previous_actions.append(previous)
                    action = self.pi(obs, goal, encoded_goals[t])
                    state = wm_actor.step(state, action, context)
                    frame = frame + 1
                    obs_ctx = self.binding.bind(frame, state=state, action=action, previous=action, context=xi)
                    next_obs = self.mdp.observation(obs_ctx)
                    obs = append_observation(obs, next_obs)
                    goal = goal_sequence[t + 1]
                    states.append(state)
                    frames.append(frame)
                    goals.append(goal)
                    actions.append(action)
                    previous = action

                def flat(xs: Sequence[torch.Tensor]) -> torch.Tensor:
                    return torch.stack(xs).flatten(0, 1)

                all_states = flat(states)
                reward_ctx = self.binding.bind(
                    flat(frames),
                    state=all_states,
                    action=flat(actions),
                    previous=flat(previous_actions),
                    context=xi[None].expand(cfg.horizon, -1, -1).flatten(0, 1),
                )
                r = self.mdp.reward(reward_ctx).view(cfg.horizon, -1)
                d = self.mdp.soft_termination(reward_ctx).view(cfg.horizon, -1)
                survival = SurvivalCumprod.apply(1 - d)
                before = torch.cat((torch.ones_like(survival[:1]), survival[:-1]), 0)
                discounts = (cfg.gamma ** torch.arange(cfg.horizon, device=self.dev))[:, None]
                objective = (discounts * before * r).sum(0)
                objective = objective + cfg.gamma**cfg.horizon * survival[-1] * self.critic(
                    self.critic_input(state, frame, context, goal)
                )
                loss = -objective.mean()
            # Backward/clip/step run outside the graphed forward; the graphed
            # loss is differentiable w.r.t. the live policy parameters.
            self.opt_pi.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
            nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
            # On-GPU non-finite guard (no host sync): sanitize gradients and
            # count occurrences; the counters surface at logging cadence.
            finite = torch.isfinite(norm)
            self._nonfinite_actor += (~finite).to(self._nonfinite_actor.dtype)
            for parameter in self.pi.parameters():
                if parameter.grad is not None:
                    parameter.grad = torch.where(
                        torch.isfinite(parameter.grad), parameter.grad, torch.zeros_like(parameter.grad)
                    )
            self.opt_pi.step()
            if self.pi_avg is not None:
                with torch.no_grad():
                    live = list(self.pi.parameters()) + list(self.pi.buffers())
                    avg = list(self.pi_avg.module.parameters()) + list(self.pi_avg.module.buffers())
                    torch._foreach_lerp_(avg, [x.detach() for x in live], 1 - self.pi_ema)
                    self.pi_avg.n_averaged.add_(1)
        finally:
            _requires_grad(self.wm, True)
            _requires_grad(self.critic, True)
        with torch.no_grad():
            stacked = torch.cat((initial[None], torch.stack(states).detach()), 0).flatten(0, 1)
            fs = torch.cat((start["f"][None], torch.stack(frames)), 0).flatten()
            gs = torch.cat((initial_goal[None], torch.stack(goals)), 0).flatten(0, 1)
            cs = context[None].expand(cfg.horizon + 1, -1, -1).flatten(0, 1)
            features = self.critic_input(stacked, fs, cs, gs).view(cfg.horizon + 1, cfg.pi_batch, -1)
            self.critic.norm.update(features)
            values = self.critic(features)
            target = values[-1]
            targets = []
            for t in reversed(range(cfg.horizon)):
                target = r[t].detach() + cfg.gamma * (1 - d[t].detach()) * (
                    (1 - cfg.lam) * values[t + 1] + cfg.lam * target
                )
                targets.append(target)
            targets = torch.stack(targets[::-1]).flatten()
            features = features[:-1].flatten(0, 1)
            normalized_features = self.critic.norm(features) if self.cfg.trainer.cache_critic_features else None
        value_loss = torch.zeros((), device=self.dev)
        for _ in range(cfg.critic_epochs):
            for ids in torch.randperm(len(targets), device=self.dev).tensor_split(cfg.critic_minibatches):
                if not len(ids):
                    continue
                if normalized_features is not None:
                    prediction = run_mlp(self.critic.net, normalized_features[ids], self.critic.amp_dtype).squeeze(-1)
                else:
                    prediction = self.critic(features[ids])
                value_loss = (prediction - targets[ids]).square().mean()
                self.opt_v.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_v.step()
        self._last_imag_survival = before[-1].detach().mean()
        return (
            loss.detach(),
            value_loss.detach(),
            (before * r).detach().mean() / self.return_normalizer,
            norm.detach(),
        )

    # ------------------------------------------------------------------ checkpoint
    def save(self, wall_time: float) -> None:
        # TODO: persist environment training state through a dedicated API and
        # collector snapshot channel. Sampler state currently restarts fresh.
        self.output.mkdir(parents=True, exist_ok=True)
        ck = {
            "wm": self.wm.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_wm": self.opt_wm.state_dict(),
            "opt_pi": self.opt_pi.state_dict(),
            "opt_v": self.opt_v.state_dict(),
            "observation_spec": self.observation_spec,
            "root_seed": self.root_seed,
            "goal_frames": self.mdp.goal_frames,
        }
        ck["pi"] = self.pi.state_dict()
        ck["pi_live"] = ck["pi"]
        if self.pi_avg is not None:
            ck["pi_avg"] = self.pi_avg.module.state_dict()
            ck["pi_avg_n_averaged"] = int(self.pi_avg.n_averaged)
        ck.update(
            iteration=self.iteration,
            sim_steps=self.sim_steps,
            wall_time=wall_time,
            policy_decay_at=self.decay_at,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            replay_resume=(
                "re-warm; replay, simulator, collector exploration and environment RNG state are not checkpointed"
            ),
        )
        evaluation = self.mdp.imagination.evaluation_metadata
        if evaluation:
            # The task's official evaluator reads config fields from 'args'.
            ck["args"] = {
                "hidden": self.cfg.agent.hidden,
                "hist": self.observation_spec.policy.history_length,
                "goal_frames": self.mdp.goal_frames,
                "pi_ref_skip": self.cfg.agent.pi_ref_skip,
                "amp": self.cfg.trainer.amp,
                "physics_dt": 0.005,
                **evaluation,
            }
        temp = self.output / "last.pt.tmp"
        torch.save(ck, temp)
        temp.replace(self.output / "last.pt")
        snapshot = {k: ck[k] for k in ("wm", "pi", "pi_live", "critic", "iteration", "sim_steps", "wall_time")}
        snapshot["observation_spec"] = self.observation_spec
        snapshot["root_seed"] = self.root_seed
        snapshot["goal_frames"] = self.mdp.goal_frames
        if "args" in ck:
            snapshot["args"] = ck["args"]
        if self.pi_avg is not None:
            snapshot["pi_avg"] = ck["pi_avg"]
            snapshot["pi_avg_n_averaged"] = ck["pi_avg_n_averaged"]
        torch.save(snapshot, self.output / f"policy_{self.iteration:07d}.pt")

    def load(self, path: str) -> None:
        self.load_checkpoint(torch.load(path, map_location=self.dev, weights_only=False))

    def load_checkpoint(self, ck: Mapping[str, Any]) -> None:
        """Restore an already-read checkpoint after construction with its root seed."""
        if ck["goal_frames"] != self.mdp.goal_frames:
            raise ValueError(
                f"Checkpoint goal_frames={ck['goal_frames']} does not match task goal_frames={self.mdp.goal_frames}"
            )
        self.root_seed = ck.get("root_seed", self.root_seed)
        self.wm.load_state_dict(ck["wm"])
        self.critic.load_state_dict(ck["critic"])
        self.opt_wm.load_state_dict(ck["opt_wm"])
        self.opt_pi.load_state_dict(ck["opt_pi"])
        self.opt_v.load_state_dict(ck["opt_v"])
        self.pi.load_state_dict(ck["pi_live"] if "pi_live" in ck else ck["pi"])
        if self.pi_avg is not None and "pi_avg" in ck:
            self.pi_avg.module.load_state_dict(ck["pi_avg"])
            if "pi_avg_n_averaged" in ck:
                self.pi_avg.n_averaged.fill_(ck["pi_avg_n_averaged"])
        self.iteration, self.sim_steps = ck["iteration"], ck["sim_steps"]
        self.decay_at = ck.get("policy_decay_at")
        # RNG is restored before a fresh replay warmup; resume is not
        # trajectory-exact.
        if "torch_rng" in ck:
            torch.set_rng_state(ck["torch_rng"].cpu() if ck["torch_rng"].is_cuda else ck["torch_rng"])
        if ck.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
