# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
# Adapted from SuperMimic (supermimic_vibed), Copyright (c) 2026 ETH Zurich,
# Chong Zhang. MIT License.

"""Collect / world-model fit / imagination learner consuming only the TaskMDP ABI.

Imagination is a differentiable loop inside ``policy_update``: the frozen
world model acts as the dynamics while the TaskMDP's analytic heads provide
observations, rewards, and soft termination. Nothing here knows a robot.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from motrix_rl_supermimic import networks
from motrix_rl_supermimic import state as st
from motrix_rl_supermimic.actor_graph import ActorLossGraph
from motrix_rl_supermimic.config import SuperMimicCfg
from motrix_rl_supermimic.mbrl_env import MbrlEnvBase, compile_task_heads
from motrix_rl_supermimic.networks import Critic, Policy, WorldModel, run_mlp
from motrix_rl_supermimic.optim import MuonAdam, policy_multiplier, trigger_decay
from motrix_rl_supermimic.replay import ReplayBuffer
from motrix_rl_supermimic.survival import survival_cumprod
from motrix_rl_supermimic.task_mdp import TaskMDP, optional_task_hook
from motrix_rl_supermimic.wm_graph import WMLossGraph

__all__ = ["Learner", "build_policy"]

MetricSink = Callable[[int, dict[str, float]], None]


def _requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def build_policy(cfg: SuperMimicCfg, mdp: TaskMDP, goal_dim: int, device: str) -> Policy:
    """One policy-construction contract for learning, collection and evaluation."""
    kw = {"hidden": (cfg.agent.hidden,) * 2} if cfg.agent.hidden else {}
    hook = optional_task_hook(mdp, "reference_action") if cfg.agent.pi_ref_skip else None
    reference_action = hook() if hook is not None else None
    return Policy(
        mdp.spec.observation_dim, goal_dim, mdp.spec.action_dim, hist=cfg.agent.hist, z_dim=cfg.agent.z_dim,
        reference_action=reference_action, **kw
    ).to(device)


def _make_optimizer(cfg: SuperMimicCfg, kind: str, model: nn.Module, mlps: list[nn.Module], lr: float,
                    betas=(0.9, 0.95), device: str = "cpu") -> torch.optim.Optimizer:
    on_cuda = torch.device(device).type == "cuda"
    if getattr(cfg.trainer, f"{kind}_opt", "adam") == "muon" and on_cuda:
        return MuonAdam(model, mlps, lr=lr, betas=betas, fused=True)
    fused = {"fused": True} if on_cuda else {}
    return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, **fused)


class Learner:
    def __init__(
        self, cfg: SuperMimicCfg, env: MbrlEnvBase, device: str, output_dir: Path, metrics: MetricSink | None = None
    ):
        self.cfg, self.env, self.dev = cfg, env, device
        init_state = getattr(env, "init_state", None)
        if init_state is not None and getattr(env, "_state", None) is None:
            init_state()
        self.mdp: TaskMDP = env.task_mdp
        spec = env.spec
        a, t = cfg.agent, cfg.trainer
        if (t.cuda_graph_wm_loss or t.cuda_graph_actor_loss) and not device.startswith("cuda"):
            raise ValueError("cuda_graph_wm_loss/cuda_graph_actor_loss require a CUDA device")
        if (t.cuda_graph_wm_loss or t.cuda_graph_actor_loss) and (t.compile or t.compile_wm):
            raise ValueError("cuda_graph_wm_loss/cuda_graph_actor_loss are mutually exclusive with compile/compile_wm")
        networks.AMP_DTYPE = torch.bfloat16 if t.amp == "bf16" else None
        self.goal_dim = spec.command_dim * t.goal_frames
        wm_width = a.wm_hidden or a.hidden
        wm_kw = {"hidden": (wm_width,) * 3} if wm_width else {}
        self.wm = WorldModel(spec, structured=a.wm_structured, contact_modes=a.wm_contact_modes, **wm_kw).to(self.dev)
        self.pi = build_policy(cfg, self.mdp, self.goal_dim, self.dev)
        self.critic = Critic(spec.critic_observation_dim + self.goal_dim, **({"hidden": (a.hidden,) * 2} if a.hidden else {})).to(
            self.dev
        )
        self.opt_wm = _make_optimizer(cfg, "wm", self.wm, [self.wm.net], lr=t.lr, betas=(0.9, 0.999), device=device)
        self.opt_pi = _make_optimizer(cfg, "pi", self.pi, [self.pi.enc, self.pi.dec], lr=t.lr * t.pi_lr_mult,
                                      device=device)
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
        import copy as _copy

        self.wm_imag = (
            _copy.deepcopy(self.wm).to(self.dev)
            if t.overlap_wm and torch.device(device).type == "cuda"
            else self.wm
        )
        if self.wm_imag is not self.wm:
            for p in self.wm_imag.parameters():
                p.requires_grad_(False)
        self.wm_side = torch.cuda.Stream() if self.wm_imag is not self.wm else None
        self.buffer = ReplayBuffer(spec, t.buffer_steps, env.num_envs, self.dev)
        self.history = torch.zeros(env.num_envs, a.hist, spec.observation_dim, device=self.dev)
        self.err_floor = st.error_floor(spec, self.dev)
        self.mirror_state = optional_task_hook(self.mdp, "mirror_state")
        self.mirror_action = optional_task_hook(self.mdp, "mirror_action")
        self.mirror_context = optional_task_hook(self.mdp, "mirror_context")
        normalizer_hook = optional_task_hook(self.mdp, "return_normalizer")
        self.return_normalizer = normalizer_hook() if normalizer_hook is not None else env.action_dt
        if a.wm_mirror > 0 and not (self.mirror_state and self.mirror_action and self.mirror_context):
            raise ValueError("agent.wm_mirror requires the task's mirror_state/mirror_action/mirror_context hooks")
        self._wm_loss_graph = None
        if t.cuda_graph_wm_loss:
            self._wm_loss_graph = WMLossGraph(
                self.wm, self.err_floor, self.mdp.normalize_context, t.wm_window, t.wm_batch,
                compile_kernel=t.compile_wm_kernel,
            )
        self._actor_loss_graph = None
        if t.cuda_graph_actor_loss:
            self._actor_loss_graph = ActorLossGraph(self)
        self.iteration = self.sim_steps = 0
        self.decay_at: int | None = None
        self._return_blocks: list = []
        self._nonfinite_actor = torch.zeros((), device=device)
        self.wall_offset = 0.0
        self.output = Path(output_dir)
        self.metrics = metrics
        self._action_limit = 20.0
        self._last_rollout_e8 = 0.0
        self._last_imag_survival = torch.zeros(())
        if t.compile or t.compile_wm:
            self.wm.step = torch.compile(self.wm.step)
        if t.compile:
            self.pi.forward = torch.compile(self.pi.forward)
            # Keep TaskMDP heads compiled as one shared tensor-only boundary;
            # this covers collection, normalization, and imagination instead
            # of compiling only reward here.
            compile_task_heads(self.mdp)

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

    # ------------------------------------------------------------------ collect
    @torch.no_grad()
    def collect(self, n) -> float:
        rewards = []
        for _ in range(n):
            obs = self.mdp.add_observation_noise(
                self.mdp.observation(self.env.state(), self.env.prev_action, self.env.context, self.env.frame)
            )
            shifted = torch.cat((self.history[:, 1:], obs[:, None]), 1)
            self.history = torch.where(
                (self.env.episode_steps == 0)[:, None, None], obs[:, None].expand_as(self.history), shifted
            )
            goal = self.mdp.command(self.env.frame, self.cfg.trainer.goal_frames).flatten(-2)
            self.pi.obs_norm.update(obs)
            self.pi.goal_norm.update(goal)
            action = self.pi(self.history, goal)
            action = (action + self.cfg.trainer.action_noise * torch.randn_like(action)).clamp(
                -self._action_limit, self._action_limit
            )
            self.env.step(action)
            transition = self.env.last_transition
            self.buffer.add(transition)
            s, s1 = (
                st.unflatten_state(self.env.spec, transition["s"]),
                st.unflatten_state(self.env.spec, transition["s1"]),
            )
            self.wm.in_norm.update(self.wm.inputs(s, transition["a"], self.mdp.normalize_context(transition["xi"])))
            self.wm.out_norm.update(self.wm.targets(s, s1))
            self.wm.inc_norm.update(self.wm.increments(s, s1))
            self.sim_steps += self.env.num_envs
            rewards.append(transition["r"].mean() / self.return_normalizer)
        return torch.stack(rewards).mean() if rewards else torch.zeros((), device=self.dev)

    # ------------------------------------------------------------------ world model
    @torch.no_grad()
    def wm_rollout_health(self, window: int = 8) -> float:
        """Mean per-step normalized error of an 8-step WM rollout from replay."""
        batch = self.buffer.sample_windows(256, window)
        state = st.unflatten_state(self.env.spec, batch["s"][:, 0])
        total, steps = 0.0, 0
        std = torch.maximum(self.wm.inc_norm.std(), self.err_floor)
        for j in range(window):
            state = self.wm.step(state, batch["a"][:, j], self.mdp.normalize_context(batch["xi"][:, j]))
            error = (
                (self.wm.state_error(state, st.unflatten_state(self.env.spec, batch["s1"][:, j])) / std).square().mean()
            )
            if j == window - 1:
                total += float(error)
            steps += 1
        return total

    def _mirror_windows(self, batch: dict, count: int) -> dict:
        """Replace the first ``count`` windows with their task-mirror image."""
        result = dict(batch)
        spec = self.env.spec
        for key in ("s", "s1"):
            mirrored = st.flatten_state(self.mirror_state(st.unflatten_state(spec, batch[key][:count])))
            result[key] = torch.cat((mirrored, batch[key][count:]))
        result["a"] = torch.cat((self.mirror_action(batch["a"][:count]), batch["a"][count:]))
        result["xi"] = torch.cat((self.mirror_context(batch["xi"][:count]), batch["xi"][count:]))
        return result

    def wm_update(self) -> float:
        cfg = self.cfg.trainer
        batch = self.buffer.sample_windows(cfg.wm_batch, cfg.wm_window)
        if self.cfg.agent.wm_mirror > 0:
            batch = self._mirror_windows(batch, int(cfg.wm_batch * self.cfg.agent.wm_mirror))
        if self._wm_loss_graph is not None:
            loss = self._wm_loss_graph(batch["s"][:, 0], batch["a"], batch["xi"], batch["s1"], batch["mask"])
        else:
            state = st.unflatten_state(self.env.spec, batch["s"][:, 0])
            predictions = []
            for j in range(cfg.wm_window):
                state = self.wm.step(state, batch["a"][:, j], self.mdp.normalize_context(batch["xi"][:, j]))
                predictions.append(state)
            states = tuple(torch.stack([s[i] for s in predictions], 1) for i in range(6))
            std = torch.maximum(self.wm.inc_norm.std(), self.err_floor)
            error = (self.wm.state_error(states, st.unflatten_state(self.env.spec, batch["s1"])) / std).square().mean(-1)
            loss = (error * batch["mask"]).sum() / cfg.wm_batch
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        return loss.detach() / cfg.wm_window

    # ------------------------------------------------------------------ imagination
    def starts(self):
        cfg = self.cfg.trainer
        valid = getattr(self.mdp, "imagination_horizon_valid", None)
        ss, hh, count = [], [], 0
        for _ in range(100):
            s, h = self.buffer.sample_starts(2 * cfg.pi_batch, self.cfg.agent.hist)
            if valid is not None:
                ok = valid(s["f"], cfg.horizon)
            else:
                ok = torch.ones(s["f"].shape, dtype=torch.bool, device=s["f"].device)
            ids = ok.nonzero().flatten()[: cfg.pi_batch - count]
            if ids.numel():
                ss.append({k: v[ids] for k, v in s.items()})
                hh.append({k: v[ids] for k, v in h.items()})
                count += ids.numel()
            if count == cfg.pi_batch:
                cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}
                return cat(ss), cat(hh)
        raise RuntimeError("Replay has insufficient valid imagination starts for requested horizon")

    def critic_input(self, state, frame, context, goal):
        return torch.cat(
            (
                self.mdp.critic_observation(
                    st.flatten_state(state) if isinstance(state, tuple) else state, frame, context
                ),
                goal,
            ),
            -1,
        )

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

    def policy_update(self) -> tuple[float, float, float, float]:
        cfg = self.cfg.trainer
        start, history = self.starts()
        state = st.unflatten_state(self.env.spec, start["s"])
        initial = state
        xi, frame, previous = start["xi"], start["f"], start["a_prev"]
        context = self.mdp.normalize_context(xi)
        with torch.no_grad():
            obs = self.mdp.add_observation_noise(
                self.mdp.observation(
                    st.flatten_state(st.unflatten_state(self.env.spec, history["s"])),
                    history["a_prev"],
                    history["xi"],
                    history["f"],
                )
            )
        goal = initial_goal = self.mdp.command(frame, cfg.goal_frames).flatten(-2)
        z_all = None
        if cfg.batch_goal_enc:
            frames_all = (frame[None] + torch.arange(cfg.horizon + 1, device=frame.device)[:, None]).reshape(-1)
            goals_all = self.mdp.command(frames_all, cfg.goal_frames).reshape(cfg.horizon + 1, frame.shape[0], -1)
            z_all = self.pi.encode_goal(goals_all)
        states, frames, goals, actions, previous_actions = [], [], [], [], []
        _requires_grad(self.wm, False)
        _requires_grad(self.critic, False)
        try:
            if self._actor_loss_graph is not None:
                loss, states, frames, goals, r, d, before = self._actor_loss_graph(
                    start["s"], xi, frame, previous, obs, initial_goal
                )
            else:
                wm_actor = self.wm_imag if self.wm_imag is not self.wm else self.wm
                for t in range(cfg.horizon):
                    previous_actions.append(previous)
                    action = self.pi(obs, goal, z_all[t] if z_all is not None else None)
                    state = wm_actor.step(state, action, context)
                    frame = frame + 1
                    next_obs = self.mdp.add_observation_noise(
                        self.mdp.observation(st.flatten_state(state), action, xi, frame)
                    )
                    obs = torch.cat((obs[:, 1:], next_obs[:, None]), 1)
                    goal = goals_all[t + 1] if z_all is not None else self.mdp.command(frame, cfg.goal_frames).flatten(-2)
                    states.append(state)
                    frames.append(frame)
                    goals.append(goal)
                    actions.append(action)
                    previous = action
                flat = lambda xs: torch.stack(xs).flatten(0, 1)
                all_states = tuple(flat([s[i] for s in states]) for i in range(6))
                reward_fn = optional_task_hook(self.mdp, "imagined_reward") or self.mdp.reward
                r = reward_fn(st.flatten_state(all_states), flat(frames), flat(actions),
                              flat(previous_actions)).view(cfg.horizon, -1)
                d = self.mdp.soft_termination(st.flatten_state(all_states), flat(frames)).view(
                    cfg.horizon, -1
                )
                survival = survival_cumprod(1 - d)
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
                    parameter.grad = torch.where(torch.isfinite(parameter.grad), parameter.grad,
                                                 torch.zeros_like(parameter.grad))
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
            stacked = tuple(
                torch.cat((initial[i][None], torch.stack([s[i].detach() for s in states])), 0).flatten(0, 1)
                for i in range(6)
            )
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
        value_loss = torch.tensor(0.0)
        for _ in range(cfg.critic_epochs):
            for ids in torch.randperm(len(targets), device=self.dev).tensor_split(cfg.critic_minibatches):
                if not len(ids):
                    continue
                prediction = self.critic(features[ids]).squeeze(-1)
                if normalized_features is not None:
                    prediction = run_mlp(self.critic.net, normalized_features[ids]).squeeze(-1)
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
        self.output.mkdir(parents=True, exist_ok=True)
        ck = {k: getattr(self, k).state_dict() for k in ("wm", "critic", "opt_wm", "opt_pi", "opt_v")}
        if self.pi_avg is not None:
            ck["pi"] = self.pi.state_dict()
            ck["pi_live"] = self.pi.state_dict()
            ck["pi_avg"] = self.pi_avg.module.state_dict()
            ck["pi_avg_n_averaged"] = int(self.pi_avg.n_averaged)
        else:
            ck["pi"] = self.pi.state_dict()
            ck["pi_live"] = self.pi.state_dict()
        ck.update(
            iteration=self.iteration,
            sim_steps=self.sim_steps,
            wall_time=wall_time,
            policy_decay_at=self.decay_at,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            replay_resume="re-warm; replay and simulator state are not checkpointed",
        )
        evaluation = optional_task_hook(self.mdp, "evaluation_metadata")
        if evaluation is not None:
            # The task's official evaluator reads config fields from 'args'.
            ck["args"] = {
                "hidden": self.cfg.agent.hidden, "hist": self.cfg.agent.hist,
                "goal_frames": self.cfg.trainer.goal_frames,
                "pi_ref_skip": self.cfg.agent.pi_ref_skip, "amp": self.cfg.trainer.amp,
                "physics_dt": 0.005, **evaluation(),
            }
        curriculum_hook = optional_task_hook(self.mdp, "curriculum_state")
        if curriculum_hook is not None:
            for key, value in curriculum_hook().items():
                ck[f"curriculum_{key}"] = value
        for key in ("bin_G", "bin_history"):
            if hasattr(self.env, key):
                ck[f"reset_{key}"] = getattr(self.env, key)
        temp = self.output / "last.pt.tmp"
        torch.save(ck, temp)
        temp.replace(self.output / "last.pt")
        snapshot = {k: ck[k] for k in ("wm", "pi", "pi_live", "critic", "iteration", "sim_steps", "wall_time")}
        if "args" in ck:
            snapshot["args"] = ck["args"]
        if self.pi_avg is not None:
            snapshot["pi_avg"] = ck["pi_avg"]
            snapshot["pi_avg_n_averaged"] = ck["pi_avg_n_averaged"]
        torch.save(snapshot, self.output / f"policy_{self.iteration:07d}.pt")

    def load(self, path: str, *, module_only: bool = False) -> None:
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        if module_only:
            for key in ("wm", "pi", "critic"):
                target = "pi_live" if key == "pi" and "pi_live" in ck else key
                module = getattr(self, key)
                if key == "pi" and self.pi_avg is not None and "pi_avg" in ck:
                    module.load_state_dict(ck["pi_live"])
                    self.pi_avg.module.load_state_dict(ck["pi_avg"])
                    if "pi_avg_n_averaged" in ck:
                        self.pi_avg.n_averaged.fill_(ck["pi_avg_n_averaged"])
                    continue
                module.load_state_dict(ck[target])
            return
        for key in ("wm", "pi", "critic", "opt_wm", "opt_pi", "opt_v"):
            module = getattr(self, key)
            if key == "pi" and self.pi_avg is not None and "pi_avg" in ck:
                self.pi.load_state_dict(ck["pi_live"])
                self.pi_avg.module.load_state_dict(ck["pi_avg"])
                if "pi_avg_n_averaged" in ck:
                    self.pi_avg.n_averaged.fill_(ck["pi_avg_n_averaged"])
                continue
            module.load_state_dict(ck[key])
        self.iteration, self.sim_steps = ck["iteration"], ck["sim_steps"]
        self.wall_offset = ck["wall_time"]
        self.decay_at = ck.get("policy_decay_at")
        curriculum_load = optional_task_hook(self.mdp, "load_curriculum")
        curriculum = {
            key.removeprefix("curriculum_"): value
            for key, value in ck.items()
            if key.startswith("curriculum_") and isinstance(value, torch.Tensor)
        }
        if curriculum_load is not None and curriculum:
            curriculum_load(curriculum)
        for key in ("bin_G", "bin_history"):
            if hasattr(self.env, key) and f"reset_{key}" in ck:
                getattr(self.env, key).copy_(ck[f"reset_{key}"])
        # RNG is restored before a fresh replay warmup; resume is not
        # trajectory-exact.
        if "torch_rng" in ck:
            torch.set_rng_state(ck["torch_rng"].cpu() if ck["torch_rng"].is_cuda else ck["torch_rng"])
        if ck.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])

    # ------------------------------------------------------------------ loop
    def train(self) -> None:
        cfg = self.cfg.trainer
        self.collect(max(cfg.warmup_steps, cfg.wm_window))
        started = time.monotonic() - self.wall_offset
        while self.iteration < cfg.iterations:
            self._apply_lr()
            self._credit_accumulator = getattr(self, "_credit_accumulator", 0.0) + 1 / cfg.utd
            n = int(self._credit_accumulator)
            self._credit_accumulator -= n
            sim_reward = self.collect(n)
            if bool(sim_reward > 0):
                self._return_blocks.append(sim_reward)
            wm_loss = 0.0
            for _ in range(cfg.wm_batches):
                wm_loss = self.wm_update()
            policy = self.policy_update() if self.iteration >= cfg.wm_only_iters else (0.0, 0.0, 0.0, 0.0)
            self.iteration += 1
            if self.iteration % cfg.log_every == 0:
                # One-shot decay trigger evaluated at log cadence on the mean
                # normalized training return over the elapsed log interval.
                aggregated = sum(self._return_blocks) / len(self._return_blocks) if self._return_blocks else 0.0
                self._return_blocks = []
                self.decay_at = trigger_decay(cfg, self.iteration, aggregated, self.decay_at)
            if self.metrics is not None and self.iteration % cfg.log_every == 0:
                self.metrics(
                    self.iteration,
                    {
                        "wm_loss": wm_loss,
                        "policy_loss": float(policy[0]),
                        "value_loss": float(policy[1]),
                        "imag_reward": float(policy[2]),
                        "grad_norm": float(policy[3]),
                        "nonfinite_actor": float(self._nonfinite_actor),
                        "sim_reward": float(sim_reward),
                        "sim_steps": float(self.sim_steps),
                        "policy_lr": self.opt_pi.param_groups[0]["lr"],
                        "policy_decay_at": float(self.decay_at) if self.decay_at is not None else float("nan"),
                        "wall_s": time.monotonic() - started,
                    },
                )
            if self.iteration % cfg.ckpt_every == 0 or self.iteration == cfg.iterations:
                self.save(time.monotonic() - started)
