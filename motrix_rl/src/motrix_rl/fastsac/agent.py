# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC agent (asymmetric-observation port of holosoma's ``FastSACAgent``).

Distributional (C51) Soft Actor-Critic with a tanh-squashed Gaussian policy on
the actor observation and twin distributional Q networks on a privileged critic
observation; auto-tuned temperature, n-step returns and running observation
normalization. CNN, left/right symmetry augmentation, ONNX export and multi-GPU
paths from the original are omitted.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import nn, optim

from motrix_rl.fastsac.buffer import EmpiricalNormalization, SimpleReplayBuffer
from motrix_rl.fastsac.config import FastSacAgentCfg, SonicSacCfg
from motrix_rl.fastsac.factory import make_actor
from motrix_rl.fastsac.networks import Critic


def _own_value(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_own_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _own_value(item) for key, item in value.items()}
    return value


def _own(outputs: tuple) -> tuple:
    """Copy compiled-region outputs out of the CUDA graph's memory pool.

    With ``mode="reduce-overhead"`` the tensors a compiled function returns live
    in the graph's pool, and the next ``cudagraph_mark_step_begin()`` invalidates
    them -- reading one afterwards raises rather than returning stale data. These
    outputs do outlive the call that made them: the metrics travel to the next log
    record, and the actor pair is carried across policy-frequency gating. So they
    are copied out here, while they are still valid.

    Nested tuples and metric mappings are copied recursively so every compiled
    output has the same ownership rule. Cloning a handful of 0-dim tensors costs
    nothing and does not synchronize or move data between devices; what the
    caller must keep avoiding is ``.item()`` / ``float()``, which does.
    See https://docs.pytorch.org/docs/2.7/torch.compiler_cudagraph_trees.html
    -- "clone tensors of a prior iteration (outside of torch.compile) before you
    begin the next run".
    """
    return tuple(_own_value(value) for value in outputs)


class FastSacAgent:
    def __init__(
        self,
        obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
        num_envs: int,
        cfg: FastSacAgentCfg,
        device: torch.device,
        sonic_cfg: SonicSacCfg | None = None,
        action_scale: torch.Tensor | None = None,
        action_bias: torch.Tensor | None = None,
        writer=None,
    ):
        """Build the actor, twin distributional critics, optimizers and replay buffer.

        The observation is asymmetric: the actor sees ``obs_dim`` features while
        the critic sees a privileged ``critic_obs_dim`` observation. ``num_envs``
        sizes the replay buffer's per-env rings. ``action_scale`` / ``action_bias``
        map the tanh-squashed policy output into the environment's action range
        (defaults to identity when ``None``). AMP autocast and ``torch.compile``
        are enabled per ``cfg`` but auto-disabled on CPU.
        """
        self.cfg = cfg
        self.device = device
        self.obs_dim = obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.act_dim = act_dim
        self.num_envs = num_envs
        self.writer = writer
        self.global_step = 0
        # Persistent gradient-update counter shared by all trainers. Used for
        # policy_frequency gating (so the actor/Q ratio is exactly 1/policy_freq
        # across calls, not just within one) and exposed to trainers for UTD /
        # publish-cadence bookkeeping.
        self.update_idx = 0
        self._last_update_timing_ms: dict[str, float] = {}

        self.sonic_cfg = sonic_cfg
        self.actor = make_actor(
            cfg,
            sonic_cfg,
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_scale=action_scale if action_scale is not None else torch.ones(act_dim, device=device),
            action_bias=action_bias if action_bias is not None else torch.zeros(act_dim, device=device),
            device=device,
        )
        critic_kwargs = dict(
            n_obs=critic_obs_dim,
            n_act=act_dim,
            num_atoms=cfg.num_atoms,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
            hidden_dim=cfg.critic_hidden_dim,
            use_layer_norm=cfg.use_layer_norm,
            num_q_networks=cfg.num_q_networks,
            device=device,
        )
        self.qnet = Critic(**critic_kwargs)
        self.qnet_target = Critic(**critic_kwargs)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        self.log_alpha = torch.tensor([math.log(cfg.alpha_init)], requires_grad=True, device=device)
        self.target_entropy = -act_dim * cfg.target_entropy_ratio

        fused = device.type == "cuda"
        self.q_optimizer = optim.AdamW(
            self.qnet.parameters(),
            lr=cfg.critic_learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
            fused=fused,
        )
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=cfg.actor_learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
            fused=fused,
        )
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=cfg.alpha_learning_rate, betas=(0.9, 0.95), fused=fused)

        if cfg.obs_normalization:
            selector_dims = 2 if sonic_cfg is not None and sonic_cfg.enabled else 0
            self.obs_normalizer: nn.Module = EmpiricalNormalization(
                shape=obs_dim, device=device, passthrough_dims=selector_dims
            )
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(shape=critic_obs_dim, device=device)
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        self.rb = SimpleReplayBuffer(
            n_env=num_envs,
            buffer_size=cfg.buffer_size,
            n_obs=obs_dim,
            n_act=act_dim,
            n_critic_obs=critic_obs_dim,
            n_steps=cfg.num_steps,
            gamma=cfg.gamma,
            device=device,
        )

        # AMP / autocast setup: auto-disabled on CPU. dtype resolved from cfg.amp_dtype.
        self._amp_enabled = bool(cfg.amp) and device.type == "cuda"
        self._amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
            cfg.amp_dtype.lower() if isinstance(cfg.amp_dtype, str) else "bf16",
            torch.bfloat16,
        )

        # Runtime callables default to the canonical modules. Checkpointing,
        # optimizer ownership and inter-process weight publication always use
        # the canonical modules so torch.compile remains a runtime-only detail.
        self._actor_runtime = self.actor
        self._qnet_runtime = self.qnet
        self._qnet_target_runtime = self.qnet_target
        self._update_main_runtime = self._update_main
        self._update_pol_runtime = self._update_pol

        # Stage A compile boundary: compile the complete learner update rather
        # than compiling individual network modules. This follows Holosoma's
        # FastSAC structure and avoids nested eager/compiled boundaries around
        # actor sampling, projection, loss, backward, and optimizers.
        if bool(cfg.compile) and device.type == "cuda":
            import torch._inductor.config as inductor_config

            inductor_config.compile_threads = 1
            # B3 experiment: reduce-overhead wraps each compiled update in a
            # CUDA graph (trees), removing per-kernel launch and region-gap CPU
            # time inside the hot learner loop.
            self._update_main_runtime = torch.compile(self._update_main, mode="reduce-overhead")
            self._update_pol_runtime = torch.compile(self._update_pol, mode="reduce-overhead")

    # --------------------------------------------------------------- normalize
    @staticmethod
    def _norm(normalizer: nn.Module, obs: torch.Tensor, update: bool) -> torch.Tensor:
        if isinstance(normalizer, EmpiricalNormalization):
            return normalizer(obs, update=update)
        return obs

    def _autocast(self):
        """torch.autocast context manager, or a no-op when AMP is disabled.

        Wrap forward+backward of a single update step. Optimizer.step() runs
        outside the context (standard PyTorch AMP pattern — grads are fp32).
        """
        if self._amp_enabled:
            return torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)
        import contextlib

        return contextlib.nullcontext()

    # --------------------------------------------------------------- updates
    def _update_main(self, b: dict):
        cfg = self.cfg
        rewards = b["rewards"]
        dones = b["dones"].bool()
        truncations = b["truncations"].bool()
        bootstrap = (truncations | ~dones).float()

        with self._autocast():
            with torch.no_grad():
                next_actions, next_logp = self._actor_runtime.get_actions_and_log_probs(b["next_obs"])
                discount = cfg.gamma ** b["effective_n_steps"]
                target_distributions = self._qnet_target_runtime.projection(
                    b["next_critic_obs"],
                    next_actions,
                    rewards - discount * bootstrap * self.log_alpha.exp() * next_logp,
                    bootstrap,
                    discount,
                )
                target_values = self._qnet_target_runtime.get_value(target_distributions)

            q_outputs = self._qnet_runtime(b["critic_obs"], b["actions"])
            critic_log_probs = F.log_softmax(q_outputs, dim=-1)
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
            qf_loss = critic_losses.mean(dim=1).sum(dim=0)

        self.q_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), cfg.max_grad_norm)
        self.q_optimizer.step()

        alpha_loss = torch.zeros((), device=self.device)
        if cfg.use_autotune:
            alpha_loss = (-self.log_alpha.exp() * (next_logp.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        return (
            qf_loss.detach().float(),
            alpha_loss.detach().float(),
            target_values.max().detach().float(),
            target_values.min().detach().float(),
        )

    def _update_pol(self, b: dict):
        with self._autocast():
            auxiliary = {}
            auxiliary_forward = getattr(self._actor_runtime, "get_actions_and_log_probs_with_aux", None)
            if auxiliary_forward is None:
                actions, log_probs = self._actor_runtime.get_actions_and_log_probs(b["obs"])
            else:
                actions, log_probs, auxiliary = auxiliary_forward(b["obs"])
            q_outputs = self._qnet_runtime(b["critic_obs"], actions)
            q_values = self._qnet_runtime.get_value(F.softmax(q_outputs, dim=-1))
            qf_value = q_values.mean(dim=0)
            policy_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()
            auxiliary_loss = auxiliary.get("total", policy_loss.new_zeros(()))
            actor_loss = policy_loss + auxiliary_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.actor_optimizer.step()
        auxiliary_metrics = {
            f"sonic_{name}": value.detach().float() for name, value in auxiliary.items() if value.numel() == 1
        }
        auxiliary_metrics["sonic_auxiliary_loss"] = auxiliary_loss.detach().float()
        return actor_loss.detach().float(), (-log_probs.mean()).detach().float(), auxiliary_metrics

    @torch.no_grad()
    def _soft_update(self):
        tau = self.cfg.tau
        src = [p.data for p in self.qnet.parameters()]
        tgt = [p.data for p in self.qnet_target.parameters()]
        torch._foreach_mul_(tgt, 1.0 - tau)
        torch._foreach_add_(tgt, src, alpha=tau)

    def update(self, num_updates: int):
        """Run ``num_updates`` gradient steps, each on a fresh batch.

        ``num_updates`` is required — callers pass either ``cfg.num_updates``
        (sync) or a UTD-governed count (async). Returns a dict of **raw GPU
        tensors** (no float conversion) for the last update, or ``None`` if
        ``num_updates <= 0``. Converting to python floats would force a
        cudaStreamSynchronize per call, draining the GPU pipeline and blocking
        kernel submission for the next env step. The caller should ``.item()``
        / ``float()`` only when actually emitting a log record.

        Those tensors are safe to hold until then: under ``mode="reduce-overhead"``
        they would otherwise live in the CUDA graph pool and be invalidated by the
        next update, so :func:`_own` copies them out at the point they are made.

        Policy-frequency gating uses the persistent ``self.update_idx`` counter
        (not a per-call ``i``), so the actor/Q update ratio is exactly
        ``1 / policy_frequency`` across calls. The counter is bumped by
        ``num_updates`` before returning.
        """
        cfg = self.cfg
        if num_updates <= 0:
            return None
        batch_per_env = max(cfg.batch_size // self.num_envs, 1)
        last = (torch.zeros((), device=self.device),) * 5
        self._last_auxiliary = {}
        timing_s = {key: 0.0 for key in ("sample_normalize", "critic_alpha", "actor", "soft_update")}
        update_started = time.perf_counter()
        # Batched data preparation (Holosoma-style): sample once and normalize
        # once per update() call, then slice views into per-gradient-step
        # batches. Removes the per-step randint/gather/normalization eager
        # overhead. Note: the observation normalizer statistics now update once
        # per call on the large batch instead of once per gradient step.
        stage_started = time.perf_counter()
        large_batch = self.rb.sample(batch_per_env * num_updates)
        large_batch["obs"] = self._norm(self.obs_normalizer, large_batch["obs"], update=True)
        large_batch["next_obs"] = self._norm(self.obs_normalizer, large_batch["next_obs"], update=False)
        large_batch["critic_obs"] = self._norm(self.critic_obs_normalizer, large_batch["critic_obs"], update=True)
        large_batch["next_critic_obs"] = self._norm(
            self.critic_obs_normalizer, large_batch["next_critic_obs"], update=False
        )
        rows = large_batch["obs"].shape[0]
        step_rows = rows // num_updates
        prepared = [
            {
                key: (value[off : off + step_rows] if torch.is_tensor(value) else value)
                for key, value in large_batch.items()
            }
            for off in range(0, rows, step_rows)
        ]
        timing_s["sample_normalize"] += time.perf_counter() - stage_started

        for i in range(num_updates):
            b = prepared[i]
            # Required with reduce-overhead (CUDA graph trees): open a new graph
            # generation for this update. NOTE it does not preserve anything --
            # it *invalidates* the previous generation's outputs, which is why
            # every output that outlives its own iteration goes through `_own`.
            torch.compiler.cudagraph_mark_step_begin()
            stage_started = time.perf_counter()
            qf_loss, alpha_loss, qf_max, qf_min = _own(self._update_main_runtime(b))
            timing_s["critic_alpha"] += time.perf_counter() - stage_started

            actor_loss, entropy = last[3], last[4]
            if (self.update_idx + i) % cfg.policy_frequency == 0:
                stage_started = time.perf_counter()
                actor_loss, entropy, auxiliary_metrics = _own(self._update_pol_runtime(b))
                self._last_auxiliary = auxiliary_metrics
                timing_s["actor"] += time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            self._soft_update()
            timing_s["soft_update"] += time.perf_counter() - stage_started
            last = (qf_loss, alpha_loss, qf_max, actor_loss, entropy)
        self.update_idx += num_updates
        timing_s["total"] = time.perf_counter() - update_started
        self._last_update_timing_ms = {key: value * 1000.0 for key, value in timing_s.items()}
        return {
            "qf_loss": last[0],
            "alpha_loss": last[1],
            "qf_max": last[2],
            "actor_loss": last[3],
            "policy_entropy": last[4],
            "alpha": self.log_alpha.exp(),
            **self._last_auxiliary,
        }

    # --------------------------------------------------------------- rollout
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample actions from the policy for a batch of actor observations.

        Normalizes ``obs`` with the (frozen, ``update=False``) observation
        normalizer, then returns tanh-squashed actions in the environment's
        action range. With ``deterministic=True`` the policy mean is used
        (no exploration noise) — the mode for evaluation / rollout replay.
        """
        norm_obs = self._norm(self.obs_normalizer, obs, update=False)
        return self._actor_runtime.explore(norm_obs, deterministic=deterministic)

    def set_train_mode(self) -> None:
        """Put actor + obs-normalizers into training mode.

        Trainers (sync or async) call this once at the start of training. Kept
        as a helper so the isinstance check against :class:`EmpiricalNormalization`
        lives next to the normalizer construction in :meth:`__init__` rather
        than leaking into every trainer.
        """
        self.actor.train()
        for m in (self.obs_normalizer, self.critic_obs_normalizer):
            if isinstance(m, EmpiricalNormalization):
                m.train()

    # --------------------------------------------------------------- io
    def state_dict(self) -> dict:
        """Serialize full training state for checkpointing.

        Includes canonical network state (actor, both critics, critic target), the temperature
        ``log_alpha``, both observation normalizers, all three optimizers and the
        ``global_step``. Optimizer state is included so training can resume
        exactly; :meth:`load_state_dict` can skip it for inference-only loads.
        Compiled runtime wrappers are deliberately excluded from the checkpoint
        contract so enabling ``torch.compile`` never changes parameter names.
        """
        return {
            "actor": self.actor.state_dict(),
            "qnet": self.qnet.state_dict(),
            "qnet_target": self.qnet_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "obs_normalizer": self.obs_normalizer.state_dict(),
            "critic_obs_normalizer": self.critic_obs_normalizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "global_step": self.global_step,
            "update_idx": self.update_idx,
            "sonic": {
                "enabled": bool(self.sonic_cfg is not None and self.sonic_cfg.enabled),
                "profile": self.sonic_cfg.profile if self.sonic_cfg is not None else None,
            },
        }

    def load_state_dict(self, ckpt: dict, load_optimizers: bool = True) -> None:
        """Restore training state from a :meth:`state_dict` checkpoint.

        Loads networks, ``log_alpha`` and both normalizers unconditionally. With
        ``load_optimizers=False`` (or when the checkpoint predates optimizer
        serialization) the optimizer states are skipped — the mode for
        inference-only / play loads that don't resume training.
        """
        self.actor.load_state_dict(ckpt["actor"])
        self.qnet.load_state_dict(ckpt["qnet"])
        self.qnet_target.load_state_dict(ckpt["qnet_target"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.obs_normalizer.load_state_dict(ckpt["obs_normalizer"])
        self.critic_obs_normalizer.load_state_dict(ckpt["critic_obs_normalizer"])
        # optimizer states are optional (older checkpoints / inference-only loads)
        if load_optimizers:
            if "q_optimizer" in ckpt:
                self.q_optimizer.load_state_dict(ckpt["q_optimizer"])
            if "actor_optimizer" in ckpt:
                self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
            if "alpha_optimizer" in ckpt:
                self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        self.update_idx = ckpt.get("update_idx", self.update_idx)
