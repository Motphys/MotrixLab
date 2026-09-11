# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Collector: samples the environment with a local (inference-only) policy copy.

Holds its own :class:`~motrix_rl.fastsac.networks.Actor` and read-only
:class:`~motrix_rl.fastsac.buffer.EmpiricalNormalization`, both refreshed from
the learner via :class:`~motrix_rl.fastsac.async_impl.shm.WeightSnapshot`. Each step
mirrors the sync collector phase (``agent.py`` collect phase) exactly: decide
action -> ``env.step`` -> push the transition batch to the shared ring -> update
episode bookkeeping. The normalizer is used read-only (``update=False``), matching
the sync ``act()`` path — only the learner ever updates normalizer stats.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

from motrix_rl.fastsac.async_impl.shm import Control, SharedTransitionRing, WeightSnapshot, bind_flat_params
from motrix_rl.fastsac.buffer import EmpiricalNormalization
from motrix_rl.fastsac.config import FastSacAgentCfg, FastSacCfg
from motrix_rl.fastsac.factory import make_actor
from motrix_rl.fastsac.wrap import FastSacEnvWrap


def resolve_collector_inference_device(device_spec: str) -> torch.device:
    """Resolve an explicit collector inference device without CPU fallback."""
    device = torch.device(device_spec)
    if device.type == "cpu":
        return device
    if device.type != "cuda":
        raise ValueError(f"collector_inference_device must be cpu or cuda, got '{device_spec}'")
    if not torch.cuda.is_available():
        raise RuntimeError(f"collector_inference_device='{device_spec}' requested CUDA, but CUDA is unavailable")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"collector_inference_device='{device_spec}' selects CUDA device {index}, "
            f"but only {torch.cuda.device_count()} device(s) are available"
        )
    return torch.device("cuda", index)


class _CollectorPolicy(nn.Module):
    """Read-only normalizer + stochastic actor callable compiled as one graph."""

    def __init__(self, actor: nn.Module, obs_normalizer: nn.Module):
        super().__init__()
        self.actor = actor
        self.obs_normalizer = obs_normalizer

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            return self.obs_normalizer(obs, update=False)
        return obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.explore(self._normalize(obs), deterministic=False)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor.explore(self._normalize(obs), deterministic=True)


class Collector:
    def __init__(
        self,
        env: FastSacEnvWrap,
        cfg: FastSacCfg,
        obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
        action_scale: torch.Tensor,
        action_bias: torch.Tensor,
        ring: SharedTransitionRing,
        weights: WeightSnapshot,
        control: Control,
        is_resume: bool = False,
    ):
        self.env = env
        self.cfg = cfg
        self.async_options = cfg.trainer.async_options
        acfg: FastSacAgentCfg = cfg.agent
        self.device = resolve_collector_inference_device(self.async_options.collector_inference_device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.env_device = torch.device("cpu")
        self.num_envs = env.num_envs
        self.act_dim = act_dim
        self.ring = ring
        self.weights = weights
        self.control = control
        self.is_resume = is_resume
        self._learning_starts = acfg.learning_starts
        self._local_version = 0

        self.actor = make_actor(
            acfg,
            getattr(cfg, "sonic", None),
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_scale=action_scale,
            action_bias=action_bias,
            device=self.device,
        )
        self.actor.eval()
        if acfg.obs_normalization:
            sonic_cfg = getattr(cfg, "sonic", None)
            selector_dims = 2 if sonic_cfg is not None and sonic_cfg.enabled else 0
            self.obs_normalizer = EmpiricalNormalization(
                shape=obs_dim, device=self.device, passthrough_dims=selector_dims
            )
        else:
            self.obs_normalizer = torch.nn.Identity()
        self.obs_normalizer.eval()  # read-only: never updates stats on the collector
        self._policy = _CollectorPolicy(self.actor, self.obs_normalizer)
        self._policy_runtime = self._policy
        self._amp_enabled = bool(self.async_options.collector_amp) and self.device.type == "cuda"
        amp_dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}
        try:
            self._amp_dtype = amp_dtypes[self.async_options.collector_amp_dtype.lower()]
        except KeyError as exc:
            raise ValueError("collector_amp_dtype must be 'fp16' or 'bf16'") from exc

        self._flat_params = bind_flat_params(self.actor) if self.device.type == "cuda" else None
        param_numel = (
            self._flat_params.numel()
            if self._flat_params is not None
            else sum(param.numel() for param in self.actor.parameters())
        )
        pin_weight_staging = self.device.type == "cuda"
        self._weight_param_staging = torch.empty(
            param_numel,
            dtype=torch.float32,
            pin_memory=pin_weight_staging,
        )
        self._weight_buffer_staging = torch.empty(
            self.weights.buffer_numel,
            dtype=torch.float32,
            pin_memory=pin_weight_staging,
        )
        self._weight_normalizer_staging = tuple(
            torch.empty(
                (1, obs_dim),
                dtype=torch.float32,
                pin_memory=pin_weight_staging,
            )
            for _ in range(3)
        )
        self._obs_host = None
        self._obs_device = None
        self._actions_host = None
        if self.device.type == "cuda":
            self._obs_host = torch.empty((self.num_envs, obs_dim), dtype=torch.float32, pin_memory=True)
            self._obs_device = torch.empty_like(self._obs_host, device=self.device)
            self._actions_host = torch.empty((self.num_envs, act_dim), dtype=torch.float32, pin_memory=True)
            if bool(self.async_options.collector_compile):
                import torch._inductor.config as inductor_config

                inductor_config.compile_threads = 1
                self._policy_runtime = torch.compile(self._policy, mode="reduce-overhead")

        self._action_scale_cpu = self.actor.action_scale.detach().cpu()
        self._action_bias_cpu = self.actor.action_bias.detach().cpu()

        # rollout state
        self.obs = None
        self.critic_obs = None
        # episode bookkeeping (owned here since the collector sees reward/done)
        self.ep_return = torch.zeros(self.num_envs, device=self.env_device)
        self.ep_len = torch.zeros(self.num_envs, device=self.env_device)
        self.recent_returns: list[float] = []
        self.recent_lengths: list[float] = []
        self.n_episodes = 0
        self.term_accum: dict[str, float] = {}
        self.term_count = 0
        self.last_env_metrics: dict[str, float] = {}
        # windowed collector timing (wall-clock spent producing one env-step batch)
        self._collect_t = 0.0
        self._sample_actions_t = 0.0
        self._env_step_t = 0.0
        self._push_t = 0.0
        self._bookkeep_t = 0.0
        self._sync_t = 0.0
        self._sync_wait_writer_t = 0.0
        self._sync_host_snapshot_t = 0.0
        self._sync_actor_load_t = 0.0
        self._wait_t = 0.0
        self._wait_started: float | None = None
        self._collect_n = 0

    def reset(self) -> None:
        self.obs, self.critic_obs = self.env.reset()

    def _autocast(self):
        if self._amp_enabled:
            return torch.autocast(device_type="cuda", dtype=self._amp_dtype)
        import contextlib

        return contextlib.nullcontext()

    @torch.no_grad()
    def _infer(self, obs: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cpu":
            return self._policy_runtime(obs)

        self._obs_host.copy_(obs)
        self._obs_device.copy_(self._obs_host, non_blocking=True)
        with self._autocast():
            device_actions = self._policy_runtime(self._obs_device)
        self._actions_host.copy_(device_actions, non_blocking=True)
        # reduce-overhead may reuse CUDA Graph output storage. Complete D2H
        # before returning so the next replay cannot overwrite env actions.
        torch.cuda.current_stream(self.device).synchronize()
        return self._actions_host

    def warmup_inference(self) -> None:
        """Compile/warm the fixed collector batch shape before the first env step."""
        if self.device.type == "cuda" and bool(self.async_options.collector_compile):
            self._infer(self.obs)
            self._infer(self.obs)

    @torch.no_grad()
    def _sample_actions(self, warming: bool) -> torch.Tensor:
        if warming and not self.is_resume:
            actions = torch.empty(self.num_envs, self.act_dim, device=self.env_device).uniform_(-1.0, 1.0)
            return actions * self._action_scale_cpu + self._action_bias_cpu
        return self._infer(self.obs)

    def sync_weights(self, *, record_timing: bool = False) -> None:
        version, wait_writer_s, host_snapshot_s, actor_load_s = self.weights.maybe_load(
            self.actor,
            self.obs_normalizer,
            self._local_version,
            param_staging=self._weight_param_staging,
            normalizer_staging=self._weight_normalizer_staging,
            buffer_staging=self._weight_buffer_staging,
            flat_params=self._flat_params,
        )
        self._local_version = version
        if record_timing:
            self._sync_wait_writer_t += wait_writer_s
            self._sync_host_snapshot_t += host_snapshot_s
            self._sync_actor_load_t += actor_load_s

    @property
    def policy_lag(self) -> int:
        """How many published versions behind the collector's local policy is."""
        return max(0, self.weights.version - self._local_version)

    # ------------------------------------------------------------------ step
    def step_once(self) -> bool:
        """Run one env-step batch and push it to the ring.

        Returns ``False`` if the ring was full (backpressure) — the caller should
        let the learner drain and retry. The env is only stepped when there is
        room, so no transition is ever dropped.
        """
        now = time.perf_counter()
        if self.ring.is_full():
            if self._wait_started is None:
                self._wait_started = now
            return False

        if self._wait_started is not None:
            self._wait_t += now - self._wait_started
            self._wait_started = None

        t0 = now
        warming = self.control.collector_steps < self._learning_starts
        t_sample_actions = time.perf_counter()
        actions = self._sample_actions(warming)
        if not torch.isfinite(actions).all():
            raise RuntimeError("collector policy produced non-finite actions")
        t_env = time.perf_counter()
        next_obs, next_critic_obs, rewards, terminated, truncated = self.env.step(actions)
        t_push = time.perf_counter()

        pushed = self.ring.push(
            self.obs.detach().cpu(),
            self.critic_obs.detach().cpu(),
            actions.detach().cpu(),
            rewards.detach().cpu(),
            terminated.detach().long().cpu(),
            truncated.detach().long().cpu(),
            next_obs.detach().cpu(),
            next_critic_obs.detach().cpu(),
        )
        assert pushed, "ring became full after is_full() check — single-producer invariant violated"
        t_bookkeep = time.perf_counter()

        # episode bookkeeping (mirrors sync agent.py collect phase)
        self.ep_return += rewards
        self.ep_len += 1
        done_idx = torch.nonzero(terminated | truncated, as_tuple=False).flatten()
        for j in done_idx.tolist():
            self.recent_returns.append(float(self.ep_return[j]))
            self.recent_lengths.append(float(self.ep_len[j]))
            self.ep_return[j] = 0.0
            self.ep_len[j] = 0.0
            self.n_episodes += 1
        self.recent_returns = self.recent_returns[-100:]
        self.recent_lengths = self.recent_lengths[-100:]

        info = self.env.last_info
        reward_terms = info.get("Reward")
        if reward_terms:
            for k, v in reward_terms.items():
                self.term_accum[k] = self.term_accum.get(k, 0.0) + float(v.mean())
            self.term_count += 1
        self.last_env_metrics = {k: float(np.mean(v)) for k, v in info.get("metrics", {}).items()}

        self.obs = next_obs
        self.critic_obs = next_critic_obs
        self.control.inc_collector_steps()
        t_sync = time.perf_counter()

        if self.control.collector_steps % max(self.async_options.weight_poll_interval, 1) == 0:
            self.sync_weights(record_timing=True)
        t_done = time.perf_counter()

        self._collect_t += t_done - t0
        self._sample_actions_t += t_env - t_sample_actions
        self._env_step_t += t_push - t_env
        self._push_t += t_bookkeep - t_push
        self._bookkeep_t += t_sync - t_bookkeep
        self._sync_t += t_done - t_sync
        self._collect_n += 1
        return True

    # ------------------------------------------------------------------ stats
    def snapshot_stats(self) -> dict:
        """Compact rollout-stats snapshot for the learner's log panel.

        Resets the windowed reward-term accumulator so each snapshot reflects the
        interval since the last call. Used by the M1 collector process to ship
        episode/reward info to the learner process (which owns logging).
        """
        rr, rl = self.recent_returns, self.recent_lengths
        term_means = {k: v / max(self.term_count, 1) for k, v in self.term_accum.items()}
        stats = {
            "return": (sum(rr) / len(rr)) if rr else float("nan"),
            "ep_len": (sum(rl) / len(rl)) if rl else float("nan"),
            "episodes": self.n_episodes,
            "reward_terms": term_means,
            "env_metrics": dict(self.last_env_metrics),
            "policy_lag": self.policy_lag,
            # Average wall-clock milliseconds per successful env-step batch.
            "timing_ms": {
                "collect": self._collect_t * 1000.0 / max(self._collect_n, 1),
                "wait": self._wait_t * 1000.0 / max(self._collect_n, 1),
                "sample_actions": self._sample_actions_t * 1000.0 / max(self._collect_n, 1),
                "env_step": self._env_step_t * 1000.0 / max(self._collect_n, 1),
                "push": self._push_t * 1000.0 / max(self._collect_n, 1),
                "bookkeep": self._bookkeep_t * 1000.0 / max(self._collect_n, 1),
                "sync": self._sync_t * 1000.0 / max(self._collect_n, 1),
                "sync_wait_writer": self._sync_wait_writer_t * 1000.0 / max(self._collect_n, 1),
                "sync_host_snapshot": self._sync_host_snapshot_t * 1000.0 / max(self._collect_n, 1),
                "sync_actor_load": self._sync_actor_load_t * 1000.0 / max(self._collect_n, 1),
            },
        }
        self.term_accum, self.term_count = {}, 0
        self._collect_t = 0.0
        self._sample_actions_t = 0.0
        self._env_step_t = 0.0
        self._push_t = 0.0
        self._bookkeep_t = 0.0
        self._sync_t = 0.0
        self._sync_wait_writer_t = 0.0
        self._sync_host_snapshot_t = 0.0
        self._sync_actor_load_t = 0.0
        self._wait_t = 0.0
        self._collect_n = 0
        return stats
