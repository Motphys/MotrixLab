# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import random
import time
from pathlib import Path

import numpy as np
import torch

from motrix_env_core import registry as env_registry
from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.renderer import RenderConfig
from motrix_env_motrixsim.torch_env import TorchEnv
from motrix_rl import checkpoints
from motrix_rl.console import TrainingPanelStats, emit_training_panel, open_training_live
from motrix_rl.contracts import TrainerBase, TrainerContext
from motrix_rl.device import resolve_default_device
from motrix_rl.system_metrics import (
    CpuLoadSampler,
    GpuMemoryUsageSampler,
    GpuUtilizationSampler,
    MemoryUsageSampler,
    sample_gpu_devices,
)
from motrix_rl.training_logger import TrainingLogger
from motrix_rl_builtin.fastsac.agent import FastSacAgent
from motrix_rl_builtin.fastsac.config import FastSacCfg
from motrix_rl_builtin.fastsac.live_metrics import ROLLOUT_MEANS, system_scalars
from motrix_rl_builtin.fastsac.wrap import FastSacEnvWrap
from motrix_rl_builtin.fastsac.wrap_np import FastSacNpEnvWrap
from motrix_rl_builtin.fastsac.wrap_torch import FastSacTorchEnvWrap

# Enable TF32 matmul on Ampere+ GPUs. SAC training has no precision concern with
# TF32 (10 mantissa bits), and the speedup is meaningful when AMP is off.
# Auto-noop on pre-Ampere GPUs.
torch.set_float32_matmul_precision("high")


class Trainer(TrainerBase):
    def __init__(
        self,
        *,
        context: TrainerContext[FastSacCfg],
    ) -> None:
        env_name = context.env_name
        self._rlcfg = context.rl_cfg
        self._env_name = env_name
        self._env_spec = env_registry.resolve(env_name, sim=context.sim)
        self._render = context.render
        self._resume_from = context.resume_from
        self._context = context

    # ------------------------------------------------------------------ setup
    def _device(self) -> torch.device:
        if self._rlcfg.device:
            return torch.device(self._rlcfg.device)
        return resolve_default_device()

    def _make_env(
        self,
        num_envs: int,
        render: RenderConfig | None,
        mode: str = "train",
    ) -> FastSacEnvWrap:
        env = self._env_spec.make(num_envs=num_envs, mode=mode, seed=self._context.seed)
        if isinstance(env, TorchEnv):
            return FastSacTorchEnvWrap(env, self._device(), render=render)
        if isinstance(env, ArrayEnv):
            return FastSacNpEnvWrap(env, self._device(), render=render)
        raise TypeError(f"FastSAC does not support environment type '{type(env).__name__}'.")

    def _make_agent(self, env: FastSacEnvWrap) -> FastSacAgent:
        inner = env.env
        obs_dim = inner.policy_observation_space.shape[-1]
        critic_obs_dim = inner.value_observation_space.shape[-1]
        act_dim = inner.action_space.shape[-1]
        device = self._device()
        low = env.action_low
        high = env.action_high
        action_scale = (high - low) / 2.0
        action_bias = (high + low) / 2.0
        return FastSacAgent(
            obs_dim=obs_dim,
            critic_obs_dim=critic_obs_dim,
            act_dim=act_dim,
            num_envs=env.num_envs,
            cfg=self._rlcfg.agent,
            device=device,
            action_scale=action_scale,
            action_bias=action_bias,
        )

    def _set_seed(self, seed: int | None) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------ train
    def train(self) -> None:
        self._set_seed(self._context.seed)

        if self._context.logging.backend != "tensorboard":
            raise ValueError("FastSAC supports only the 'tensorboard' logging backend.")

        run_dir = self._context.run_dir
        training_log = TrainingLogger(run_dir, self._context.metrics)

        try:
            env = self._make_env(self._context.num_envs, render=self._render)
            agent = self._make_agent(env)
            num_iterations = self._rlcfg.trainer.num_learning_iterations

            if self._resume_from:
                ckpt = torch.load(self._resume_from, map_location=agent.device, weights_only=False)
                agent.load_state_dict(ckpt, load_optimizers=True)
                print(f"[motrix.fastsac sync] resumed from {self._resume_from} at global_step={agent.global_step}")
                training_log.event(
                    "resumed", step=agent.global_step, path=str(self._resume_from), global_step=agent.global_step
                )
                if agent.global_step >= num_iterations:
                    print(
                        f"[motrix.fastsac sync] WARNING: checkpoint global_step ({agent.global_step}) >= target "
                        f"iters ({num_iterations}); nothing to do. Increase trainer iterations in the RL config to "
                        "train further."
                    )

            print(
                f"[motrix.fastsac sync] training '{self._env_name}' on {agent.device} "
                f"num_envs={env.num_envs} iters={num_iterations} (from {agent.global_step})"
            )
            training_log.event("run_started", step=agent.global_step)

            def _record_periodic_checkpoint(path) -> None:
                # Record each periodic checkpoint as the latest resumable training
                # state so an interrupted run can still be resumed / played from the
                # manifest, not just after train() completes.
                checkpoints.record_checkpoint_artifact(
                    run_dir,
                    checkpoints.LATEST_TRAINING_STATE,
                    path,
                    checkpoints.TRAINING_STATE,
                    checkpoint_format=self._context.checkpoint_format,
                )

            self._run_loop(
                agent,
                env,
                training_log,
                num_iterations=num_iterations,
                logging_interval=self._context.logging.interval,
                save_interval=self._context.checkpoint.interval,
                record_checkpoint=_record_periodic_checkpoint,
            )

            ckpt_path = checkpoints.final_checkpoint_path(self._context.checkpoint_format, run_dir)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(agent.state_dict(), ckpt_path)
            checkpoints.record_checkpoint_artifact(
                run_dir,
                checkpoints.LATEST_TRAINING_STATE,
                ckpt_path,
                checkpoints.TRAINING_STATE,
                checkpoint_format=self._context.checkpoint_format,
            )
            checkpoints.record_checkpoint_artifact(
                run_dir,
                checkpoints.BEST_POLICY,
                ckpt_path,
                checkpoints.POLICY,
                checkpoint_format=self._context.checkpoint_format,
            )
            print(f"[motrix.fastsac sync] saved checkpoint to {ckpt_path}")
            training_log.event("checkpoint_saved", step=agent.global_step, path=str(ckpt_path))
        finally:
            # Close on failure too, so buffered TensorBoard events are flushed.
            training_log.close()

    # -------------------------------------------------------- sync train loop
    def _run_loop(
        self,
        agent: FastSacAgent,
        env: FastSacEnvWrap,
        training_log: TrainingLogger,
        *,
        num_iterations: int,
        logging_interval: int,
        save_interval: int,
        record_checkpoint,
    ) -> None:
        """Synchronous collect-then-train loop.

        Mirrors the structure of the async trainer (:func:`motrix_rl_builtin.fastsac.
        async_impl.worker.run_*_process`) but without cross-process plumbing:
        a single process alternates a collector phase (env.step + replay buffer
        write) and a learner phase (``num_updates`` gradient steps on a sampled
        batch), sharing the same GPU.

        The agent exposes the primitives (``act`` / ``update`` / ``rb`` /
        ``state_dict`` / ``set_train_mode``); this method owns the orchestration:
        warmup, episode bookkeeping, panel emit, tensorboard logging and
        periodic checkpointing.
        """
        cfg = agent.cfg
        device = agent.device
        agent.set_train_mode()

        obs, critic_obs = env.reset()
        ep_return = torch.zeros(agent.num_envs, device=device)
        ep_len = torch.zeros(agent.num_envs, device=device)
        recent_returns: list[float] = []
        recent_lengths: list[float] = []

        # window accumulators for the console panel
        start_time = time.time()
        last_log_time = start_time
        last_log_step = agent.global_step
        term_accum: dict[str, float] = {}
        term_count = 0
        n_episodes = 0
        # per-iter phase timing accumulators (collector vs learner), reset each log window
        t_collect = 0.0
        t_learn = 0.0
        cpu_sampler = CpuLoadSampler()
        gpu_sampler = GpuUtilizationSampler()
        memory_sampler = MemoryUsageSampler()
        gpu_memory_sampler = GpuMemoryUsageSampler()
        last_checkpoint_path: str | None = None

        # optional live console that refreshes one panel in place
        console, live = open_training_live()

        def emit_msg(msg: str) -> None:
            if console is not None:
                console.print(msg)
            else:
                print(msg)

        # `local` counts steps since this loop started; it gates the
        # warmup / learning_starts so a resumed run (global_step > 0) still
        # re-fills the (empty) replay buffer before training.
        # On a fresh start we use random actions during warmup (exploration). On
        # resume we already have a trained policy, so warm up WITH the policy:
        # this re-fills the buffer with good on-policy data and avoids perturbing
        # the loaded policy (random warmup caused a large transient return drop).
        is_resume = agent.global_step > 0
        local = 0
        try:
            while agent.global_step < num_iterations:
                # ---------------- collector phase: decision + env.step + buffer write + bookkeeping
                t0 = time.perf_counter()
                warming = local < cfg.learning_starts
                if cfg.learning_starts > 0 and local == cfg.learning_starts:
                    training_log.event("warmup_done", step=agent.global_step)
                if warming and not is_resume:
                    actions = torch.empty(agent.num_envs, agent.act_dim, device=device).uniform_(-1.0, 1.0)
                    actions = actions * agent.actor.action_scale + agent.actor.action_bias
                else:
                    actions = agent.act(obs, deterministic=False)

                result = env.step(actions)
                next_obs, next_critic_obs = result.obs, result.critic_obs
                rewards, terminated, truncated = result.reward, result.terminated, result.truncated

                agent.rb.extend(obs, critic_obs, actions, rewards, terminated.long(), truncated.long())

                ep_return += rewards
                ep_len += 1
                done_idx = torch.nonzero(terminated | truncated, as_tuple=False).flatten()
                if done_idx.numel():
                    # Vectorized: one batched gather + clear instead of
                    # per-episode Python-level scalar indexing.
                    recent_returns.extend(ep_return[done_idx].tolist())
                    recent_lengths.extend(ep_len[done_idx].tolist())
                    ep_return[done_idx] = 0.0
                    ep_len[done_idx] = 0.0
                    n_episodes += int(done_idx.numel())
                recent_returns = recent_returns[-100:]
                recent_lengths = recent_lengths[-100:]

                obs = next_obs
                critic_obs = next_critic_obs

                # accumulate per-term reward means over the logging window.
                # info["Reward"] holds each term's per-env contribution as a (N,)
                # array (shared convention across backends); mean over envs here.
                reward_terms = result.info.get("Reward")
                if reward_terms:
                    for k, v in reward_terms.items():
                        term_accum[k] = term_accum.get(k, 0.0) + float(v.mean())
                    term_count += 1
                t_collect += time.perf_counter() - t0

                # ---------------- learner phase: num_updates gradient steps on a sampled batch
                if not warming:
                    t1 = time.perf_counter()
                    agent.update(cfg.num_updates)
                    t_learn += time.perf_counter() - t1

                if agent.global_step % logging_interval == 0:
                    now = time.time()
                    sps = (agent.global_step - last_log_step) * agent.num_envs / max(now - last_log_time, 1e-6)
                    # Window-aggregated diagnostics (means + extremes since the
                    # last log window), shared semantics with the async
                    # trainer's payload.
                    metrics_log = agent.window_metrics.stats_and_reset() if not warming else None
                    mean_ret = sum(recent_returns) / len(recent_returns) if recent_returns else float("nan")
                    mean_len = sum(recent_lengths) / len(recent_lengths) if recent_lengths else float("nan")
                    term_means = {k: v / max(term_count, 1) for k, v in term_accum.items()}
                    # window-mean per-iter phase timings (ms); ratio shows collector/learner balance
                    n_iter_win = max(agent.global_step - last_log_step, 1)
                    collect_ms = t_collect * 1000.0 / n_iter_win
                    learn_ms = t_learn * 1000.0 / n_iter_win
                    phase_total = t_collect + t_learn
                    learn_pct = 100.0 * t_learn / max(phase_total, 1e-9)
                    info = result.info
                    # generic env metrics channel: the wrapper merges the state's
                    # metrics snapshot into infos["metrics"], so backends log them
                    # without hardcoding any metric name.
                    env_metrics = {k: float(np.mean(v)) for k, v in info.get("metrics", {}).items()}
                    stats = TrainingPanelStats(
                        iteration=agent.global_step,
                        total_iterations=num_iterations,
                        steps_per_second=sps,
                        elapsed_seconds=now - start_time,
                        mean_return=mean_ret,
                        mean_episode_length=mean_len,
                        episodes=n_episodes,
                        buffer_size=agent.rb.num_stored * agent.num_envs,
                        buffer_capacity=agent.rb.buffer_size * agent.num_envs,
                        collect_ms=collect_ms,
                        learn_ms=learn_ms,
                        learn_percent=learn_pct,
                        warming=warming,
                        training_metrics=metrics_log,
                        reward_terms=term_means,
                        env_metrics=env_metrics,
                        cpu_load=cpu_sampler.sample(),
                        gpu_utilization_percent=gpu_sampler.sample(),
                        memory_usage=memory_sampler.sample(),
                        gpu_memory_usage=gpu_memory_sampler.sample(),
                        gpu_devices=sample_gpu_devices(gpu_sampler, gpu_memory_sampler),
                        checkpoint_path=last_checkpoint_path,
                    )
                    emit_training_panel(live, stats, title=f"{self._env_name}/motrix.fastsac")
                    scalars = {
                        "rollout/mean_return": mean_ret,
                        "rollout/mean_ep_len": mean_len,
                        "rollout/env_reward": float(rewards.mean()),
                        "perf/env_steps_per_s": sps,
                        "perf/collect_ms_per_iter": collect_ms,
                        "perf/learn_ms_per_iter": learn_ms,
                        "perf/learn_pct": learn_pct,
                        **{f"metrics/{k}": v for k, v in env_metrics.items()},
                        **{f"reward/{k}": v for k, v in term_means.items()},
                        **{f"train/{k}": v for k, v in (metrics_log or {}).items()},
                    }
                    # Iteration global_step has already collected (the counter
                    # advances at the end of the loop), so it is the
                    # (global_step + 1)-th batch of num_envs transitions.
                    training_log.scalars(
                        scalars,
                        agent.global_step,
                        env_steps=(agent.global_step + 1) * agent.num_envs,
                        live_extra=lambda: system_scalars(stats),
                        placeholders=() if recent_returns else ROLLOUT_MEANS,
                    )
                    last_log_time, last_log_step = now, agent.global_step
                    term_accum, term_count = {}, 0
                    t_collect, t_learn = 0.0, 0.0

                if save_interval > 0 and self._context.checkpoint_dir is not None and agent.global_step > 0:
                    if agent.global_step % save_interval == 0:
                        path = Path(self._context.checkpoint_dir) / f"model_{agent.global_step:07d}.pt"
                        torch.save(agent.state_dict(), path)
                        record_checkpoint(path)
                        training_log.event("checkpoint_saved", step=agent.global_step, path=str(path))
                        if console is not None:
                            last_checkpoint_path = str(path)
                        else:
                            emit_msg(f"saved checkpoint {path}")

                local += 1
                agent.global_step += 1
        finally:
            if live is not None:
                live.stop()

    # ------------------------------------------------------------------ play
    def play(self, policy: str) -> None:
        self._set_seed(self._context.seed)
        env = self._make_env(
            self._context.play_num_envs,
            render=self._render,
            mode="play",
        )
        agent = self._make_agent(env)
        ckpt = torch.load(policy, map_location=agent.device, weights_only=False)
        agent.load_state_dict(ckpt)
        agent.actor.eval()
        if hasattr(agent.obs_normalizer, "eval"):
            agent.obs_normalizer.eval()

        obs, _ = env.reset()
        try:
            while True:
                actions = agent.act(obs, deterministic=True)
                obs = env.step(actions).obs
                if env.render() is False:
                    break
        finally:
            env.close()
