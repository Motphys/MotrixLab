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
from motrix_rl.fastsac.agent import FastSacAgent
from motrix_rl.fastsac.config import FastSacCfg
from motrix_rl.fastsac.wrap import FastSacEnvWrap
from motrix_rl.fastsac.wrap_np import FastSacNpEnvWrap
from motrix_rl.fastsac.wrap_torch import FastSacTorchEnvWrap
from motrix_rl.frameworks import TrainerBase, TrainerContext
from motrix_rl.system_metrics import CpuLoadSampler, GpuMemoryUsageSampler, GpuUtilizationSampler, MemoryUsageSampler

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
        self._writer = None

    # ------------------------------------------------------------------ setup
    def _device(self) -> torch.device:
        if self._rlcfg.device:
            return torch.device(self._rlcfg.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            writer=self._writer,
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

        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(run_dir))
        except Exception:  # tensorboard optional
            self._writer = None

        env = self._make_env(self._context.num_envs, render=self._render)
        agent = self._make_agent(env)
        num_iterations = self._rlcfg.trainer.num_learning_iterations

        if self._resume_from:
            ckpt = torch.load(self._resume_from, map_location=agent.device, weights_only=False)
            agent.load_state_dict(ckpt, load_optimizers=True)
            print(f"[motrix.fastsac sync] resumed from {self._resume_from} at global_step={agent.global_step}")
            if agent.global_step >= num_iterations:
                print(
                    f"[motrix.fastsac sync] WARNING: checkpoint global_step ({agent.global_step}) >= target iters "
                    f"({num_iterations}); nothing to do. Increase trainer iterations in the RL config to train further."
                )

        print(
            f"[motrix.fastsac sync] training '{self._env_name}' on {agent.device} "
            f"num_envs={env.num_envs} iters={num_iterations} (from {agent.global_step})"
        )

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
        if self._writer is not None:
            self._writer.close()

    # -------------------------------------------------------- sync train loop
    def _run_loop(
        self,
        agent: FastSacAgent,
        env: FastSacEnvWrap,
        *,
        num_iterations: int,
        logging_interval: int,
        save_interval: int,
        record_checkpoint,
    ) -> None:
        """Synchronous collect-then-train loop.

        Mirrors the structure of the async trainer (:func:`motrix_rl.fastsac.
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
                if warming and not is_resume:
                    actions = torch.empty(agent.num_envs, agent.act_dim, device=device).uniform_(-1.0, 1.0)
                    actions = actions * agent.actor.action_scale + agent.actor.action_bias
                else:
                    actions = agent.act(obs, deterministic=False)

                next_obs, next_critic_obs, rewards, terminated, truncated = env.step(actions)

                agent.rb.extend(
                    obs, critic_obs, actions, rewards, terminated.long(), truncated.long(), next_obs, next_critic_obs
                )

                ep_return += rewards
                ep_len += 1
                done_idx = torch.nonzero(terminated | truncated, as_tuple=False).flatten()
                for j in done_idx.tolist():
                    recent_returns.append(float(ep_return[j]))
                    recent_lengths.append(float(ep_len[j]))
                    ep_return[j] = 0.0
                    ep_len[j] = 0.0
                    n_episodes += 1
                recent_returns = recent_returns[-100:]
                recent_lengths = recent_lengths[-100:]

                obs = next_obs
                critic_obs = next_critic_obs

                # accumulate per-term reward means over the logging window.
                # info["Reward"] holds each term's per-env contribution as a (N,)
                # array (shared convention across backends); mean over envs here.
                reward_terms = env.last_info.get("Reward")
                if reward_terms:
                    for k, v in reward_terms.items():
                        term_accum[k] = term_accum.get(k, 0.0) + float(v.mean())
                    term_count += 1
                t_collect += time.perf_counter() - t0

                # ---------------- learner phase: num_updates gradient steps on a sampled batch
                metrics = None
                if not warming:
                    t1 = time.perf_counter()
                    metrics = agent.update(cfg.num_updates)
                    t_learn += time.perf_counter() - t1

                if agent.global_step % logging_interval == 0:
                    now = time.time()
                    sps = (agent.global_step - last_log_step) * agent.num_envs / max(now - last_log_time, 1e-6)
                    # Materialize metrics to floats here (and only here). update()
                    # returns raw tensors to avoid per-iter cudaStreamSynchronize.
                    metrics_log = {k: float(v) for k, v in metrics.items()} if metrics is not None else None
                    mean_ret = sum(recent_returns) / len(recent_returns) if recent_returns else float("nan")
                    mean_len = sum(recent_lengths) / len(recent_lengths) if recent_lengths else float("nan")
                    term_means = {k: v / max(term_count, 1) for k, v in term_accum.items()}
                    # window-mean per-iter phase timings (ms); ratio shows collector/learner balance
                    n_iter_win = max(agent.global_step - last_log_step, 1)
                    collect_ms = t_collect * 1000.0 / n_iter_win
                    learn_ms = t_learn * 1000.0 / n_iter_win
                    phase_total = t_collect + t_learn
                    learn_pct = 100.0 * t_learn / max(phase_total, 1e-9)
                    info = env.last_info
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
                        checkpoint_path=last_checkpoint_path,
                    )
                    emit_training_panel(live, stats, title=f"{self._env_name}/motrix.fastsac")
                    if self._writer is not None:
                        self._writer.add_scalar("rollout/mean_return", mean_ret, agent.global_step)
                        self._writer.add_scalar("rollout/mean_ep_len", mean_len, agent.global_step)
                        self._writer.add_scalar("rollout/env_reward", float(rewards.mean()), agent.global_step)
                        self._writer.add_scalar("perf/env_steps_per_s", sps, agent.global_step)
                        self._writer.add_scalar("perf/collect_ms_per_iter", collect_ms, agent.global_step)
                        self._writer.add_scalar("perf/learn_ms_per_iter", learn_ms, agent.global_step)
                        self._writer.add_scalar("perf/learn_pct", learn_pct, agent.global_step)
                        for k, v in env_metrics.items():
                            self._writer.add_scalar(f"metrics/{k}", v, agent.global_step)
                        for k, v in term_means.items():
                            self._writer.add_scalar(f"reward/{k}", v, agent.global_step)
                        if metrics is not None:
                            for k, v in metrics_log.items():
                                self._writer.add_scalar(f"train/{k}", v, agent.global_step)
                    last_log_time, last_log_step = now, agent.global_step
                    term_accum, term_count = {}, 0
                    t_collect, t_learn = 0.0, 0.0

                if save_interval > 0 and self._context.checkpoint_dir is not None and agent.global_step > 0:
                    if agent.global_step % save_interval == 0:
                        path = Path(self._context.checkpoint_dir) / f"model_{agent.global_step:07d}.pt"
                        torch.save(agent.state_dict(), path)
                        record_checkpoint(path)
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
        self._writer = None
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
                obs, _, _, _, _ = env.step(actions)
                if env.render() is False:
                    break
        finally:
            env.close()
