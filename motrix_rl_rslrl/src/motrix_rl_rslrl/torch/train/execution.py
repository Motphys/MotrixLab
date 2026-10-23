# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""RSLRL rollout/update adapter for the shared synchronous execution engine."""

import logging
import statistics
import sys
import time

import torch

from motrix_rl import checkpoints
from motrix_rl.console import TrainingPanelStats, emit_training_panel, open_training_live
from motrix_rl.execution.contracts import Cancellation, Parameters
from motrix_rl.execution.synchronous import SyncExecution
from motrix_rl.system_metrics import (
    CpuLoadSampler,
    GpuMemoryUsageSampler,
    GpuUtilizationSampler,
    MemoryUsageSampler,
    sample_gpu_devices,
)


def run_sync(runner, context, *, cancellation: Cancellation | None = None) -> None:
    """One handoff is a full on-policy rollout followed by one PPO update.

    RSLRL owns storage, returns, optimizer epochs and recurrent/RND semantics.
    Model state is shared serially, not copied through Sample. Checkpoints retain
    OnPolicyRunner's format; iter is the completed rollout/update count.
    """
    alg, env, log = runner.alg, runner.env, runner.logger
    cfg = runner.cfg
    total_iterations = cfg["max_iterations"]
    start_iteration = runner.current_learning_iteration
    steps_per_env = cfg["num_steps_per_env"]
    obs = env.get_observations().to(runner.device)
    alg.train_mode()
    if runner.is_distributed:
        alg.broadcast_parameters()
    log.init_logging_writer()
    primary = not log.disable_logs
    cpu = CpuLoadSampler()
    gpu = GpuUtilizationSampler()
    memory = MemoryUsageSampler()
    gpu_memory = GpuMemoryUsageSampler()
    live = None
    collect_time = learn_time = 0.0
    episodes = 0
    losses = {}
    reward_terms = {}
    env_metrics = {}
    reward_count = metrics_count = 0
    last_checkpoint = None
    started = time.perf_counter()
    window_collect = window_learn = 0.0
    window_iterations = 0
    window_losses = {}

    def save(path, *, final=False):
        nonlocal last_checkpoint
        path.parent.mkdir(parents=True, exist_ok=True)
        runner.save(str(path))
        checkpoints.record_checkpoint_artifact(
            context.run_dir,
            checkpoints.LATEST_TRAINING_STATE,
            path,
            checkpoints.TRAINING_STATE,
            checkpoint_format=context.checkpoint_format,
        )
        if final:
            checkpoints.record_checkpoint_artifact(
                context.run_dir,
                checkpoints.BEST_POLICY,
                path,
                checkpoints.POLICY,
                checkpoint_format=context.checkpoint_format,
            )
        last_checkpoint = str(path)
        context.metrics.event("checkpoint_saved", runner.current_learning_iteration, path=str(path))

    def collect(parameters, token):
        nonlocal obs, collect_time, episodes, reward_count, metrics_count
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(steps_per_env):
                token.raise_if_cancelled()
                actions = alg.act(obs)
                obs, rewards, dones, extras = env.step(actions.to(env.device))
                obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                alg.process_env_step(obs, rewards, dones, extras)
                intrinsic = alg.intrinsic_rewards if cfg["algorithm"]["rnd_cfg"] else None
                log.process_env_step(rewards, dones, extras, intrinsic)
                episodes += int((dones > 0).sum().item())
                for key, value in extras.get("Reward", {}).items():
                    reward_terms[key] = reward_terms.get(key, 0.0) + float(torch.as_tensor(value).float().mean())
                reward_count += 1
                for key, value in extras.get("metrics", {}).items():
                    env_metrics[key] = env_metrics.get(key, 0.0) + float(torch.as_tensor(value).float().mean())
                metrics_count += 1
            token.raise_if_cancelled()
            alg.compute_returns(obs)
        collect_time = time.perf_counter() - t0
        # Storage is algorithm-owned and shared only across serial phases.
        return None

    def learn(sample, token):
        nonlocal learn_time, losses
        t0 = time.perf_counter()
        losses = alg.update()
        learn_time = time.perf_counter() - t0
        return Parameters(None, sample.policy_version + 1)

    def on_step(progress):
        nonlocal window_collect, window_learn, window_iterations, reward_count, metrics_count
        runner.current_learning_iteration = start_iteration + progress.iterations
        iteration = runner.current_learning_iteration
        if not primary:
            return
        if context.checkpoint.interval > 0 and iteration < total_iterations:
            if iteration % context.checkpoint.interval == 0:
                save(context.checkpoint_dir / f"model_{iteration}.pt")
        window_collect += collect_time
        window_learn += learn_time
        window_iterations += 1
        for key, value in losses.items():
            window_losses[key] = window_losses.get(key, 0.0) + float(value)
        if iteration % context.logging.interval != 0 and iteration != total_iterations:
            return
        mean_losses = {key: value / window_iterations for key, value in window_losses.items()}
        # The native writer and shared publication consume the same completed
        # learner-iteration window. Episode extras stay buffered until here.
        log.log(
            it=iteration,
            start_it=start_iteration,
            total_it=total_iterations,
            collect_time=window_collect,
            learn_time=window_learn,
            loss_dict=mean_losses,
            learning_rate=alg.learning_rate,
            action_std=alg.get_policy().output_std,
            rnd_weight=alg.rnd.weight if cfg["algorithm"]["rnd_cfg"] else None,
            num_iterations=window_iterations,
        )
        duration = window_collect + window_learn
        transitions = window_iterations * steps_per_env * env.num_envs * runner.gpu_world_size
        mean_return = statistics.mean(log.rewbuffer) if log.rewbuffer else float("nan")
        mean_length = statistics.mean(log.lenbuffer) if log.lenbuffer else float("nan")
        train_metrics = {
            **mean_losses,
            "learning_rate": alg.learning_rate,
            "action_std": float(alg.get_policy().output_std.mean()),
        }
        terms = {k: v / reward_count for k, v in reward_terms.items()}
        metrics = {k: v / metrics_count for k, v in env_metrics.items()}
        stats = TrainingPanelStats(
            iteration=iteration,
            learning_iteration=iteration,
            total_iterations=total_iterations,
            steps_per_second=transitions / max(duration, 1e-9),
            elapsed_seconds=time.perf_counter() - started,
            mean_return=mean_return,
            mean_episode_length=mean_length,
            episodes=episodes,
            buffer_size=None,
            buffer_capacity=None,
            collect_ms=1000 * window_collect / window_iterations,
            learn_ms=1000 * window_learn / window_iterations,
            learn_percent=100 * window_learn / max(duration, 1e-9),
            training_metrics=train_metrics,
            reward_terms=terms,
            env_metrics=metrics,
            cpu_load=cpu.sample(),
            gpu_utilization_percent=gpu.sample(),
            memory_usage=memory.sample(),
            gpu_memory_usage=gpu_memory.sample(),
            gpu_devices=sample_gpu_devices(gpu, gpu_memory),
            checkpoint_path=last_checkpoint,
        )
        emit_training_panel(live, stats, title=f"{context.env_name}/rslrl.ppo")
        scalars = {
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"reward/{k}": v for k, v in terms.items()},
            **{f"metrics/{k}": v for k, v in metrics.items()},
            "perf/env_steps_per_s": stats.steps_per_second,
            "perf/collect_ms_per_iter": stats.collect_ms,
            "perf/learn_ms_per_iter": stats.learn_ms,
            "progress/learning_iteration": iteration,
        }
        if log.rewbuffer:
            scalars.update({"rollout/mean_return": mean_return, "rollout/mean_ep_len": mean_length})
        context.metrics.scalars(
            scalars,
            iteration,
            env_steps=iteration * steps_per_env * env.num_envs * runner.gpu_world_size,
        )
        reward_terms.clear()
        env_metrics.clear()
        window_losses.clear()
        reward_count = metrics_count = 0
        window_collect = window_learn = 0.0
        window_iterations = 0

    try:
        if primary:
            _, live = open_training_live()
            context.metrics.event("run_started", start_iteration)
        SyncExecution().run(
            collect,
            learn,
            initial_parameters=Parameters(None, start_iteration),
            max_iterations=max(0, total_iterations - start_iteration),
            cancellation=cancellation,
            on_step=on_step,
        )
        if primary:
            save(checkpoints.final_checkpoint_path(context.checkpoint_format, context.run_dir), final=True)
    finally:
        if live is not None:
            active_error = sys.exc_info()[1]
            try:
                live.stop()
            except BaseException:
                if active_error is None:
                    raise
                logging.getLogger(__name__).warning("closing training console failed", exc_info=True)
