# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Async (heterogeneous collector/learner) FastSAC trainer.

``train()`` spawns a collector process (CPU env + inference) and a learner
process (GPU training) that exchange data through the shared-memory ring /
weight snapshot, so CPU simulation and GPU training overlap. The parent process
reads env dims, allocates the shared-memory primitives, spawns both workers and
manages their lifecycle (stop flag, join, crash propagation, cleanup).

Env dims and the play path reuse the sync FastSAC building blocks; the
per-process training bodies live in ``worker.py``. Select this topology with
``algo.asynchronous=true`` on the shared ``motrix.fastsac`` task.
"""

from __future__ import annotations

import random
import time
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp

from motrix_env_core import registry as env_registry
from motrix_env_core.renderer import RenderConfig
from motrix_rl.fastsac.agent import FastSacAgent
from motrix_rl.fastsac.async_impl.collector import resolve_collector_inference_device
from motrix_rl.fastsac.async_impl.shm import Control, SharedTransitionRing, WeightSnapshot
from motrix_rl.fastsac.async_impl.worker import (
    actor_param_numel,
    build_env,
    run_collector_process,
    run_learner_process,
)
from motrix_rl.fastsac.config import FastSacCfg
from motrix_rl.fastsac.wrap import FastSacEnvWrap
from motrix_rl.frameworks import TrainerBase, TrainerContext

torch.set_float32_matmul_precision("high")


class Trainer(TrainerBase):
    def __init__(self, *, context: TrainerContext[FastSacCfg]) -> None:
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
        device: torch.device | None = None,
        mode: str = "train",
    ) -> FastSacEnvWrap:
        return build_env(
            self._env_spec,
            num_envs,
            device or self._device(),
            render=render,
            mode=mode,
            seed=self._context.seed,
        )

    def _dims(self, env: FastSacEnvWrap):
        inner = env.env
        obs_dim = inner.policy_observation_space.shape[-1]
        critic_obs_dim = inner.value_observation_space.shape[-1]
        act_dim = inner.action_space.shape[-1]
        low = env.action_low
        high = env.action_high
        action_scale = (high - low) / 2.0
        action_bias = (high + low) / 2.0
        return obs_dim, critic_obs_dim, act_dim, action_scale, action_bias

    def _make_agent(self, env: FastSacEnvWrap) -> FastSacAgent:
        obs_dim, critic_obs_dim, act_dim, action_scale, action_bias = self._dims(env)
        return FastSacAgent(
            obs_dim=obs_dim,
            critic_obs_dim=critic_obs_dim,
            act_dim=act_dim,
            num_envs=env.num_envs,
            cfg=self._rlcfg.agent,
            device=self._device(),
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
        cfg = self._rlcfg
        num_iterations = cfg.trainer.num_learning_iterations
        async_options = cfg.trainer.async_options
        logging_interval = self._context.logging.interval
        save_interval = self._context.checkpoint.interval

        if self._context.logging.backend != "tensorboard":
            raise ValueError("FastSAC supports only the 'tensorboard' logging backend.")

        # Read dims once in the parent (cheap 1-env build), then discard. Each
        # child rebuilds its own env because env objects are not picklable across spawn.
        probe = self._make_env(1, render=None, device=torch.device("cpu"))
        obs_dim, critic_obs_dim, act_dim, action_scale, action_bias = self._dims(probe)
        probe.close()
        dims = (obs_dim, critic_obs_dim, act_dim)

        learner_device = self._device()
        collector_device = resolve_collector_inference_device(async_options.collector_inference_device)
        param_numel = actor_param_numel(cfg, dims, action_scale, action_bias)

        # shared-memory primitives allocated in the parent, inherited by children.
        num_envs = self._context.num_envs
        ring = SharedTransitionRing(async_options.ring_capacity, num_envs, obs_dim, critic_obs_dim, act_dim)
        weights = WeightSnapshot(param_numel=param_numel, obs_dim=obs_dim)
        control = Control()

        resume_step = 0
        is_resume = False
        if self._resume_from:
            # Peek global_step so the collector warms up with the loaded policy and
            # the shared async counters continue from the checkpoint iteration.
            ckpt = torch.load(self._resume_from, map_location="cpu", weights_only=False)
            resume_step = int(ckpt.get("global_step", 0))
            is_resume = resume_step > 0
            control.collector_steps = resume_step
            control.global_step = resume_step

        ctx = mp.get_context("spawn")
        stats_queue = ctx.Queue(maxsize=8)
        error_queue = ctx.Queue(maxsize=8)
        reported_errors: set[tuple[str, str]] = set()
        seed = self._context.seed

        def _drain_child_errors() -> list[tuple[str, str]]:
            errors = []
            while True:
                try:
                    process_name, traceback_text = error_queue.get_nowait()
                except Empty:
                    break
                key = (process_name, traceback_text)
                if key in reported_errors:
                    continue
                reported_errors.add(key)
                errors.append(key)

                error_dir = self._context.run_dir / "async_errors"
                error_dir.mkdir(parents=True, exist_ok=True)
                error_path = error_dir / f"{process_name}_error.log"
                error_path.write_text(traceback_text)
                print(f"[motrix.fastsac async] {process_name} traceback written to {error_path}")
                print(traceback_text.rstrip())
            return errors

        print(
            f"[motrix.fastsac async] two-process training '{self._env_name}' learner={learner_device} "
            f"collector_env=cpu collector_inference={collector_device} num_envs={num_envs} iters={num_iterations} "
            f"from={resume_step} utd_mode={async_options.utd_mode}"
        )

        p_learner = ctx.Process(
            target=run_learner_process,
            args=(
                cfg,
                num_envs,
                dims,
                action_scale,
                action_bias,
                ring,
                weights,
                control,
                stats_queue,
                error_queue,
                num_iterations,
                logging_interval,
                save_interval,
                str(self._context.run_dir),
                self._env_name,
                str(self._context.checkpoint_dir),
                self._context.checkpoint_format,
                self._resume_from,
                seed,
            ),
            name="fastsac-async-learner",
        )
        p_collector = ctx.Process(
            target=run_collector_process,
            args=(
                self._env_spec,
                cfg,
                num_envs,
                dims,
                action_scale,
                action_bias,
                ring,
                weights,
                control,
                stats_queue,
                error_queue,
                num_iterations,
                logging_interval,
                is_resume,
                seed,
            ),
            name="fastsac-async-collector",
        )

        p_learner.start()
        p_collector.start()
        try:
            # monitor: exit when the learner finishes; abort both if either crashes.
            while True:
                if not p_learner.is_alive():
                    if p_learner.exitcode not in (0, None):
                        print(f"[motrix.fastsac async] learner crashed (exit {p_learner.exitcode}); stopping.")
                        _drain_child_errors()
                    break
                if not p_collector.is_alive() and p_collector.exitcode not in (0, None):
                    print(f"[motrix.fastsac async] collector crashed (exit {p_collector.exitcode}); stopping.")
                    _drain_child_errors()
                    break
                time.sleep(0.5)
        finally:
            control.set_stop()
            p_collector.join(timeout=30)
            p_learner.join(timeout=30)
            for p in (p_collector, p_learner):
                if p.is_alive():
                    print(f"[motrix.fastsac async] force-terminating {p.name}")
                    p.terminate()
                    p.join(timeout=10)
            _drain_child_errors()
            # drain the queue so the feeder thread can shut down cleanly.
            try:
                while True:
                    stats_queue.get_nowait()
            except Exception:
                pass
            stats_queue.close()
            error_queue.close()

        if p_learner.exitcode not in (0, None):
            raise RuntimeError(f"motrix.fastsac async learner process failed with exit code {p_learner.exitcode}")
        if p_collector.exitcode not in (0, None):
            raise RuntimeError(f"motrix.fastsac async collector process failed with exit code {p_collector.exitcode}")

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
