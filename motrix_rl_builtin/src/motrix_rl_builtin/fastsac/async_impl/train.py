# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC algorithm plans for the shared asynchronous training runtime.

The adapter owns topology, tensor transports, resume interpretation and metrics
semantics. Process lifecycle, message consumption and presentation belong to
``AsyncTrainerBase``; worker algorithm bodies live in ``worker.py``.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass, field, replace

import torch

from motrix_env_core import registry as env_registry
from motrix_env_core.renderer import RenderConfig
from motrix_rl.contracts import TrainerContext
from motrix_rl.device import resolve_default_device
from motrix_rl.execution.building import BuildRuntime
from motrix_rl.execution.trainer import AsyncTrainerBase
from motrix_rl.execution.training import (
    TrainingDecision,
    TrainingEvent,
    TrainingMetrics,
    TrainingPlan,
    TrainingProgress,
    WorkerReports,
    WorkerSpec,
)
from motrix_rl.execution.worker_runtime import set_seed
from motrix_rl.execution.workers import WorkerSettings
from motrix_rl.runs import annotate_run
from motrix_rl.system_metrics import capture_system_info
from motrix_rl_builtin.fastsac.agent import FastSacAgent
from motrix_rl_builtin.fastsac.async_impl.metrics import FastSacMetrics
from motrix_rl_builtin.fastsac.async_impl.topology import TrainerTopology, resolve_trainer_topology
from motrix_rl_builtin.fastsac.async_impl.transport.common import Control
from motrix_rl_builtin.fastsac.async_impl.transport.handshake import StartupHandshake
from motrix_rl_builtin.fastsac.async_impl.transport.ring import FinalObsStorage, RingCursors, SharedTransitionRing
from motrix_rl_builtin.fastsac.async_impl.transport.weight_channel import WeightChannelShared
from motrix_rl_builtin.fastsac.async_impl.worker import (
    actor_param_numel,
    build_env,
    run_collector_process,
    run_learner_process,
)
from motrix_rl_builtin.fastsac.config import FastSacCfg
from motrix_rl_builtin.fastsac.wrap import FastSacEnvWrap

torch.set_float32_matmul_precision("high")


@dataclass
class TrainingState:
    """Parent-owned FastSAC state, without execution handles or callbacks."""

    control: Control
    topology: TrainerTopology
    budget: int
    metrics: FastSacMetrics
    resume_step: int
    resume_learning_iteration: int
    learner_ids: tuple[str, ...]
    pending_events: list[TrainingEvent] = field(default_factory=list)


class Trainer(AsyncTrainerBase):
    def __init__(self, *, context: TrainerContext[FastSacCfg]) -> None:
        super().__init__(context=context)
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
        return resolve_default_device()

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
        set_seed(seed)

    # ------------------------------------------------------------------ train
    def build_training(self, runtime: BuildRuntime) -> TrainingPlan:
        """Declare FastSAC workers and their algorithm data connections."""
        cfg = self._rlcfg
        options = cfg.trainer.async_options
        budget = cfg.trainer.num_learning_iterations
        num_envs = self._context.num_envs
        num_collectors = options.num_collectors
        num_learners = options.num_learners
        logging_interval = self._context.logging.interval
        seed = self._context.seed

        if self._context.logging.backend != "tensorboard":
            raise ValueError("FastSAC supports only the 'tensorboard' logging backend.")
        if num_learners < 1:
            raise ValueError(f"num_learners must be >= 1, got {num_learners}")
        if num_learners > 1 and num_collectors % num_learners != 0:
            raise ValueError(f"num_collectors={num_collectors} must divide evenly across num_learners={num_learners}")
        if cfg.agent.batch_size % num_learners != 0:
            raise ValueError(
                f"agent.batch_size={cfg.agent.batch_size} must divide evenly across num_learners={num_learners}"
            )

        # The CPU probe is short-lived; spawned collectors construct their own
        # environments rather than attempting to pickle simulator resources.
        with runtime.enter_context(ExitStack()) as probe_resources:
            probe = self._make_env(1, render=None, device=torch.device("cpu"))
            probe_resources.callback(probe.close)
            obs_dim, critic_obs_dim, act_dim, action_scale, action_bias = self._dims(probe)
            max_episode_steps = probe.max_episode_steps
        dims = (obs_dim, critic_obs_dim, act_dim)
        topology = resolve_trainer_topology(
            num_envs,
            num_collectors,
            num_learners,
            options.learner_devices,
            options.collector_inference_device,
            self._device(),
            options,
            actor_param_numel(cfg, dims, action_scale, action_bias),
            cpus_per_collector=options.cpus_per_collector,
        )
        annotate_run(self._context.run_dir, system=capture_system_info(topology=topology))

        # Shared tensors have no explicit close operation: references in the
        # plan's worker arguments keep their storage alive until peers are reaped.
        # One ring and weight channel per collector preserve SPSC ownership.
        final_stores = (
            [FinalObsStorage(options.ring_capacity, shard, obs_dim, critic_obs_dim) for shard in topology.env_shards]
            if max_episode_steps is not None
            else None
        )
        rings = [
            RingCursors()
            if collector.ring_ipc
            else SharedTransitionRing(
                options.ring_capacity,
                collector.num_envs,
                obs_dim,
                critic_obs_dim,
                act_dim,
                final_obs_store=final_stores[i] if final_stores is not None else None,
            )
            for i, collector in enumerate(topology.collectors)
        ]
        weights = [
            WeightChannelShared(obs_dim=obs_dim, obs_normalization=cfg.agent.obs_normalization)
            for _ in topology.collectors
        ]
        control = Control(num_collectors)
        resume_step = 0
        resume_learning_iteration = 0
        pending_events = []
        if self._resume_from:
            checkpoint = torch.load(self._resume_from, map_location="cpu", weights_only=False)
            resume_step = int(checkpoint.get("global_step", 0))
            resume_learning_iteration = int(checkpoint.get("learning_iteration", 0))
            control.resume_collector_steps(resume_step)
            control.global_step = resume_step
            pending_events.append(
                TrainingEvent(
                    "resumed",
                    step=resume_step,
                    data={"path": str(self._resume_from), "global_step": resume_step},
                )
            )

        # This handshake carries only algorithm slot tensors. Readiness and
        # start/stop barriers are supplied by the public worker runtime.
        handshake = StartupHandshake(runtime, num_collectors)
        rendezvous_file = None
        if num_learners > 1:
            rendezvous_path = (self._context.run_dir / "ddp_rendezvous").resolve()
            rendezvous_path.parent.mkdir(parents=True, exist_ok=True)
            rendezvous_path.unlink(missing_ok=True)
            rendezvous_file = str(rendezvous_path)

        learner_ids = tuple(
            "fastsac-async-learner" if num_learners == 1 else f"fastsac-async-learner-{rank}"
            for rank in range(num_learners)
        )
        workers = []
        for rank, learner in enumerate(topology.learners):
            owned = slice(rank * topology.per_learner, (rank + 1) * topology.per_learner)
            workers.append(
                WorkerSpec(
                    id=learner_ids[rank],
                    role="learner",
                    target=run_learner_process,
                    kwargs={
                        "cfg": cfg,
                        "num_envs": num_envs,
                        "dims": dims,
                        "action_scale": action_scale,
                        "action_bias": action_bias,
                        "rings": topology.ring_slice_for_rank(rings, rank),
                        "weights": weights[owned],
                        "control": control,
                        "num_iterations": budget,
                        "logging_interval": logging_interval,
                        "save_interval": self._context.checkpoint.interval,
                        "run_dir": str(self._context.run_dir),
                        "checkpoint_dir": str(self._context.checkpoint_dir),
                        "checkpoint_format": self._context.checkpoint_format,
                        "resume_from": self._resume_from,
                        "handshake": handshake,
                        "rank": rank,
                        "num_learners": num_learners,
                        "rendezvous_file": rendezvous_file,
                        "learner_device": str(learner.device),
                        "notify_checkpoints": self._context.metrics.enabled,
                        "all_rings": rings,
                        "weight_ipc": [collector.weight_ipc for collector in topology.collectors[owned]],
                        "final_stores": final_stores[owned] if final_stores is not None else None,
                    },
                    settings=WorkerSettings(
                        seed=None if seed is None else seed + rank,
                        numa_node=learner.numa_node,
                        cpus=tuple(learner.cpus),
                        cpu_spec=getattr(options, "learner_cpu_cores", None),
                    ),
                )
            )
        for i, collector in enumerate(topology.collectors):
            collector_seed = None if seed is None else seed + i
            workers.append(
                WorkerSpec(
                    id=f"fastsac-async-collector-{i}",
                    role="collector",
                    target=run_collector_process,
                    kwargs={
                        "env_spec": self._env_spec,
                        "cfg": cfg,
                        "num_envs": collector.num_envs,
                        "dims": dims,
                        "action_scale": action_scale,
                        "action_bias": action_bias,
                        "ring": rings[i],
                        "weights": weights[i],
                        "control": control,
                        "num_iterations": budget,
                        "logging_interval": logging_interval,
                        "is_resume": resume_step > 0,
                        "seed": collector_seed,
                        "collector_id": i,
                        "collector_device": str(collector.device),
                        "handshake": handshake,
                        "final_store": final_stores[i] if final_stores is not None else None,
                    },
                    settings=WorkerSettings(
                        seed=collector_seed,
                        numa_node=collector.numa_node,
                        cpus=tuple(collector.cpus),
                        cpu_spec=getattr(options, "collector_cpu_cores", None),
                    ),
                )
            )

        state = TrainingState(
            control=control,
            topology=topology,
            budget=budget,
            metrics=FastSacMetrics(
                num_envs=num_envs,
                num_collectors=num_collectors,
                num_learners=num_learners,
                budget=budget,
                learning_starts=cfg.agent.learning_starts,
                logging_interval=logging_interval,
                resume_step=resume_step,
                resume_learning_iteration=resume_learning_iteration,
            ),
            resume_step=resume_step,
            resume_learning_iteration=resume_learning_iteration,
            learner_ids=learner_ids,
            pending_events=pending_events,
        )
        return TrainingPlan(workers=tuple(workers), state=state)

    def evaluate_progress(self, state: TrainingState, progress: TrainingProgress) -> TrainingDecision:
        """Validate finish declarations against FastSAC's first-shard budget."""
        # Preserve the existing budget semantics: any collector reaching the
        # target establishes the algorithm boundary; slower peers may stop short.
        boundary = any(
            state.control.collector_steps_at(i) >= state.budget for i in range(state.topology.num_collectors)
        )
        completed = frozenset(
            worker_id for worker_id, status in progress.workers.items() if boundary and status.finished
        )
        return TrainingDecision(
            complete=boundary and all(progress.workers[worker_id].finished for worker_id in state.learner_ids),
            completed_workers=completed,
        )

    def collect_metrics(self, state: TrainingState, reports: WorkerReports) -> TrainingMetrics:
        """Map consumed worker reports to algorithm metrics, without I/O."""
        state.metrics.consume(reports)
        frame = state.metrics.frame(
            now=reports.now if reports.now is not None else time.time(),
            collector_steps=state.control.collector_steps,
            force=reports.force,
        )
        if state.pending_events:
            frame = replace(frame, events=tuple(state.pending_events) + frame.events)
            state.pending_events.clear()
        return frame

    # ------------------------------------------------------------------ play
    def play(self, policy: str) -> None:
        self._set_seed(self._context.seed)
        self._writer = None
        with ExitStack() as resources:
            env = self._make_env(
                self._context.play_num_envs,
                render=self._render,
                mode="play",
            )
            resources.callback(env.close)
            agent = self._make_agent(env)
            ckpt = torch.load(policy, map_location=agent.device, weights_only=False)
            agent.load_state_dict(ckpt)
            agent.actor.eval()
            if hasattr(agent.obs_normalizer, "eval"):
                agent.obs_normalizer.eval()

            obs, _ = env.reset()
            while True:
                actions = agent.act(obs, deterministic=True)
                obs = env.step(actions).obs
                if env.render() is False:
                    break
