# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Async trainer: spawn the CPU collector, run the learner on this process.

A CPU physics collector optionally runs policy inference on the GPU. The
learner ingests available transition spans up to the configured block size,
updates replay and normalizers, then spends update credit on world-model and
imagination updates. Policy snapshots travel back through the inference-bundle weight
channel (live weights plus normalizers). The learner-side task is built
without a second physics instance via ``assemble_task`` on the learner device.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from multiprocessing.connection import Connection

import torch

from motrix_env_core import registry as env_registry
from motrix_rl.contracts import TrainerBase, TrainerContext
from motrix_rl.device import resolve_default_device
from motrix_rl.numa import gpu_numa_node
from motrix_rl.topology import same_cuda_device
from motrix_rl.transport.gpu_ipc_ring import GpuIpcTransitionRing
from motrix_rl.transport.inference_bundle import InferenceBundleSpec, ModuleSection, NormalizerSection
from motrix_rl.transport.ring import HostTransitionRing, RingCursors
from motrix_rl.transport.weight_channel import GpuIpcWeightSender, HostWeightSender, WeightChannelShared
from motrix_rl.worker import WorkerHandle, WorkerPlacement
from motrix_rl_supermimic.collector import CollectorWorker
from motrix_rl_supermimic.config import SuperMimicCfg
from motrix_rl_supermimic.env import SuperMimicEnv, SuperMimicEnvCfg
from motrix_rl_supermimic.imagined_task import ImaginedTask
from motrix_rl_supermimic.learner import Learner
from motrix_rl_supermimic.networks import NormStatsView, build_policy, resolve_amp_dtype
from motrix_rl_supermimic.randomness import derive_seed, model_initialization, resolve_seed
from motrix_rl_supermimic.replay import transition_schema
from motrix_rl_supermimic.scheduling import UpdateCredit
from motrix_rl_supermimic.transitions import Transitions
from motrix_rl_supermimic.transport import GpuRingConnection


class SuperMimicTrainer(TrainerBase):
    def __init__(self, *, context: TrainerContext[SuperMimicCfg]) -> None:
        self._context = context
        self._rlcfg = context.rl_cfg
        self._env_spec = env_registry.resolve(context.env_name, sim=context.sim)

    def _device(self) -> torch.device:
        if self._rlcfg.device:
            return torch.device(self._rlcfg.device)
        return resolve_default_device()

    def train(self) -> None:
        context = self._context
        cfg = self._rlcfg
        t, col = cfg.trainer, cfg.collector
        device = str(self._device())
        checkpoint = (
            torch.load(context.resume_from, map_location=device, weights_only=False) if context.resume_from else None
        )
        # Older persisted checkpoints predate root-seed metadata and retain
        # their CLI-seed fallback. Resolve before any reference/model factory.
        root_seed = (
            checkpoint["root_seed"]
            if checkpoint is not None and "root_seed" in checkpoint
            else resolve_seed(context.seed)
        )
        # Replay, augmentation and learner noise intentionally share this
        # process default stream; model construction does not consume it.
        torch.manual_seed(derive_seed(root_seed, "learner"))

        env_cfg = self._env_spec.env_cfg
        if not isinstance(env_cfg, SuperMimicEnvCfg):
            raise TypeError("SuperMimic requires a Manager task configuration inheriting SuperMimicEnvCfg")
        task = ImaginedTask(env_cfg.for_imagined(), device)
        learner = Learner(
            cfg, task, context.num_envs, float(env_cfg.ctrl_dt), device, context.checkpoint_dir, seed=root_seed
        )
        if checkpoint is not None:
            learner.load_checkpoint(checkpoint)
        logging.getLogger(__name__).info(
            "SuperMimic root seed: %d; collector/environment/exploration start fresh (warm resume)", learner.root_seed
        )
        # Transition and weight transport: CUDA-IPC device slots when the learner and the
        # collector's policy share one GPU (collector env physics stays on
        # CPU; only the slot memory moves to the device), else host memory.
        device_t = torch.device(device)
        collector_dev = torch.device(col.policy_device)
        if collector_dev.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("collector.policy_device requires CUDA, but CUDA is unavailable")
        if collector_dev.type == "cuda" and collector_dev.index is None:
            collector_dev = torch.device("cuda", 0)
        same_gpu = same_cuda_device(device_t, collector_dev)
        # Place the collector's spawn-time and runtime allocations on its
        # inference GPU's local NUMA node; None keeps the OS default.
        collector_node = (
            gpu_numa_node(collector_dev.index, runtime="hip" if torch.version.hip else "cuda")
            if collector_dev.type == "cuda"
            else None
        )
        modules = {"policy": learner.pi}
        normalizers = {}
        normalizer_sections = []
        for name, norm, dim in (
            ("obs_norm", learner.pi.obs_norm, task.heads.observation_dim),
            ("goal_norm", learner.pi.goal_norm, task.heads.goal_dim),
        ):
            normalizers[name] = NormStatsView(norm)
            normalizer_sections.append(NormalizerSection(name, dim))
        bundle_spec = InferenceBundleSpec(
            modules=tuple(
                ModuleSection(name, sum(p.numel() for p in policy.parameters())) for name, policy in modules.items()
            ),
            normalizers=tuple(normalizer_sections),
        )
        shared = WeightChannelShared(bundle_spec)
        sender = GpuIpcWeightSender(shared, device_t) if same_gpu else HostWeightSender(shared)
        sender.publish(modules, normalizers)

        ctx = mp.get_context("spawn")
        stop = ctx.Event()
        error_rx, error_tx = ctx.Pipe(duplex=False)
        startup_rx, startup_tx = ctx.Pipe()
        collector = CollectorWorker(
            env_name=context.env_name,
            sim=context.sim,
            num_envs=context.num_envs,
            seed=learner.root_seed,
            action_noise=t.action_noise,
            action_limit=20.0,
            block_steps=col.block_steps,
            agent=cfg.agent,
            policy_device=str(collector_dev),
            weight_shared=shared,
            weight_params=sender.params,
            stop=stop,
            errors=error_tx,
            startup=startup_tx,
            preregister=col.preregister,
            amp_dtype=resolve_amp_dtype(t.amp),
        )
        credit = UpdateCredit(t.utd, max(t.warmup_steps, t.wm_window))
        started = time.monotonic()
        last_received = started
        return_blocks: list[torch.Tensor] = []
        try:
            worker = collector.start(
                ctx, name="supermimic-collector", placement=WorkerPlacement(numa_node=collector_node)
            )
            error_tx.close()
            startup_tx.close()
            while not startup_rx.poll(0.05):
                self._check_worker(worker, error_rx)
                if not worker.is_alive:
                    raise RuntimeError("Collector exited before startup metadata")
                if time.monotonic() - started > col.worker_timeout:
                    raise TimeoutError(f"No collector startup metadata for {col.worker_timeout}s")
            schema, observation_spec, action_space = startup_rx.recv()
            if (
                observation_spec.policy.history_length != task.heads.observation_spec.policy.history_length
                or observation_spec.value_or_policy.history_length
                != task.heads.observation_spec.value_or_policy.history_length
            ):
                raise ValueError("Real and imagined observation history lengths must match")
            if (
                observation_spec.policy.frame_space.shape != task.heads.observation_spec.policy.frame_space.shape
                or observation_spec.value_or_policy.frame_space.shape
                != task.heads.observation_spec.value_or_policy.frame_space.shape
                or action_space.shape != task.heads.action_space.shape
            ):
                raise ValueError("Real and imagined policy/value observation and action dimensions must match")
            if (
                observation_spec["goal"].space.shape != task.heads.observation_spec["goal"].space.shape
                or observation_spec["goal"].history_length != task.heads.observation_spec["goal"].history_length
            ):
                raise ValueError("Real and imagined goal observation specifications must match")
            expected_schema = transition_schema(
                state_dim=task.world_state.size, context_dim=task.world_context.size, action_dim=task.heads.action_dim
            )
            if schema.fields != expected_schema.fields:
                raise ValueError("Real and imagined transition schemas must match")
            if same_gpu:
                cursors = RingCursors()
                slots = torch.zeros(
                    (col.queue_blocks, context.num_envs, schema.feature_width), dtype=torch.float32, device=device_t
                )
                ring = GpuIpcTransitionRing(cursors, slots, schema, col.queue_blocks, context.num_envs)
                startup_rx.send(GpuRingConnection(schema, slots, cursors))
            else:
                ring = HostTransitionRing(schema, col.queue_blocks, context.num_envs)
                startup_rx.send(ring)
            last_received = time.monotonic()
            while learner.iteration < t.iterations:
                self._check_worker(worker, error_rx)
                if not ring.has_next():
                    if time.monotonic() - last_received > col.worker_timeout:
                        raise TimeoutError(f"No collector block for {col.worker_timeout}s")
                    time.sleep(0.001)
                    continue
                last_received = time.monotonic()
                count, views = ring.read_span()
                take = min(count, col.block_steps)
                # CUDA-IPC slots fuse integer fields into float32. Restore the
                # schema dtypes at the transport boundary, before consumption.
                transitions = Transitions.from_dict(
                    {field.name: views[field.name][:take].to(dtype=field.dtype) for field in schema.fields}
                )
                span_reward = learner.ingest(transitions)
                return_blocks.append(span_reward)
                ring.commit_reads(take)
                credit.receive(take)
                while credit.available and learner.iteration < t.iterations:
                    self._check_worker(worker, error_rx)
                    wm_loss, policy_row = learner.train_iteration()
                    credit.consume()
                    if learner.iteration % col.snapshot_every == 0:
                        sender.publish(modules, normalizers)
                    if learner.iteration % t.log_every == 0:
                        # One-shot policy-lr decay trigger at log cadence on
                        # the mean normalized training return of the interval.
                        aggregated = float(torch.stack(return_blocks).mean()) if return_blocks else 0.0
                        return_blocks.clear()
                        from motrix_rl_supermimic.optim import trigger_decay

                        learner.decay_at = trigger_decay(t, learner.iteration, aggregated, learner.decay_at)
                    if context.metrics.enabled and learner.iteration % t.log_every == 0:
                        context.metrics.scalars(
                            {
                                "wm_loss": float(wm_loss),
                                "sim_steps": float(learner.sim_steps),
                                "sim_reward": float(span_reward),
                                "policy_loss": float(policy_row[0]),
                                "value_loss": float(policy_row[1]),
                                "imag_reward": float(policy_row[2]),
                                "imag_survival": float(learner._last_imag_survival),
                                "nonfinite_actor": float(learner._nonfinite_actor),
                                "grad_norm": float(policy_row[3]),
                                "policy_lr": learner.opt_pi.param_groups[0]["lr"],
                                "snapshots_published": float(sender.version),
                                "ring_backlog": float(ring.size()),
                                "wall_s": time.monotonic() - started,
                            },
                            step=learner.iteration,
                        )
                    if learner.iteration % t.ckpt_every == 0 or learner.iteration == t.iterations:
                        learner.save(time.monotonic() - started)
        finally:
            stop.set()
            worker.join(timeout=10.0)
            if worker.is_alive:
                worker.terminate()
                worker.join(timeout=5.0)
            error_rx.close()
            startup_rx.close()

    def play(self, policy: str) -> None:
        cfg = self._rlcfg
        env_cfg = self._env_spec.env_cfg
        if not isinstance(env_cfg, SuperMimicEnvCfg):
            raise TypeError("SuperMimic requires a Manager task configuration inheriting SuperMimicEnvCfg")
        env = self._env_spec.make(num_envs=self._context.play_num_envs, mode="play")
        device = self._device()
        if not isinstance(env, SuperMimicEnv):
            raise TypeError("SuperMimic requires an environment inheriting SuperMimicEnv")
        observation_spec = env.observation_spec
        obs_dim = observation_spec.policy.frame_space.shape[0]
        action_dim = env.action_space.shape[0]
        goal_dim = observation_spec["goal"].space.shape[0]
        checkpoint = torch.load(policy, map_location=device, weights_only=False)
        if checkpoint["goal_frames"] != env.cfg.goal_frames:
            raise ValueError(
                f"Checkpoint goal_frames={checkpoint['goal_frames']} "
                f"does not match task goal_frames={env.cfg.goal_frames}"
            )
        root_seed = checkpoint["root_seed"] if "root_seed" in checkpoint else resolve_seed(self._context.seed)
        with model_initialization(derive_seed(root_seed, "model/policy")):
            reference_action = env.cfg.reference_action() if cfg.agent.pi_ref_skip else None
            pi = build_policy(
                cfg.agent,
                obs_dim,
                goal_dim,
                action_dim,
                "cpu",
                observation_spec=observation_spec.policy,
                reference_action=reference_action,
                amp_dtype=resolve_amp_dtype(cfg.trainer.amp),
            )
        pi = pi.to(device)
        pi.load_state_dict(checkpoint["pi"])
        pi.eval()
        env.initialize()
        result = env.state
        with torch.inference_mode():
            while True:
                obs = (
                    result.obs["policy"]
                    .to(device)
                    .reshape(env.num_envs, observation_spec.policy.history_length, obs_dim)
                )
                goal = result.obs["goal"].to(device)
                result = env.step(pi(obs, goal).to(env.device))
                time.sleep(0.02)

    @staticmethod
    def _check_worker(worker: WorkerHandle, error_rx: Connection) -> None:
        if not worker.is_alive and worker.exitcode not in (0, None):
            raise RuntimeError(f"Collector process died (exitcode {worker.exitcode})")
        if error_rx.poll():
            raise RuntimeError(f"Collector failed:\n{error_rx.recv()}")
