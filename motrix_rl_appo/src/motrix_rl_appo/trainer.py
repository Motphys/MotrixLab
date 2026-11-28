# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Single-collector asynchronous APPO orchestration and warm checkpoints."""

from __future__ import annotations

import logging
import multiprocessing as mp
import random
import time
from multiprocessing.connection import Connection
from pathlib import Path
from queue import Empty

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from motrix_env_core import registry as env_registry
from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.renderer import create_renderer
from motrix_rl import checkpoints
from motrix_rl.contracts import TrainerBase, TrainerContext
from motrix_rl.device import resolve_default_device
from motrix_rl.transport.ring import HostTransitionRing, RingCursors
from motrix_rl.transport.weight_channel import (
    GpuIpcWeightSender,
    HostWeightSender,
    WeightChannelShared,
)
from motrix_rl.worker import WorkerHandle, WorkerRuntime
from motrix_rl_appo.collector import CollectorWorker
from motrix_rl_appo.config import AppoCfg
from motrix_rl_appo.learner import APPOLearner
from motrix_rl_appo.networks import build_actor
from motrix_rl_appo.transport import (
    CollectorConnection,
    GpuRingConnection,
    chunk_schema,
    inference_spec,
    resolve_transport,
    take_chunk,
)

logger = logging.getLogger(__name__)


def check_worker(worker: WorkerHandle, errors: Connection) -> None:
    if errors.poll():
        try:
            message = errors.recv()
        except EOFError:
            message = None
        if message is not None:
            raise RuntimeError(f"APPO collector failed:\n{message}")
    if not worker.is_alive:
        raise RuntimeError(f"APPO collector exited unexpectedly (exitcode={worker.exitcode})")


def wait_ready(worker: WorkerHandle, errors: Connection, ready: Connection, timeout: float) -> None:
    started = time.monotonic()
    while not ready.poll(0.01):
        check_worker(worker, errors)
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"No APPO collector startup metadata for {timeout}s")


def stop_worker(worker: WorkerHandle) -> None:
    worker.join(timeout=5)
    if worker.is_alive:
        worker.terminate()
        worker.join(timeout=5)
    if worker.is_alive:
        worker.kill()
        worker.join(timeout=5)
    worker.process.close()


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


class AppoTrainer(TrainerBase):
    def __init__(self, *, context: TrainerContext[AppoCfg]) -> None:
        self.context = context

    def _device(self) -> torch.device:
        device = torch.device(self.context.rl_cfg.device) if self.context.rl_cfg.device else resolve_default_device()
        return torch.device("cuda:0") if device.type == "cuda" and device.index is None else device

    def _save(self, learner: APPOLearner, seed: int, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"learner": learner.state_dict(), "rng": rng_state(), "seed": seed, "iteration": learner.update_count}, path
        )
        checkpoints.record_checkpoint_artifact(
            self.context.run_dir,
            checkpoints.LATEST_TRAINING_STATE,
            path,
            checkpoints.TRAINING_STATE,
            checkpoint_format=self.context.checkpoint_format,
        )
        checkpoints.record_checkpoint_artifact(
            self.context.run_dir,
            checkpoints.BEST_POLICY,
            path,
            checkpoints.POLICY,
            checkpoint_format=self.context.checkpoint_format,
        )

    def train(self) -> None:
        context = self.context
        cfg = context.rl_cfg
        if context.render is not None:
            raise ValueError("APPO asynchronous training does not support render; use play instead")
        if context.logging.backend != "tensorboard":
            raise ValueError("APPO supports logging.backend=tensorboard only")
        device = self._device()
        collector_device = torch.device(cfg.collector_device or str(device))
        if collector_device.type == "cuda" and collector_device.index is None:
            collector_device = torch.device("cuda:0")
        transport = resolve_transport(cfg.transition_transport, device, collector_device)
        weight_transport = resolve_transport(cfg.weight_transport, device, collector_device)
        chunk_device = collector_device if transport == "cuda_ipc" else torch.device("cpu")
        logger.info(
            "APPO transition transport=%s weight_transport=%s chunk_device=%s learner=%s collector=%s",
            transport,
            weight_transport,
            chunk_device,
            device,
            collector_device,
        )
        checkpoint = (
            torch.load(context.resume_from, map_location=device, weights_only=False) if context.resume_from else None
        )
        seed = (
            checkpoint["seed"]
            if checkpoint
            else (context.seed if context.seed is not None else random.randrange(2**31))
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        ctx = mp.get_context("spawn")
        stop = ctx.Event()
        error_rx, error_tx = ctx.Pipe(duplex=False)
        startup_rx, startup_tx = ctx.Pipe()
        metrics_queue = ctx.Queue(maxsize=cfg.queue_chunks + 1)
        worker: WorkerHandle | None = None
        writer: SummaryWriter | None = None
        started = time.monotonic()
        try:
            worker = CollectorWorker(
                context.env_name,
                context.sim,
                context.num_envs,
                seed,
                cfg,
                str(collector_device),
                stop,
                error_tx,
                startup_tx,
                metrics_queue,
            ).start(ctx, name="appo-collector", runtime=WorkerRuntime(torch_num_threads=1, seed=seed))
            error_tx.close()
            startup_tx.close()
            wait_ready(worker, error_rx, startup_rx, cfg.worker_timeout)
            dims = startup_rx.recv()
            learner = APPOLearner(cfg, dims.obs_dim, dims.critic_dim, dims.action_dim, device)
            if checkpoint is not None:
                learner.load_state_dict(checkpoint["learner"])
                restore_rng(checkpoint["rng"])
            modules = {"actor": learner.actor, "critic": learner.critic}
            norms = {"actor_norm": learner.actor.obs_normalizer, "critic_norm": learner.critic.obs_normalizer}
            spec = inference_spec(learner.actor, learner.critic, dims)
            # Keep shared stats and sender allocations alive until the child exits.
            weights = WeightChannelShared(spec)
            sender = (
                GpuIpcWeightSender(weights, device) if weight_transport == "cuda_ipc" else HostWeightSender(weights)
            )
            sender.publish(modules, norms)
            schema = chunk_schema(cfg.rollout_steps, dims)
            if transport == "cuda_ipc":
                # Parent owns the allocation until the collector has stopped.
                slots = torch.empty(
                    (cfg.queue_chunks, context.num_envs, schema.feature_width), device=device, dtype=torch.float32
                )
                ring_connection = GpuRingConnection(RingCursors(), slots, schema, cfg.queue_chunks, context.num_envs)
                ring = ring_connection.endpoint()
            else:
                ring = HostTransitionRing(schema, cfg.queue_chunks, context.num_envs)
                ring_connection = ring
            startup_rx.send(CollectorConnection(ring_connection, weights, sender.params, str(chunk_device)))
            writer = SummaryWriter(log_dir=str(context.run_dir))
            last_received = time.monotonic()
            episode_returns: list[float] = []
            episode_lengths: list[int] = []
            while learner.update_count < cfg.max_iterations:
                check_worker(worker, error_rx)
                if not ring.has_next():
                    if time.monotonic() - last_received > cfg.worker_timeout:
                        raise TimeoutError(f"No APPO collector chunk for {cfg.worker_timeout}s")
                    stop.wait(0.001)
                    continue
                ingest_started = time.monotonic()
                batch = take_chunk(ring, device)
                # Host wall time for owned copies/enqueues; no extra CUDA sync.
                ingest_s = time.monotonic() - ingest_started
                try:
                    record = metrics_queue.get(timeout=cfg.worker_timeout)
                except Empty as exc:
                    check_worker(worker, error_rx)
                    raise TimeoutError("Missing APPO collector metrics for published chunk") from exc
                last_received = time.monotonic()
                update_started = time.monotonic()
                metrics = learner.update(batch)
                optimize_s = time.monotonic() - update_started
                publish_started = time.monotonic()
                sender.publish(modules, norms)
                weight_publish_s = time.monotonic() - publish_started
                # Wall-clock boundaries only; publication owns its required CUDA completion.
                update_s = time.monotonic() - update_started
                episode_returns.extend(record.returns)
                episode_lengths.extend(record.lengths)
                metrics.update(
                    {
                        "collector/latency_s": record.latency_s,
                        "collector/weight_lag": float(record.published_version - record.weight_version),
                        "collector/queue_depth": float(ring.size()),
                        "learner/update_s": update_s,
                        "learner/optimize_s": optimize_s,
                        "learner/weight_publish_s": weight_publish_s,
                        "learner/ingest_s": ingest_s,
                        "env_steps": float(learner.global_step),
                        "wall_s": time.monotonic() - started,
                    }
                )
                metrics.update({f"env/{key}": value for key, value in record.env_metrics.items()})
                iteration = learner.update_count
                if iteration % max(context.logging.interval, 1) == 0 or iteration == cfg.max_iterations:
                    if episode_returns:
                        # Raw completed-episode return, without timeout bootstrap.
                        metrics["Train/mean_reward"] = float(np.mean(episode_returns))
                        metrics["Train/mean_episode_length"] = float(np.mean(episode_lengths))
                        episode_returns.clear()
                        episode_lengths.clear()
                    for name, value in metrics.items():
                        writer.add_scalar(name, value, iteration)
                    context.metrics.scalars(metrics, step=iteration, env_steps=learner.global_step)
                if context.checkpoint.interval > 0 and iteration % context.checkpoint.interval == 0:
                    self._save(learner, seed, context.checkpoint_dir / f"checkpoint_{iteration}.pt")
            self._save(learner, seed, checkpoints.final_checkpoint_path(context.checkpoint_format, context.run_dir))
        finally:
            stop.set()
            if worker is not None:
                stop_worker(worker)
            for pipe in (error_rx, error_tx, startup_rx, startup_tx):
                pipe.close()
            metrics_queue.close()
            metrics_queue.join_thread()
            if writer is not None:
                writer.close()

    def play(self, policy: str) -> None:
        context = self.context
        device = self._device()
        env = env_registry.make(
            context.env_name,
            sim=context.sim,
            num_envs=context.play_num_envs,
            mode="play",
            seed=context.seed,
            headless=context.render.headless if context.render else False,
            record=context.render.record if context.render else None,
        )
        if not isinstance(env, ArrayEnv):
            raise TypeError("APPO requires a Torch ArrayEnv frontend")
        actor = build_actor(
            context.rl_cfg.actor, env.observation_spec.policy.space.shape[0], env.action_space.shape[0]
        ).to(device)
        checkpoint = torch.load(policy, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["learner"]["actor"])
        actor.eval()
        renderer = create_renderer(env, context.render)
        try:
            state = env.initialize()
            with torch.no_grad():
                while True:
                    state = env.step(actor.mean(state.obs.policy.to(device)).to(state.obs.policy.device))
                    if renderer is not None and renderer.render() is False:
                        break
                    if context.render is None or context.render.record is None:
                        time.sleep(float(env.cfg.ctrl_dt))
        except KeyboardInterrupt:
            pass
        finally:
            if renderer is not None:
                renderer.close()
