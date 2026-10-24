# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Process entry points and shared builders for the M1 two-process trainer.

The collector process (CPU) runs the env + inference policy and feeds the shared
transition ring; the learner process (GPU) drains the ring, trains, publishes
weights, and owns logging + checkpointing. Both are spawned by
:class:`~motrix_rl_builtin.fastsac.async_impl.train.Trainer`. Entry functions are module-level
so ``torch.multiprocessing`` (``spawn``) can pickle them by qualified name.

Build helpers (``build_env`` / ``build_agent``) are shared with the single-process
(M0) path so both trainers construct env/agent identically.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.registry import EnvBuildSpec
from motrix_env_core.renderer import RenderConfig
from motrix_env_motrixsim.torch_env import TorchEnv
from motrix_rl import checkpoints
from motrix_rl.device import resolve_default_device
from motrix_rl.execution.workers import WorkerRuntime
from motrix_rl_builtin.fastsac.agent import FastSacAgent
from motrix_rl_builtin.fastsac.async_impl.collector import Collector
from motrix_rl_builtin.fastsac.async_impl.learner import CollectorEndpoint, Learner
from motrix_rl_builtin.fastsac.async_impl.stats import timing_mean
from motrix_rl_builtin.fastsac.async_impl.transport import (
    Control,
    IpcTransitionRing,
    RingCursors,
    SharedTransitionRing,
)
from motrix_rl_builtin.fastsac.async_impl.transport.handshake import StartupHandshake
from motrix_rl_builtin.fastsac.async_impl.transport.ring import FinalObsStorage
from motrix_rl_builtin.fastsac.async_impl.transport.weight_channel import (
    GpuIpcWeightSender,
    HostWeightSender,
    WeightChannelShared,
    weight_receiver_for,
)
from motrix_rl_builtin.fastsac.config import FastSacCfg
from motrix_rl_builtin.fastsac.wrap import FastSacEnvWrap
from motrix_rl_builtin.fastsac.wrap_np import FastSacNpEnvWrap
from motrix_rl_builtin.fastsac.wrap_torch import FastSacTorchEnvWrap

logger = logging.getLogger(__name__)


def build_env(
    env_spec: EnvBuildSpec,
    num_envs: int,
    device: torch.device,
    render: RenderConfig | None = None,
    mode: str = "train",
    seed: int | None = None,
) -> FastSacEnvWrap:
    env = env_spec.make(num_envs=num_envs, mode=mode, seed=seed)
    if isinstance(env, TorchEnv):
        return FastSacTorchEnvWrap(env, device, render=render)
    if isinstance(env, ArrayEnv):
        return FastSacNpEnvWrap(env, device, render=render)
    raise TypeError(f"FastSAC does not support environment type '{type(env).__name__}'.")


def actor_param_numel(cfg: FastSacCfg, dims, action_scale, action_bias) -> int:
    """Total actor parameter count, used to size the shared weight buffer.

    Builds a throwaway CPU actor mirroring the learner's — device-independent, so
    the parent can size shared memory without touching CUDA.
    """
    from motrix_rl_builtin.fastsac.networks import Actor

    obs_dim, _critic_obs_dim, act_dim = dims
    a = cfg.agent
    actor = Actor(
        n_obs=obs_dim,
        n_act=act_dim,
        hidden_dim=a.actor_hidden_dim,
        log_std_max=a.log_std_max,
        log_std_min=a.log_std_min,
        use_tanh=a.use_tanh,
        use_layer_norm=a.use_layer_norm,
        action_scale=action_scale,
        action_bias=action_bias,
        device="cpu",
    )
    return sum(p.numel() for p in actor.parameters())


def build_agent(
    cfg: FastSacCfg,
    dims: tuple[int, int, int],
    num_envs: int,
    device: torch.device,
    action_scale: torch.Tensor | None,
    action_bias: torch.Tensor | None,
    writer: SummaryWriter | None = None,
    world_size: int = 1,
    final_obs_bootstrap: bool = False,
) -> FastSacAgent:
    """Build the learner's ``FastSacAgent``.

    ``dims`` is ``(obs_dim, critic_obs_dim, act_dim)``. ``action_scale`` /
    ``action_bias`` map the tanh-squashed policy output to the env action
    range (``None`` = identity). ``writer`` is the parent-created
    TensorBoard writer; ``world_size`` is the DDP learner-rank count.
    ``final_obs_bootstrap`` enables the replay buffer's terminal-observation
    storage for time-limit-truncated bootstrapping.
    """
    obs_dim, critic_obs_dim, act_dim = dims
    return FastSacAgent(
        obs_dim=obs_dim,
        critic_obs_dim=critic_obs_dim,
        act_dim=act_dim,
        num_envs=num_envs,
        cfg=cfg.agent,
        device=device,
        action_scale=action_scale,
        action_bias=action_bias,
        writer=writer,
        world_size=world_size,
        final_obs_bootstrap=final_obs_bootstrap,
    )


# ------------------------------------------------------------------ collector process


def run_collector_process(
    runtime: WorkerRuntime,
    env_spec: EnvBuildSpec,
    cfg: FastSacCfg,
    num_envs: int,
    dims: tuple[int, int, int],
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    ring: SharedTransitionRing | RingCursors,
    weights: WeightChannelShared,
    control: Control,
    num_iterations: int,
    logging_interval: int,
    is_resume: bool,
    seed: int | None,
    handshake: StartupHandshake,
    collector_id: int = 0,
    collector_device: str | None = None,
    final_store: FinalObsStorage | None = None,
) -> None:
    control.bind_stop(runtime.stop_event)
    try:
        opts = cfg.trainer.async_options
        # The parent's topology pass resolved every collector's inference
        # device explicitly (generic "cuda" co-located with the owning
        # learner's GPU), so no generic spec is re-resolved in-process here.
        if collector_device is not None:
            opts.collector_inference_device = collector_device
        obs_dim, critic_obs_dim, act_dim = dims
        device = torch.device("cpu")
        env_started = time.perf_counter()
        env = build_env(env_spec, num_envs, device, seed=seed)
        runtime.callback(env.close)
        logger.info(
            "collector startup: env build (scene + manager kernels) finished in %.3fs",
            time.perf_counter() - env_started,
        )
        # Handshake: build the endpoints from the slot tensors the learners
        # shipped before the collector is wired up. Weight slots come from
        # the publishing learner (rank 0), CUDA-IPC transition-ring slots
        # from the learner that drains this collector's ring (its owner —
        # always rank 0 in the single-learner topology).
        weight_slots = handshake.await_weight_slots(collector_id, control)
        ring_slots = handshake.await_ring_slots(collector_id, control)
        weight_rx = weight_receiver_for(weights, weight_slots)
        if isinstance(ring, RingCursors):
            if ring_slots is None:
                raise RuntimeError("the learner shipped no transition-ring slots for the IPC ring handshake")
            ring = IpcTransitionRing(
                ring,
                ring_slots,
                opts.ring_capacity,
                num_envs,
                obs_dim,
                critic_obs_dim,
                act_dim,
                final_obs_store=final_store,
            )
        collector = Collector(
            env,
            cfg,
            obs_dim,
            act_dim,
            action_scale,
            action_bias,
            ring,
            weight_rx,
            control,
            is_resume=is_resume,
            collector_id=collector_id,
        )
        collector.reset()
        collector.sync_weights()
        warmup_started = time.perf_counter()
        collector.warmup_inference()
        logger.info(
            "collector startup: inference warmup (torch.compile + CUDA graphs) finished in %.3fs",
            time.perf_counter() - warmup_started,
        )

        runtime.ready()

        # First snapshot once a few steps have landed: the parent's panel
        # quiescence gate opens early (not after a full logging window of
        # silence) while the handoff progress view gets to show real steps,
        # and the snapshot already carries a little real data. The
        # per-interval send below then replaces it.
        first_stats_steps = 10
        first_stats_sent = False

        while not control.stop and control.collector_steps_at(collector_id) < num_iterations:
            if not collector.step_once():
                time.sleep(opts.idle_sleep_s)  # ring full -> backpressure
                continue
            steps = control.collector_steps_at(collector_id)
            if not first_stats_sent and steps >= first_stats_steps:
                first_stats_sent = True
                runtime.report({"type": "collector_stats", "stats": collector.snapshot_stats()})
            elif steps % max(logging_interval, 1) == 0:
                runtime.report({"type": "collector_stats", "stats": collector.snapshot_stats()})
        if control.collector_steps_at(collector_id) >= num_iterations and not runtime.should_stop():
            runtime.finish({"reason": "peer_budget"})
    finally:
        # Algorithm completion stops peers, but never requests public cancellation.
        if control.collector_steps_at(collector_id) >= num_iterations:
            control.set_stop()


# ------------------------------------------------------------------ learner process
def run_learner_process(
    runtime: WorkerRuntime,
    cfg: FastSacCfg,
    num_envs: int,
    dims: tuple[int, int, int],
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    rings: list[SharedTransitionRing | RingCursors],
    weights: list[WeightChannelShared],
    control: Control,
    num_iterations: int,
    logging_interval: int,
    save_interval: int,
    run_dir: str,
    checkpoint_dir: str,
    checkpoint_format: str,
    resume_from: str | None,
    handshake: StartupHandshake,
    all_rings: list[SharedTransitionRing],
    weight_ipc: list[bool],
    rank: int = 0,
    num_learners: int = 1,
    rendezvous_file: str | None = None,
    learner_device: str | None = None,
    final_stores: list[FinalObsStorage] | None = None,
    notify_checkpoints: bool = False,
) -> None:
    """One learner process; ``rank 0`` additionally owns stats aggregation,
    logging and checkpointing.

    ``final_stores`` is either absent for all owned collectors or contains one
    store per owned ring; final-observation support is uniform across collectors.

    Multi-learner (DDP) specifics: ``rings`` is this rank's slice (the parent
    partitions by collector ownership) while ``all_rings`` is the full list
    (rank-0 logging only); ``weights`` is this rank's slice of the weight
    channels — every learner publishes to its OWN collectors only; update
    counts are derived from the global progress basis (see
    ``Learner.maybe_train``).
    """
    is_primary = num_learners == 1 or rank == 0
    if num_learners > 1:
        import torch.distributed as dist
    else:
        dist = None
    control.bind_stop(runtime.stop_event)
    try:
        opts = cfg.trainer.async_options
        num_collectors = len(all_rings)
        if learner_device is not None:
            device = torch.device(learner_device)
        elif cfg.device:
            device = torch.device(cfg.device)
        else:
            device = resolve_default_device()
        if num_learners > 1:
            if device.type != "cuda" or device.index is None:
                raise ValueError(f"multi-learner requires explicit indexed CUDA learner devices, got {device}")
            torch.cuda.set_device(device)
            # Bounded collective timeout: with the default 30-minute timeout a
            # rank waiting on a crashed/blocked peer hangs until the parent's
            # force-kill instead of unwinding through its own error path.
            import datetime as _dt

            dist.init_process_group(
                "nccl",
                init_method=f"file://{rendezvous_file}",
                rank=rank,
                world_size=num_learners,
                timeout=_dt.timedelta(seconds=90),
            )
        # Replay buffer sized by THIS rank's env shard: each learner drains only
        # its collectors' rings, so every ingested batch carries
        # num_envs // num_learners envs (env-level sharding keeps each env's
        # full trajectory — and n-step adjacency — inside one rank). The batch
        # math still divides the GLOBAL batch_size by world_size, so per-rank
        # sample rows == batch_size / num_learners, matching DDP averaging.
        agent_started = time.perf_counter()
        agent = build_agent(
            cfg,
            dims,
            num_envs // num_learners,
            device,
            action_scale,
            action_bias,
            world_size=num_learners,
            final_obs_bootstrap=final_stores is not None,
        )
        if resume_from:
            ckpt = torch.load(resume_from, map_location=device, weights_only=False)
            agent.load_state_dict(ckpt, load_optimizers=True)
        logger.info(
            "learner startup: agent build%s finished in %.3fs",
            " + checkpoint load" if resume_from else "",
            time.perf_counter() - agent_started,
        )

        resume_step = control.collector_steps // num_collectors  # full-batch equivalents
        per_collector_envs = num_envs // num_collectors
        per_learner = num_collectors // num_learners
        # Build one sender endpoint per collector OWNED by this rank (collector
        # i belongs to rank i // per_learner) and ship the slot tensors BEFORE
        # the first publish so each collector (blocking on its handshake queue)
        # builds the matching receiver from step one. The IPC transition-ring
        # slots ship in the same one-shot message: this process is the owner
        # (it allocated the device tensor and must keep it alive), the
        # collector maps it through the queue's CUDA-IPC reducers.
        # Transport per channel was decided by the parent topology pass
        # (TrainerTopology.weight_ipc, indexed per collector); this process
        # only constructs the endpoints. One throwaway CPU actor counts the
        # parameters that size both transports' buffers.
        param_numel = actor_param_numel(cfg, dims, action_scale, action_bias)
        weight_txs = []
        for j, shared in enumerate(weights):
            channel = rank * per_learner + j
            # CUDA-IPC device slots when the topology resolved IPC for this
            # channel (the collector infers on this rank's GPU), host
            # shared-memory slots otherwise; this rank is the exporter (its
            # CUDA context must keep the slot tensors alive).
            tx = (
                GpuIpcWeightSender(shared, param_numel, device)
                if weight_ipc[j]
                else HostWeightSender(shared, param_numel)
            )
            handshake.ship_weight_slots(channel, tx.params)
            weight_txs.append(tx)
        # Wrap RingCursors into IpcTransitionRing (allocating the device slots
        # and shipping them to the collector) BEFORE building the endpoints —
        # the endpoints must hold the FINAL ring objects the drain path uses.
        for j, ring in enumerate(rings):
            ring_slots = None
            if isinstance(ring, RingCursors):
                if device.type != "cuda":
                    raise RuntimeError("the IPC transition ring was selected but the learner device is not CUDA")
                obs_dim, critic_obs_dim, act_dim = dims
                feat = obs_dim + critic_obs_dim + act_dim + 3
                ring_slots = torch.zeros(
                    opts.ring_capacity, per_collector_envs, feat, dtype=torch.float32, device=device
                )
                rings[j] = IpcTransitionRing(
                    ring,
                    ring_slots,
                    opts.ring_capacity,
                    per_collector_envs,
                    obs_dim,
                    critic_obs_dim,
                    act_dim,
                    final_obs_store=final_stores[j] if final_stores is not None else None,
                )
            handshake.ship_ring_slots(rank * per_learner + j, ring_slots)

        endpoints = [CollectorEndpoint(ring=rings[j], weight_sender=weight_txs[j]) for j in range(len(weight_txs))]

        learner = Learner(
            agent,
            cfg,
            endpoints,
            control,
            ddp_rank=rank if num_learners > 1 else None,
        )
        learner._last_train_gstep = resume_step  # lockstep basis continues from the checkpoint
        learner.publish_weights()  # give every collector an initial policy before it warms up

        # Elapsed/ETA anchor. The collector's env build (scene compile, numba
        # JIT) runs concurrently with the learner build and can outlast it by
        # tens of seconds; timing from this point would bill that startup wait
        # to elapsed and the first window's rates. The anchor moves to the
        # first ingested collector batch below.
        start_time = time.time()
        start_anchored = False
        last_log_time = start_time
        last_log_step = resume_step
        train_started = False
        # Initial and first-update payloads keep console readiness independent
        # of publication cadence. Subsequent windows follow learner iterations,
        # not collection progress (which can run far ahead in learner-bound mode).
        initial_payload_sent = False
        next_log = (
            ((learner.learning_iteration // logging_interval) + 1) * logging_interval if logging_interval > 0 else 0
        )
        next_save = ((resume_step // save_interval) + 1) * save_interval if save_interval > 0 else 0
        t_learn_win = 0.0  # wall-clock spent in learner train calls this log window
        learner_train_samples_ms: list[float] = []
        learner_drain_samples_ms: list[float] = []
        learner_breakdown_samples_ms: dict[str, list[float]] = {}
        learner_ring_wait_samples_ms: list[float] = []
        learner_gate_wait_samples_ms: list[float] = []
        last_checkpoint_path: str | None = None
        # The parent process renders the panel and writes TensorBoard: this
        # worker ships one compact payload per log window (all learner ranks
        # send — workers are peers, the parent is the recorder).

        # Startup barrier: report readiness and hold until every worker booted
        # (see run_collector_process). Weight slots were already shipped and
        # the initial weights published, so collectors blocking on their
        # handshakes complete as soon as every learner reaches this point.
        runtime.ready()

        # Each collector runs until its own counter reaches num_iterations; the
        # aggregate (sum) therefore reaches num_iterations * num_collectors.
        total_collector_steps = num_iterations * num_collectors
        while not control.stop and control.collector_steps < total_collector_steps:
            # Progress basis: full num_envs-batch equivalents — the aggregate
            # counter counts per-collector batches of per_collector_envs
            # transitions each, so dividing by num_collectors recovers the
            # sync-trainer "iteration collects num_envs transitions" unit
            # (identical to the raw counter when num_collectors == 1). All
            # ranks read the same shared value.
            step = control.collector_steps // num_collectors
            control.global_step = step
            t_drain = time.perf_counter()
            ingested = learner.drain()
            if ingested:
                if not start_anchored:
                    start_time = time.time()
                    last_log_time = start_time
                    start_anchored = True
                learner_drain_samples_ms.append((time.perf_counter() - t_drain) * 1000.0)
            t_l = time.perf_counter()
            # One decision path for both topologies: updates are gated on the
            # global progress basis (identical on every DDP rank; the single
            # learner's own position when num_collectors == 1).
            trained = learner.maybe_train(step)
            first_update = trained and not train_started
            if trained:
                train_started = True
                elapsed_learn_s = time.perf_counter() - t_l
                t_learn_win += elapsed_learn_s
                learner_train_samples_ms.append(elapsed_learn_s * 1000.0)
                # Keep the raw total cost of one agent.update(n) call. The
                # timing tree uses total-cost semantics, not per-gradient-step
                # normalization.
                for key, value in learner.agent._last_update_timing_ms.items():
                    learner_breakdown_samples_ms.setdefault(key, []).append(value)
                learner_breakdown_samples_ms.setdefault("publish", []).append(learner._last_publish_ms)
            else:
                # warmup or starved: avoid a hot spin.
                t_idle = time.perf_counter()
                time.sleep(opts.idle_sleep_s)
                idle_ms = (time.perf_counter() - t_idle) * 1000.0
                if ingested == 0:
                    # No ring slot was available: learner is waiting for collector data.
                    learner_ring_wait_samples_ms.append(idle_ms)
                else:
                    # Data arrived, but replay/batch readiness still gated training.
                    learner_gate_wait_samples_ms.append(idle_ms)
            if not initial_payload_sent or first_update or (trained and learner.learning_iteration >= next_log):
                # Publish first-update completion immediately: the startup
                # handoff must not wait an entire user logging interval.
                # One compact payload per log window to the parent (panel +
                # TensorBoard live there; see train.py). All ranks send.
                now = time.time()
                sps = (step - last_log_step) * num_envs / max(now - last_log_time, 1e-6)
                warming = step < agent.cfg.learning_starts
                metrics_log = None
                if train_started and not warming:
                    # Window-aggregated diagnostics only (means + extremes
                    # since the last log window): single-update events between
                    # log points stay visible, mirroring holosoma's averaged
                    # meter logging. train_started just gates "training
                    # started".
                    metrics_log = learner.agent.window_metrics.stats_and_reset()
                learn_pct = 100.0 * t_learn_win / max(now - last_log_time, 1e-9)
                learner_items: dict[str, Any] = {}
                if learner_drain_samples_ms:
                    learner_items["drain"] = timing_mean(learner_drain_samples_ms)
                if learner_ring_wait_samples_ms:
                    learner_items["ring wait"] = timing_mean(learner_ring_wait_samples_ms)
                if learner_gate_wait_samples_ms:
                    learner_items["gate wait"] = timing_mean(learner_gate_wait_samples_ms)
                update_items = {key: timing_mean(values) for key, values in learner_breakdown_samples_ms.items()}
                if update_items:
                    # publish is a child stage of the learner update in the
                    # panel, so include it in the displayed update total too.
                    if "publish" in update_items and "total" in update_items:
                        update_items["total"] += update_items["publish"]
                    learner_items["update"] = update_items
                payload = {
                    "rank": rank,
                    "step": step,
                    "sps": sps,
                    "updates": learner.update_idx,
                    "update_idx": learner.update_idx,
                    "learning_iteration": learner.learning_iteration,
                    "training_started": train_started,
                    "metrics": metrics_log,
                    "warming": warming,
                    "learn_ms": timing_mean(learner_train_samples_ms) if learner_train_samples_ms else 0.0,
                    "learn_pct": learn_pct,
                    "learner_timing": learner_items,
                    "buffer_size": agent.rb.num_stored * agent.num_envs,
                    "buffer_capacity": agent.rb.buffer_size * agent.num_envs,
                    # bare RingCursors (IPC transport) carry no size; the
                    # wrapped IpcTransitionRing lives in this worker.
                    "ring_fill": [None if isinstance(ring, RingCursors) else ring.size() for ring in all_rings],
                    "weight_version": max((tx.version for tx in weight_txs), default=None),
                    "checkpoint_path": last_checkpoint_path,
                }
                # Never block the training loop on a slow panel consumer:
                # drop the window's payload if the queue is full (the next
                # window's numbers supersede it anyway).
                runtime.report({"type": "learner_payload", "payload": payload})
                last_log_time, last_log_step = now, step
                t_learn_win = 0.0
                learner_train_samples_ms = []
                learner_drain_samples_ms = []
                learner_breakdown_samples_ms = {}
                learner_ring_wait_samples_ms = []
                learner_gate_wait_samples_ms = []
                # Recompute the window edge (not +=) so the immediate first
                # payload cannot shift every later window off the interval.
                initial_payload_sent = True
                next_log = (
                    ((learner.learning_iteration // logging_interval) + 1) * logging_interval
                    if logging_interval > 0
                    else 0
                )

            if is_primary and save_interval > 0 and step >= next_save and step > 0:
                agent.global_step = step
                learner.wait_ingest()  # checkpoint reads the rb tensors
                path = Path(checkpoint_dir) / f"model_{step:07d}.pt"
                torch.save(agent.state_dict(), path)
                checkpoints.record_checkpoint_artifact(
                    Path(run_dir),
                    checkpoints.LATEST_TRAINING_STATE,
                    path,
                    checkpoints.TRAINING_STATE,
                    checkpoint_format=checkpoint_format,
                )
                last_checkpoint_path = str(path)
                logger.info("saved checkpoint %s", path)
                # A notice per save for live metrics, separate from the log-window
                # payload: this step's payload already went out above, and a save
                # after the last log window would otherwise never reach the parent.
                # Never blocks the learner, like the log payloads.
                if notify_checkpoints:
                    runtime.report({"type": "checkpoint_saved", "step": step, "path": str(path)})
                next_save += save_interval

        # Flush the unfinished learner window with the final counters. Empty
        # windows stay empty; never relabel a preceding window's diagnostics
        # with a newer learner iteration.
        runtime.report(
            {
                "type": "learner_payload",
                "payload": {
                    "rank": rank,
                    "learner_counters": {
                        "learning_iteration": learner.learning_iteration,
                        "update_idx": learner.update_idx,
                        "updates": learner.update_idx,
                        "metrics": agent.window_metrics.stats_and_reset() if train_started else None,
                    },
                },
            }
        )

        if num_learners > 1:
            # All ranks have consumed the identical global step count, so the
            # collectives inside update() are matched; this barrier aligns
            # teardown so no rank destroys the process group while another is
            # still inside a collective.
            dist.barrier()
        if is_primary:
            # final checkpoint (identical structure to sync fastsac)
            agent.global_step = control.collector_steps // num_collectors
            learner.wait_ingest()  # checkpoint reads the rb tensors
            ckpt_path = checkpoints.final_checkpoint_path(checkpoint_format, Path(run_dir))
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(agent.state_dict(), ckpt_path)
            checkpoints.record_checkpoint_artifact(
                Path(run_dir),
                checkpoints.LATEST_TRAINING_STATE,
                ckpt_path,
                checkpoints.TRAINING_STATE,
                checkpoint_format=checkpoint_format,
            )
            checkpoints.record_checkpoint_artifact(
                Path(run_dir),
                checkpoints.BEST_POLICY,
                ckpt_path,
                checkpoints.POLICY,
                checkpoint_format=checkpoint_format,
            )
            logger.info("saved checkpoint to %s", ckpt_path)
            runtime.report({"type": "checkpoint_saved", "step": agent.global_step, "path": str(ckpt_path)})
        if not runtime.should_stop():
            runtime.finish({"reason": "completed"})
    finally:
        if num_learners > 1 and dist is not None and dist.is_initialized():
            dist.destroy_process_group()
