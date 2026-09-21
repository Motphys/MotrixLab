# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Process entry points and shared builders for the M1 two-process trainer.

The collector process (CPU) runs the env + inference policy and feeds the shared
transition ring; the learner process (GPU) drains the ring, trains, publishes
weights, and owns logging + checkpointing. Both are spawned by
:class:`~motrix_rl.fastsac.async_impl.train.Trainer`. Entry functions are module-level
so ``torch.multiprocessing`` (``spawn``) can pickle them by qualified name.

Build helpers (``build_env`` / ``build_agent``) are shared with the single-process
(M0) path so both trainers construct env/agent identically.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
import traceback
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import torch

from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.registry import EnvBuildSpec
from motrix_env_core.renderer import RenderConfig
from motrix_env_motrixsim.torch_env import TorchEnv
from motrix_rl import checkpoints
from motrix_rl.console import TrainingPanelStats, emit_training_panel, open_training_live
from motrix_rl.fastsac.agent import FastSacAgent
from motrix_rl.fastsac.async_impl.collector import Collector, resolve_collector_inference_device
from motrix_rl.fastsac.async_impl.learner import Learner
from motrix_rl.fastsac.async_impl.shm import Control, SharedTransitionRing
from motrix_rl.fastsac.async_impl.shm.weight_channel import (
    GpuIpcWeightSender,
    HostWeightSender,
    WeightChannelShared,
    WeightSender,
    weight_receiver_for,
)
from motrix_rl.fastsac.config import FastSacCfg
from motrix_rl.fastsac.wrap import FastSacEnvWrap
from motrix_rl.fastsac.wrap_np import FastSacNpEnvWrap
from motrix_rl.fastsac.wrap_torch import FastSacTorchEnvWrap
from motrix_rl.system_metrics import CpuLoadSampler, GpuMemoryUsageSampler, GpuUtilizationSampler, MemoryUsageSampler


def _timing_mean(values: list[float]) -> float:
    """Mean of a non-empty timing sample list (ms)."""
    return sum(values) / len(values)


# ------------------------------------------------------------------ builders
def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    from motrix_rl.fastsac.factory import make_actor

    obs_dim, _critic_obs_dim, act_dim = dims
    actor = make_actor(
        cfg,
        dims=(obs_dim, act_dim),
        action_scale=action_scale,
        action_bias=action_bias,
        device="cpu",
    )
    return sum(p.numel() for p in actor.parameters())


def actor_buffer_numel(cfg: FastSacCfg, dims, action_scale, action_bias) -> int:
    """Total persistent actor-buffer count for shared snapshot sizing."""
    from motrix_rl.fastsac.factory import make_actor

    obs_dim, _critic_obs_dim, act_dim = dims
    actor = make_actor(
        cfg,
        dims=(obs_dim, act_dim),
        action_scale=action_scale,
        action_bias=action_bias,
        device="cpu",
    )
    return sum(buffer.numel() for buffer in actor.buffers() if buffer.numel())


def build_agent(cfg: FastSacCfg, dims, num_envs, device, action_scale, action_bias, writer=None) -> FastSacAgent:
    obs_dim, critic_obs_dim, act_dim = dims
    return FastSacAgent(
        obs_dim=obs_dim,
        critic_obs_dim=critic_obs_dim,
        act_dim=act_dim,
        num_envs=num_envs,
        cfg=cfg,
        device=device,
        action_scale=action_scale,
        action_bias=action_bias,
        writer=writer,
    )


# ------------------------------------------------------------------ collector process
def _available_cpu_ids() -> set[int]:
    """CPU ids this process may run on (Linux affinity mask, Windows process mask)."""
    if hasattr(os, "sched_getaffinity"):
        return set(os.sched_getaffinity(0))
    if sys.platform == "win32":
        import ctypes

        process_mask = ctypes.c_ulonglong()
        system_mask = ctypes.c_ulonglong()
        kernel32 = ctypes.windll.kernel32
        got = kernel32.GetProcessAffinityMask(
            kernel32.GetCurrentProcess(), ctypes.byref(process_mask), ctypes.byref(system_mask)
        )
        if got:
            return {i for i in range(64) if process_mask.value & (1 << i)}
    return set(range(os.cpu_count() or 1))


def _set_cpu_affinity(cpus: set[int]) -> None:
    """Pin this process to ``cpus``; best-effort and never fatal.

    Raises OSError (or skips with a warning on platforms without an affinity
    API) so the caller can decide; permission-constrained environments
    (cgroups/cpusets) degrade to a warning instead of killing the worker.
    """
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    elif sys.platform == "win32":
        import ctypes

        mask = ctypes.c_ulonglong(sum(1 << c for c in cpus))
        current = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.kernel32.SetProcessAffinityMask(current, mask):
            raise OSError(f"SetProcessAffinityMask failed for cpus {sorted(cpus)}")
    else:
        logging.getLogger(__name__).warning(
            "CPU affinity is unsupported on this platform; skipping pinning to %s", sorted(cpus)
        )


def _resolve_cpu_set(spec: str | None, field: str) -> set[int]:
    """Resolve one worker's CPU affinity from a spec like ``"0:5,7"``.

    Each comma-separated item is a single core id or an inclusive ``A:B``
    range; cores outside the process's available set are dropped. An empty
    spec or an empty effective set disables pinning.
    """
    if not spec:
        return set()
    available = _available_cpu_ids()
    cpus: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            if ":" in item:
                start_s, _, end_s = item.partition(":")
                start, end = int(start_s), int(end_s)
            else:
                start = end = int(item)
        except ValueError as exc:
            raise ValueError(
                f"{field}='{spec}': invalid core spec item '{item}' "
                "(expected core ids or inclusive A:B ranges, e.g. '0:5,7')"
            ) from exc
        if start > end:
            raise ValueError(f"{field}='{spec}': range '{item}' has start > end")
        # Clamp to the available core span instead of enumerating the raw
        # range: a typo like 0:1000000000 must not iterate billions of ids.
        lo, hi = min(available), max(available)
        cpus.update(range(max(start, lo), min(end, hi) + 1))
    return cpus & available


def _pin_worker_cpus(cpus: set[int]) -> None:
    """Pin this worker process and cap its torch threads to its CPU slice.

    The thread cap applies even when the affinity call fails (cgroup/cpuset
    constraints): limiting torch to the configured core count still reduces
    CPU contention when running unpinned.
    """
    if not cpus:
        return
    try:
        _set_cpu_affinity(cpus)
    except OSError as exc:
        logging.getLogger(__name__).warning("CPU pinning to %s failed (%s); continuing unpinned", sorted(cpus), exc)
    torch.set_num_threads(max(len(cpus), 1))


def _configure_process_logging() -> None:
    """Surface INFO logs (e.g. manager env startup) from spawned worker processes.

    ``basicConfig`` only applies when the root logger has no handlers yet; when
    a handler is already configured, raising the root level is enough for the
    startup INFO records to be emitted through the existing setup.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(logging.INFO)
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _build_weight_sender(
    shared: WeightChannelShared,
    cfg: FastSacCfg,
    dims: tuple[int, int, int],
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    device: torch.device,
) -> WeightSender:
    """Construct the sender-side weight endpoint per the configured transport.

    CUDA-IPC device slots only when learner and collector inference share one
    GPU and the actor parameters reach the configured size threshold; host
    shared-memory slots otherwise. The learner process is the only place both
    transports' requirements can be met (the IPC-handle exporter needs the
    CUDA context and must keep the tensors alive).
    """
    opts = cfg.trainer.async_options
    mode = opts.weight_ipc
    # YAML 1.1 parses unquoted ``on``/``off`` scalars as booleans; accept that
    # form so ``weight_ipc: on`` in a config behaves like the documented string.
    if isinstance(mode, bool):
        mode = "on" if mode else "off"
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"async_options.weight_ipc must be auto, on or off, got {mode!r}")
    collector_device = resolve_collector_inference_device(opts.collector_inference_device)
    same_gpu = device.type == "cuda" and collector_device.type == "cuda"
    if same_gpu:
        # ``learner=`` without an index means the current device; resolve it so
        # the comparison never treats "cuda" as matching an explicit different
        # index. Only resolve under the cuda branch: a CPU learner has no CUDA
        # context and torch.cuda.current_device() would raise.
        learner_index = device.index if device.index is not None else torch.cuda.current_device()
        same_gpu = learner_index == collector_device.index
    if mode == "on" and not same_gpu:
        reason = (
            "collector inference device is not CUDA"
            if collector_device.type != "cuda"
            else f"learner device {device} and collector device {collector_device} are different GPUs"
        )
        logging.getLogger(__name__).warning(
            "async_options.weight_ipc=on requires learner and collector inference on the same GPU, "
            "but %s; falling back to host shared-memory transport",
            reason,
        )
    param_numel = actor_param_numel(cfg, dims, action_scale, action_bias)
    use_gpu = same_gpu and (mode == "on" or (mode == "auto" and param_numel * 4 >= opts.weight_ipc_min_bytes))
    if use_gpu:
        return GpuIpcWeightSender(shared, param_numel, device)
    return HostWeightSender(shared, param_numel)


def _init_ring_transport(
    ring: SharedTransitionRing, cfg: FastSacCfg, collector_device: torch.device, ring_slot_queue: Queue
) -> None:
    """Collector: enable the CUDA-IPC ring transport when both ends share one GPU.

    The ring's big fields (obs/critic_obs) move into device-memory slots the
    collector allocates here; the slot tensors are shipped to the learner
    through ``ring_slot_queue`` (reduction turns them into cudaIpcMemHandles).
    Always sends a decision message so the learner never blocks past startup.
    """
    opts = cfg.trainer.async_options
    mode = opts.ring_ipc
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"async_options.ring_ipc must be auto, on or off, got {mode!r}")
    learner_device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    learner_index = (
        learner_device.index
        if learner_device.index is not None
        else (torch.cuda.current_device() if learner_device.type == "cuda" else None)
    )
    same_gpu = (
        learner_device.type == "cuda"
        and collector_device.type == "cuda"
        and learner_index == collector_device.index
    )
    if mode == "on" and not same_gpu:
        raise ValueError("async_options.ring_ipc=on requires learner and collector inference on the same GPU")
    if not same_gpu or mode == "off":
        ring_slot_queue.put(("host",))
        return
    device_obs, device_critic_obs = ring.init_device_slots(collector_device)
    ring_slot_queue.put(("gpu", device_obs, device_critic_obs))


def _bind_ring_transport(ring: SharedTransitionRing, ring_slot_queue: Queue, timeout: float = 300.0) -> None:
    """Learner: bind the transport the collector decided on (blocking handshake)."""
    message = ring_slot_queue.get(timeout=timeout)
    kind = message[0]
    if kind == "gpu":
        ring.bind_device_slots(message[1], message[2])
    elif kind != "host":
        raise RuntimeError(f"unexpected ring transport message {kind!r}")


def run_collector_process(
    env_spec: EnvBuildSpec,
    cfg: FastSacCfg,
    num_envs: int,
    dims: tuple[int, int, int],
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    ring: SharedTransitionRing,
    weights: WeightChannelShared,
    control: Control,
    stats_queue: Queue,
    error_queue: Queue,
    num_iterations: int,
    logging_interval: int,
    is_resume: bool,
    seed: int | None,
    slot_queue: Queue,
    ring_slot_queue: Queue,
) -> None:
    try:
        _configure_process_logging()
        set_seed(seed)
        opts = cfg.trainer.async_options
        _pin_worker_cpus(_resolve_cpu_set(opts.collector_cpu_cores, "collector_cpu_cores"))
        async_options = cfg.trainer.async_options
        obs_dim, critic_obs_dim, act_dim = dims
        device = torch.device("cpu")
        env = build_env(env_spec, num_envs, device, seed=seed)
        # Handshake: build the receiver from the slot tensors the learner
        # shipped (host shm or CUDA-IPC), before the collector is wired up.
        try:
            slots = slot_queue.get(timeout=60.0)
        except Empty as exc:
            raise RuntimeError(
                "timed out waiting for the learner to ship the weight-slot tensors "
                "(learner startup — agent build / checkpoint load / CUDA warmup — "
                "likely failed or took over 60s; check the learner process's error queue/log)"
            ) from exc
        weight_rx = weight_receiver_for(weights, slots)
        collector = Collector(
            env,
            cfg,
            obs_dim,
            critic_obs_dim,
            act_dim,
            action_scale,
            action_bias,
            ring,
            weight_rx,
            control,
            is_resume=is_resume,
        )
        # Decide the ring transport now that the inference device is known and
        # unblock the learner's handshake before the (slow) env warmup below.
        _init_ring_transport(ring, cfg, collector.device, ring_slot_queue)
        collector.reset()
        collector.sync_weights()
        collector.warmup_inference()

        while not control.stop and control.collector_steps < num_iterations:
            if not collector.step_once():
                time.sleep(async_options.idle_sleep_s)  # ring full -> backpressure
                continue
            if collector.control.collector_steps % max(logging_interval, 1) == 0:
                # replace any stale snapshot so the learner always sees the latest.
                try:
                    while True:
                        stats_queue.get_nowait()
                except Empty:
                    pass
                stats_queue.put(collector.snapshot_stats())
    except BaseException:
        error_queue.put(("collector", traceback.format_exc()))
        raise
    finally:
        control.set_stop()  # signal the learner if the collector exits for any reason


# ------------------------------------------------------------------ learner process
def run_learner_process(
    cfg: FastSacCfg,
    num_envs: int,
    dims: tuple[int, int, int],
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    ring: SharedTransitionRing,
    weights: WeightChannelShared,
    control: Control,
    stats_queue: Queue,
    error_queue: Queue,
    num_iterations: int,
    logging_interval: int,
    save_interval: int,
    run_dir: str,
    env_name: str,
    checkpoint_dir: str,
    checkpoint_format: str,
    resume_from: str | None,
    seed: int | None,
    slot_queue: Queue,
    ring_slot_queue: Queue,
) -> None:
    _configure_process_logging()
    console, live = open_training_live()
    try:
        set_seed(seed)
        opts = cfg.trainer.async_options
        _pin_worker_cpus(_resolve_cpu_set(opts.learner_cpu_cores, "learner_cpu_cores"))
        async_options = cfg.trainer.async_options
        device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=run_dir)
        except Exception:
            writer = None

        agent = build_agent(cfg, dims, num_envs, device, action_scale, action_bias, writer=writer)
        if resume_from:
            ckpt = torch.load(resume_from, map_location=device, weights_only=False)
            agent.load_state_dict(ckpt, load_optimizers=True)

        # Build the sender endpoint and ship its slot tensors BEFORE the first
        # publish so the collector (blocking on the handshake queue) builds the
        # matching receiver and takes the agreed transport from step one.
        weight_tx = _build_weight_sender(weights, cfg, dims, action_scale, action_bias, device)
        slot_queue.put(weight_tx.params)
        # Bind the ring transport the collector decided on (host shm or
        # CUDA-IPC device slots) before the first drain. Ordered after the
        # weight handshake above: the collector only decides the ring
        # transport once it has the weight slots, so binding any earlier
        # would deadlock both startups against each other.
        _bind_ring_transport(ring, ring_slot_queue)
        learner = Learner(agent, cfg, ring, weight_tx, control)
        learner.publish_weights()  # give the collector an initial policy before it warms up

        start_time = time.time()
        last_log_time = start_time
        resume_step = control.collector_steps
        last_log_step = resume_step
        last_update_idx = 0
        last_stats = {
            "return": float("nan"),
            "ep_len": float("nan"),
            "episodes": 0,
            "reward_terms": {},
            "env_metrics": {},
            "policy_lag": 0,
            "timing_ms": {},
        }
        last_metrics = None
        next_log = ((resume_step // logging_interval) + 1) * logging_interval if logging_interval > 0 else 0
        next_save = ((resume_step // save_interval) + 1) * save_interval if save_interval > 0 else 0
        t_learn_win = 0.0  # wall-clock spent in learner train calls this log window
        learner_train_samples_ms: list[float] = []
        learner_drain_samples_ms: list[float] = []
        learner_breakdown_samples_ms: dict[str, list[float]] = {}
        learner_ring_wait_samples_ms: list[float] = []
        learner_gate_wait_samples_ms: list[float] = []
        cpu_sampler = CpuLoadSampler()
        gpu_sampler = GpuUtilizationSampler()
        memory_sampler = MemoryUsageSampler()
        gpu_memory_sampler = GpuMemoryUsageSampler()
        last_checkpoint_path: str | None = None

        def _drain_stats():
            nonlocal last_stats
            try:
                while True:
                    last_stats = stats_queue.get_nowait()
            except Empty:
                pass

        while not control.stop and control.collector_steps < num_iterations:
            t_drain = time.perf_counter()
            ingested = learner.drain()
            if ingested:
                learner_drain_samples_ms.append((time.perf_counter() - t_drain) * 1000.0)
            t_l = time.perf_counter()
            metrics = learner.maybe_train(ingested)
            if metrics is not None:
                last_metrics = metrics
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
                time.sleep(async_options.idle_sleep_s)
                idle_ms = (time.perf_counter() - t_idle) * 1000.0
                if ingested == 0:
                    # No ring slot was available: learner is waiting for collector data.
                    learner_ring_wait_samples_ms.append(idle_ms)
                else:
                    # Data arrived, but replay/batch readiness still gated training.
                    learner_gate_wait_samples_ms.append(idle_ms)
            step = control.collector_steps
            control.global_step = step

            if step >= next_log:
                _drain_stats()
                now = time.time()
                sps = (step - last_log_step) * num_envs / max(now - last_log_time, 1e-6)
                warming = step < agent.cfg.learning_starts
                metrics_log = {k: float(v) for k, v in last_metrics.items()} if (last_metrics and not warming) else None
                updates = learner.update_idx
                utd = updates / max(step, 1)
                # Per-process timing (collector and learner run concurrently, so
                # these do NOT sum to 100% like the sync panel):
                #   timing_ms[collect] — collector's avg ms per env-step batch (from queue)
                #   timing_ms[wait] — avg ring-backpressure wait per batch
                #   learn_ms   — learner's avg ms per train call (one UTD
                #                execution, which may run several gradient
                #                updates); idle waits are not included
                #   learn_pct  — fraction of learner wall-clock spent updating vs
                #                idle/starved (≈100% when GPU-bound, lower if the
                #                collector can't keep the buffer fed)
                # Window means only: live-panel percentiles are noise at these
                # sample counts (benchmarks own the tail statistics).
                learn_pct = 100.0 * t_learn_win / max(now - last_log_time, 1e-9)
                collector_timing_ms = last_stats.get("timing_ms", {})
                collector_timing_detail_ms = {
                    key: value for key, value in collector_timing_ms.items() if key != "collect"
                }
                # Panel tree is per-process; the headline collect/learn means
                # live on TrainingPanelStats, sub-stages nest under "sync" /
                # "update" branches.
                collector_items: dict[str, Any] = {}
                sync_items: dict[str, float] = {}
                for key, value in collector_timing_detail_ms.items():
                    if key.startswith("sync_"):
                        sync_items[key[len("sync_") :]] = value
                    else:
                        collector_items[key] = value
                if sync_items:
                    collector_items["sync"] = {"total": collector_items.pop("sync", 0.0), **sync_items}
                timing_groups = {"collector": collector_items}
                learner_items: dict[str, Any] = {}
                drain_ms = _timing_mean(learner_drain_samples_ms) if learner_drain_samples_ms else 0.0
                ring_wait_ms = _timing_mean(learner_ring_wait_samples_ms) if learner_ring_wait_samples_ms else 0.0
                gate_wait_ms = _timing_mean(learner_gate_wait_samples_ms) if learner_gate_wait_samples_ms else 0.0
                if learner_drain_samples_ms:
                    learner_items["drain"] = drain_ms
                if learner_ring_wait_samples_ms:
                    learner_items["ring wait"] = ring_wait_ms
                if learner_gate_wait_samples_ms:
                    learner_items["gate wait"] = gate_wait_ms
                update_items = {key: _timing_mean(values) for key, values in learner_breakdown_samples_ms.items()}
                if update_items:
                    # publish is a child stage of the learner update in the
                    # panel, so include it in the displayed update total too.
                    if "publish" in update_items and "total" in update_items:
                        update_items["total"] += update_items["publish"]
                    learner_items["update"] = update_items
                if learner_items:
                    timing_groups["learner"] = learner_items
                learn_ms = _timing_mean(learner_train_samples_ms) if learner_train_samples_ms else 0.0
                stats = TrainingPanelStats(
                    iteration=step,
                    total_iterations=num_iterations,
                    steps_per_second=sps,
                    elapsed_seconds=now - start_time,
                    mean_return=last_stats["return"],
                    mean_episode_length=last_stats["ep_len"],
                    episodes=last_stats["episodes"],
                    buffer_size=agent.rb.num_stored * num_envs,
                    buffer_capacity=agent.rb.buffer_size * num_envs,
                    collect_ms=collector_timing_ms.get("collect", 0.0),
                    learn_ms=learn_ms,
                    learn_percent=learn_pct,
                    warming=warming,
                    training_metrics=metrics_log,
                    reward_terms=last_stats["reward_terms"],
                    env_metrics=last_stats["env_metrics"],
                    timing_groups=timing_groups,
                    diagnostics={"UTD": utd},
                    cpu_load=cpu_sampler.sample(),
                    gpu_utilization_percent=gpu_sampler.sample(),
                    memory_usage=memory_sampler.sample(),
                    gpu_memory_usage=gpu_memory_sampler.sample(),
                    checkpoint_path=last_checkpoint_path,
                )
                emit_training_panel(live, stats, title=f"{env_name}/motrix.fastsac")
                if writer is not None:
                    writer.add_scalar("rollout/mean_return", last_stats["return"], step)
                    writer.add_scalar("rollout/mean_ep_len", last_stats["ep_len"], step)
                    writer.add_scalar("perf/env_steps_per_s", sps, step)
                    writer.add_scalar(
                        "perf/updates_per_s", (updates - last_update_idx) / max(now - last_log_time, 1e-6), step
                    )
                    writer.add_scalar("async/policy_lag", last_stats["policy_lag"], step)
                    writer.add_scalar("async/ring_fill", ring.size(), step)
                    writer.add_scalar("async/weight_version", weight_tx.version, step)
                    writer.add_scalar("async/utd", utd, step)
                    writer.add_scalar("perf/collect_ms_per_batch", collector_timing_ms.get("collect", 0.0), step)
                    for k, v in collector_timing_detail_ms.items():
                        writer.add_scalar(f"perf/collector_{k}_ms", v, step)
                    writer.add_scalar("perf/learn_ms_total", learn_ms, step)
                    writer.add_scalar("perf/learn_pct", learn_pct, step)
                    for k, v in last_stats["env_metrics"].items():
                        writer.add_scalar(f"metrics/{k}", v, step)
                    for k, v in last_stats["reward_terms"].items():
                        writer.add_scalar(f"reward/{k}", v, step)
                    if metrics_log is not None:
                        for k, v in metrics_log.items():
                            writer.add_scalar(f"train/{k}", v, step)
                last_log_time, last_log_step, last_update_idx = now, step, updates
                t_learn_win = 0.0
                learner_train_samples_ms = []
                learner_drain_samples_ms = []
                learner_breakdown_samples_ms = {}
                learner_ring_wait_samples_ms = []
                learner_gate_wait_samples_ms = []
                next_log += logging_interval

            if save_interval > 0 and step >= next_save and step > 0:
                agent.global_step = step
                path = Path(checkpoint_dir) / f"model_{step:07d}.pt"
                torch.save(agent.state_dict(), path)
                checkpoints.record_checkpoint_artifact(
                    Path(run_dir),
                    checkpoints.LATEST_TRAINING_STATE,
                    path,
                    checkpoints.TRAINING_STATE,
                    checkpoint_format=checkpoint_format,
                )
                if console is not None:
                    last_checkpoint_path = str(path)
                else:
                    print(f"[motrix.fastsac async] saved checkpoint {path}")
                next_save += save_interval

        # final checkpoint (identical structure to sync fastsac)
        agent.global_step = control.collector_steps
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
        (console.print if console else print)(f"[motrix.fastsac async] saved checkpoint to {ckpt_path}")
        if writer is not None:
            writer.close()
    except BaseException:
        error_queue.put(("learner", traceback.format_exc()))
        raise
    finally:
        if live is not None:
            live.stop()
        control.set_stop()  # tell the collector to exit
