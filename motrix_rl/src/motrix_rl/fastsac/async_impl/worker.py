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
from contextlib import contextmanager
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty, Full
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from motrix_env_core.array.env import ArrayEnv
from motrix_env_core.registry import EnvBuildSpec
from motrix_env_core.renderer import RenderConfig
from motrix_env_motrixsim.torch_env import TorchEnv
from motrix_rl import checkpoints
from motrix_rl.fastsac.agent import FastSacAgent
from motrix_rl.fastsac.async_impl.collector import Collector
from motrix_rl.fastsac.async_impl.learner import CollectorEndpoint, Learner
from motrix_rl.fastsac.async_impl.numa import apply_binding
from motrix_rl.fastsac.async_impl.stats import timing_mean
from motrix_rl.fastsac.async_impl.transport import (
    Control,
    IpcTransitionRing,
    RingCursors,
    SharedTransitionRing,
)
from motrix_rl.fastsac.async_impl.transport.handshake import StartupHandshake
from motrix_rl.fastsac.async_impl.transport.weight_channel import (
    GpuIpcWeightSender,
    HostWeightSender,
    WeightChannelShared,
    weight_receiver_for,
)
from motrix_rl.fastsac.config import FastSacCfg
from motrix_rl.fastsac.wrap import FastSacEnvWrap
from motrix_rl.fastsac.wrap_np import FastSacNpEnvWrap
from motrix_rl.fastsac.wrap_torch import FastSacTorchEnvWrap

logger = logging.getLogger(__name__)


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
    from motrix_rl.fastsac.networks import Actor

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
) -> FastSacAgent:
    """Build the learner's ``FastSacAgent``.

    ``dims`` is ``(obs_dim, critic_obs_dim, act_dim)``. ``action_scale`` /
    ``action_bias`` map the tanh-squashed policy output to the env action
    range (``None`` = identity). ``writer`` is the parent-created
    TensorBoard writer; ``world_size`` is the DDP learner-rank count.
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


def _install_graceful_sigint(control: Control) -> None:
    """Turn SIGINT (terminal Ctrl+C hits the whole process group) into the
    shared stop flag so the worker unwinds at its next loop safe point instead
    of dying mid-CUDA/queue operation with a traceback.

    A worker blocked inside a NCCL collective never returns to the interpreter,
    so the handler cannot fire there — the parent's escalation ladder
    (join -> terminate -> kill, see train.py) is the backstop for that case.
    """
    import signal

    def _handler(signum, frame):  # noqa: ARG001
        control.set_stop()

    try:
        signal.signal(signal.SIGINT, _handler)
    except ValueError:  # not on the main thread (defensive; workers are processes)
        pass


@contextmanager
def inherit_log_stdio(log_file: Path):
    """Parent-side spawn guard: point fd 1/2 at the worker's log file while starting it.

    Spawn children inherit the parent's terminal fds and re-import
    torch/simulator modules BEFORE their entry function runs
    (``_configure_process_logging``), so import-time prints (library banners,
    profiler messages, warnings) would land raw on the shared terminal and
    jitter the parent's live panel. Holding the log file on fd 1/2 across
    ``Process.start()`` makes the child inherit it from its first
    instruction; the parent's own fds are restored immediately afterwards.
    """
    log_fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    saved_stdout, saved_stderr = os.dup(1), os.dup(2)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(log_fd)


def _configure_process_logging(log_file: Path) -> None:
    """Route worker logs (e.g. manager env startup) to a file, not the terminal.

    The parent renders the live panel on the shared terminal; raw worker log
    bytes would tear the rich Live frames apart. Startup records stay
    inspectable under the run's ``logs/`` directory; errors additionally
    travel the error-queue channel as before.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        filename=str(log_file),
        filemode="a",
        force=True,
    )
    # Hard hat: dup the log file onto stdout/stderr so C-level output
    # (third-party import banners, library warnings) can never reach the
    # shared terminal and tear the parent's Rich panel.
    log_fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)


def run_collector_process(
    env_spec: EnvBuildSpec,
    cfg: FastSacCfg,
    num_envs: int,
    dims: tuple[int, int, int],
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    ring: SharedTransitionRing | RingCursors,
    weights: WeightChannelShared,
    control: Control,
    stats_queue: Queue,
    error_queue: Queue,
    num_iterations: int,
    logging_interval: int,
    is_resume: bool,
    seed: int | None,
    run_dir: str,
    handshake: StartupHandshake,
    collector_id: int = 0,
    numa_node: int | None = None,
    cpus: list[int] | None = None,
    collector_device: str | None = None,
) -> None:
    role = f"collector[{collector_id}]"
    _install_graceful_sigint(control)
    try:
        log_dir = Path(run_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _configure_process_logging(log_dir / f"collector{collector_id}.log")
        # Bind before any env / staging allocation: the memory policy only
        # affects future pages, so this must be the first thing the worker does.
        # The binding itself was decided by the parent topology pass.
        apply_binding(role, numa_node, cpus or [])
        set_seed(seed)
        opts = cfg.trainer.async_options
        # Multi-learner: the parent resolves the generic "cuda" spec to the
        # owning learner's GPU; an explicit spec passes through unchanged.
        if collector_device is not None:
            opts.collector_inference_device = collector_device
        _pin_worker_cpus(_resolve_cpu_set(opts.collector_cpu_cores, "collector_cpu_cores"))
        obs_dim, critic_obs_dim, act_dim = dims
        device = torch.device("cpu")
        env_started = time.perf_counter()
        env = build_env(env_spec, num_envs, device, seed=seed)
        logger.info(
            "collector startup: env build (scene + manager kernels) finished in %.3fs",
            time.perf_counter() - env_started,
        )
        # Handshake: build the endpoints from the slot tensors the learners
        # shipped before the collector is wired up. Weight slots come from
        # the publishing learner (rank 0), CUDA-IPC transition-ring slots
        # from the learner that drains this collector's ring (its owner —
        # always rank 0 in the single-learner topology).
        weight_slots = handshake.await_weight_slots(collector_id)
        ring_slots = handshake.await_ring_slots(collector_id)
        weight_rx = weight_receiver_for(weights, weight_slots)
        if isinstance(ring, RingCursors):
            if ring_slots is None:
                raise RuntimeError("the learner shipped no transition-ring slots for the IPC ring handshake")
            ring = IpcTransitionRing(ring, ring_slots, opts.ring_capacity, num_envs, obs_dim, critic_obs_dim, act_dim)
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

        # Startup barrier: report readiness and hold until every worker (all
        # collectors AND learners) has booted, so stepping starts in lockstep
        # — warmup counters, weight generations and generation merge align
        # from step one instead of converging mid-run.
        handshake.report_ready("collector", collector_id)
        handshake.wait_for_start(control)
        if control.stop:
            return

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
                stats_queue.put(collector.snapshot_stats())
            elif steps % max(logging_interval, 1) == 0:
                # replace any stale snapshot so the learner always sees the latest.
                try:
                    while True:
                        stats_queue.get_nowait()
                except Empty:
                    pass
                stats_queue.put(collector.snapshot_stats())
    except BaseException:
        if not isinstance(sys.exc_info()[1], (KeyboardInterrupt, SystemExit)):
            # Ctrl+C is an operator action, not a defect: the stop flag is
            # already set by the SIGINT handler; don't spam async_errors.
            error_queue.put((role, traceback.format_exc()))
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
    rings: list[SharedTransitionRing | RingCursors],
    weights: list[WeightChannelShared],
    control: Control,
    error_queue,
    num_iterations: int,
    logging_interval: int,
    save_interval: int,
    run_dir: str,
    checkpoint_dir: str,
    checkpoint_format: str,
    resume_from: str | None,
    seed: int | None,
    handshake: StartupHandshake,
    all_rings: list[SharedTransitionRing],
    weight_ipc: list[bool],
    panel_queue: Queue,
    learner_cpus: list[int] | None = None,
    learner_numa_node: int | None = None,
    rank: int = 0,
    num_learners: int = 1,
    rendezvous_file: str | None = None,
    learner_device: str | None = None,
) -> None:
    """One learner process; ``rank 0`` additionally owns stats aggregation,
    logging and checkpointing.

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
    _install_graceful_sigint(control)
    _learner_log_dir = Path(run_dir) / "logs"
    _learner_log_dir.mkdir(parents=True, exist_ok=True)
    _configure_process_logging(_learner_log_dir / (f"learner{rank}.log" if num_learners > 1 else "learner.log"))
    try:
        apply_binding(f"learner[{rank}]" if num_learners > 1 else "learner", learner_numa_node, learner_cpus)
        set_seed(None if seed is None else seed + rank)
        opts = cfg.trainer.async_options
        _pin_worker_cpus(_resolve_cpu_set(opts.learner_cpu_cores, "learner_cpu_cores"))
        num_collectors = len(all_rings)
        device = (
            torch.device(learner_device)
            if learner_device is not None
            else torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        )
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
        last_metrics = None
        # First payload goes out immediately (step 0) so the parent's panel
        # quiescence gate opens as soon as collectors take their first steps,
        # instead of after a silent full log window; resume keeps window
        # alignment.
        if logging_interval > 0 and resume_step:
            next_log = ((resume_step // logging_interval) + 1) * logging_interval
        else:
            next_log = 0
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
        handshake.report_ready("learner", rank)
        handshake.wait_for_start(control)
        if control.stop:
            return

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
            metrics = learner.maybe_train(step)
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
                time.sleep(opts.idle_sleep_s)
                idle_ms = (time.perf_counter() - t_idle) * 1000.0
                if ingested == 0:
                    # No ring slot was available: learner is waiting for collector data.
                    learner_ring_wait_samples_ms.append(idle_ms)
                else:
                    # Data arrived, but replay/batch readiness still gated training.
                    learner_gate_wait_samples_ms.append(idle_ms)
            if step >= next_log:
                # One compact payload per log window to the parent (panel +
                # TensorBoard live there; see train.py). All ranks send.
                now = time.time()
                sps = (step - last_log_step) * num_envs / max(now - last_log_time, 1e-6)
                warming = step < agent.cfg.learning_starts
                metrics_log = {k: float(v) for k, v in last_metrics.items()} if (last_metrics and not warming) else None
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
                try:
                    panel_queue.put_nowait(payload)
                except Full:
                    pass
                last_log_time, last_log_step = now, step
                t_learn_win = 0.0
                learner_train_samples_ms = []
                learner_drain_samples_ms = []
                learner_breakdown_samples_ms = {}
                learner_ring_wait_samples_ms = []
                learner_gate_wait_samples_ms = []
                # Recompute the window edge (not +=) so the immediate first
                # payload cannot shift every later window off the interval.
                next_log = ((step // logging_interval) + 1) * logging_interval if logging_interval > 0 else 0

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
                next_save += save_interval

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
    except BaseException:
        if not isinstance(sys.exc_info()[1], (KeyboardInterrupt, SystemExit)):
            error_queue.put((f"learner[{rank}]" if num_learners > 1 else "learner", traceback.format_exc()))
        raise
    finally:
        if num_learners > 1:
            dist.destroy_process_group()
        control.set_stop()  # tell the collectors to exit
