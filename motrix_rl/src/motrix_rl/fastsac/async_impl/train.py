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

import os
import random
import time
from pathlib import Path
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp

from motrix_env_core import registry as env_registry
from motrix_env_core.renderer import RenderConfig
from motrix_rl.console import TrainingPanelStats, emit_training_panel, open_training_live
from motrix_rl.fastsac.agent import FastSacAgent
from motrix_rl.fastsac.async_impl.collector import resolve_collector_inference_device
from motrix_rl.fastsac.async_impl.numa import spawn_placement
from motrix_rl.fastsac.async_impl.panels import BootPanel
from motrix_rl.fastsac.async_impl.stats import aggregate_collector_stats, nest_timing_path, timing_mean
from motrix_rl.fastsac.async_impl.topology import resolve_learner_devices, resolve_trainer_topology
from motrix_rl.fastsac.async_impl.transport import Control, RingCursors, SharedTransitionRing
from motrix_rl.fastsac.async_impl.transport.handshake import StartupHandshake
from motrix_rl.fastsac.async_impl.transport.weight_channel import WeightChannelShared
from motrix_rl.fastsac.async_impl.worker import (
    actor_param_numel,
    build_env,
    inherit_log_stdio,
    run_collector_process,
    run_learner_process,
)
from motrix_rl.fastsac.config import FastSacCfg
from motrix_rl.fastsac.wrap import FastSacEnvWrap
from motrix_rl.frameworks import TrainerBase, TrainerContext
from motrix_rl.system_metrics import (
    CpuLoadSampler,
    GpuMemoryUsageSampler,
    GpuUtilizationSampler,
    MemoryUsageSampler,
    sample_gpu_devices,
)

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

        num_collectors = async_options.num_collectors
        num_learners = async_options.num_learners
        if num_learners < 1:
            raise ValueError(f"num_learners must be >= 1, got {num_learners}")
        if num_learners > 1 and num_collectors % num_learners != 0:
            raise ValueError(f"num_collectors={num_collectors} must divide evenly across num_learners={num_learners}")
        if cfg.agent.batch_size % num_learners != 0:
            raise ValueError(
                f"agent.batch_size={cfg.agent.batch_size} must divide evenly across num_learners={num_learners}"
            )
        # NUMA placement (see async_impl/topology.py): each learner binds to
        # its GPU's PCIe-local node and every collector follows its owning
        # learner, so a collector/learner pair never straddles a NUMA node.
        # Falls back to the OS default placement on single-node hosts.
        learner_devices = resolve_learner_devices(async_options.learner_devices, num_learners, self._device())
        cpus_per_collector = async_options.cpus_per_collector
        num_envs = self._context.num_envs
        # Generic "cuda" collector inference resolves to the owning learner's
        # GPU (num_collectors // num_learners collectors per learner); explicit
        # specs pass through. The single-learner path keeps the in-process
        # resolution (collector_device=None) byte-identical.
        collector_device_specs = None
        if num_learners > 1:
            per_learner = num_collectors // num_learners
            spec = async_options.collector_inference_device
            collector_device_specs = [
                spec
                if not (spec == "cuda" and learner_devices[i // per_learner].type == "cuda")
                else f"cuda:{learner_devices[i // per_learner].index}"
                for i in range(num_collectors)
            ]
        # Shared-memory primitives allocated in the parent, inherited by children.
        # One SPSC ring + one weight channel per collector: every shared quantity
        # keeps exactly one producer and one consumer, so the lock-free
        # single-writer invariants are unchanged by the collector count.
        #
        # The ring transport is decided per LEARNER RANK (all-or-nothing over
        # its collectors, so the rank's drain stays transport-homogeneous):
        # when every collector of the rank infers on the rank's GPU, its rings
        # are bare cursors and each learner allocates the CUDA-IPC device
        # slots, shipping them to its collectors through the one-shot
        # ring-slot queue; otherwise the rank's rings are host shared memory.
        # Collectors are co-located with their owning learner's GPU by
        # default (collector_inference_device="cuda" resolves per owner), so
        # the IPC path is the default whenever both sides share that GPU.
        if collector_device_specs is not None:
            collector_devices = [torch.device(spec) for spec in collector_device_specs]
        else:
            collector_devices = [collector_device] * num_collectors
        # One resolution pass derives the whole compute layout: env shards,
        # NUMA placement, and per-collector transports (rings + weights).
        param_numel = actor_param_numel(cfg, dims, action_scale, action_bias)
        topology = resolve_trainer_topology(
            num_envs,
            num_collectors,
            num_learners,
            learner_devices,
            collector_devices,
            self._device(),
            async_options,
            param_numel,
            cpus_per_collector=cpus_per_collector,
        )
        numa_nodes = [collector.numa_node for collector in topology.collectors]
        learner_numa_nodes = [learner.numa_node for learner in topology.learners]
        env_shards = topology.env_shards
        ring_ipc = [collector.ring_ipc for collector in topology.collectors]
        rings: list[SharedTransitionRing | RingCursors] = [
            RingCursors()
            if ipc
            else SharedTransitionRing(async_options.ring_capacity, shard, obs_dim, critic_obs_dim, act_dim)
            for ipc, shard in zip(ring_ipc, env_shards)
        ]
        weights = [WeightChannelShared(obs_dim=obs_dim) for _ in range(num_collectors)]
        control = Control(num_collectors)

        resume_step = 0
        is_resume = False
        if self._resume_from:
            # Peek global_step so the collector warms up with the loaded policy and
            # the shared async counters continue from the checkpoint iteration.
            ckpt = torch.load(self._resume_from, map_location="cpu", weights_only=False)
            resume_step = int(ckpt.get("global_step", 0))
            is_resume = resume_step > 0
            control.resume_collector_steps(resume_step)
            control.global_step = resume_step

        ctx = mp.get_context("spawn")
        stats_queues = [ctx.Queue(maxsize=8) for _ in range(num_collectors)]
        error_queue = ctx.Queue(maxsize=8)
        # Per-window worker payloads for the parent-rendered panel: rollout
        # snapshots from every collector (stats_queues below) plus one learner
        # payload per rank. All workers are peers; the parent is the recorder.
        panel_queue = ctx.Queue(maxsize=8 * max(num_collectors + num_learners, 1))
        # Startup handshake (parent-arbitrated, see transport/handshake.py):
        # learners ship slot tensors through the per-collector one-shot
        # queues; workers report readiness, the parent renders a boot panel
        # and releases everyone at once.
        handshake = StartupHandshake(ctx, num_collectors)
        reported_errors: set[tuple[str, str]] = set()
        seed = self._context.seed

        def _record_error_trace(process_name: str, traceback_text: str) -> Path:
            error_dir = self._context.run_dir / "async_errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            error_path = error_dir / f"{process_name}_error.log"
            error_path.write_text(traceback_text)
            return error_path

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

                error_path = _record_error_trace(process_name, traceback_text)
                print(f"[motrix.fastsac async] {process_name} traceback written to {error_path}")
                print(traceback_text.rstrip())
            return errors

        print(
            f"[motrix.fastsac async] collector/learner training '{self._env_name}' learner={learner_device} "
            f"learner_replicas={num_learners} "
            + (f"learner_devices={[str(d) for d in learner_devices]} " if num_learners > 1 else "")
            + f"collector_env=cpu collector_inference={collector_device} num_collectors={num_collectors} "
            f"numa_nodes={numa_nodes} learner_numa_nodes={learner_numa_nodes} "
            f"num_envs={num_envs} iters={num_iterations} "
            f"from={resume_step} utd_mode={async_options.utd_mode}"
        )

        rendezvous_file = None
        if num_learners > 1:
            # Resolve to an absolute path: init_method="file://<relative>" is
            # URL-parsed (host/path), which mangles a relative run_dir into a
            # bogus absolute path and blocks forever inside the file store.
            rendezvous_path = (self._context.run_dir / "ddp_rendezvous").resolve()
            rendezvous_path.parent.mkdir(parents=True, exist_ok=True)
            rendezvous_path.unlink(missing_ok=True)  # the file store requires a fresh file
            rendezvous_file = str(rendezvous_path)
        per_learner = num_collectors // num_learners
        p_learners = [
            ctx.Process(
                target=run_learner_process,
                args=(
                    cfg,
                    num_envs,
                    dims,
                    action_scale,
                    action_bias,
                    topology.ring_slice_for_rank(rings, rank),
                    # every rank publishes to its OWN collectors; rank 0 is
                    # additionally the logger / checkpointer.
                    weights[rank * per_learner : (rank + 1) * per_learner],
                    control,
                    error_queue,
                    num_iterations,
                    logging_interval,
                    save_interval,
                    str(self._context.run_dir),
                    str(self._context.checkpoint_dir),
                    self._context.checkpoint_format,
                    self._resume_from,
                    seed,
                    handshake,
                ),
                kwargs={
                    "rank": rank,
                    "panel_queue": panel_queue,
                    "num_learners": num_learners,
                    "rendezvous_file": rendezvous_file,
                    "learner_device": str(learner_devices[rank]) if learner_devices else None,
                    "all_rings": rings,
                    "weight_ipc": [
                        c.weight_ipc for c in topology.collectors[rank * per_learner : (rank + 1) * per_learner]
                    ],
                    "learner_cpus": topology.learners[rank].cpus if learner_numa_nodes else None,
                    "learner_numa_node": learner_numa_nodes[rank] if learner_numa_nodes else None,
                },
                name="fastsac-async-learner" if num_learners == 1 else f"fastsac-async-learner-{rank}",
            )
            for rank in range(num_learners)
        ]
        p_collectors = [
            ctx.Process(
                target=run_collector_process,
                kwargs={
                    "env_spec": self._env_spec,
                    "cfg": cfg,
                    "num_envs": env_shards[i],
                    "dims": dims,
                    "action_scale": action_scale,
                    "action_bias": action_bias,
                    "ring": rings[i],
                    "weights": weights[i],
                    "control": control,
                    "stats_queue": stats_queues[i],
                    "error_queue": error_queue,
                    "num_iterations": num_iterations,
                    "logging_interval": logging_interval,
                    "is_resume": is_resume,
                    "seed": None if seed is None else seed + i,
                    "collector_id": i,
                    "numa_node": numa_nodes[i],
                    "cpus": topology.collectors[i].cpus,
                    "collector_device": collector_device_specs[i] if collector_device_specs is not None else None,
                    "run_dir": str(self._context.run_dir),
                    "handshake": handshake,
                },
                name=f"fastsac-async-collector-{i}",
            )
            for i in range(num_collectors)
        ]

        # Start each child while the parent is pre-placed on the child's NUMA
        # node: spawn children re-import torch/simulator before their entry
        # function runs, and affinity + memory policy survive fork+exec, so
        # the child's import-time allocations are node-local from the start
        # (the worker re-binds itself afterwards as a no-op refinement).
        log_dir = self._context.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        learner_log_name = (lambda rank: f"learner{rank}.log") if num_learners > 1 else (lambda rank: "learner.log")
        for rank, p in enumerate(p_learners):
            with (
                spawn_placement(learner_numa_nodes[rank] if learner_numa_nodes else None, f"learner-spawn[{rank}]"),
                inherit_log_stdio(log_dir / learner_log_name(rank)),
            ):
                p.start()
        for i, p in enumerate(p_collectors):
            with (
                spawn_placement(numa_nodes[i], f"collector-spawn[{i}]"),
                inherit_log_stdio(log_dir / f"collector{i}.log"),
            ):
                p.start()

        # -------------------------------------------------- startup panel + barrier
        # Workers boot at different speeds (env/numba compile vs CUDA init and
        # the DDP rendezvous). The parent tracks readiness on a live table and
        # releases everyone at once, so stepping starts in lockstep. A worker
        # crash or the boot timeout aborts via the shared stop flag.
        expected = {("collector", i) for i in range(num_collectors)} | {("learner", r) for r in range(num_learners)}
        ready: set[tuple[str, int]] = set()
        boot_deadline = time.time() + max(float(os.environ.get("MOTRIX_BOOT_TIMEOUT_S", "600")), 30.0)
        from rich.live import Live

        log_names = [f"collector{i}.log" for i in range(num_collectors)]
        log_names += [learner_log_name(rank) for rank in range(num_learners)]
        boot_panel = BootPanel(
            title=f"{self._env_name}/motrix.fastsac",
            log_dir=log_dir,
            log_names=log_names,
            workers=sorted(expected),
        )
        interactive = boot_panel.interactive

        # Panel-data state must exist BEFORE the boot barrier: the handoff
        # loop below drains the same queues the training panel will use, so
        # the transition between panels is a data-source switch, not a wait.
        per_collector_stats: dict[int, dict | None] = {i: None for i in range(num_collectors)}
        learner_payloads: dict[int, dict | None] = {r: None for r in range(num_learners)}

        def _drain_worker_stats() -> None:
            for queue in stats_queues:
                try:
                    while True:
                        snapshot = queue.get_nowait()
                        per_collector_stats[snapshot["collector_id"]] = snapshot
                except Empty:
                    pass
            try:
                while True:
                    payload = panel_queue.get_nowait()
                    learner_payloads[payload["rank"]] = payload
            except Empty:
                pass

        def _panel_data_ready() -> bool:
            """The training panel's quiescence gate (see _render_panel)."""
            return (
                all(v is not None for v in per_collector_stats.values())
                and any(v is not None for v in learner_payloads.values())
                and control.collector_steps > 0
            )

        def _handoff_gate_line() -> str:
            """One-line panel-data readiness summary for the handoff view."""
            stats_marks = " ".join(
                f"c{i} {'✓' if per_collector_stats[i] is not None else '…'}" for i in range(num_collectors)
            )
            payload_marks = " ".join(
                f"l{r} {'✓' if learner_payloads[r] is not None else '…'}" for r in range(num_learners)
            )
            return f"panel data — stats: {stats_marks} | payloads: {payload_marks} | steps: {control.collector_steps}"

        try:
            # Bare prints inside Live would tear its frame; collect abort
            # reasons here and print them after Live has exited.
            abort_reason: str | None = None
            # On a real terminal the panel runs on the alternate screen
            # (htop-style): every refresh redraws the whole screen, so no
            # frame can ever survive to tear — even when the frame fills the
            # full terminal height, where cursor-up erasure is unreliable.
            # On pipes/files Live cannot redraw in place at all; the
            # per-refresh prints are skipped and stop() emits one final frame.
            with Live(
                boot_panel.render(ready),
                refresh_per_second=4,
                auto_refresh=interactive,
                screen=interactive,
            ) as boot_live:
                while len(ready) < len(expected):
                    fresh = handshake.drain_ready()
                    if fresh:
                        ready |= fresh
                    if interactive:
                        boot_live.update(boot_panel.render(ready))  # log tail advances even without new readiness
                    try:
                        process_name, traceback_text = error_queue.get_nowait()
                    except Empty:
                        process_name = None
                    if process_name is not None:
                        control.set_stop()
                        _record_error_trace(process_name, traceback_text)
                        abort_reason = (
                            f"worker {process_name} failed during startup "
                            f"(traceback: async_errors/{process_name}_error.log)"
                        )
                        break
                    if time.time() > boot_deadline:
                        control.set_stop()
                        missing = sorted(expected - ready)
                        abort_reason = f"worker startup timed out after 600s; missing: {missing}"
                        break
                    time.sleep(0.25)
                # ------------------------------------------------ handoff phase
                # Release the barrier NOW: the handoff gate below waits for
                # collectors to step (shared counters), and workers held at
                # wait_for_start cannot step — waiting for the gate before
                # releasing is a parent/child deadlock.
                if abort_reason is None:
                    handshake.release()
                # The barrier released every worker, but the training panel's
                # queue-fed stats have not converged yet. Keep the SAME Live
                # (and alt screen) running with a shared-counter progress
                # view — every field is well-defined immediately — until the
                # quiescence gate opens, then fall through to the training
                # panel whose first frame is therefore fully aggregated.
                if abort_reason is None:
                    while not _panel_data_ready():
                        _drain_worker_stats()
                        try:
                            process_name, traceback_text = error_queue.get_nowait()
                        except Empty:
                            process_name = None
                        if process_name is not None:
                            control.set_stop()
                            abort_reason = f"worker {process_name} failed after startup"
                            break
                        if interactive:
                            boot_live.update(boot_panel.render(ready, gate=_handoff_gate_line(), starting=True))
                        time.sleep(0.25)
            # Live has exited — plain prints are safe on the terminal again.
            if abort_reason is not None:
                print(f"[motrix.fastsac async] {abort_reason}; aborting.")
                raise RuntimeError(abort_reason)
            print(f"[motrix.fastsac async] all {len(expected)} workers ready — training starts")
        finally:
            handshake.release()  # idempotent; frees workers on an aborted boot
        # -------------------------------------------------- parent-owned panel
        # The parent is the only process that sees every worker (collectors'
        # rollout snapshots + every learner rank's payload) AND the whole
        # machine: it is unbound, so the system panel shows all cores, and
        # workers stay peers with no "primary renderer" role.
        console, live = None, None  # opened lazily once real data flows (see below)
        panel_open_attempted = False
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=str(self._context.run_dir))
        except Exception:
            tb_writer = None
        panel_cpu_sampler = CpuLoadSampler()
        panel_gpu_sampler = GpuUtilizationSampler()
        panel_memory_sampler = MemoryUsageSampler()
        panel_gpu_memory_sampler = GpuMemoryUsageSampler()
        panel_start_time = time.time()
        panel_anchored = False
        # Rolling (time, step) samples for the DISPLAYED env-steps/s — a
        # time-boxed rate decoupled from the TensorBoard log-window
        # bookkeeping below, which never resets the panel's number.
        panel_rate_samples: list[tuple[float, int]] = []
        panel_last_render = 0.0
        panel_last_log_time = panel_start_time
        panel_last_log_step = resume_step
        panel_last_updates = 0
        panel_last_weight_version = 0
        panel_next_log = (
            ((resume_step // max(logging_interval, 1)) + 1) * logging_interval if logging_interval > 0 else 0
        )

        def _render_panel(force: bool = False) -> None:
            nonlocal panel_anchored, panel_start_time, panel_last_render, console, live, panel_open_attempted
            nonlocal panel_rate_samples
            nonlocal panel_last_log_time, panel_last_log_step, panel_last_updates, panel_next_log
            nonlocal panel_last_weight_version
            # ALWAYS drain first: the quiescence gate below waits for learner
            # payloads that can only arrive through this drain — draining
            # after the gate would deadlock the workers on a full queue.
            _drain_worker_stats()
            if console is None and not panel_open_attempted:
                # Quiescence gate (see _panel_data_ready): stay off the
                # terminal until every collector has reported at least one
                # snapshot, a learner payload exists and collection started.
                if not _panel_data_ready():
                    return
                panel_open_attempted = True
                console, live = open_training_live()
                if console is not None:
                    console.clear()
                    panel_last_render = 0.0  # render immediately after the gate opens
                # non-TTY (nohup/redirect): open_training_live returns Nones —
                # fall through to emit_training_panel's plain-frame path and
                # never retry the open on every tick.
            if not force and time.time() - panel_last_render < 1.0:
                return
            panel_last_render = time.time()
            step = control.collector_steps // num_collectors
            if step > resume_step and not panel_anchored:
                panel_start_time = time.time()
                panel_last_log_time = panel_start_time
                panel_anchored = True
            last_stats = aggregate_collector_stats(per_collector_stats)
            payloads = [p for p in learner_payloads.values() if p]
            primary = payloads[0] if payloads else None
            now = time.time()
            # Windowed rate. The window is meaningless right after the anchor
            # (Δt ≈ 0 → absurd rate) or before any step landed (Δstep = 0);
            # once it has accumulated real elapsed time a PARTIAL first
            # window is already a honest rate, so report it instead of
            # waiting a full logging interval on "warming up".
            # Rolling rate over the last ~10s of render samples: monotone
            # updates, no reset when a TB log point fires, honest from the
            # first ~5s of stepping (below that there is not enough span).
            panel_rate_samples.append((now, step))
            while len(panel_rate_samples) >= 2 and panel_rate_samples[1][0] <= now - 10.0:
                panel_rate_samples.pop(0)
            t0, s0 = panel_rate_samples[0]
            t1, s1 = panel_rate_samples[-1]
            # Short-window init: as soon as steps flow, compute over the
            # available span (floor 1s against div-by-tiny) so the panel
            # shows a real — if rough — rate immediately; None (warming up)
            # only means "no steps flowed in the window" (pre-start or a
            # genuine throughput stall).
            if s1 > s0:
                span = max(t1 - t0, 1.0)
                sps = (s1 - s0) * num_envs / span
                iters = (s1 - s0) / span
            else:
                sps = None
                iters = None
            warming = step < cfg.agent.learning_starts
            # learn_ms is 0.0 (not None) in the immediate step-0 warmup
            # payload — filter on None, not truthiness, or the list empties
            # and timing_mean divides by zero.
            learn_ms = timing_mean([p["learn_ms"] for p in payloads if p["learn_ms"] is not None]) if payloads else 0.0
            learn_pct = timing_mean([p["learn_pct"] for p in payloads]) if payloads else 0.0
            updates = primary["updates"] if primary else panel_last_updates
            utd = updates / max(step, 1)
            collector_timing_ms = last_stats.get("timing_ms", {})
            collector_timing_detail_ms = {k: v for k, v in collector_timing_ms.items() if k != "collect"}
            collector_items: dict = {}
            for key, value in collector_timing_detail_ms.items():
                nest_timing_path(collector_items, tuple(key.split(".")), value)
            timing_groups: dict = {"collector": collector_items}
            for payload in payloads:
                rank_key = "learner" if num_learners == 1 else f"learner[{payload['rank']}]"
                if payload["learner_timing"]:
                    timing_groups[rank_key] = payload["learner_timing"]
            stats = TrainingPanelStats(
                iteration=step,
                total_iterations=num_iterations,
                steps_per_second=sps,
                iterations_per_second=iters,
                elapsed_seconds=now - panel_start_time,
                mean_return=last_stats["return"],
                mean_episode_length=last_stats["ep_len"],
                episodes=last_stats["episodes"],
                buffer_size=primary["buffer_size"] if primary else 0,
                buffer_capacity=primary["buffer_capacity"] if primary else 0,
                collect_ms=collector_timing_ms.get("collect", 0.0),
                learn_ms=learn_ms,
                learn_percent=learn_pct,
                warming=warming,
                training_metrics=primary["metrics"] if primary else None,
                reward_terms=last_stats["reward_terms"],
                env_metrics=last_stats["env_metrics"],
                timing_groups=timing_groups,
                diagnostics={"UTD": utd},
                cpu_load=panel_cpu_sampler.sample(),
                gpu_utilization_percent=panel_gpu_sampler.sample(),
                memory_usage=panel_memory_sampler.sample(),
                gpu_memory_usage=panel_gpu_memory_sampler.sample(),
                gpu_devices=sample_gpu_devices(panel_gpu_sampler, panel_gpu_memory_sampler),
                checkpoint_path=primary["checkpoint_path"] if primary else None,
            )
            emit_training_panel(live, stats, title=f"{self._env_name}/motrix.fastsac")
            if tb_writer is not None and step >= panel_next_log:
                tb_writer.add_scalar("rollout/mean_return", last_stats["return"], step)
                tb_writer.add_scalar("rollout/mean_ep_len", last_stats["ep_len"], step)
                if sps is not None:
                    tb_writer.add_scalar("perf/env_steps_per_s", sps, step)
                tb_writer.add_scalar(
                    "perf/updates_per_s", (updates - panel_last_updates) / max(now - panel_last_log_time, 1e-6), step
                )
                tb_writer.add_scalar("async/policy_lag", last_stats["policy_lag"], step)
                if primary and primary.get("weight_version") is not None:
                    tb_writer.add_scalar(
                        "async/weight_publishes_per_iter",
                        (primary["weight_version"] - panel_last_weight_version) / max(step - panel_last_log_step, 1),
                        step,
                    )
                    panel_last_weight_version = primary["weight_version"]
                if primary:
                    fills = primary.get("ring_fill") or []
                    tb_writer.add_scalar("async/ring_fill", sum(f for f in fills if f is not None), step)
                    if primary.get("weight_version") is not None:
                        tb_writer.add_scalar("async/weight_version", primary["weight_version"], step)
                tb_writer.add_scalar("async/utd", utd, step)
                if num_collectors > 1:
                    for i in range(num_collectors):
                        stats_i = per_collector_stats.get(i)
                        if stats_i:
                            tb_writer.add_scalar(f"async/policy_lag_collector{i}", stats_i["policy_lag"], step)
                        if primary:
                            fill = (primary.get("ring_fill") or [None] * num_collectors)[i]
                            if fill is not None:
                                tb_writer.add_scalar(f"async/ring_fill_collector{i}", fill, step)
                tb_writer.add_scalar("perf/collect_ms_per_batch", collector_timing_ms.get("collect", 0.0), step)
                for k, v in collector_timing_detail_ms.items():
                    tb_writer.add_scalar(f"perf/collector_{k}_ms", v, step)
                tb_writer.add_scalar("perf/learn_ms_total", learn_ms, step)
                tb_writer.add_scalar("perf/learn_pct", learn_pct, step)
                for k, v in last_stats["env_metrics"].items():
                    tb_writer.add_scalar(f"metrics/{k}", v, step)
                for k, v in last_stats["reward_terms"].items():
                    tb_writer.add_scalar(f"reward/{k}", v, step)
                if primary and primary["metrics"] is not None:
                    for k, v in primary["metrics"].items():
                        tb_writer.add_scalar(f"train/{k}", v, step)
                panel_last_log_time, panel_last_log_step, panel_last_updates = now, step, updates
                panel_next_log += logging_interval

        try:
            # monitor: exit when every learner has finished; abort all if any worker crashes.
            while True:
                crashed = [p for p in (*p_learners, *p_collectors) if not p.is_alive() and p.exitcode not in (0, None)]
                learners_done = [p for p in p_learners if not p.is_alive()]
                if crashed:
                    for p in crashed:
                        print(f"[motrix.fastsac async] {p.name} crashed (exit {p.exitcode}); stopping.")
                    _drain_child_errors()
                    break
                if learners_done:
                    _drain_child_errors()
                    if len(learners_done) == len(p_learners):
                        break
                _render_panel()
                time.sleep(0.5)
        finally:
            # Ctrl+C reaches the parent as KeyboardInterrupt and every worker via
            # its SIGINT handler (stop-flag unwind). Shutdown is an escalation
            # ladder with per-step exception isolation: one stuck child must
            # never skip the cleanup of the others (that is how orphans holding
            # GPU memory were produced).
            control.set_stop()

            def _shutdown(processes, grace_s):
                for p in processes:
                    try:
                        p.join(timeout=grace_s)
                        if not p.is_alive():
                            continue
                        print(f"[motrix.fastsac async] terminating {p.name}")
                        p.terminate()  # SIGTERM: unwinds queue feeder threads
                        p.join(timeout=3.0)
                        if p.is_alive():
                            print(f"[motrix.fastsac async] killing {p.name}")
                            p.kill()  # SIGKILL: e.g. blocked inside a NCCL collective
                            p.join(timeout=3.0)
                    except Exception as exc:  # noqa: BLE001 - isolation by design
                        print(f"[motrix.fastsac async] shutdown of {p.name} failed: {exc}")

            _shutdown(p_collectors, grace_s=8.0)
            _shutdown(p_learners, grace_s=8.0)
            try:
                _drain_child_errors()
            except Exception:
                pass
            try:
                _render_panel(force=True)
            except Exception:
                pass
            if live is not None:
                try:
                    live.stop()
                except Exception:
                    pass
            if tb_writer is not None:
                try:
                    tb_writer.close()
                except Exception:
                    pass
            try:
                panel_queue.close()
            except Exception:
                pass
            handshake.close()
            for stats_queue in stats_queues:
                try:
                    while True:
                        stats_queue.get_nowait()
                except Exception:
                    pass
                stats_queue.close()
            try:
                error_queue.close()
            except Exception:
                pass

        for p in p_learners:
            if p.exitcode not in (0, None):
                raise RuntimeError(
                    f"motrix.fastsac async learner process '{p.name}' failed with exit code {p.exitcode}"
                )
        for i, p in enumerate(p_collectors):
            if p.exitcode not in (0, None):
                raise RuntimeError(f"motrix.fastsac async collector {i} process failed with exit code {p.exitcode}")

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
