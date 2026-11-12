# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Async trainer: spawn the CPU collector, run the learner on this process.

Topology mirrors the validated reference runs: a CPU physics collector with
optional GPU policy inference, and a learner that ingests exactly one
collector block per cycle (fixed update timing), fills the replay and
normalizers, then spends update credit on world-model and imagination
updates. Policy snapshots travel back through the inference-bundle weight
channel (live weights plus normalizers). The learner-side task is built
without a second physics instance via ``build_task`` on the learner device.
"""

from __future__ import annotations

import multiprocessing as mp
import time

import torch

from motrix_env_core import registry as env_registry
from motrix_rl.contracts import TrainerBase, TrainerContext
from motrix_rl.transport.inference_bundle import InferenceBundleSpec, ModuleSection, NormalizerSection
from motrix_rl.transport.ring import HostTransitionRing
from motrix_rl.transport.gpu_ipc_ring import GpuIpcTransitionRing
from motrix_rl.transport.weight_channel import HostWeightSender, WeightChannelShared
from motrix_rl_supermimic.async_impl.collector import collector_process
from motrix_rl_supermimic.async_impl.credit import UpdateCredit
from motrix_rl_supermimic.async_impl.schema import transition_schema
from motrix_rl_supermimic.config import SuperMimicCfg
from motrix_rl_supermimic.learner import Learner
from motrix_rl_supermimic.mbrl_env import MbrlEnvBase
from motrix_rl_supermimic.task_mdp import optional_task_hook


def collector_dev_available() -> bool:
    return torch.cuda.is_available()


class _TaskView:
    """Learner-side task view over a physics-free TaskMDP."""

    def __init__(self, task, num_envs: int, action_dt: float):
        self.task_mdp = task
        self.spec = task.spec
        self.num_envs = num_envs
        self.action_dt = action_dt

    def init_state(self) -> None:  # the async learner never runs the lifecycle
        pass


class AsyncTrainer(TrainerBase):
    def __init__(self, *, context: TrainerContext[SuperMimicCfg]):
        self._context = context
        self._rlcfg = context.rl_cfg
        self._env_spec = env_registry.resolve(context.env_name, sim=context.sim)
        self._stop_requested = False

    def _device(self) -> torch.device:
        if self._rlcfg.device:
            return torch.device(self._rlcfg.device)
        from motrix_rl.device import resolve_default_device

        return resolve_default_device()

    def request_stop(self) -> None:
        """Ask the training loop to finish at the next iteration boundary."""
        self._stop_requested = True

    def train(self) -> None:
        context = self._context
        cfg = self._rlcfg
        t, col = cfg.trainer, cfg.collector
        if context.seed is not None:
            torch.manual_seed(context.seed)

        env_cfg = self._env_spec.env_cfg
        device = str(self._device())
        task = self._env_spec.env_cls.build_task(env_cfg, device)
        env = _TaskView(task, context.num_envs, float(getattr(env_cfg, "ctrl_dt", 0.1)))
        learner = Learner(cfg, env, device, context.checkpoint_dir, metrics=self._metrics_sink())
        if context.resume_from:
            learner.load(context.resume_from)

        schema = transition_schema(env.spec, t.goal_frames, task)
        # Ring transport: CUDA-IPC device slots when the learner and the
        # collector's policy share one GPU (collector env physics stays on
        # CPU; only the slot memory moves to the device), else host memory.
        device_t = torch.device(device)
        collector_dev = torch.device(col.policy_device) if collector_dev_available() else torch.device("cpu")
        if collector_dev.type == "cuda" and collector_dev.index is None:
            collector_dev = torch.device("cuda", 0)
        same_gpu = device_t.type == "cuda" and collector_dev.type == "cuda" and device_t == collector_dev
        cursors = None
        slots = None
        if same_gpu:
            from motrix_rl.transport.ring import RingCursors

            cursors = RingCursors()
            slots = torch.zeros((col.queue_blocks, context.num_envs, schema.feature_width),
                                dtype=torch.float32, device=device_t)
            ring = GpuIpcTransitionRing(cursors, slots, schema, col.queue_blocks, context.num_envs)
        else:
            ring = HostTransitionRing(schema, col.queue_blocks, context.num_envs)
        policy_numel = sum(p.numel() for p in learner.pi.parameters())
        bundle_spec = InferenceBundleSpec(
            modules=(ModuleSection("policy", policy_numel),),
            normalizers=(
                NormalizerSection("obs_norm", env.spec.observation_dim),
                NormalizerSection("goal_norm", env.spec.command_dim * t.goal_frames),
            ),
        )
        shared = WeightChannelShared(bundle_spec)
        sender = HostWeightSender(shared)
        from motrix_rl_supermimic.networks import NormStatsView

        sender.publish(
            {"policy": learner.pi},
            {"obs_norm": NormStatsView(learner.pi.obs_norm), "goal_norm": NormStatsView(learner.pi.goal_norm)},
        )

        ctx = mp.get_context("spawn")
        stop = ctx.Event()
        error_rx, error_tx = ctx.Pipe(duplex=False)
        worker = ctx.Process(
            target=collector_process,
            name="supermimic-collector",
            args=(
                context.env_name,
                context.sim,
                context.num_envs,
                None if context.seed is None else context.seed + 1,
                t.goal_frames,
                t.action_noise,
                20.0,
                col.block_steps,
                cfg.agent.hidden,
                cfg.agent.hist,
                cfg.agent.pi_ref_skip,
                col.policy_device,
                cursors,
                slots,
                ring,
                shared,
                sender.params,
                stop,
                error_tx,
                col.preregister,
            ),
        )
        credit = UpdateCredit(t.utd, max(t.warmup_steps, t.wm_window))
        started = time.monotonic()
        last_received = started
        try:
            worker.start()
            error_tx.close()
            while learner.iteration < t.iterations and not self._stop_requested:
                self._check_worker(worker, error_rx)
                if not ring.has_next():
                    if time.monotonic() - last_received > col.worker_timeout:
                        raise TimeoutError(f"No collector block for {col.worker_timeout}s")
                    time.sleep(0.001)
                    continue
                last_received = time.monotonic()
                count, views = ring.read_span()
                take = min(count, col.block_steps)
                self._ingest(learner, task, take, views)
                ring.commit_reads(take)
                credit.receive(take)
                while credit.available and learner.iteration < t.iterations and not self._stop_requested:
                    self._check_worker(worker, error_rx)
                    learner._apply_lr()
                    if t.overlap_wm and learner.wm_imag is not learner.wm:
                        # WM updates on the side stream while the actor BPTT
                        # runs on the main stream with the iteration-start WM.
                        learner.sync_imag_wm()
                        with torch.cuda.stream(learner.wm_side):
                            for _ in range(t.wm_batches):
                                wm_loss = learner.wm_update()
                    else:
                        wm_loss = 0.0
                        for _ in range(t.wm_batches):
                            wm_loss = learner.wm_update()
                    policy_row = (0.0, 0.0, 0.0, 0.0)
                    if learner.iteration >= t.wm_only_iters:
                        policy_row = learner.policy_update()
                    learner.iteration += 1
                    credit.consume()
                    if learner.iteration % col.snapshot_every == 0:
                        sender.publish(
                            {"policy": learner.pi},
                            {"obs_norm": NormStatsView(learner.pi.obs_norm),
                             "goal_norm": NormStatsView(learner.pi.goal_norm)},
                        )
                    if learner.iteration % t.log_every == 0:
                        # One-shot policy-lr decay trigger at log cadence on
                        # the mean normalized training return of the interval.
                        aggregated = (
                            sum(learner._return_blocks) / len(learner._return_blocks) if learner._return_blocks else 0.0
                        )
                        learner._return_blocks = []
                        from motrix_rl_supermimic.optim import trigger_decay

                        learner.decay_at = trigger_decay(t, learner.iteration, aggregated, learner.decay_at)
                    if learner.metrics is not None and learner.iteration % t.log_every == 0:
                        learner.metrics(
                            learner.iteration,
                            {
                                "wm_loss": wm_loss,
                                "sim_steps": float(learner.sim_steps),
                                "sim_reward": getattr(learner, "_span_reward", 0.0),
                                "policy_loss": policy_row[0],
                                "value_loss": policy_row[1],
                                "imag_reward": policy_row[2],
                                "imag_survival": learner._last_imag_survival,
                                "policy_lr": learner.opt_pi.param_groups[0]["lr"],
                                "snapshots_published": float(sender.version),
                                "ring_backlog": float(ring.size()),
                                "wall_s": time.monotonic() - started,
                            },
                        )
                    if learner.iteration % t.ckpt_every == 0 or learner.iteration == t.iterations:
                        learner.save(time.monotonic() - started)
        finally:
            stop.set()
            worker.join(timeout=10.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5.0)
            error_rx.close()

    def play(self, policy: str) -> None:
        from motrix_rl_supermimic.sync import Trainer

        Trainer(context=self._context).play(policy)

    @staticmethod
    def _ingest(learner: Learner, task, count: int, views: dict) -> None:
        replay_fields = set(learner.buffer.data)
        device = learner.dev
        from motrix_rl_supermimic import state as st

        spec = learner.env.spec
        transport = optional_task_hook(task, "transport_fields")
        buffers = transport() if transport is not None else {}
        reward_sum = torch.zeros((), device=device)
        action_mag = torch.zeros((), device=device)
        for step in range(count):
            row = {k: views[k][step].to(device) for k in replay_fields}
            reward_sum += row["r"].mean()
            action_mag += row["a"].abs().mean()
            learner.buffer.add(row)
            learner.pi.obs_norm.update(views["obs"][step].to(device))
            learner.pi.goal_norm.update(views["goal"][step].to(device))
            # World-model normalizer statistics (the async learner never runs
            # collect; reference implementations update these during ingest).
            s = st.unflatten_state(spec, row["s"])
            s1 = st.unflatten_state(spec, row["s1"])
            learner.wm.in_norm.update(learner.wm.inputs(s, row["a"], learner.mdp.normalize_context(row["xi"])))
            learner.wm.out_norm.update(learner.wm.targets(s, s1))
            learner.wm.inc_norm.update(learner.wm.increments(s, s1))
            learner.sim_steps += row["s"].shape[0]
            with torch.no_grad():
                for name, buffer in buffers.items():
                    if name in views:
                        # Transport fields are shared task state; the collector
                        # clones one value into every env row of the step.
                        buffer.copy_(views[name][step][0].to(buffer.device))
        learner._span_reward = reward_sum / max(count, 1) / learner.return_normalizer
        learner._return_blocks.append(learner._span_reward)
        learner._span_action_mag = action_mag / max(count, 1)

    def _metrics_sink(self):
        metrics = self._context.metrics

        def sink(iteration: int, values: dict[str, float]) -> None:
            if metrics.enabled:
                metrics.scalars({k: float(v) for k, v in values.items()}, step=iteration)

        return sink

    @staticmethod
    def _check_worker(worker, error_rx) -> None:
        if not worker.is_alive() and worker.exitcode not in (0, None):
            raise RuntimeError(f"Collector process died (exitcode {worker.exitcode})")
        if error_rx.poll():
            raise RuntimeError(f"Collector failed:\n{error_rx.recv()}")
