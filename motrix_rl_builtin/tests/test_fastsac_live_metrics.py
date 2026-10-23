# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Live metrics emitted by the FastSAC sync and async trainers.

The TensorBoard tag sets below are a compatibility contract: live scalars reuse
the tags as canonical keys (see wiki/design/live-metrics.md), so renaming a tag
breaks dashboards as well as existing TensorBoard comparisons.
"""

from __future__ import annotations

import contextlib
import math
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest
import torch
import torch.utils.tensorboard
from fastsac_async_mocks import ThreadSpawnContext
from motrix_rl_builtin.fastsac.wrap import FastSacStepResult

from motrix_rl import checkpoints
from motrix_rl.config import CheckpointConfig, LoggingConfig
from motrix_rl.contracts import TrainerContext
from motrix_rl.metrics import JsonlSink, MetricsLogger
from motrix_rl.system_metrics import CpuLoad, GpuDeviceUsage, MemoryUsage

SYNC_TB_TAGS = {
    "progress/collection_step",
    "progress/learning_iteration",
    "progress/update_idx",
    "rollout/mean_return",
    "rollout/mean_ep_len",
    "rollout/env_reward",
    "perf/env_steps_per_s",
    "perf/collect_ms_per_iter",
    "perf/learn_ms_per_iter",
    "perf/learn_pct",
    "metrics/height",
    "reward/alive",
    "train/qf_loss",
    "train/actor_loss",
}

ASYNC_TB_TAGS = {
    "progress/collection_step",
    "progress/learning_iteration",
    "progress/update_idx",
    "rollout/mean_return",
    "rollout/mean_ep_len",
    "perf/env_steps_per_s",
    "perf/updates_per_s",
    "async/policy_lag",
    "async/weight_publishes_per_collection_step",
    "async/ring_fill",
    "async/weight_version",
    "async/utd",
    "async/policy_lag_collector0",
    "async/policy_lag_collector1",
    "async/ring_fill_collector0",
    "async/ring_fill_collector1",
    "perf/collect_ms_per_batch",
    "perf/collector_env_step_ms",
    "perf/collector_sync_ms",
    "perf/collector_sync.wait_writer_ms",
    "perf/learn_ms_total",
    "perf/learn_pct",
    "metrics/height",
    "reward/alive",
    "train/qf_loss",
    "train/actor_loss",
}

SYS_KEY = re.compile(r"^sys/(cpu_percent|mem_used_gb|gpu\d+/(util_percent|mem_used_gb))$")


# ---------------------------------------------------------------- shared helpers
class _RecordingSink(JsonlSink):
    """The real jsonl sink, also keeping each record so a test can wait on them."""

    def __init__(self, path) -> None:
        super().__init__(path)
        self.records: list[dict] = []
        self._lock = threading.Lock()

    def write(self, record) -> None:
        super().write(record)
        with self._lock:
            self.records.append(record)

    def count(self, kind: str) -> int:
        with self._lock:
            return sum(1 for record in self.records if record["kind"] == kind)


class _RecordingWriter:
    """SummaryWriter stand-in recording ``(tag, value, step)``."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag, value, step) -> None:
        self.scalars.append((tag, float(value), int(step)))

    def close(self) -> None:
        pass


def _context(tmp_path, rl_cfg, metrics, *, num_envs, logging_interval, save_interval, resume_from=None):
    (tmp_path / "checkpoints").mkdir(exist_ok=True)
    return TrainerContext(
        run=None,
        env_name="cartpole",
        run_dir=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        sim=None,
        checkpoint_format="pt",
        num_envs=num_envs,
        play_num_envs=1,
        seed=None,
        rl_cfg=rl_cfg,
        logging=LoggingConfig(backend="tensorboard", interval=logging_interval),
        checkpoint=CheckpointConfig(interval=save_interval),
        resume_from=resume_from,
        metrics=metrics,
    )


def _logger(tmp_path) -> tuple[MetricsLogger, _RecordingSink]:
    sink = _RecordingSink(tmp_path / "metrics.jsonl")
    return MetricsLogger("unit-run", jsonl=sink), sink


def _events(sink: _RecordingSink) -> list[tuple[str, int]]:
    return [(record["data"]["name"], record["step"]) for record in sink.records if record["kind"] == "event"]


def _paths(sink: _RecordingSink) -> list[tuple[str, int]]:
    return [
        (record["data"]["path"], record["step"])
        for record in sink.records
        if record["kind"] == "event" and record["data"]["name"] == "checkpoint_saved"
    ]


ROLLOUT_MEANS = {"rollout/mean_return", "rollout/mean_ep_len"}


def _event(sink: _RecordingSink, name: str) -> dict:
    (record,) = [r for r in sink.records if r["kind"] == "event" and r["data"]["name"] == name]
    return record["data"]


def _assert_live_matches_tensorboard(
    sink: _RecordingSink, writer: _RecordingWriter, *, no_episode_steps: tuple[int, ...] = ()
) -> None:
    """Each TB log point has one live point: the same tags and values plus fixed ``sys/*`` keys.

    The one difference: at ``no_episode_steps`` (no episode has ended yet) the
    rollout means are a NaN placeholder that TensorBoard keeps and live leaves out.
    """
    log_points: list[tuple[int, dict[str, float]]] = []
    for tag, value, step in writer.scalars:
        if not log_points or log_points[-1][0] != step or tag in log_points[-1][1]:
            log_points.append((step, {}))
        log_points[-1][1][tag] = value
    records = [r for r in sink.records if r["kind"] == "scalars"]
    assert [r["step"] for r in records] == [step for step, _ in log_points]
    for record, (step, tensorboard) in zip(records, log_points):
        live = record["data"]
        placeholders = ROLLOUT_MEANS if step in no_episode_steps else set()
        assert {key for key in live if not key.startswith("sys/")} == set(tensorboard) - placeholders
        assert all(SYS_KEY.match(key) for key in live if key.startswith("sys/"))
        for tag, value in tensorboard.items():
            if tag in placeholders:
                continue
            assert live[tag] == (None if math.isnan(value) else pytest.approx(value)), tag


def test_system_scalars_use_the_fixed_key_list():
    from motrix_rl_builtin.fastsac.live_metrics import system_scalars

    gib = 1024**3
    stats = SimpleNamespace(
        cpu_load=CpuLoad(
            utilization_percent=37.5,
            used_logical_cpus=3.0,
            logical_cpu_count=8,
            physical_core_count=4,
            iowait_percent=0.0,
            steal_percent=0.0,
            per_core_percent=(37.5,) * 8,
        ),
        memory_usage=MemoryUsage(used_bytes=3 * gib, total_bytes=16 * gib),
        gpu_devices=[
            GpuDeviceUsage(index=0, utilization_percent=55.0, memory=MemoryUsage(used_bytes=2 * gib, total_bytes=gib)),
            GpuDeviceUsage(index=1, utilization_percent=None, memory=None),
        ],
    )

    assert system_scalars(stats) == {
        "sys/cpu_percent": 37.5,
        "sys/mem_used_gb": 3.0,
        "sys/gpu0/util_percent": 55.0,
        "sys/gpu0/mem_used_gb": 2.0,
    }
    assert system_scalars(SimpleNamespace(cpu_load=None, memory_usage=None, gpu_devices=None)) == {}


# ---------------------------------------------------------------- sync trainer
class _SyncEnv:
    """Two-step episodes, one reward term and one env metric; rewards turn NaN from step ``diverge_at``."""

    def __init__(self, num_envs: int, diverge_at: int | None = None) -> None:
        self.num_envs = num_envs
        self._t = 0
        self._diverge_at = diverge_at

    def close(self):
        pass

    def reset(self):
        return torch.zeros(self.num_envs, 2), torch.zeros(self.num_envs, 3)

    def step(self, actions):
        self._t += 1
        n = self.num_envs
        diverged = self._diverge_at is not None and self._t >= self._diverge_at
        return FastSacStepResult(
            obs=torch.zeros(n, 2),
            critic_obs=torch.zeros(n, 3),
            reward=torch.full((n,), math.nan if diverged else 1.0),
            terminated=torch.full((n,), self._t % 2 == 0),
            truncated=torch.zeros(n, dtype=torch.bool),
            final_obs=None,
            info={"Reward": {"alive": np.full(n, 0.5)}, "metrics": {"height": np.full(n, 0.9)}},
        )


class _SyncAgent:
    """The agent primitives the sync loop orchestrates."""

    def __init__(self, num_envs: int, learning_starts: int) -> None:
        self.cfg = SimpleNamespace(learning_starts=learning_starts, num_updates=1)
        self.device = torch.device("cpu")
        self.num_envs = num_envs
        self.act_dim = 1
        self.actor = SimpleNamespace(action_scale=torch.ones(1), action_bias=torch.zeros(1))
        self.rb = SimpleNamespace(extend=lambda *columns: None, num_stored=0, buffer_size=4)
        self.global_step = 0
        self.learning_iteration = 0
        self.update_idx = 0
        self.window_metrics = SimpleNamespace(stats_and_reset=lambda: {"qf_loss": 0.25, "actor_loss": -1.5})

    def set_train_mode(self) -> None:
        pass

    def act(self, obs, deterministic):
        return torch.zeros(self.num_envs, 1)

    def update(self, num_updates) -> None:
        if num_updates > 0:
            self.learning_iteration += 1
            self.update_idx += num_updates

    def state_dict(self):
        return {
            "global_step": self.global_step,
            "learning_iteration": self.learning_iteration,
            "update_idx": self.update_idx,
        }

    def load_state_dict(self, state, load_optimizers=False) -> None:
        self.global_step = state["global_step"]
        self.learning_iteration = state.get("learning_iteration", 0)
        self.update_idx = state.get("update_idx", 0)


def _train_sync(tmp_path, monkeypatch, *, resume_from=None, diverge_at=None):
    from motrix_rl_builtin.fastsac.sync.train import Trainer

    writer = _RecordingWriter()
    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", lambda **kwargs: writer)
    num_envs = 3
    monkeypatch.setattr(Trainer, "_make_env", lambda self, *args, **kwargs: _SyncEnv(num_envs, diverge_at))
    monkeypatch.setattr(Trainer, "_make_agent", lambda self, env: _SyncAgent(num_envs, learning_starts=3))
    rl_cfg = SimpleNamespace(trainer=SimpleNamespace(num_learning_iterations=8))
    metrics, sink = _logger(tmp_path)
    context = _context(
        tmp_path, rl_cfg, metrics, num_envs=num_envs, logging_interval=2, save_interval=4, resume_from=resume_from
    )

    Trainer(context=context).train()
    return sink, writer


def test_sync_trainer_emits_tensorboard_tags_as_live_scalars(tmp_path, monkeypatch):
    sink, writer = _train_sync(tmp_path, monkeypatch)

    scalars = [r for r in sink.records if r["kind"] == "scalars"]
    # Three warmup collections, then five positive learning calls: interval two
    # publishes learner steps two/four and the final remainder at step five.
    assert [r["step"] for r in scalars] == [0, 2, 4, 5]
    assert [r["axes"]["env_steps"] for r in scalars] == [3, 15, 21, 24]
    assert [r["data"]["progress/collection_step"] for r in scalars] == [1, 5, 7, 8]
    assert [r["data"]["progress/learning_iteration"] for r in scalars] == [0, 2, 4, 5]
    assert {tag for tag, _, _ in writer.scalars} == SYNC_TB_TAGS
    _assert_live_matches_tensorboard(sink, writer, no_episode_steps=(0,))


def test_sync_trainer_emits_lifecycle_events(tmp_path, monkeypatch):
    sink, _ = _train_sync(tmp_path, monkeypatch)

    assert _events(sink) == [
        ("run_started", 0),
        ("warmup_done", 3),  # learning_starts iterations in
        ("checkpoint_saved", 4),
        ("checkpoint_saved", 8),
    ]
    assert _paths(sink) == [
        (str(tmp_path / "checkpoints" / "model_0000004.pt"), 4),
        (str(checkpoints.final_checkpoint_path("pt", tmp_path)), 8),
    ]


def test_sync_trainer_leaves_placeholder_rollout_means_out_of_live(tmp_path, monkeypatch):
    sink, writer = _train_sync(tmp_path, monkeypatch)

    # No episode has ended at step 0: TensorBoard keeps its NaN, live leaves the keys out
    # instead of spending their one nonfinite event on the placeholder.
    assert all(math.isnan(v) for tag, v, step in writer.scalars if tag in ROLLOUT_MEANS and step == 0)
    first = next(r for r in sink.records if r["kind"] == "scalars")
    assert first["step"] == 0 and not ROLLOUT_MEANS & set(first["data"])
    assert "nonfinite" not in {name for name, _ in _events(sink)}


def test_sync_trainer_flags_a_real_divergence(tmp_path, monkeypatch):
    sink, _ = _train_sync(tmp_path, monkeypatch, diverge_at=4)

    nonfinite = {
        r["data"]["key"]: r["step"] for r in sink.records if r["kind"] == "event" and r["data"]["name"] == "nonfinite"
    }
    # Episode four diverges; the next learner log is iteration two, after
    # collection five. Learner-axis labels are not collection-step labels.
    assert nonfinite["rollout/mean_return"] == 2


def test_sync_trainer_reports_resume(tmp_path, monkeypatch):
    resume_from = tmp_path / "resume.pt"
    torch.save({"global_step": 4, "learning_iteration": 6, "update_idx": 18}, resume_from)

    sink, _ = _train_sync(tmp_path, monkeypatch, resume_from=str(resume_from))

    assert _event(sink, "resumed") == {"name": "resumed", "path": str(resume_from), "global_step": 4}
    assert _events(sink) == [
        ("resumed", 4),
        ("run_started", 4),
        ("warmup_done", 7),  # warmup restarts on resume: the replay buffer is empty
        ("checkpoint_saved", 8),
    ]
    # Replay rewarms for three collections without advancing learner progress
    # or republishing the already-saved learner step six.
    assert [(r["step"], r["axes"]["env_steps"]) for r in sink.records if r["kind"] == "scalars"] == [(7, 24)]


# ---------------------------------------------------------------- async trainer (parent process)
NUM_ENVS = 6
NUM_COLLECTORS = 2
LEARNING_STARTS = 20


class _FastTime:
    """``time`` for the async trainer module, running faster so its throttles and sleeps stay short."""

    def __init__(self, speedup: float) -> None:
        self._origin = time.time()
        self._speedup = speedup

    def time(self) -> float:
        return self._origin + (time.time() - self._origin) * self._speedup

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds / self._speedup)

    @staticmethod
    def monotonic() -> float:
        return time.monotonic()

    @staticmethod
    def perf_counter() -> float:
        return time.perf_counter()


class _ProbeEnv:
    env = SimpleNamespace(
        policy_observation_space=SimpleNamespace(shape=(2,)),
        value_observation_space=SimpleNamespace(shape=(3,)),
        action_space=SimpleNamespace(shape=(1,)),
    )
    action_low = torch.tensor([-1.0])
    action_high = torch.tensor([1.0])
    max_episode_steps = None

    def close(self) -> None:
        pass


class _Stage(NamedTuple):
    """One scripted learner stage.

    The learner moves the per-collector step counters to ``steps``, publishes a
    window of collector snapshots (``episodes`` finished per collector, rollout
    means ``mean_return``) and, if ``payload``, a learner payload, then saves
    one periodic checkpoint per entry of ``saves``. That is the real learner
    order: the payload of a step goes out before that step's checkpoint.
    """

    steps: tuple[int, int]
    saves: tuple[int, ...] = ()
    payload: bool = True
    episodes: int = 2
    mean_return: float = 1.5
    updates: int | None = None


def _snapshot(collector_id: int, stage: _Stage) -> dict:
    no_episode = stage.episodes == 0
    return {
        "collector_id": collector_id,
        "return": math.nan if no_episode else stage.mean_return,
        "ep_len": math.nan if no_episode else 12.0,
        "episodes": stage.episodes,
        "reward_terms": {"alive": 0.5},
        "env_metrics": {"height": 0.9},
        "policy_lag": collector_id + 1,
        "timing_ms": {"collect": 2.0, "env_step": 1.0, "sync": 0.5, "sync.wait_writer": 0.2},
    }


def _payload(step: int, checkpoint_path: str | None) -> dict:
    warming = step < LEARNING_STARTS
    return {
        "rank": 0,
        "step": step,
        "sps": 100.0,
        "updates": 2 * step,
        "update_idx": 2 * step,
        "learning_iteration": step // 2,
        "training_started": True,
        "metrics": None if warming else {"qf_loss": 0.25, "actor_loss": -1.5},
        "warming": warming,
        "learn_ms": 1.0,
        "learn_pct": 40.0,
        "learner_timing": {},
        "buffer_size": 6,
        "buffer_capacity": 12,
        "ring_fill": [1, 2],
        "weight_version": step,
        "checkpoint_path": checkpoint_path,
    }


class _Workers:
    """Scripted learner and collectors.

    The learner drives each ``_Stage`` and, after each payload, waits until
    the parent has logged it live (or, with ``live=False``, taken it off the
    queue). After the last stage it either
    writes the final checkpoint and exits cleanly, or makes ``crash_collector``
    report a traceback and exit with code 1.
    """

    def __init__(self, sink: _RecordingSink, stages, *, crash_collector: int | None = None, live: bool = True) -> None:
        from motrix_rl_builtin.fastsac.async_impl.worker import run_collector_process, run_learner_process

        self.sink = sink
        self.stages = stages
        self.crash_collector = crash_collector
        self.live = live
        self.exitcodes: dict[int, int] = {}
        self.ctx = ThreadSpawnContext({run_learner_process: self.learner, run_collector_process: self.collector})

    def learner(self, process) -> int:
        args = process.arguments
        control, handshake = args["control"], args["handshake"]
        handshake.report_ready("learner", 0)
        handshake.wait_for_start(control)
        collectors = [p.arguments for p in self.ctx.processes if p is not process]
        logged = self.sink.count("scalars")
        checkpoint_path = None
        next_learning_log = 10
        for stage in self.stages:
            for collector_id, target in enumerate(stage.steps):
                while control.collector_steps_at(collector_id) < target:
                    control.inc_collector_steps(collector_id)
            for collector in collectors:
                collector["stats_queue"].put(_snapshot(collector["collector_id"], stage))
            if stage.payload:
                payload = _payload(control.collector_steps // NUM_COLLECTORS, checkpoint_path)
                if stage.updates is not None:
                    payload["updates"] = stage.updates
                    payload["update_idx"] = stage.updates
                    payload["learning_iteration"] = stage.updates // 4
                    payload["training_started"] = stage.updates > 0
                args["panel_queue"].put(payload)
                if payload["updates"] == 0:
                    self._wait(lambda: args["panel_queue"].empty() or control.stop)
                    continue
                due = payload["learning_iteration"] >= next_learning_log
                if due:
                    next_learning_log = (payload["learning_iteration"] // 10 + 1) * 10
                    logged += 1
                if self.live and due:
                    self._wait(lambda: self.sink.count("scalars") >= logged or control.stop)
                else:
                    self._wait(lambda: args["panel_queue"].empty() or control.stop)
            for step in stage.saves:
                checkpoint_path = str(Path(args["checkpoint_dir"]) / f"model_{step:07d}.pt")
                args["panel_queue"].put({"rank": 0, "checkpoint_saved": {"step": step, "path": checkpoint_path}})
        if self.crash_collector is not None:
            args["error_queue"].put((f"collector[{self.crash_collector}]", "Traceback: boom\n"))
            self.exitcodes[self.crash_collector] = 1
            self._wait(lambda: control.stop)
            return 0
        self.complete(process)
        final_path = checkpoints.final_checkpoint_path("pt", args["run_dir"])
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(b"")
        return 0

    def complete(self, process) -> None:
        # Real collectors reach a target before triggering clean peer shutdown.
        args = process.arguments
        while args["control"].collector_steps_at(0) < args["num_iterations"]:
            args["control"].inc_collector_steps(0)

    def collector(self, process) -> int:
        args = process.arguments
        control, handshake, collector_id = args["control"], args["handshake"], args["collector_id"]
        handshake.report_ready("collector", collector_id)
        handshake.wait_for_start(control)
        self._wait(lambda: control.stop or collector_id in self.exitcodes)
        return self.exitcodes.get(collector_id, 0)

    @staticmethod
    def _wait(predicate, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.005)


def _train_async(tmp_path, monkeypatch, workers: _Workers, *, num_iterations: int, resume_from=None, live=True):
    from motrix_rl_builtin.fastsac.async_impl import train as async_train
    from motrix_rl_builtin.fastsac.async_impl.topology import CollectorInfo, LearnerInfo, TrainerTopology

    cpu = torch.device("cpu")
    topology = TrainerTopology(
        learners=[LearnerInfo(rank=0, device=cpu, numa_node=None, cpus=[])],
        collectors=[
            CollectorInfo(i, 0, NUM_ENVS // NUM_COLLECTORS, cpu, None, [], ring_ipc=False, weight_ipc=False)
            for i in range(NUM_COLLECTORS)
        ],
    )
    writer = _RecordingWriter()
    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", lambda **kwargs: writer)
    monkeypatch.setattr(async_train, "time", _FastTime(10.0))
    monkeypatch.setattr(async_train, "mp", SimpleNamespace(get_context=lambda method: workers.ctx))
    monkeypatch.setattr(async_train, "resolve_trainer_topology", lambda *args, **kwargs: topology)
    monkeypatch.setattr(async_train, "actor_param_numel", lambda *args: 0)
    monkeypatch.setattr(async_train, "capture_system_info", lambda **kwargs: {})
    monkeypatch.setattr(async_train, "spawn_placement", lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(async_train, "inherit_log_stdio", lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(async_train.Trainer, "_make_env", lambda self, *args, **kwargs: _ProbeEnv())
    (tmp_path / "metadata.json").write_text("{}")
    rl_cfg = SimpleNamespace(
        device="cpu",
        agent=SimpleNamespace(batch_size=8, learning_starts=LEARNING_STARTS),
        trainer=SimpleNamespace(
            num_learning_iterations=num_iterations,
            async_options=SimpleNamespace(
                num_collectors=NUM_COLLECTORS,
                num_learners=1,
                cpus_per_collector=None,
                learner_devices=None,
                collector_inference_device="cpu",
                ring_capacity=2,
                utd_mode="strict",
            ),
        ),
    )
    metrics = MetricsLogger("unit-run", jsonl=workers.sink) if live else MetricsLogger.disabled()
    context = _context(
        tmp_path, rl_cfg, metrics, num_envs=NUM_ENVS, logging_interval=10, save_interval=0, resume_from=resume_from
    )
    return async_train.Trainer(context=context), writer


def test_async_panel_waits_for_first_learner_update(tmp_path, monkeypatch):
    from motrix_rl_builtin.fastsac.async_impl import train as async_train

    sink = _RecordingSink(tmp_path / "metrics.jsonl")
    workers = _Workers(sink, [_Stage((10, 10), updates=0), _Stage((20, 20))])
    trainer, _ = _train_async(tmp_path, monkeypatch, workers, num_iterations=30)
    panels = []
    monkeypatch.setattr(async_train, "emit_training_panel", lambda live, stats, **kwargs: panels.append(stats))

    trainer.train()

    assert panels
    assert all(panel.iteration >= 20 for panel in panels)


@pytest.mark.parametrize("final_metrics", [{"qf_loss": 0.75}, None])
def test_async_panel_keeps_collection_progress_separate_from_final_learner_counters(
    tmp_path, monkeypatch, final_metrics
):
    from motrix_rl_builtin.fastsac.async_impl import train as async_train

    class FinalCountersWorkers(_Workers):
        def complete(self, process):
            super().complete(process)
            process.arguments["panel_queue"].put(
                {
                    "rank": 0,
                    "learner_counters": {
                        "learning_iteration": 13,
                        "update_idx": 52,
                        "updates": 52,
                        "metrics": final_metrics,
                    },
                }
            )

    sink = _RecordingSink(tmp_path / "metrics.jsonl")
    workers = FinalCountersWorkers(sink, [_Stage((20, 20))])
    trainer, _ = _train_async(tmp_path, monkeypatch, workers, num_iterations=30)
    panels = []
    monkeypatch.setattr(async_train, "emit_training_panel", lambda live, stats, **kwargs: panels.append(stats))

    trainer.train()

    assert panels[-1].iteration == 25  # (30 + 20) / two collectors
    assert panels[-1].total_iterations == 30
    assert panels[-1].learning_iteration == 13
    assert panels[-1].update_idx == 52
    assert panels[-1].training_metrics == {"qf_loss": 0.25, "actor_loss": -1.5}
    assert any(panel.learning_iteration == 10 and panel.update_idx == 40 for panel in panels[:-1])
    records = [r for r in sink.records if r["kind"] == "scalars"]
    assert [r["step"] for r in records] == [10, 13]  # force final sub-interval publication
    assert records[-1]["axes"]["env_steps"] == 150
    assert records[-1]["data"]["progress/collection_step"] == 25
    final_train = {key: value for key, value in records[-1]["data"].items() if key.startswith("train/")}
    assert final_train == {f"train/{key}": value for key, value in (final_metrics or {}).items()}
    assert panels[-1].progress_unit == "collection steps"


def test_async_run_can_finish_without_any_training_update(tmp_path, monkeypatch):
    sink = _RecordingSink(tmp_path / "metrics.jsonl")
    workers = _Workers(sink, [_Stage((10, 10), updates=0)])
    trainer, _ = _train_async(tmp_path, monkeypatch, workers, num_iterations=10)

    trainer.train()

    assert checkpoints.final_checkpoint_path("pt", tmp_path).exists()


def test_async_panel_reports_zero_throughput_after_collection_stalls(tmp_path, monkeypatch):
    from motrix_rl_builtin.fastsac.async_impl import train as async_train

    panels = []

    class StalledWorkers(_Workers):
        def complete(self, process):
            # Hold fixed collection progress across two measured windows before
            # allowing the normal final collection target to complete.
            self._wait(lambda: len(panels) >= 3 or process.arguments["control"].stop)
            super().complete(process)

    sink = _RecordingSink(tmp_path / "metrics.jsonl")
    workers = StalledWorkers(sink, [_Stage((20, 20))])
    trainer, _ = _train_async(tmp_path, monkeypatch, workers, num_iterations=30)
    monkeypatch.setattr(async_train, "emit_training_panel", lambda live, stats, **kwargs: panels.append(stats))

    trainer.train()

    assert len(panels) >= 3
    stalled = [panel for panel in panels if panel.iteration == 20]
    assert stalled[-1].steps_per_second == 0.0
    assert stalled[-1].iterations_per_second == 0.0
    assert not stalled[-1].warming
    assert [r["step"] for r in sink.records if r["kind"] == "scalars"] == [10]


def test_async_trainer_emits_tensorboard_tags_events_and_collector_env_steps(tmp_path, monkeypatch):
    sink = _RecordingSink(tmp_path / "metrics.jsonl")
    # Uneven collectors in the first stage: 21 batches -> step 10, but 21 * 3 env steps.
    workers = _Workers(sink, [_Stage((11, 10)), _Stage((20, 20), saves=(20,)), _Stage((40, 40))])
    trainer, writer = _train_async(tmp_path, monkeypatch, workers, num_iterations=40)

    trainer.train()

    scalars = [r for r in sink.records if r["kind"] == "scalars"]
    assert sorted({r["step"] for r in scalars}) == [10, 20]
    # Learner x-axis is independent of the collected transition count.
    collector_steps = {10: 40, 20: 80}
    assert all(r["axes"]["env_steps"] == collector_steps[r["step"]] * 3 for r in scalars)
    assert {tag for tag, _, _ in writer.scalars} == ASYNC_TB_TAGS
    assert all(r["data"]["progress/learning_iteration"] == r["step"] for r in scalars)
    assert all(r["data"]["progress/update_idx"] == 4 * r["step"] for r in scalars)
    _assert_live_matches_tensorboard(sink, writer)

    # The checkpoint notice and the step-20 payload can land in one drain: their order is not fixed.
    assert sorted(_events(sink), key=lambda event: event[1]) in (
        [("run_started", 0), ("warmup_done", 20), ("checkpoint_saved", 20), ("checkpoint_saved", 40)],
        [("run_started", 0), ("checkpoint_saved", 20), ("warmup_done", 20), ("checkpoint_saved", 40)],
    )
    assert _event(sink, "run_started")["workers"] == NUM_COLLECTORS + 1
    assert _paths(sink) == [
        (str(tmp_path / "checkpoints" / "model_0000020.pt"), 20),
        (str(checkpoints.final_checkpoint_path("pt", tmp_path)), 40),
    ]

    # The logger holds a thread and locks: it must never be pickled into a worker.
    assert len(workers.ctx.processes) == NUM_COLLECTORS + 1
    for process in workers.ctx.processes:
        values = (*process.args, *process.kwargs.values())
        assert not any(isinstance(value, (MetricsLogger, TrainerContext)) for value in values), process.name


@pytest.mark.parametrize("live", [True, False])
def test_learner_sends_checkpoint_notices_only_for_live_metrics(tmp_path, monkeypatch, live):
    # A notice is a put on the learner's hot loop; without live metrics nothing reads it.
    sink = _RecordingSink(tmp_path / "metrics.jsonl")
    workers = _Workers(sink, [_Stage((20, 20))], live=live)
    trainer, _ = _train_async(tmp_path, monkeypatch, workers, num_iterations=20, live=live)

    trainer.train()

    (learner,) = [process for process in workers.ctx.processes if "learner" in process.name]
    assert learner.kwargs["notify_checkpoints"] is live


def test_async_trainer_reports_worker_crash(tmp_path, monkeypatch):
    sink = _RecordingSink(tmp_path / "metrics.jsonl")
    workers = _Workers(sink, [_Stage((11, 10))], crash_collector=1)
    trainer, _ = _train_async(tmp_path, monkeypatch, workers, num_iterations=40)

    with pytest.raises(RuntimeError, match="collector-1"):
        trainer.train()

    crash = _event(sink, "worker_crashed")
    assert crash == {
        "name": "worker_crashed",
        "worker": "fastsac-async-collector-1",
        "exitcode": 1,
        "error_log": "async_errors/collector[1]_error.log",
    }
    assert (tmp_path / crash["error_log"]).read_text() == "Traceback: boom\n"
    assert "checkpoint_saved" not in {name for name, _ in _events(sink)}


def test_async_trainer_reports_resume(tmp_path, monkeypatch):
    resume_from = tmp_path / "resume.pt"
    torch.save({"global_step": 10, "learning_iteration": 7}, resume_from)
    sink = _RecordingSink(tmp_path / "metrics.jsonl")
    workers = _Workers(sink, [_Stage((20, 20))])
    trainer, _ = _train_async(tmp_path, monkeypatch, workers, num_iterations=20, resume_from=str(resume_from))

    trainer.train()

    assert _event(sink, "resumed") == {"name": "resumed", "path": str(resume_from), "global_step": 10}
    assert _events(sink)[:3] == [("resumed", 10), ("run_started", 10), ("warmup_done", 20)]
    assert {(r["step"], r["axes"]["env_steps"]) for r in sink.records if r["kind"] == "scalars"} == {(10, 120)}


def test_async_trainer_emits_live_scalars_without_tensorboard(tmp_path, monkeypatch):
    sink = _RecordingSink(tmp_path / "metrics.jsonl")
    workers = _Workers(sink, [_Stage((11, 10)), _Stage((20, 20))])
    trainer, _ = _train_async(tmp_path, monkeypatch, workers, num_iterations=20)

    def _unavailable(**kwargs):
        raise ImportError("tensorboard is not installed")

    monkeypatch.setattr(torch.utils.tensorboard, "SummaryWriter", _unavailable)
    trainer.train()

    scalars = [r for r in sink.records if r["kind"] == "scalars"]
    assert sorted({r["step"] for r in scalars}) == [10]
    assert {"rollout/mean_return", "train/qf_loss"} <= set(scalars[-1]["data"])


def test_async_trainer_reports_checkpoints_saved_after_the_last_log_window(tmp_path, monkeypatch):
    sink = _RecordingSink(tmp_path / "metrics.jsonl")
    # Real learner order: the step-10 payload goes out before the step-10 save, then two more
    # saves land after it and the loop ends before the next log window (step 20) is reached.
    workers = _Workers(sink, [_Stage((10, 10), saves=(10,)), _Stage((14, 14), saves=(12, 14), payload=False)])
    trainer, _ = _train_async(tmp_path, monkeypatch, workers, num_iterations=15)

    trainer.train()

    checkpoint_dir = tmp_path / "checkpoints"
    assert _paths(sink) == [
        (str(checkpoint_dir / "model_0000010.pt"), 10),
        (str(checkpoint_dir / "model_0000012.pt"), 12),
        (str(checkpoint_dir / "model_0000014.pt"), 14),
        (str(checkpoints.final_checkpoint_path("pt", tmp_path)), 14),
    ]


def test_async_trainer_leaves_placeholder_rollout_means_out_of_live(tmp_path, monkeypatch):
    sink = _RecordingSink(tmp_path / "metrics.jsonl")
    stages = [
        _Stage((20, 20), episodes=0),  # learner iteration 10: NaN placeholders
        _Stage((40, 40)),
        _Stage((60, 60), mean_return=math.nan),  # learner iteration 30: real divergence
    ]
    workers = _Workers(sink, stages)
    trainer, writer = _train_async(tmp_path, monkeypatch, workers, num_iterations=60)

    trainer.train()

    _assert_live_matches_tensorboard(sink, writer, no_episode_steps=(10,))
    scalars = {r["step"]: r["data"] for r in sink.records if r["kind"] == "scalars"}
    assert not ROLLOUT_MEANS & set(scalars[10])
    assert scalars[20]["rollout/mean_return"] == pytest.approx(1.5)
    nonfinite = [
        (r["data"]["key"], r["step"]) for r in sink.records if r["kind"] == "event" and r["data"]["name"] == "nonfinite"
    ]
    assert nonfinite == [("rollout/mean_return", 30)]
