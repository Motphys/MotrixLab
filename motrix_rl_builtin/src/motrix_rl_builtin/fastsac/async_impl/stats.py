# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC async statistics aggregation and terminal consumers."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.queues import Queue as ProcessQueue
from queue import Empty
from typing import Any, Protocol, TypedDict

from motrix_rl.console import PanelSystemSampler, TrainingPanelSession, TrainingPanelStats
from motrix_rl.training_logger import TrainingLogger
from motrix_rl_builtin.fastsac.live_metrics import ROLLOUT_MEANS, system_scalars

TimingTree = dict[str, object]


def timing_mean(values: list[float]) -> float:
    """Mean of a non-empty timing sample list (ms)."""
    return sum(values) / len(values)


def nest_timing_path(tree: dict[str, Any], parts: tuple[str, ...], value: float) -> None:
    """Insert one dotted timing path into a nested mapping."""
    head, rest = parts[0], parts[1:]
    node = tree.get(head)
    if not rest:
        if isinstance(node, dict):
            node["total"] = value
        else:
            tree[head] = value
    else:
        if not isinstance(node, dict):
            node = {"total": node} if node is not None else {}
            tree[head] = node
        nest_timing_path(node, rest, value)


def aggregate_collector_stats(per_collector: dict[int, dict | None]) -> dict:
    """Merge per-collector rollout snapshots into one aggregate stats dict."""
    snapshots = [stats for stats in per_collector.values() if stats]

    def _nanmean(key: str) -> float:
        values = [stats[key] for stats in snapshots if not math.isnan(stats.get(key, float("nan")))]
        return (sum(values) / len(values)) if values else float("nan")

    def _mean_dicts(key: str) -> dict[str, float]:
        keys = {k for stats in snapshots for k in stats.get(key, {})}
        return {
            k: sum(stats[key][k] for stats in snapshots if k in stats.get(key, {}))
            / sum(1 for stats in snapshots if k in stats.get(key, {}))
            for k in keys
        }

    return {
        "return": _nanmean("return"),
        "ep_len": _nanmean("ep_len"),
        "episodes": sum(stats.get("episodes", 0) for stats in snapshots),
        "reward_terms": _mean_dicts("reward_terms"),
        "env_metrics": _mean_dicts("env_metrics"),
        "policy_lag": max((stats.get("policy_lag", 0) for stats in snapshots), default=0),
        "timing_ms": _mean_dicts("timing_ms"),
    }


class ControlLike(Protocol):
    collector_steps: int


CollectorSnapshot = TypedDict(
    "CollectorSnapshot",
    {
        "collector_id": int,
        "return": float,
        "ep_len": float,
        "episodes": int,
        "reward_terms": dict[str, float],
        "env_metrics": dict[str, float],
        "policy_lag": float,
        "timing_ms": dict[str, float],
    },
    total=False,
)


class LearnerPayload(TypedDict, total=False):
    rank: int
    training_started: bool
    updates: int
    learn_ms: float | None
    learn_pct: float
    learner_timing: dict[str, float | dict]
    buffer_size: float
    buffer_capacity: float
    metrics: dict[str, float] | None
    ring_fill: list[float | None]
    weight_version: int
    checkpoint_path: str | None


class CheckpointNotice(TypedDict):
    step: int
    path: str


@dataclass
class AsyncTrainingStatsSource:
    """Parent-owned worker queues and mutable training-stat snapshots."""

    collector_data_queues: Sequence[ProcessQueue]
    learner_data_queue: ProcessQueue
    control: ControlLike
    num_envs: int
    num_collectors: int
    num_learners: int
    num_iterations: int
    resume_step: int
    per_collector_stats: dict[int, CollectorSnapshot | None]
    learner_payloads: dict[int, LearnerPayload | None]

    def drain(self) -> list[CheckpointNotice]:
        checkpoint_notices: list[CheckpointNotice] = []
        for queue in self.collector_data_queues:
            try:
                while True:
                    snapshot = queue.get_nowait()
                    self.per_collector_stats[snapshot["collector_id"]] = snapshot
            except Empty:
                pass
        try:
            while True:
                payload = self.learner_data_queue.get_nowait()
                notice = payload.get("checkpoint_saved")
                if notice is not None:
                    checkpoint_notices.append(notice)
                else:
                    self.learner_payloads[payload["rank"]] = payload
        except Empty:
            pass
        return checkpoint_notices

    def data_ready(self) -> bool:
        return (
            all(value is not None for value in self.per_collector_stats.values())
            and all(
                value is not None
                and (
                    value["training_started"]
                    or self.control.collector_steps >= self.num_iterations * self.num_collectors
                )
                for value in self.learner_payloads.values()
            )
            and self.control.collector_steps > 0
        )

    def handoff_gate_line(self) -> str:
        stats_marks = " ".join(
            f"c{i} {'✓' if self.per_collector_stats[i] is not None else '…'}" for i in range(self.num_collectors)
        )
        payload_marks = " ".join(
            f"l{rank} " + ("✓" if payload is not None and payload["training_started"] else "initializing first update…")
            for rank, payload in self.learner_payloads.items()
        )
        return f"panel data — stats: {stats_marks} | payloads: {payload_marks} | steps: {self.control.collector_steps}"


@dataclass(frozen=True)
class TrainingLogRecord:
    """One logging-window payload produced independently of terminal rendering."""

    values: Mapping[str, float]
    step: int
    env_steps: int
    placeholders: tuple[str, ...]


@dataclass(frozen=True)
class AsyncTrainingStats:
    """Normalized training statistics emitted for independent consumers."""

    stats: TrainingPanelStats
    step: int
    warmup_done: bool
    updates: int
    log_record: TrainingLogRecord | None = None
    checkpoint_notices: tuple[CheckpointNotice, ...] = ()


class AsyncTrainingStatsAggregator:
    """Aggregate source payloads into normalized training statistics."""

    def __init__(self, *, learning_starts: int, logging_interval: int) -> None:
        self._learning_starts = learning_starts
        self._logging_interval = logging_interval
        self._num_envs = 0
        self._num_collectors = 0
        self._num_learners = 0
        self._num_iterations = 0
        self._system_sampler = PanelSystemSampler()
        self._panel_start_time = time.time()
        self._panel_anchored = False
        self._rate_samples: list[tuple[float, int]] = []
        self._last_log_time = self._panel_start_time
        self._last_log_step = 0
        self._last_updates = 0
        self._last_weight_version = 0
        self._next_log = 0
        self._was_warming = learning_starts > 0

    def collect(self, source: AsyncTrainingStatsSource) -> AsyncTrainingStats | None:
        checkpoint_notices = source.drain()
        if not source.data_ready():
            return None
        self._num_envs = source.num_envs
        self._num_collectors = source.num_collectors
        self._num_learners = source.num_learners
        self._num_iterations = source.num_iterations
        if self._next_log == 0:
            self._last_log_step = source.resume_step
            self._next_log = (
                ((source.resume_step // max(self._logging_interval, 1)) + 1) * self._logging_interval
                if self._logging_interval > 0
                else 0
            )
            self._was_warming = source.resume_step < self._learning_starts

        collector_steps = source.control.collector_steps
        step = collector_steps // self._num_collectors
        now = time.time()
        if step > self._last_log_step and not self._panel_anchored:
            self._panel_start_time = now
            self._last_log_time = now
            self._panel_anchored = True

        last_stats = aggregate_collector_stats(source.per_collector_stats)
        payloads = [payload for payload in source.learner_payloads.values() if payload]
        primary = payloads[0] if payloads else None
        self._rate_samples.append((now, step))
        while len(self._rate_samples) >= 2 and self._rate_samples[1][0] <= now - 10.0:
            self._rate_samples.pop(0)
        t0, s0 = self._rate_samples[0]
        span = now - t0
        steps_per_second = (step - s0) * self._num_envs / max(span, 1.0) if span > 0 else None
        iterations_per_second = (step - s0) / max(span, 1.0) if span > 0 else None
        warming = step < self._learning_starts
        learn_ms = (
            timing_mean([payload["learn_ms"] for payload in payloads if payload["learn_ms"] is not None])
            if payloads
            else 0.0
        )
        learn_percent = timing_mean([payload["learn_pct"] for payload in payloads]) if payloads else 0.0
        updates = primary["updates"] if primary else self._last_updates
        utd = updates / max(step, 1)
        collector_timing_ms = last_stats.get("timing_ms", {})
        collector_detail_ms = {key: value for key, value in collector_timing_ms.items() if key != "collect"}
        collector_items: TimingTree = {}
        for key, value in collector_detail_ms.items():
            nest_timing_path(collector_items, tuple(key.split(".")), value)
        timing_groups: dict[str, TimingTree] = {"collector": collector_items}
        for payload in payloads:
            rank_key = "learner" if self._num_learners == 1 else f"learner[{payload['rank']}]"
            if payload["learner_timing"]:
                timing_groups[rank_key] = payload["learner_timing"]
        system = self._system_sampler.sample()
        panel_stats = TrainingPanelStats(
            iteration=step,
            total_iterations=self._num_iterations,
            steps_per_second=steps_per_second,
            iterations_per_second=iterations_per_second,
            elapsed_seconds=now - self._panel_start_time,
            mean_return=last_stats["return"],
            mean_episode_length=last_stats["ep_len"],
            episodes=last_stats["episodes"],
            buffer_size=primary["buffer_size"] if primary else 0,
            buffer_capacity=primary["buffer_capacity"] if primary else 0,
            collect_ms=collector_timing_ms.get("collect", 0.0),
            learn_ms=learn_ms,
            learn_percent=learn_percent,
            warming=warming,
            training_metrics=primary["metrics"] if primary else None,
            reward_terms=last_stats["reward_terms"],
            env_metrics=last_stats["env_metrics"],
            timing_groups=timing_groups,
            diagnostics={"UTD": utd},
            cpu_load=system.cpu_load,
            gpu_utilization_percent=system.gpu_utilization_percent,
            memory_usage=system.memory_usage,
            gpu_memory_usage=system.gpu_memory_usage,
            gpu_devices=system.gpu_devices,
            checkpoint_path=primary["checkpoint_path"] if primary else None,
        )
        scalars = self._scalars(
            last_stats,
            primary,
            source.per_collector_stats,
            step,
            now,
            steps_per_second,
            updates,
            utd,
            learn_ms,
            learn_percent,
        )
        warmup_done = self._was_warming and not warming
        self._was_warming = warming
        log_record = None
        if step >= self._next_log:
            log_record = TrainingLogRecord(
                values=scalars,
                step=step,
                env_steps=collector_steps * (self._num_envs // self._num_collectors),
                placeholders=() if last_stats["episodes"] else ROLLOUT_MEANS,
            )
            self.mark_logged(
                step=step,
                now=now,
                updates=updates,
                weight_version=primary.get("weight_version") if primary else None,
            )
        return AsyncTrainingStats(
            stats=panel_stats,
            step=step,
            warmup_done=warmup_done,
            updates=updates,
            log_record=log_record,
            checkpoint_notices=tuple(checkpoint_notices),
        )

    def mark_logged(self, *, step: int, now: float, updates: int, weight_version: int | None) -> None:
        self._last_log_time, self._last_log_step, self._last_updates = now, step, updates
        if weight_version is not None:
            self._last_weight_version = weight_version
        self._next_log += self._logging_interval

    def _scalars(
        self,
        last_stats: Mapping[str, object],
        primary: LearnerPayload | None,
        per_collector_stats: Mapping[int, CollectorSnapshot | None],
        step: int,
        now: float,
        steps_per_second: float | None,
        updates: int,
        utd: float,
        learn_ms: float,
        learn_percent: float,
    ) -> dict[str, float]:
        scalars: dict[str, float] = {
            "rollout/mean_return": last_stats["return"],
            "rollout/mean_ep_len": last_stats["ep_len"],
        }
        if steps_per_second is not None:
            scalars["perf/env_steps_per_s"] = steps_per_second
        scalars["perf/updates_per_s"] = (updates - self._last_updates) / max(now - self._last_log_time, 1e-6)
        scalars["async/policy_lag"] = last_stats["policy_lag"]
        if primary and primary.get("weight_version") is not None:
            scalars["async/weight_publishes_per_iter"] = (primary["weight_version"] - self._last_weight_version) / max(
                step - self._last_log_step, 1
            )
        if primary:
            fills = primary.get("ring_fill") or []
            scalars["async/ring_fill"] = sum(fill for fill in fills if fill is not None)
            if primary.get("weight_version") is not None:
                scalars["async/weight_version"] = primary["weight_version"]
        scalars["async/utd"] = utd
        if self._num_collectors > 1:
            for index in range(self._num_collectors):
                stats = per_collector_stats.get(index)
                if stats:
                    scalars[f"async/policy_lag_collector{index}"] = stats["policy_lag"]
                if primary:
                    fill = (primary.get("ring_fill") or [None] * self._num_collectors)[index]
                    if fill is not None:
                        scalars[f"async/ring_fill_collector{index}"] = fill
        collector_timing_ms = last_stats.get("timing_ms", {})
        collector_detail = {key: value for key, value in collector_timing_ms.items() if key != "collect"}
        scalars["perf/collect_ms_per_batch"] = collector_timing_ms.get("collect", 0.0)
        scalars.update({f"perf/collector_{key}_ms": value for key, value in collector_detail.items()})
        scalars["perf/learn_ms_total"] = learn_ms
        scalars["perf/learn_pct"] = learn_percent
        scalars.update({f"metrics/{key}": value for key, value in last_stats["env_metrics"].items()})
        scalars.update({f"reward/{key}": value for key, value in last_stats["reward_terms"].items()})
        if primary and primary["metrics"] is not None:
            scalars.update({f"train/{key}": value for key, value in primary["metrics"].items()})
        return scalars


def log_training_stats(training_log: TrainingLogger, sample: AsyncTrainingStats) -> None:
    """Forward one normalized sample to TensorBoard/live training logging."""
    for notice in sample.checkpoint_notices:
        training_log.event("checkpoint_saved", step=notice["step"], path=notice["path"])
    if sample.warmup_done:
        training_log.event("warmup_done", step=sample.step)
    if sample.log_record is not None:
        record = sample.log_record
        training_log.scalars(
            record.values,
            record.step,
            env_steps=record.env_steps,
            live_extra=lambda: system_scalars(sample.stats),
            placeholders=record.placeholders,
        )


class AsyncPanelRenderer:
    """Render normalized training statistics in the terminal panel."""

    def __init__(self, *, env_name: str) -> None:
        self._title = f"{env_name}/motrix.fastsac"
        self._panel: TrainingPanelSession | None = None

    def render(self, sample: AsyncTrainingStats, *, force: bool = False) -> None:
        if self._panel is None:
            self._panel = TrainingPanelSession.open(title=self._title)
            if self._panel.console is not None:
                self._panel.console.clear()
        self._panel.submit(sample.stats, force=force)

    def close(self) -> None:
        if self._panel is not None:
            self._panel.close()
