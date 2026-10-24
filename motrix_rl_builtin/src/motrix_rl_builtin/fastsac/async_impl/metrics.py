# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
"""Pure FastSAC async metrics aggregation."""

from __future__ import annotations

from typing import Any

from motrix_rl.console import TrainingPanelStats
from motrix_rl.execution.rates import RollingRate
from motrix_rl.execution.training import TrainingEvent, TrainingMetrics, WorkerReports

from .stats import aggregate_collector_stats, nest_timing_path, timing_mean


class FastSacMetrics:
    """Accumulate opaque worker reports and produce algorithm metrics frames."""

    def __init__(
        self,
        *,
        num_envs: int,
        num_collectors: int,
        num_learners: int,
        budget: int,
        learning_starts: int,
        logging_interval: int,
        resume_step: int,
        resume_learning_iteration: int,
    ) -> None:
        self.num_envs = num_envs
        self.num_collectors = num_collectors
        self.num_learners = num_learners
        self.budget = budget
        self.learning_starts = int(learning_starts)
        self.logging_interval = int(logging_interval)
        self.collectors: dict[int, dict] = {}
        self.learners: dict[int, dict] = {}
        self.counters: dict[int, dict] = {}
        self.checkpoints: list[TrainingEvent] = []
        self._rate = RollingRate(10.0)
        self._anchor_time: float | None = None
        self._last_log_time: float | None = None
        self._last_weight_version = 0
        self._last_render: float | None = None
        self._last_log_learning = int(resume_learning_iteration)
        self._last_log_step = int(resume_step)
        self._last_collection_steps = int(resume_step) * num_collectors
        self._last_scalar_collection_steps = self._last_collection_steps
        self._last_log_updates = 0
        self._next_log = self._next_log_at(self._last_log_learning)
        self._was_warming = self._last_log_step < self.learning_starts
        self._saw_payload = False

    def _next_log_at(self, iteration: int) -> int:
        return ((iteration // self.logging_interval) + 1) * self.logging_interval if self.logging_interval > 0 else 0

    def consume(self, reports: WorkerReports) -> None:
        """Consume already-drained worker payloads; no queues or process state involved."""
        for worker_id, messages in reports.reports.items():
            if worker_id is None:
                continue
            for message in messages:
                if message is None or not isinstance(message, dict):
                    continue
                kind = message.get("type")
                if kind == "collector_stats":
                    stats = message.get("stats")
                    if stats:
                        self.collectors[int(stats["collector_id"])] = stats
                elif kind == "learner_payload":
                    self._saw_payload = True
                    payload = message.get("payload")
                    if payload:
                        rank = int(payload["rank"])
                        if "learner_counters" in payload:
                            self.counters[rank] = payload["learner_counters"]
                        else:
                            self.learners[rank] = payload
                            self.counters[rank] = self._payload_counters(payload)
                elif kind == "learner_counters":
                    rank = int(message["rank"])
                    self.counters[rank] = message["counters"]
                elif kind == "checkpoint_saved":
                    self.checkpoints.append(
                        TrainingEvent("checkpoint_saved", step=message.get("step"), data={"path": message.get("path")})
                    )

    @staticmethod
    def _payload_counters(payload: dict) -> dict:
        return {k: payload.get(k) for k in ("learning_iteration", "update_idx", "updates", "metrics")}

    def _ready(self, step: int, collector_steps: int) -> bool:
        if len(self.collectors) < self.num_collectors or step <= 0:
            return False
        if len(self.learners) < self.num_learners:
            return False
        return all(
            p.get("training_started", False) or collector_steps >= self.budget * self.num_collectors
            for p in self.learners.values()
        )

    def frame(self, now: float, collector_steps: int | None = None, force: bool = False) -> TrainingMetrics:
        """Return a pure frame; invalid frames preserve the first-update gate."""
        payloads = list(self.learners.values())
        primary = self.learners.get(0) or (payloads[0] if payloads else None)
        if collector_steps is None:
            collector_steps = self._last_log_step * self.num_collectors
        step = int(collector_steps) // self.num_collectors
        if not self._ready(step, int(collector_steps)):
            return TrainingMetrics(events=self._take_events(), valid=False)
        if not force and self._last_render is not None and now - self._last_render < 1.0:
            return TrainingMetrics(events=self._take_events(), valid=False)
        self._last_render = now
        if self._anchor_time is None:
            self._anchor_time = now
            self._last_log_time = now
        stats = aggregate_collector_stats(self.collectors)
        rate = self._rate.sample(now, step)
        iters = None if rate is None else rate
        sps = None if rate is None else rate * self.num_envs
        counters = self.counters.get(0) or self._payload_counters(primary or {})
        learning = counters.get("learning_iteration")
        updates = counters.get("update_idx")
        warming = step < self.learning_starts
        events = self._take_events()
        if self._was_warming and not warming:
            events.append(TrainingEvent("warmup_done", step=step))
        self._was_warming = warming
        learn_values = [p.get("learn_ms", 0.0) for p in payloads if p.get("learn_ms") is not None]
        learn_ms = timing_mean(learn_values) if learn_values else 0.0
        learn_pct = timing_mean([p.get("learn_pct", 0.0) for p in payloads]) if payloads else 0.0
        timing = stats.get("timing_ms", {})
        detail = {k: v for k, v in timing.items() if k != "collect"}
        collector_items: dict[str, Any] = {}
        for key, value in detail.items():
            nest_timing_path(collector_items, tuple(key.split(".")), value)
        groups: dict[str, Any] = {"collector": collector_items}
        for p in payloads:
            if p.get("learner_timing"):
                groups["learner" if self.num_learners == 1 else f"learner[{p.get('rank', 0)}]"] = p["learner_timing"]
        panel = TrainingPanelStats(
            iteration=step,
            total_iterations=self.budget,
            progress_unit="collection steps",
            steps_per_second=sps,
            iterations_per_second=iters,
            elapsed_seconds=now - self._anchor_time,
            mean_return=stats["return"],
            mean_episode_length=stats["ep_len"],
            episodes=stats["episodes"],
            buffer_size=(primary or {}).get("buffer_size", 0),
            buffer_capacity=(primary or {}).get("buffer_capacity", 0),
            collect_ms=timing.get("collect", 0.0),
            learn_ms=learn_ms,
            learn_percent=learn_pct,
            warming=warming,
            learning_iteration=learning,
            update_idx=updates,
            training_metrics=(primary or {}).get("metrics"),
            reward_terms=stats["reward_terms"],
            env_metrics=stats["env_metrics"],
            timing_groups=groups,
            diagnostics={"UTD": (updates or 0) / max(step, 1)},
            checkpoint_path=(primary or {}).get("checkpoint_path"),
        )
        scalars: dict[str, float] = {}
        if (
            learning is not None
            and learning > self._last_log_learning
            and (force or learning >= self._next_log)
            and (collector_steps > self._last_scalar_collection_steps or not self._saw_payload)
        ):
            scalars.update(
                {
                    "rollout/mean_return": stats["return"],
                    "rollout/mean_ep_len": stats["ep_len"],
                    "progress/collection_step": step,
                    "progress/learning_iteration": learning,
                    "progress/update_idx": updates or 0,
                }
            )
            if sps is not None:
                scalars["perf/env_steps_per_s"] = sps
            scalars["perf/updates_per_s"] = ((updates or 0) - self._last_log_updates) / max(
                now - self._last_log_time, 1e-6
            )
            scalars["async/policy_lag"] = stats["policy_lag"]
            if primary.get("weight_version") is not None:
                scalars["async/weight_publishes_per_collection_step"] = (
                    primary["weight_version"] - self._last_weight_version
                ) / max(step - self._last_log_step, 1)
                self._last_weight_version = primary["weight_version"]
                scalars["async/weight_version"] = primary["weight_version"]
            fills = primary.get("ring_fill") or [None] * self.num_collectors
            scalars["async/ring_fill"] = sum(fill for fill in fills if fill is not None)
            if self.num_collectors > 1:
                for collector_id in range(self.num_collectors):
                    scalars[f"async/policy_lag_collector{collector_id}"] = self.collectors[collector_id]["policy_lag"]
                    if fills[collector_id] is not None:
                        scalars[f"async/ring_fill_collector{collector_id}"] = fills[collector_id]
            scalars["perf/collect_ms_per_batch"] = timing.get("collect", 0.0)
            for key, value in detail.items():
                scalars[f"perf/collector_{key}_ms"] = value
            scalars["perf/learn_ms_total"] = learn_ms
            scalars["perf/learn_pct"] = learn_pct
            scalars["async/utd"] = (updates or 0) / max(step, 1)
            for key, value in stats["env_metrics"].items():
                scalars[f"metrics/{key}"] = value
            for key, value in stats["reward_terms"].items():
                scalars[f"reward/{key}"] = value
            for key, value in (counters.get("metrics") or {}).items():
                scalars[f"train/{key}"] = value
            self._last_log_learning, self._last_log_step, self._last_log_updates = learning, step, updates or 0
            self._next_log = self._next_log_at(learning)
            self._last_log_time = now
        scalar_env_steps = self._last_scalar_collection_steps * (self.num_envs // self.num_collectors)
        if scalars:
            self._last_scalar_collection_steps = collector_steps
            scalar_env_steps = collector_steps * (self.num_envs // self.num_collectors)
        return TrainingMetrics(
            panel=panel,
            scalars=scalars,
            step=learning,
            env_steps=scalar_env_steps,
            placeholders=frozenset()
            if stats["episodes"]
            else frozenset(("rollout/mean_return", "rollout/mean_ep_len")),
            events=tuple(events),
            valid=True,
        )

    def _take_events(self) -> list[TrainingEvent]:
        events, self.checkpoints = self.checkpoints, []
        return events
