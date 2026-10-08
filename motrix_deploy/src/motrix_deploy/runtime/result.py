# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Structured rollout results and latency summaries."""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LatencySummary:
    """Aggregate latency in milliseconds."""

    mean_ms: float
    max_ms: float
    p95_ms: float


@dataclass(frozen=True)
class RolloutResult:
    """Machine-readable outcome of one deployment run."""

    success: bool
    exit_reason: str
    completed_steps: int
    wall_time_s: float
    overrun_count: int
    trace_sha256: str
    error: str | None = None
    latency: dict[str, LatencySummary] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class SimulationRolloutResult(RolloutResult):
    """Simulation outcome with model-only physical time, excluding readiness and damping."""

    policy_simulation_time_s: float
    real_time_factor: float


class LatencyRecorder:
    """Collect latency samples and produce stable summaries."""

    def __init__(self) -> None:
        self._samples_ns: dict[str, list[int]] = {
            name: [] for name in ("input", "read", "observation", "inference", "action", "write", "loop")
        }

    def add(self, name: str, elapsed_ns: int) -> None:
        self._samples_ns[name].append(elapsed_ns)

    def summarize(self) -> dict[str, LatencySummary]:
        result: dict[str, LatencySummary] = {}
        for name, samples in self._samples_ns.items():
            if not samples:
                continue
            values_ms = np.asarray(samples, dtype=np.float64) / 1e6
            result[name] = LatencySummary(
                mean_ms=float(np.mean(values_ms)),
                max_ms=float(np.max(values_ms)),
                p95_ms=float(np.percentile(values_ms, 95)),
            )
        return result
