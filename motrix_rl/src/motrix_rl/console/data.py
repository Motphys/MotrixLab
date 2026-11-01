# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral data samplers used by RL training panels.

These helpers aggregate plain Python values only. Provider code remains
responsible for extracting values from its framework and for defining what a
training iteration or completed episode means.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from motrix_rl.system_metrics import (
    CpuLoad,
    CpuLoadSampler,
    GpuDeviceUsage,
    GpuMemoryUsageSampler,
    GpuUtilizationSampler,
    MemoryUsage,
    MemoryUsageSampler,
    sample_gpu_devices,
)


@dataclass(frozen=True)
class EpisodeWindowStats:
    mean_return: float | None
    mean_length: float | None
    episodes: int


class PanelClock:
    """Monotonic clock for a panel or training phase."""

    def __init__(self, *, start: float | None = None) -> None:
        self._start = time.monotonic() if start is None else start

    def elapsed(self, *, now: float | None = None) -> float:
        return max(0.0, (time.monotonic() if now is None else now) - self._start)

    def restart(self, *, now: float | None = None) -> None:
        self._start = time.monotonic() if now is None else now


class RateTracker:
    """Rolling-window rate for any monotonic counter."""

    def __init__(self, *, window_seconds: float = 10.0) -> None:
        if window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = window_seconds
        self._samples: deque[tuple[float, float]] = deque()

    def update(self, value: float, *, now: float | None = None) -> float | None:
        timestamp = time.monotonic() if now is None else now
        self._samples.append((timestamp, value))
        cutoff = timestamp - self.window_seconds
        while len(self._samples) >= 2 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        if len(self._samples) < 2:
            return None
        first_time, first_value = self._samples[0]
        span = timestamp - first_time
        if span <= 0.0:
            return None
        return (value - first_value) / span

    def reset(self) -> None:
        self._samples.clear()


class EpisodeWindow:
    """Keep a bounded recent window of completed episode summaries."""

    def __init__(self, *, max_episodes: int = 100) -> None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        self._returns: deque[float] = deque(maxlen=max_episodes)
        self._lengths: deque[float] = deque(maxlen=max_episodes)
        self._total_episodes = 0

    def add(self, returns: Iterable[float], lengths: Iterable[float]) -> None:
        return_values = list(returns)
        length_values = list(lengths)
        if len(return_values) != len(length_values):
            raise ValueError("returns and lengths must have the same number of values")
        self._returns.extend(float(value) for value in return_values)
        self._lengths.extend(float(value) for value in length_values)
        self._total_episodes += len(return_values)

    def snapshot(self) -> EpisodeWindowStats:
        return EpisodeWindowStats(
            mean_return=sum(self._returns) / len(self._returns) if self._returns else None,
            mean_length=sum(self._lengths) / len(self._lengths) if self._lengths else None,
            episodes=self._total_episodes,
        )


class TimingWindow:
    """Accumulate phase durations and return mean milliseconds per sample."""

    def __init__(self) -> None:
        self._seconds: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def add(self, phase: str, seconds: float) -> None:
        self._seconds[phase] = self._seconds.get(phase, 0.0) + seconds
        self._counts[phase] = self._counts.get(phase, 0) + 1

    def mean_ms(self, phase: str, *, default: float = 0.0) -> float:
        count = self._counts.get(phase, 0)
        return 1000.0 * self._seconds.get(phase, 0.0) / count if count else default

    def total_seconds(self, phase: str) -> float:
        return self._seconds.get(phase, 0.0)

    def reset(self) -> None:
        self._seconds.clear()
        self._counts.clear()


class ScalarWindow:
    """Aggregate Python scalar observations over a drain/reset window."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._minimums: dict[str, float] = {}
        self._maximums: dict[str, float] = {}

    def add(self, values: Mapping[str, Any], *, track_range: bool = False) -> None:
        for key, value in values.items():
            number = float(value)
            self._sums[key] = self._sums.get(key, 0.0) + number
            self._counts[key] = self._counts.get(key, 0) + 1
            if track_range:
                self._minimums[key] = min(number, self._minimums.get(key, number))
                self._maximums[key] = max(number, self._maximums.get(key, number))

    def drain(self, *, include_range: bool = True) -> dict[str, float]:
        result = {key: self._sums[key] / self._counts[key] for key in self._sums}
        if include_range:
            result.update({f"{key}_min": value for key, value in self._minimums.items()})
            result.update({f"{key}_max": value for key, value in self._maximums.items()})
        self._sums.clear()
        self._counts.clear()
        self._minimums.clear()
        self._maximums.clear()
        return result


@dataclass(frozen=True)
class PanelSystemStats:
    cpu_load: CpuLoad | None
    gpu_utilization_percent: float | None
    memory_usage: MemoryUsage | None
    gpu_memory_usage: MemoryUsage | None
    gpu_devices: Sequence[GpuDeviceUsage] | None


class PanelSystemSampler:
    """Sample host resources once for a panel frame."""

    def __init__(self) -> None:
        self._cpu = CpuLoadSampler()
        self._gpu = GpuUtilizationSampler()
        self._memory = MemoryUsageSampler()
        self._gpu_memory = GpuMemoryUsageSampler()

    def sample(self) -> PanelSystemStats:
        return PanelSystemStats(
            cpu_load=self._cpu.sample(),
            gpu_utilization_percent=self._gpu.sample(),
            memory_usage=self._memory.sample(),
            gpu_memory_usage=self._gpu_memory.sample(),
            gpu_devices=sample_gpu_devices(self._gpu, self._gpu_memory),
        )
