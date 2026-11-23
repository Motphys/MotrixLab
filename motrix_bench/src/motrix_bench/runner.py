# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Measurement engine: prepared cases, measurement policy, and result schema.

The engine is task-neutral. It knows how to warm up, time, and summarize a
callable workload described by :class:`PreparedCase`; it never knows what the
workload measures (environments, simulators, collectors, ...).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .stats import percentiles

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Typed measurement policy, independent of any command-line parser."""

    steps: int
    warmup: int = 0

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.warmup < 0:
            raise ValueError("warmup must not be negative")


@dataclass(frozen=True, slots=True)
class BenchmarkRequest(BenchmarkConfig):
    """Compatibility name for :class:`BenchmarkConfig`."""


@dataclass(slots=True)
class PreparedCase:
    """Prepared, executable benchmark case.

    A case owns its lifecycle. ``close`` is called at most once, including when
    a step or synchronization callback raises. ``synchronize`` is invoked after
    every warmup and measured step so asynchronous backends are measured fairly.
    ``before_step`` runs before each warmup/sample outside the timed region.
    """

    step: Callable[[], object]
    close: Callable[[], object] | None = None
    synchronize: Callable[[], object] | None = None
    before_step: Callable[[], object] | None = None
    throughput: float | None = None
    items_per_step: float | None = None
    # Unit name for the counted items ("env steps", "transitions", ...);
    # flows into the result schema and rendered column headers.
    unit: str = "items"
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def measured_items_per_step(self) -> float | None:
        return self.items_per_step if self.items_per_step is not None else self.throughput

    def synchronize_step(self) -> None:
        if self.synchronize is not None:
            self.synchronize()

    def close_once(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.close is not None:
            self.close()

    def __enter__(self) -> PreparedCase:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.close_once()
        return False


@dataclass(frozen=True, slots=True)
class Measurement:
    """Raw timed samples and their aggregate timing metadata."""

    samples_seconds: tuple[float, ...]
    elapsed_seconds: float

    @property
    def samples_ms(self) -> tuple[float, ...]:
        return tuple(sample * 1000.0 for sample in self.samples_seconds)

    @property
    def steps(self) -> int:
        return len(self.samples_seconds)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Stable result schema for every benchmark in the framework.

    ``equivalent_items_per_second`` is the median-latency reciprocal
    (``items_per_step / median``); ``sustained_items_per_second`` divides total
    produced items by the total timed window. The two differ under jitter and
    are reported separately so consumers never mistake one for the other.
    """

    schema_version: int
    benchmark: str
    label: str
    steps: int
    warmup: int
    elapsed_seconds: float
    samples_ms: tuple[float, ...]
    p10_ms: float
    median_ms: float
    p90_ms: float
    items_per_step: float | None
    sustained_items_per_second: float | None
    equivalent_items_per_second: float | None
    config: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    unit: str = "items"

    @property
    def throughput(self) -> float | None:
        """Sustained throughput; historical spelling."""
        return self.sustained_items_per_second

    @property
    def items_per_second(self) -> float | None:
        return self.sustained_items_per_second

    @property
    def samples(self) -> tuple[float, ...]:
        """Measured durations in milliseconds (stable, serializable schema)."""
        return self.samples_ms

    @property
    def measurement(self) -> Measurement:
        return Measurement(tuple(sample / 1000.0 for sample in self.samples_ms), self.elapsed_seconds)


def run_case(
    case: PreparedCase,
    request: BenchmarkConfig,
    *,
    benchmark: str | None = None,
    label: str = "",
    config: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> BenchmarkResult:
    """Warm up, measure a prepared case, and close it deterministically."""

    primary_error: BaseException | None = None
    try:
        for _ in range(request.warmup):
            if case.before_step is not None:
                case.before_step()
            case.step()
            case.synchronize_step()
        samples: list[float] = []
        started = time.perf_counter()
        for _ in range(request.steps):
            if case.before_step is not None:
                case.before_step()
            tick = time.perf_counter_ns()
            case.step()
            case.synchronize_step()
            samples.append((time.perf_counter_ns() - tick) / 1e9)
        elapsed = time.perf_counter() - started
        items = case.measured_items_per_step
        samples_ms = tuple(sample * 1000.0 for sample in samples)
        p10, median, p90 = percentiles(samples_ms, 0.10, 0.50, 0.90)
        sustained = None if items is None else items * request.steps / elapsed
        equivalent = None if items is None or median <= 0 else items / (median / 1000.0)
        return BenchmarkResult(
            SCHEMA_VERSION,
            benchmark or "",
            label,
            request.steps,
            request.warmup,
            elapsed,
            samples_ms,
            p10,
            median,
            p90,
            items,
            sustained,
            equivalent,
            config or {},
            metadata or {},
            unit=case.unit,
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            case.close_once()
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            primary_error.add_note(f"benchmark cleanup failed: {cleanup_error!r}")


__all__ = [
    "BenchmarkConfig",
    "BenchmarkRequest",
    "BenchmarkResult",
    "Measurement",
    "PreparedCase",
    "SCHEMA_VERSION",
    "run_case",
]
