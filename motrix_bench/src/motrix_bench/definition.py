# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-neutral contract between the dispatcher and a benchmark.

A spec wires three concerns without knowing what the benchmark measures:

* ``configure`` / ``build_config`` — the CLI adapter: parsed arguments become
  the benchmark's own typed config ``C``.
* ``prepare`` — the workload contract: a typed config becomes a measurable
  :class:`~motrix_bench.runner.PreparedCase`.
* ``run`` — optional orchestration for benchmarks whose execution exceeds one
  prepared case (batch-size sweeps, NUMA sharding, profiling). It receives the
  typed config and returns structured results; the dispatcher renders them.

The spec never mentions environments, robots, simulators, or providers:
those concepts exist only inside concrete benchmark configs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from .runner import BenchmarkResult, PreparedCase

C = TypeVar("C")


@dataclass(frozen=True, slots=True)
class BenchmarkSpec(Generic[C]):
    name: str
    description: str
    configure: Callable[[argparse.ArgumentParser], None]
    build_config: Callable[[argparse.Namespace], C]
    prepare: Callable[[C], PreparedCase]
    run: Callable[[C], Sequence[BenchmarkResult]] | None = None


__all__ = ["BenchmarkSpec"]
