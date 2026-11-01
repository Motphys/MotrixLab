# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generic process-backed worker abstraction."""

import multiprocessing as mp
import os
from dataclasses import dataclass

from motrix_rl.worker import Worker, WorkerHandle, WorkerPlacement


@dataclass
class _WorkerConfig:
    value: int


class _ExampleWorker(Worker):
    def __init__(self, result_queue, config: _WorkerConfig) -> None:
        self.result_queue = result_queue
        self.config = config

    def run(self) -> None:
        self.result_queue.put(self.config.value)


class _SlottedWorker(Worker):
    __slots__ = ("result_queue", "value")

    def __init__(self, result_queue, value: int) -> None:
        self.result_queue = result_queue
        self.value = value

    def run(self) -> None:
        self.result_queue.put(self.value)


def test_worker_start_returns_parent_handle() -> None:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    available_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    cpus = tuple(available_cpus[:2])
    worker = _ExampleWorker(result_queue, _WorkerConfig(42))

    handle = worker.start(ctx, name="example-worker", placement=WorkerPlacement(cpus=cpus))

    assert isinstance(handle, WorkerHandle)
    assert handle.pid is not None
    assert handle.exitcode is None
    handle.join(timeout=10)

    assert handle.exitcode == 0
    assert not handle.is_alive
    assert result_queue.get(timeout=2) == 42
    assert handle.placement.cpus == cpus


def test_slotted_worker_spawn_preserves_state() -> None:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    handle = _SlottedWorker(result_queue, 42).start(ctx, name="slotted-worker")

    handle.join(timeout=10)

    assert handle.exitcode == 0
    assert result_queue.get(timeout=2) == 42
