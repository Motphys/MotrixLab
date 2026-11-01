# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generic process-backed worker abstraction."""

import multiprocessing as mp
import os
from dataclasses import dataclass

import pytest

from motrix_rl.worker import Worker, WorkerPlacement


@dataclass
class _WorkerConfig:
    value: int


class _SlottedWorker(Worker):
    __slots__ = ("result_queue", "value")

    def __init__(self, result_queue, value: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.result_queue = result_queue
        self.value = value

    def run(self) -> None:
        self.result_queue.put((self.name, self.value))


class _ExampleWorker(Worker):
    def __init__(self, result_queue, config: _WorkerConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.result_queue = result_queue
        self.config = config

    def run(self) -> None:
        self.result_queue.put((self.name, self.config.value, self.placement.cpus))


def test_worker_requires_start_before_lifecycle_operations() -> None:
    worker = _ExampleWorker(None, _WorkerConfig(1))

    with pytest.raises(RuntimeError, match="has not been started"):
        worker.join()
    with pytest.raises(RuntimeError, match="has not been started"):
        worker.terminate()
    with pytest.raises(RuntimeError, match="has not been started"):
        worker.kill()


def test_worker_spawn_runs_subclass_method_and_exposes_lifecycle() -> None:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    available_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    cpus = tuple(available_cpus[:2])
    worker = _ExampleWorker(
        result_queue,
        _WorkerConfig(42),
        name="example-worker",
        placement=WorkerPlacement(cpus=cpus),
    )

    assert worker.pid is None
    assert worker.exitcode is None
    assert not worker.is_alive

    worker.start(ctx)
    worker.join(timeout=10)

    assert worker.exitcode == 0
    assert not worker.is_alive
    assert result_queue.get(timeout=2) == ("example-worker", 42, cpus)

    with pytest.raises(RuntimeError, match="already started"):
        worker.start(ctx)


def test_slotted_worker_spawn_preserves_state() -> None:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    worker = _SlottedWorker(result_queue, 42, name="slotted-worker")

    worker.start(ctx)
    worker.join(timeout=10)

    assert worker.exitcode == 0
    assert result_queue.get(timeout=2) == ("slotted-worker", 42)


def test_worker_state_excludes_parent_process() -> None:
    worker = _ExampleWorker(None, _WorkerConfig(1))
    state = worker.__getstate__()

    assert state["_process"] is None
    assert state["_started"] is False
