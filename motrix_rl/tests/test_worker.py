# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Tests for the generic process-backed worker abstraction."""

import multiprocessing as mp
import os
import random
from dataclasses import dataclass

import numpy as np
import pytest

from motrix_rl.worker import Worker, WorkerHandle, WorkerPlacement, WorkerRuntime


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


class _RuntimeProbe(Worker):
    def __init__(self, results) -> None:
        self.results = results

    def run(self) -> None:
        import torch

        self.results.put(
            (
                torch.get_num_threads(),
                torch.initial_seed(),
                torch.rand(4).tolist(),
                [random.random() for _ in range(4)],
                np.random.random(4).tolist(),
            )
        )


@pytest.mark.integration
@pytest.mark.parametrize("seed", [0, 23, 2**40 + 23])
def test_worker_runtime_is_applied_before_task_and_does_not_change_parent(seed: int) -> None:
    import torch

    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    parent_threads = torch.get_num_threads()
    parent_rng = torch.get_rng_state().clone()
    parent_python_rng = random.getstate()
    parent_numpy_rng = np.random.get_state()
    handles = []
    try:
        for _ in range(2):
            handles.append(_RuntimeProbe(results).start(ctx, runtime=WorkerRuntime(torch_num_threads=1, seed=seed)))
        observed = [results.get(timeout=20) for _ in handles]
        for handle in handles:
            handle.join(timeout=10)
            assert handle.exitcode == 0
        assert observed[0] == observed[1]
        assert observed[0][:2] == (1, seed)
        expected = torch.rand(4, generator=torch.Generator().manual_seed(seed)).tolist()
        assert observed[0][2] == expected
        python_rng = random.Random(seed)
        assert observed[0][3] == [python_rng.random() for _ in range(4)]
        assert observed[0][4] == np.random.RandomState(seed % (2**32)).random(4).tolist()
        assert torch.get_num_threads() == parent_threads
        assert torch.equal(torch.get_rng_state(), parent_rng)
        assert random.getstate() == parent_python_rng
        numpy_rng = np.random.get_state()
        assert numpy_rng[0] == parent_numpy_rng[0]
        np.testing.assert_array_equal(numpy_rng[1], parent_numpy_rng[1])
        assert numpy_rng[2:] == parent_numpy_rng[2:]
    finally:
        for handle in handles:
            if handle.is_alive:
                handle.terminate()
                handle.join(timeout=5)
            handle.process.close()
        results.close()
        results.join_thread()


def test_worker_runtime_rejects_nonpositive_thread_budget() -> None:
    with pytest.raises(ValueError, match="positive"):
        WorkerRuntime(torch_num_threads=0)


def test_worker_start_returns_parent_handle() -> None:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    available_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    cpus = tuple(available_cpus[:2])
    worker = _ExampleWorker(result_queue, _WorkerConfig(42))

    handle = worker.start(ctx, name="example-worker", placement=WorkerPlacement(cpus=cpus))

    assert isinstance(handle, WorkerHandle)
    assert handle.pid is not None
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
