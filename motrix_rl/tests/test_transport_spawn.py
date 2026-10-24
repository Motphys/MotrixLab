# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Real spawn ownership tests for tensor transport, independent of SAC fields."""

import queue
import time
import traceback

import pytest
import torch
import torch.multiprocessing as mp

from motrix_rl.transport.cuda_ring import CudaTensorRing
from motrix_rl.transport.ring import RingCursors, SharedTensorRing, TensorField

_CAPACITY = 2
_COUNT = 13
_UNROLL_SHAPE = (3, 2, 4)


def _wait(predicate, deadline):
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("transport worker failed to make progress")
        time.sleep(0.001)


def _values(index, cuda):
    # An unroll layout, not FastSAC's observation/action transition schema.
    values = torch.arange(24, dtype=torch.float32).reshape(_UNROLL_SHAPE) + 100 * index
    if cuda:
        return (values.cuda(),)
    return values, torch.full((3, 2), index % 2, dtype=torch.bool)


def _endpoint(resource, cuda):
    return CudaTensorRing(*resource) if cuda else resource


def _producer(resource, cuda, filled, held, checked, results):
    try:
        torch.set_num_threads(1)
        ring = _endpoint(resource, cuda)
        deadline = time.monotonic() + 30
        # Keep CUDA copy sources alive through completion, including retry loops.
        sources = [_values(index, cuda) for index in range(_COUNT)]
        for index, values in enumerate(sources):
            if index == _CAPACITY:
                _wait(held.is_set, deadline)
                assert ring.is_full()
                assert not ring.push(*values)
                checked.set()
            _wait(lambda: ring.push(*values), deadline)
            if index == _CAPACITY - 1:
                _wait(lambda: ring.write_completed_count == _CAPACITY, deadline)
                filled.set()
        _wait(lambda: ring.write_completed_count == _COUNT, deadline)
        _wait(lambda: ring.read_completed_count == _COUNT, deadline)
        results.put(("producer", None))
    except BaseException:
        results.put(("producer", traceback.format_exc()))
        raise


def _consumer(resource, cuda, filled, held, checked, results):
    try:
        torch.set_num_threads(1)
        ring = _endpoint(resource, cuda)
        deadline = time.monotonic() + 30
        _wait(filled.is_set, deadline)
        consumed = 0
        while consumed < _COUNT:
            _wait(ring.has_next, deadline)
            count, views = ring.read_span()
            columns = (views,) if cuda else views
            if consumed == 0:
                # Neither taking a view nor copying it implicitly releases slots.
                held.set()
                _wait(checked.is_set, deadline)
            owned = tuple(column.clone() for column in columns)
            # CUDA acknowledgements must follow reads on the same stream.
            ring.commit_reads(count)
            for offset in range(count):
                expected = _values(consumed + offset, False)
                for actual, value in zip(owned, expected):
                    torch.testing.assert_close(actual[offset].cpu(), value)
            consumed += count
        # Flush event-proven acknowledgements even after the last data read.
        _wait(lambda: ring.read_completed_count == _COUNT, deadline)
        results.put(("consumer", None))
    except BaseException:
        results.put(("consumer", traceback.format_exc()))
        raise


def _exercise_spawn(resource, cuda):
    ctx = mp.get_context("spawn")
    filled, held, checked = (ctx.Event() for _ in range(3))
    results = ctx.Queue()
    processes = [
        ctx.Process(target=target, args=(resource, cuda, filled, held, checked, results))
        for target in (_producer, _consumer)
    ]
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        received = {}
        for _ in processes:
            try:
                name, error = results.get(timeout=45)
            except queue.Empty:
                pytest.fail(f"transport workers timed out; exits={[p.exitcode for p in started]}")
            assert error is None, error
            received[name] = error
        assert set(received) == {"producer", "consumer"}
        for process in started:
            process.join(timeout=10)
            assert not process.is_alive()
            assert process.exitcode == 0
    finally:
        for process in started:
            if process.is_alive():
                process.terminate()
        for process in started:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        results.close()
        results.join_thread()


@pytest.mark.integration
def test_spawn_shared_ring_preserves_unrolls_and_explicit_release():
    ring = SharedTensorRing(
        _CAPACITY,
        (TensorField("unroll", _UNROLL_SHAPE), TensorField("mask", (3, 2), torch.bool)),
    )
    _exercise_spawn(ring, False)
    assert ring.write_completed_count == ring.read_completed_count == _COUNT


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA IPC requires a GPU")
def test_spawn_cuda_ring_preserves_unrolls_and_event_acknowledgements():
    cursors = RingCursors()
    # The exporting parent owns this allocation until both IPC clients exit.
    slots = torch.zeros((_CAPACITY, *_UNROLL_SHAPE), device="cuda")
    _exercise_spawn((cursors, slots), True)
    assert cursors.write_completed_count == cursors.read_completed_count == _COUNT
