# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Actual CUDA-IPC mixed-scope transport, including a spawned producer.

Run with a CUDA runtime: python -m pytest motrix_rl/tests/test_transport_global_gpu.py -m integration
"""

import multiprocessing as mp
import time

import pytest
import torch

from motrix_rl.transport.gpu_ipc_ring import GpuIpcTransitionRing
from motrix_rl.transport.ring import RingCursors
from motrix_rl.transport.schema import TransitionField, TransitionSchema

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


def _schema() -> TransitionSchema:
    return TransitionSchema(
        (
            TransitionField("state", (2,), torch.float32),
            TransitionField("matrix", (2, 3), torch.float32, scope="global"),
            TransitionField("scalar", (), torch.int64, scope="global"),
        )
    )


def _batch(step: int, num_envs: int) -> dict[str, torch.Tensor]:
    return {
        "state": torch.full((num_envs, 2), float(step)),
        "matrix": torch.arange(6, dtype=torch.float32).reshape(2, 3) + step,
        "scalar": torch.tensor(step, dtype=torch.int64),
    }


def _producer(cursors: RingCursors, slots: torch.Tensor, globals_: torch.Tensor, ready, release) -> None:
    torch.set_num_threads(2)
    ring = GpuIpcTransitionRing(cursors, slots, _schema(), 3, 2, global_slots=globals_)
    deadline = time.monotonic() + 60
    for step in range(9):
        while not ring.push(_batch(step, 2)):
            if time.monotonic() > deadline:
                raise TimeoutError("producer remained backpressured")
            time.sleep(0.001)
        # The producer must poll its own events to publish completed H2D.
        torch.cuda.synchronize()
        ring.write_completed_count
    ready.set()
    if not release.wait(60):
        raise TimeoutError("consumer did not release CUDA IPC mapping")


def test_spawned_gpu_ipc_preserves_global_native_shapes_across_ring_wraps() -> None:
    torch.set_num_threads(2)
    ctx = mp.get_context("spawn")
    cursors = RingCursors()
    slots = torch.zeros(3, 2, _schema().feature_width, device="cuda")
    globals_ = torch.zeros(3, _schema().global_feature_width, device="cuda")
    ring = GpuIpcTransitionRing(cursors, slots, _schema(), 3, 2, global_slots=globals_)
    ready, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_producer, args=(cursors, slots, globals_, ready, release))
    child.start()
    expected = 0
    snapshots = []
    deadline = time.monotonic() + 60
    try:
        while expected < 9:
            if not ring.has_next():
                if time.monotonic() > deadline or child.exitcode not in (None, 0):
                    raise AssertionError(f"producer failed or timed out: {child.exitcode}")
                time.sleep(0.001)
                continue
            count, views = ring.read_span()
            assert views["state"].shape == (count, 2, 2)
            assert views["matrix"].shape == (count, 2, 3)
            assert views["scalar"].shape == (count,)
            # Clone on-device before commit. The commit event must protect
            # reads of BOTH fused buffers against the producer's next wrap.
            owned = {name: value.clone() for name, value in views.items()}
            ring.commit_reads(count)
            for offset in range(count):
                reference = _batch(expected + offset, 2)
                for name, value in reference.items():
                    torch.testing.assert_close(owned[name][offset].cpu(), value.to(torch.float32))
            snapshots.append((expected, owned))
            expected += count
            torch.cuda.synchronize()
            ring.read_completed_count
        assert ready.wait(60)
        for start, owned in snapshots:
            for offset in range(owned["scalar"].shape[0]):
                for name, value in _batch(start + offset, 2).items():
                    torch.testing.assert_close(owned[name][offset].cpu(), value.to(torch.float32))
        assert ring.read_payload_span(0) == []
    finally:
        release.set()
        child.join(10)
        if child.is_alive():
            child.terminate()
            child.join(10)
    assert child.exitcode == 0


@pytest.mark.parametrize("count", [2])
def test_gpu_global_span_preserves_matrix_and_scalar_values(count: int) -> None:
    """A contiguous global fused slice reshapes to native ``(count, *shape)``."""
    schema = _schema()
    cursors = RingCursors()
    slots = torch.zeros(3, 2, schema.feature_width, device="cuda")
    globals_ = torch.zeros(3, schema.global_feature_width, device="cuda")
    owner = GpuIpcTransitionRing(cursors, slots, schema, 3, 2, global_slots=globals_)
    receiver = GpuIpcTransitionRing(cursors, slots, schema, 3, 2, global_slots=globals_)
    first, second = _batch(11, 2), _batch(12, 2)
    assert owner.push(first)
    assert owner.push(second)
    torch.cuda.synchronize()
    assert owner.write_completed_count == 2
    read_count, views = receiver.read_span()
    assert read_count == count
    assert views["matrix"].shape == (2, 2, 3)
    assert views["scalar"].shape == (2,)
    torch.testing.assert_close(views["matrix"].cpu(), torch.stack((first["matrix"], second["matrix"])))
    torch.testing.assert_close(views["scalar"].cpu(), torch.tensor([11.0, 12.0]))
    receiver.commit_reads(read_count)
    torch.cuda.synchronize()


def test_gpu_global_fields_require_matching_shared_allocation() -> None:
    schema = _schema()
    cursors = RingCursors()
    slots = torch.zeros(3, 2, schema.feature_width, device="cuda")
    with pytest.raises(ValueError, match="global slots"):
        GpuIpcTransitionRing(cursors, slots, schema, 3, 2)
    with pytest.raises(ValueError, match="shape"):
        GpuIpcTransitionRing(cursors, slots, schema, 3, 2, global_slots=torch.zeros(3, 1, device="cuda"))
