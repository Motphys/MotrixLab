# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Ownership/FIFO contracts independent of FastSAC's transition schema."""

import pytest
import torch

from motrix_rl.transport.cuda_ring import CudaTensorRing
from motrix_rl.transport.ring import RingCursors, SharedTensorRing, TensorField


def test_unroll_layout_preserves_fields_backpressure_and_views_across_wraps():
    fields = (
        TensorField("observations", (3, 2, 4)),
        TensorField("behavior_logp", (3, 2)),
        TensorField("policy_versions", (3, 2), torch.int64),
        TensorField("valid", (3, 2), torch.bool),
    )
    ring = SharedTensorRing(2, fields)
    assert ring.read_span() == (0, ())
    for base in range(0, 8, 2):
        batches = [tuple(torch.full(f.shape, base + i, dtype=f.dtype) for f in fields) for i in range(2)]
        for batch in batches:
            assert ring.push(*batch)
        assert ring.is_full()
        assert not ring.push(*batches[0])
        count, views = ring.read_span()
        assert count == 2
        assert views[0].untyped_storage().data_ptr() == ring.tensors[0].untyped_storage().data_ptr()
        # Reading does not release storage; a stalled consumer cannot be overwritten.
        assert not ring.push(*batches[0])
        for i, batch in enumerate(batches):
            for view, expected in zip(views, batch):
                torch.testing.assert_close(view[i], expected)
        ring.commit_reads(count)
        assert ring.size() == 0
    assert ring.write_completed_count == ring.read_completed_count == 8


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_commit_reads_rejects_invalid_counts(count):
    ring = SharedTensorRing(2, (TensorField("value", (), torch.int64),))
    with pytest.raises(ValueError, match="commit"):
        ring.commit_reads(count)


@pytest.mark.parametrize("machine", ["aarch64", "arm64"])
def test_shared_transport_rejects_unsupported_memory_ordering(monkeypatch, machine):
    monkeypatch.setattr("motrix_rl.transport.ring.platform.machine", lambda: machine)
    with pytest.raises(RuntimeError, match="x86-64 TSO"):
        RingCursors()


@pytest.mark.parametrize("bad", [torch.zeros(1), torch.zeros(2, dtype=torch.int64)])
def test_field_layout_rejection_does_not_publish_partial_slot(bad):
    ring = SharedTensorRing(2, (TensorField("value", (2,)),))
    with pytest.raises(ValueError, match="shape/dtype"):
        ring.push(bad)
    assert ring.write_completed_count == 0
    assert ring.push(torch.ones(2))
    _, (values,) = ring.read_span()
    torch.testing.assert_close(values[0], torch.ones(2))


def test_commit_reads_rejects_more_than_readable():
    ring = SharedTensorRing(2, (TensorField("value", (), torch.int64),))
    ring.push(torch.tensor(1))
    with pytest.raises(ValueError, match="readable"):
        ring.commit_reads(2)


def test_read_span_stops_at_physical_wrap_and_keeps_fifo():
    ring = SharedTensorRing(3, (TensorField("sequence", (), torch.int64),))
    for value in range(3):
        assert ring.push(torch.tensor(value))
    ring.commit_reads(2)
    assert ring.push(torch.tensor(3))
    assert ring.push(torch.tensor(4))
    count, (values,) = ring.read_span()
    assert count == 1 and values.tolist() == [2]
    ring.commit_reads(count)
    count, (values,) = ring.read_span()
    assert count == 2 and values.tolist() == [3, 4]


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA slots require a GPU")
def test_cuda_unroll_slots_publish_and_release_after_copy_completion():
    cursors = RingCursors()
    slots = torch.zeros((2, 3, 2, 5), device="cuda")
    producer, consumer = CudaTensorRing(cursors, slots), CudaTensorRing(cursors, slots)
    with pytest.raises(ValueError, match="shape and dtype"):
        producer.push(torch.zeros(5, device="cuda"))
    assert producer.write_issued_count == 0
    for cycle in range(3):
        source = torch.full((3, 2, 5), float(cycle), device="cuda")
        assert producer.push(source)
        assert producer.push(source)
        assert not producer.push(source)
        torch.cuda.synchronize()
        producer.size()  # The producer publishes its own completed events.
        count, view = consumer.read_span()
        assert count == 2
        owned = view.clone()
        with pytest.raises(ValueError, match="readable"):
            consumer.commit_reads(count + 1)
        with pytest.raises(ValueError, match="nonnegative"):
            consumer.commit_reads(-1)
        consumer.commit_reads(count)
        with pytest.raises(ValueError, match="readable"):
            consumer.commit_reads(1)
        torch.cuda.synchronize()
        consumer.size()  # Release only after the clone's reads have landed.
        assert not producer.is_full()
        torch.testing.assert_close(owned, source.expand_as(owned))
    assert cursors.write_completed_count == cursors.read_completed_count == 6
