# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the schema-driven SPSC transition ring.

These tests deliberately use a NON-FastSAC field set (privileged state
vectors, scalar int64 episode metadata, a boolean flag) to pin the public
contract the rings exist to provide: the slot layout, addressing and
backpressure follow the declared :class:`~motrix_rl.transport.schema.TransitionSchema`,
not any particular trainer's field names. A regression here is what lets a
new asynchronous backend adopt the transport without touching it.
"""

from __future__ import annotations

import pytest
import torch

from motrix_rl.transport.ring import HostTransitionRing, RingCursors
from motrix_rl.transport.schema import TransitionField, TransitionSchema

N_ENV = 4
CAPACITY = 3

SCHEMA = TransitionSchema(
    fields=(
        TransitionField("state", (71,), torch.float32),
        TransitionField("next_state", (71,), torch.float32),
        TransitionField("action", (29,), torch.float32),
        TransitionField("frame", (), torch.int64),
        TransitionField("ep_step", (), torch.int32),
        TransitionField("done", (), torch.bool),
    )
)


def _batch(tag: int) -> dict[str, torch.Tensor]:
    return {
        "state": torch.full((N_ENV, 71), float(tag)),
        "next_state": torch.full((N_ENV, 71), float(tag + 1)),
        "action": torch.full((N_ENV, 29), float(-tag)),
        "frame": torch.full((N_ENV,), tag, dtype=torch.int64),
        "ep_step": torch.full((N_ENV,), tag, dtype=torch.int32),
        "done": torch.full((N_ENV,), tag % 2 == 0),
    }


def _ring() -> HostTransitionRing:
    return HostTransitionRing(SCHEMA, CAPACITY, N_ENV)


def test_schema_rejects_duplicate_and_empty_field_sets():
    with pytest.raises(ValueError, match="unique"):
        TransitionSchema(fields=(TransitionField("a", (2,), torch.float32), TransitionField("a", (3,), torch.float32)))
    with pytest.raises(ValueError, match="at least one field"):
        TransitionSchema(fields=())


def test_ring_allocates_per_field_shapes_and_dtypes_from_schema():
    ring = _ring()
    assert set(ring.data) == set(SCHEMA.field_names)
    assert ring.data["state"].shape == (CAPACITY, N_ENV, 71)
    assert ring.data["state"].dtype == torch.float32
    assert ring.data["frame"].shape == (CAPACITY, N_ENV)
    assert ring.data["frame"].dtype == torch.int64
    assert ring.data["ep_step"].dtype == torch.int32
    assert ring.data["done"].dtype == torch.bool
    assert SCHEMA.feature_width == 71 + 71 + 29 + 3


def test_roundtrip_preserves_values_and_dtypes_by_name_across_wraps():
    ring = _ring()
    pushed = [_batch(t) for t in range(CAPACITY * 5)]
    expected = 0
    for batch in pushed:
        while not ring.push(batch):
            k, views = ring.read_span()
            _assert_views(views, pushed, expected, k)
            ring.commit_reads(k)
            expected += k
        k, views = ring.read_span()
        _assert_views(views, pushed, expected, k)
        ring.commit_reads(k)
        expected += k
    assert expected == len(pushed)
    assert ring.size() == 0
    assert ring.write_completed_count == ring.read_completed_count == len(pushed)


def _assert_views(views: dict[str, torch.Tensor], pushed: list, start: int, count: int) -> None:
    for i in range(count):
        batch = pushed[start + i]
        torch.testing.assert_close(views["state"][i], batch["state"])
        torch.testing.assert_close(views["next_state"][i], batch["next_state"])
        torch.testing.assert_close(views["action"][i], batch["action"])
        assert views["frame"][i].tolist() == batch["frame"].tolist()
        assert views["ep_step"][i].tolist() == batch["ep_step"].tolist()
        assert views["done"][i].tolist() == batch["done"].tolist()
        assert views["frame"].dtype == torch.int64
        assert views["done"].dtype == torch.bool


def test_read_span_returns_longest_contiguous_run_and_never_crosses_wrap():
    ring = _ring()
    for tag in range(CAPACITY):
        assert ring.push(_batch(tag))
    # consume one slot so the next run starts at physical slot 1
    k, _ = ring.read_span()
    ring.commit_reads(1)
    assert ring.push(_batch(CAPACITY))  # wraps to physical slot 0
    k, views = ring.read_span()
    assert k == CAPACITY - 1  # slots 1..CAPACITY-1 only; the wrap splits the run
    ring.commit_reads(k)
    k, views = ring.read_span()
    assert k == 1
    torch.testing.assert_close(views["state"][0], _batch(CAPACITY)["state"])
    ring.commit_reads(k)


def test_backpressure_blocks_at_capacity_and_resumes_after_commit():
    ring = _ring()
    for tag in range(CAPACITY):
        assert ring.push(_batch(tag))
    assert ring.is_full()
    assert not ring.push(_batch(CAPACITY))
    assert ring.write_completed_count == CAPACITY
    ring.commit_reads(1)
    assert not ring.is_full()
    assert ring.push(_batch(CAPACITY))
    assert ring.size() == CAPACITY


def test_push_rejects_field_sets_that_do_not_match_the_schema():
    ring = _ring()
    batch = _batch(0)
    del batch["done"]
    with pytest.raises(ValueError, match="do not match the schema"):
        ring.push(batch)
    with pytest.raises(ValueError, match="do not match the schema"):
        ring.push({**batch, "done": torch.zeros(N_ENV, dtype=torch.bool), "extra": torch.zeros(N_ENV)})


def test_empty_span_is_reported_without_advancing_cursors():
    ring = _ring()
    assert ring.read_span() == (0, {})
    assert not ring.has_next()
    cursors = ring.cursors
    assert cursors.write_completed_count == cursors.read_completed_count == 0


def test_cursors_are_the_channel_protocol_object():
    # One cursor pair is one collector->learner channel; the ring exposes it so
    # the parent can pre-create it and a second transport endpoint (e.g. the
    # CUDA-IPC ring of the same channel) can advance the same counters.
    cursors = RingCursors()
    ring = HostTransitionRing(SCHEMA, CAPACITY, N_ENV, cursors=cursors)
    assert ring.cursors is cursors
    assert ring.push(_batch(0))
    assert cursors.write_completed_count == 1
    assert cursors.read_completed_count == 0
    ring.commit_reads(1)
    assert cursors.read_completed_count == 1
