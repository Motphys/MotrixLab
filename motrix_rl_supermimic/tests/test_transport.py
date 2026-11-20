# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Collector connections carry shared storage rather than live endpoints."""

import multiprocessing as mp
from multiprocessing.connection import Connection

import pytest
import torch
from motrix_rl_supermimic.replay import transition_schema
from motrix_rl_supermimic.transport import GpuRingConnection, HostRingConnection, connect_producer

from motrix_rl.transport.gpu_ipc_ring import GpuIpcTransitionRing
from motrix_rl.transport.ring import HostTransitionRing, RingCursors


def _row() -> dict[str, torch.Tensor]:
    schema = transition_schema(state_dim=2, context_dim=0, action_dim=1)
    return {field.name: torch.full(schema.slot_shape(field, 3), 1, dtype=field.dtype) for field in schema.fields}


def test_host_connection_attaches_algorithm_transition_fields() -> None:
    schema = transition_schema(state_dim=2, context_dim=0, action_dim=1)
    owner = HostTransitionRing(schema, 2, 3)
    connection = HostRingConnection(schema, 2, 3, owner.cursors, owner.data)
    producer = connect_producer(connection)
    assert isinstance(producer, HostTransitionRing)
    assert producer.data is owner.data
    row = _row()
    for _ in range(4):
        assert producer.push(row)
        count, views = owner.read_span()
        assert count == 1
        for name, expected in row.items():
            torch.testing.assert_close(views[name][0], expected)
        owner.commit_reads(1)
    assert producer.read_completed_count == owner.write_completed_count == 4


def _gpu_descriptor_producer(channel: Connection) -> None:
    connection = channel.recv()
    producer = connect_producer(connection)
    assert producer.push(_row())
    torch.cuda.synchronize()
    assert producer.write_completed_count == 1
    channel.send("published")
    assert channel.recv() == "release"
    channel.close()


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA IPC")
def test_spawned_gpu_connection_serializes_shared_storage_without_endpoint_state() -> None:
    schema = transition_schema(state_dim=2, context_dim=0, action_dim=1)
    cursors = RingCursors()
    slots = torch.zeros(2, 3, schema.feature_width, device="cuda")
    consumer = GpuIpcTransitionRing(cursors, slots, schema, 2, 3)
    connection = GpuRingConnection(schema, slots, cursors)
    parent, child_channel = mp.get_context("spawn").Pipe()
    child = mp.get_context("spawn").Process(target=_gpu_descriptor_producer, args=(child_channel,))
    child.start()
    child_channel.close()
    try:
        parent.send(connection)
        assert parent.poll(60), "Spawned producer did not publish the GPU slot"
        assert parent.recv() == "published"
        count, views = consumer.read_span()
        assert count == 1
        for name, expected in _row().items():
            torch.testing.assert_close(views[name][0].cpu(), expected.to(views[name].dtype))
        consumer.commit_reads(count)
        torch.cuda.synchronize()
        assert consumer.read_completed_count == 1
        parent.send("release")
    finally:
        parent.close()
        child.join(10)
        if child.is_alive():
            child.terminate()
            child.join(10)
    assert child.exitcode == 0
