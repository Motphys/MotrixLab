# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Real CUDA IPC ownership/publication across spawn, opt in with APPO_TEST_CUDA=1."""

import multiprocessing as mp
import os
import time
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event

import pytest
import torch
from motrix_rl_appo.collector import publish_chunk
from motrix_rl_appo.transport import CollectorMetadata, GpuRingConnection, chunk_schema, take_chunk

from motrix_rl.transport.ring import RingCursors
from motrix_rl.transport.schema import TransitionSchema


def _payload(schema: TransitionSchema, num_envs: int, index: int, device: str = "cpu") -> dict[str, torch.Tensor]:
    result = {}
    for field in schema.fields:
        value = torch.arange(num_envs * field.per_env_numel, device=device).reshape(schema.slot_shape(field, num_envs))
        result[field.name] = (value + index) % 2 == 0 if field.dtype == torch.bool else value.float() + 100 * index
    return result


def _ipc_producer(pipe: Connection, stop: Event, carrier: str) -> None:
    try:
        connection: GpuRingConnection = pipe.recv()
        torch.cuda.set_device(connection.slots.device)
        ring = connection.endpoint()
        source = {name: value.to(carrier) for name, value in _payload(ring.schema, ring.num_envs, 0).items()}
        if carrier == "cuda:0":
            mixed = dict(source)
            mixed["dones"] = mixed["dones"].cpu()
            with pytest.raises(ValueError, match="must all be on"):
                ring.push(mixed)
            if torch.cuda.device_count() > 1:
                mixed["dones"] = source["dones"].to("cuda:1")
                with pytest.raises(ValueError, match="must all be on"):
                    ring.push(mixed)

        mutation_stream = torch.cuda.Stream()

        def publish(index: int) -> None:
            # A delayed source write exposes cursor publication before copies
            # complete, rather than relying on incidental GPU scheduling.
            if carrier == "cuda:0":
                torch.cuda._sleep(10_000_000)
            for name, value in _payload(ring.schema, ring.num_envs, index, carrier).items():
                source[name].copy_(value)
            assert publish_chunk(ring, source, stop, 20)
            # Rewrite on an independent stream: same-stream ordering must not
            # hide a publish_chunk return that precedes D2D copy completion.
            if carrier == "cuda:0":
                with torch.cuda.stream(mutation_stream):
                    for value in source.values():
                        value.zero_()
                mutation_stream.synchronize()
            else:
                for value in source.values():
                    value.zero_()

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Fill both slots and verify true backpressure without data writes.
            for index in range(connection.capacity):
                publish(index)
            before = ring.slots.clone()
            assert not ring.push(source)
            torch.testing.assert_close(ring.slots, before)
            pipe.send("full")
            for index in range(connection.capacity, 9):
                publish(index)
        # Last publication must complete with no further push/poll from collect.
        pipe.send(("published", ring.write_completed_count))
        assert stop.wait(20)
    finally:
        pipe.close()


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("APPO_TEST_CUDA") != "1", reason="set APPO_TEST_CUDA=1 for real CUDA IPC")
@pytest.mark.parametrize("carrier", ["cpu", "cuda:0"])
def test_cuda_spawn_wrap_backpressure_owned_bool_and_dims(carrier: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime required")
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    parent, child = ctx.Pipe()
    schema = chunk_schema(3, CollectorMetadata(2, 4, 1))
    slots = torch.empty((2, 3, schema.feature_width), device="cuda:0")
    connection = GpuRingConnection(RingCursors(), slots, schema, 2, 3)
    ring = connection.endpoint()
    process = ctx.Process(target=_ipc_producer, args=(child, stop, carrier))
    process.start()
    child.close()
    saved = []
    try:
        parent.send(connection)
        assert parent.poll(30), "producer did not fill ring"
        assert parent.recv() == "full"
        assert ring.size() == 2
        for index in range(9):
            deadline = time.monotonic() + 20
            while not ring.has_next():
                assert process.is_alive(), f"producer exited {process.exitcode}"
                assert time.monotonic() < deadline, "IPC chunk was not published"
                stop.wait(0.001)
            owned = take_chunk(ring, torch.device("cuda:0"))
            expected = _payload(schema, 3, index)
            for field in schema.fields:
                value = expected[field.name]
                if field.name != "last_critic_obs":
                    value = value.transpose(0, 1).contiguous()
                assert owned[field.name].dtype == field.dtype
                assert owned[field.name].is_contiguous()
                torch.testing.assert_close(owned[field.name].cpu(), value)
            saved.append(owned)
        assert parent.poll(20)
        assert parent.recv() == ("published", 9)
        # All retained batches survive repeated physical-slot reuse.
        for index, owned in enumerate(saved):
            for name, expected in _payload(schema, 3, index).items():
                if name != "last_critic_obs":
                    expected = expected.transpose(0, 1).contiguous()
                torch.testing.assert_close(owned[name].cpu(), expected)
    finally:
        stop.set()
        process.join(20)
        if process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
    assert process.exitcode == 0
    process.close()
