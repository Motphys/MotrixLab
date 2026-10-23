# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""CPU spawn roundtrip for the algorithm-neutral execution primitives."""

from __future__ import annotations

import queue
import time
import traceback

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from motrix_rl.execution.processes import ProcessSupervisor
from motrix_rl.transport.ring import SharedTensorRing, TensorField
from motrix_rl.transport.snapshot import (
    HostSnapshotReceiver,
    HostSnapshotSender,
    SnapshotTensorSpec,
    TensorSnapshotShared,
)

_COUNT = 7
_CAPACITY = 2


def _wait(predicate, deadline: float) -> None:
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("execution transport roundtrip timed out")
        time.sleep(0.001)


def _make_channel():
    shared = TensorSnapshotShared(
        {
            "generation": SnapshotTensorSpec((1,), torch.int64, host_scalar=True),
            "gain": SnapshotTensorSpec((1,), torch.float32),
        }
    )
    # The sender owns these shared parameter slots; only the slots are shipped
    # to the collector, while the sender itself is constructed in the learner.
    sender = HostSnapshotSender(shared, 1)
    return shared, sender.params


def _collector(ring, shared, params, stop, done, results):
    try:
        actor = nn.Linear(1, 1, bias=False)
        receiver = HostSnapshotReceiver(shared, params)
        generation = torch.zeros((1,), dtype=torch.int64)
        gain = torch.zeros((1,), dtype=torch.float32)
        deadline = time.monotonic() + 20
        observed = []

        for sequence in range(_COUNT):

            def snapshot_ready():
                receiver.maybe_load(actor, {"generation": generation, "gain": gain})
                return receiver.loaded_version >= sequence + 1

            _wait(snapshot_ready, deadline)
            parameter = float(actor.weight.detach().reshape(-1)[0])
            assert parameter == pytest.approx(float(sequence))
            assert int(generation.item()) == sequence
            assert float(gain.item()) == pytest.approx(float(sequence) + 0.5)
            payload = (
                torch.tensor(sequence, dtype=torch.int64),
                torch.tensor(parameter, dtype=torch.float32),
            )
            _wait(lambda: ring.push(*payload), deadline)
            observed.append((sequence, parameter, int(generation.item())))

        results.put(("collector", observed, None))
        done.set()
        _wait(stop.is_set, deadline)
    except BaseException:
        results.put(("collector", None, traceback.format_exc()))
        stop.set()
        raise


def _learner(ring, shared, params, stop, done, results):
    try:
        sender = HostSnapshotSender(shared, 1)
        sender.params = params
        flat = torch.tensor([0.0], dtype=torch.float32)
        aux = {
            "generation": torch.tensor([0], dtype=torch.int64),
            "gain": torch.tensor([0.5], dtype=torch.float32),
        }
        sender.publish_flat(flat, aux)
        deadline = time.monotonic() + 20
        consumed = []

        while len(consumed) < _COUNT:
            _wait(ring.has_next, deadline)
            count, views = ring.read_span()
            sequences, parameters = (view.clone() for view in views)
            ring.commit_reads(count)
            for sequence, parameter in zip(sequences.tolist(), parameters.tolist()):
                expected = len(consumed)
                assert sequence == expected
                assert parameter == pytest.approx(float(expected))
                consumed.append(sequence)
                flat.fill_(float(expected + 1))
                aux["generation"].fill_(expected + 1)
                aux["gain"].fill_(float(expected + 1) + 0.5)
                sender.publish_flat(flat, aux)

        results.put(("learner", consumed, None))
        done.set()
        _wait(stop.is_set, deadline)
    except BaseException:
        results.put(("learner", None, traceback.format_exc()))
        stop.set()
        raise


@pytest.mark.integration
def test_spawn_execution_transport_roundtrip_preserves_ownership_and_versions():
    ctx = mp.get_context("spawn")
    ring = SharedTensorRing(
        _CAPACITY,
        (
            TensorField("sequence", (), torch.int64),
            TensorField("parameter", (), torch.float32),
        ),
    )
    shared, params = _make_channel()
    stop, done = ctx.Event(), ctx.Event()
    results = ctx.Queue()
    collector = ctx.Process(target=_collector, args=(ring, shared, params, stop, done, results))
    learner = ctx.Process(target=_learner, args=(ring, shared, params, stop, done, results))

    received = {}
    try:
        with ProcessSupervisor(stop.set, grace_s=2, terminate_s=1, kill_s=1) as supervisor:
            supervisor.start(collector)
            supervisor.start(learner)
            deadline = time.monotonic() + 25
            while len(received) < 2:
                supervisor.check_exits(allow_clean_exit=lambda process: done.is_set())
                try:
                    name, payload, error = results.get(timeout=0.05)
                except queue.Empty:
                    if time.monotonic() >= deadline:
                        pytest.fail(f"workers timed out; exits={[collector.exitcode, learner.exitcode]}")
                    continue
                assert error is None, error
                received[name] = payload
            assert done.is_set()

        assert received["collector"] == [(i, float(i), i) for i in range(_COUNT)]
        assert received["learner"] == list(range(_COUNT))
        assert ring.write_completed_count == ring.read_completed_count == _COUNT
        assert collector.exitcode == learner.exitcode == 0
    finally:
        results.close()
        results.join_thread()
