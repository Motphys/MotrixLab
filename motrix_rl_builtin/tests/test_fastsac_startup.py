# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""FastSAC boot protocol delegates cancellation to public startup primitives."""

from queue import Queue
from types import SimpleNamespace

import pytest
from motrix_rl_builtin.fastsac.async_impl.transport.handshake import StartupHandshake

from motrix_rl.execution.contracts import ExecutionCancelled


class Context:
    queue = staticmethod(lambda **kwargs: Queue(**kwargs))


@pytest.mark.parametrize("kind", ["weight", "ring"])
def test_slot_wait_is_cancelled_before_resource_arrives(kind):
    handshake = StartupHandshake(Context(), 1)
    receive = handshake.await_weight_slots if kind == "weight" else handshake.await_ring_slots
    with pytest.raises(ExecutionCancelled):
        receive(0, SimpleNamespace(stop=True))


def test_host_ring_none_and_weight_slots_keep_distinct_protocols():
    handshake = StartupHandshake(Context(), 1)
    slots = [object(), object()]
    handshake.ship_weight_slots(0, slots)
    handshake.ship_ring_slots(0, None)
    assert handshake.await_weight_slots(0) is slots
    assert handshake.await_ring_slots(0) is None
