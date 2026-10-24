# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Control protocol states are distinct from algorithm report contents."""

from queue import Queue
from types import SimpleNamespace

import pytest

from motrix_rl.execution.messages import MessageInbox


def message(kind, worker_id="actor", payload=None):
    return SimpleNamespace(kind=kind, worker_id=worker_id, payload=payload)


def test_ready_finish_error_and_opaque_reports_are_not_conflated():
    queue = Queue()
    inbox = MessageInbox(queue, {"actor"})
    for item in (
        message("ready"),
        message("report", payload={"batches": 2}),
        message("finish", payload={"batches": 3}),
    ):
        queue.put(item)
    inbox.drain()
    assert inbox.ready == {"actor"}
    assert inbox.finished == {"actor"}
    assert inbox.take_reports() == {"actor": ({"batches": 2}, {"batches": 3})}
    assert inbox.take_reports() == {}
    queue.put(message("error", payload="late teardown failure"))
    inbox.drain()
    assert inbox.errors[0].payload == "late teardown failure"


@pytest.mark.parametrize(
    "items",
    [
        [message("ready", worker_id="unknown")],
        [message("finish")],
        [message("ready"), message("ready")],
        [message("ready"), message("finish"), message("finish")],
        [message("invalid")],
    ],
)
def test_protocol_violations_are_failures(items):
    queue = Queue()
    for item in items:
        queue.put(item)
    with pytest.raises(RuntimeError):
        MessageInbox(queue, {"actor"}).drain()
