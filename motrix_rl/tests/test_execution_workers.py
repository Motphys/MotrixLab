# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing as mp
from queue import Empty

import pytest

from motrix_rl.execution.contracts import ExecutionCancelled
from motrix_rl.execution.workers import WorkerMessage, WorkerRuntime, WorkerSettings, worker_entry


def _runtime():
    ctx = mp.get_context("spawn")
    return WorkerRuntime("w", ctx.Queue(), ctx.Event(), ctx.Event())


def test_report_finish_and_flags():
    runtime = _runtime()
    runtime.report({"step": 1})
    runtime.finish("done")
    messages = [runtime.event_queue.get(timeout=1), runtime.event_queue.get(timeout=1)]
    assert messages == [
        WorkerMessage("report", "w", {"step": 1}),
        WorkerMessage("finish", "w", "done"),
    ]
    assert runtime.finished is True
    assert runtime.ready_reported is False


def test_ready_waits_for_release_and_can_cancel():
    runtime = _runtime()
    runtime.request_stop()
    with pytest.raises(ExecutionCancelled):
        runtime.ready()
    assert runtime.ready_reported is True
    assert runtime.event_queue.get(timeout=1).kind == "ready"


def test_return_does_not_finish_even_before_ready():
    runtime = _runtime()

    def target(rt):
        return None

    worker_entry(runtime, target, settings=WorkerSettings())
    with pytest.raises(Empty):
        runtime.event_queue.get_nowait()
    assert runtime.finished is False
    assert runtime.ready_reported is False


def test_cleanup_errors_do_not_mask_target_error():
    runtime = _runtime()

    def target(rt):
        rt.callback(lambda: (_ for _ in ()).throw(RuntimeError("cleanup one")))
        rt.callback(lambda: (_ for _ in ()).throw(RuntimeError("cleanup two")))
        raise ValueError("target failure")

    with pytest.raises(ValueError, match="target failure"):
        worker_entry(runtime, target)
    message = runtime.event_queue.get(timeout=1)
    assert message.kind == "error"
    assert message.payload["phase"] == "target"
    assert message.payload["message"] == "target failure"
    assert "ValueError: target failure" in message.payload["traceback"]
    with pytest.raises(Empty):
        runtime.event_queue.get_nowait()


def test_cleanup_error_raises_when_target_succeeds():
    runtime = _runtime()

    def target(rt):
        rt.callback(lambda: (_ for _ in ()).throw(RuntimeError("cleanup one")))
        rt.callback(lambda: (_ for _ in ()).throw(RuntimeError("cleanup two")))

    with pytest.raises(RuntimeError):
        worker_entry(runtime, target)


def _spawn_cancel_worker(queue, start_event, stop_event):
    runtime = WorkerRuntime("spawned", queue, start_event, stop_event)

    def target(rt):
        rt.request_stop()
        raise ExecutionCancelled("requested")

    worker_entry(runtime, target)


def test_spawn_cancellation_exits_cleanly():
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    start_event = ctx.Event()
    stop_event = ctx.Event()
    process = ctx.Process(target=_spawn_cancel_worker, args=(queue, start_event, stop_event))
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 0
    with pytest.raises(Empty):
        queue.get(timeout=0.2)
