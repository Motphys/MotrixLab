# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import argparse
import multiprocessing as mp

import pytest

from bench.env_benchmark import runner as duration
from motrix_rl.numa import NodePlacement


def test_window_counts_only_completions_within_deadline():
    times = iter([100, 120, 125, 150, 155, 201, 202])
    calls = []
    result = duration.measure_window(lambda: calls.append(1), 8, 100, 200, lambda: next(times))
    assert len(calls) == 3
    assert result["completed_calls"] == 2
    assert result["completed_env_steps"] == 16
    assert result["last_counted_completion_ns"] == 150


def test_deadline_completion_counts_but_no_new_call_starts():
    times = iter([100, 200, 200])
    result = duration.measure_window(lambda: None, 3, 100, 200, lambda: next(times))
    assert result["completed_env_steps"] == 3


def test_late_worker_does_not_extend_window():
    result = duration.measure_window(lambda: pytest.fail("must not step"), 8, 100, 200, lambda: 205)
    assert result["completed_calls"] == 0
    assert result["actual_start_offset_ns"] == 105


def test_worker_waits_for_common_start(monkeypatch):
    times = iter([80, 100, 110, 120])
    sleeps = []
    monkeypatch.setattr(duration.time, "sleep", sleeps.append)
    result = duration.measure_window(lambda: None, 2, 100, 120, lambda: next(times))
    assert sleeps
    assert result["actual_start_ns"] == 100
    assert result["completed_calls"] == 1


def fake_worker(connection, config):
    """Importable spawn target that exercises the real pipe protocol without sim."""
    try:
        for _name in [config["env"]] + ([config["compare_env"]] if config["compare_env"] else []):
            connection.send({"kind": "ready"})
            window = connection.recv()
            calls = config["worker_index"] + 2
            connection.send(
                {
                    "kind": "result",
                    "worker": {
                        "completed_calls": calls,
                        "completed_env_steps": calls * config["worker_num_envs"],
                        "num_envs": config["worker_num_envs"],
                        "window": window,
                    },
                }
            )
        connection.recv()
    finally:
        connection.close()


def failing_worker(connection, config):
    if config["worker_index"] == 1:
        connection.close()
        raise RuntimeError("fake worker failure")
    fake_worker(connection, config)


def stalled_worker(connection, config):
    # Wait without advertising readiness; parent must time out and reap us.
    try:
        connection.recv()
    finally:
        connection.close()


def test_startup_timeout_reaps_spawn_workers(monkeypatch):
    processes = install_test_context(monkeypatch, stalled_worker)
    monkeypatch.setattr(duration, "STARTUP_TIMEOUT", 0.2)
    with pytest.raises(TimeoutError, match="timeout"):
        duration.run_duration(_args(), 8)
    assert all(process.exitcode is not None for process in processes)


def _args():
    return argparse.Namespace(
        numa="shard", duration=0.05, env="reference", compare_env="compared", mode="train", warmup=0, zero_actions=True
    )


def install_test_context(monkeypatch, target):
    context = mp.get_context("spawn")
    processes = []

    class TestContext:
        def Pipe(self, *args, **kwargs):
            return context.Pipe(*args, **kwargs)

        def Process(self, *args, **kwargs):
            kwargs["target"] = target
            process = context.Process(*args, **kwargs)
            processes.append(process)
            return process

    def get_context(method):
        assert method == "spawn"
        return TestContext()

    monkeypatch.setattr(duration.mp, "get_context", get_context)
    monkeypatch.setattr(duration, "discover_nodes", lambda: [NodePlacement(0, (0,)), NodePlacement(2, (2,))])
    return processes


def test_spawn_workers_share_window_and_aggregate_counts(monkeypatch):
    processes = install_test_context(monkeypatch, fake_worker)
    results = duration.run_duration(_args(), 5)
    assert [result["env"] for result in results] == ["reference", "compared"]
    for result in results:
        assert result["completed_env_steps"] == 2 * 3 + 3 * 2
        assert result["env_steps_per_s"] == 12 / 0.05
        windows = [worker["window"] for worker in result["workers"]]
        assert windows[0] == windows[1]
        assert windows[0]["start_ns"] == result["window_start_ns"]
        assert windows[0]["end_ns"] == result["window_end_ns"]
    assert all(process.exitcode == 0 for process in processes)


def test_worker_failure_propagates_and_reaps_processes(monkeypatch):
    processes = install_test_context(monkeypatch, failing_worker)
    with pytest.raises(RuntimeError, match="worker exited"):
        duration.run_duration(_args(), 8)
    assert all(process.exitcode is not None for process in processes)
