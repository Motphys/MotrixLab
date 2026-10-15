# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Step latency and sustained throughput measurement, without CLI dependencies."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import partial
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING

from motrix_rl.numa import NodePlacement, apply_placement, discover_nodes

from . import workload

if TYPE_CHECKING:
    from motrix_env_core.perf import Perf, PerfNode

STARTUP_TIMEOUT = 180.0
MEASUREMENT_TIMEOUT = 30.0
START_DELAY_NS = 100_000_000


@dataclass(frozen=True, slots=True)
class ProfileResult:
    elapsed_seconds: float
    root: PerfNode


def _profile_calls(perf: Perf, function: Callable[[], object], count: int, root_name: str) -> ProfileResult:
    was_enabled = perf.enabled
    perf.reset()
    if not was_enabled:
        perf.enable()
    started = time.perf_counter()
    try:
        for _ in range(count):
            function()
    finally:
        elapsed_seconds = time.perf_counter() - started
        if not was_enabled:
            perf.disable()
    roots = {root.name: root for root in perf.snapshot()}
    try:
        root = roots[root_name]
    except KeyError as error:
        raise RuntimeError(f"Perf did not record expected root scope {root_name!r}.") from error
    return ProfileResult(elapsed_seconds=elapsed_seconds, root=root)


def run_steps(
    args: argparse.Namespace, num_envs: int
) -> tuple[list[dict[str, object]], list[tuple[str, ProfileResult | None]]]:
    """Measure reference/compare latency in this process using the same actions."""
    results = []
    profiles = []
    actions = None
    expected_spec = None
    names = [args.env] + ([args.compare_env] if args.compare_env is not None else [])
    with redirect_stdout(sys.stderr) if args.json else redirect_stdout(sys.stdout):
        np, _ = workload.load_runtime()
        for name in names:
            env, actions, expected_spec = workload.prepare_env(name, args, num_envs, actions, expected_spec)
            try:
                workload.warmup(env, actions, args.warmup)
                samples_ns = np.empty((args.steps,), dtype=np.int64)
                started_wall = time.perf_counter()
                started_cpu = time.process_time()
                for index in range(args.steps):
                    started_ns = time.perf_counter_ns()
                    env.step(actions)
                    samples_ns[index] = time.perf_counter_ns() - started_ns
                elapsed_wall = time.perf_counter() - started_wall
                elapsed_cpu = time.process_time() - started_cpu
                percentiles_ns = np.percentile(samples_ns, (10, 50, 90))
                median_seconds = float(percentiles_ns[1] / 1e9)
                results.append(
                    {
                        "env": name,
                        "mode": args.mode,
                        "num_envs": num_envs,
                        "steps": args.steps,
                        "warmup": args.warmup,
                        "random_actions": not args.zero_actions,
                        "p10_ms": float(percentiles_ns[0] / 1e6),
                        "median_ms": float(percentiles_ns[1] / 1e6),
                        "p90_ms": float(percentiles_ns[2] / 1e6),
                        "env_steps_per_s": float(num_envs / median_seconds),
                        "cpu_util_percent": float(elapsed_cpu / elapsed_wall * 100.0),
                    }
                )
                profile = None
                if args.breakdown:
                    profile = _profile_calls(env.perf, partial(env.step, actions), args.steps, "step")
                profiles.append((name, profile))
            finally:
                workload.close_env(env)
    return results, profiles


def split_num_envs(total: int, nodes: list[NodePlacement]) -> list[int]:
    """Split a total batch as evenly as possible, preserving the total."""
    if not nodes or total < len(nodes):
        raise ValueError(f"Total num_envs={total} must be at least {len(nodes)} usable NUMA nodes")
    quotient, remainder = divmod(total, len(nodes))
    return [quotient + (index < remainder) for index in range(len(nodes))]


def measure_window(
    step: Callable[[], object],
    num_envs: int,
    start_ns: int,
    end_ns: int,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]:
    """Count only calls begun and completed inside the shared closed window.

    No latency samples are retained. A call completing after the deadline is
    excluded, and no new call is begun at or after the deadline.
    """
    now = clock()
    while now < start_ns:
        time.sleep(min((start_ns - now) / 1e9, 0.001))
        now = clock()
    actual_start = now
    completed = 0
    first_call = None
    last_completion = None
    while now < end_ns:
        if first_call is None:
            first_call = now
        step()
        finished = clock()
        if finished <= end_ns:
            completed += 1
            last_completion = finished
        now = clock()
    return {
        "num_envs": num_envs,
        "completed_calls": completed,
        "completed_env_steps": completed * num_envs,
        "actual_start_ns": actual_start,
        "actual_start_offset_ns": actual_start - start_ns,
        "first_call_start_ns": first_call,
        "last_counted_completion_ns": last_completion,
        "actual_finish_ns": now,
    }


def _receive_all(connections: list[Connection], processes: list[BaseProcess], timeout: float) -> list[dict]:
    pending = set(connections)
    messages = {}
    deadline = time.monotonic() + timeout
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Benchmark workers exceeded their startup/measurement timeout")
        for connection in wait(pending, min(remaining, 0.1)):
            try:
                message = connection.recv()
            except EOFError as error:
                raise RuntimeError("Benchmark worker exited without a result") from error
            if message.get("kind") == "error":
                raise RuntimeError(f"Benchmark worker failed:\n{message['error']}")
            messages[connection] = message
            pending.remove(connection)
        for connection, process in zip(connections, processes):
            if connection in pending and process.exitcode is not None:
                raise RuntimeError(f"Benchmark worker exited with status {process.exitcode}")
    return [messages[connection] for connection in connections]


def run_duration(args: argparse.Namespace, num_envs: int) -> list[dict[str, object]]:
    """Measure reference/compare in all workers using a fresh common window."""
    nodes = discover_nodes() if args.numa == "shard" else [NodePlacement(None, ())]
    batches = split_num_envs(num_envs, nodes)
    connections: list[Connection] = []
    processes: list[BaseProcess] = []
    results = []
    context = mp.get_context("spawn")
    try:
        for index, (node, batch) in enumerate(zip(nodes, batches)):
            connection, child_connection = context.Pipe()
            connections.append(connection)
            config = dict(vars(args), worker_index=index, worker_num_envs=batch, node_id=node.node_id, cpus=node.cpus)
            try:
                process = context.Process(
                    target=worker_main,
                    args=(child_connection, config),
                    name=f"bench-env-worker-{index}",
                )
                process.start()
                processes.append(process)
            finally:
                child_connection.close()
        names = [args.env] + ([args.compare_env] if args.compare_env is not None else [])
        for name in names:
            ready = _receive_all(connections, processes, STARTUP_TIMEOUT)
            if any(message.get("kind") != "ready" for message in ready):
                raise RuntimeError("Invalid worker readiness message")
            start_ns = time.perf_counter_ns() + START_DELAY_NS
            end_ns = start_ns + max(1, round(args.duration * 1e9))
            for connection in connections:
                connection.send({"start_ns": start_ns, "end_ns": end_ns})
            messages = _receive_all(connections, processes, args.duration + MEASUREMENT_TIMEOUT)
            if any(message.get("kind") != "result" for message in messages):
                raise RuntimeError("Invalid worker result message")
            workers = [message["worker"] for message in messages]
            count = sum(worker["completed_env_steps"] for worker in workers)
            calls = sum(worker["completed_calls"] for worker in workers)
            elapsed_seconds = (end_ns - start_ns) / 1e9
            results.append(
                {
                    "env": name,
                    "mode": args.mode,
                    "num_envs": num_envs,
                    "num_envs_scope": "total",
                    "duration_seconds": args.duration,
                    "elapsed_seconds": elapsed_seconds,
                    "numa": args.numa,
                    "worker_count": len(workers),
                    "warmup": args.warmup,
                    "random_actions": not args.zero_actions,
                    "window_start_ns": start_ns,
                    "window_end_ns": end_ns,
                    "completed_calls": calls,
                    "completed_env_steps": count,
                    "env_steps_per_s": count / elapsed_seconds,
                    "workers": workers,
                }
            )
        for connection in connections:
            connection.send({"kind": "stop"})
        for process in processes:
            process.join(timeout=MEASUREMENT_TIMEOUT)
            if process.is_alive():
                raise TimeoutError("Benchmark worker exceeded its shutdown timeout")
            if process.exitcode:
                raise RuntimeError(f"Benchmark worker exited with status {process.exitcode}")
        return results
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for connection in connections:
            connection.close()


def worker_main(connection: Connection, config: dict) -> None:
    """Spawn target: placement is completed before any heavy runtime imports."""
    # Redirect native extension writes as well as Python output to preserve JSONL.
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    try:
        args = argparse.Namespace(**config)
        placement = apply_placement(NodePlacement(args.node_id, tuple(args.cpus)))
        if args.numa == "single":
            placement.pop("placement_warning", None)
        actions = None
        expected_spec = None
        names = [args.env] + ([args.compare_env] if args.compare_env is not None else [])
        for name in names:
            env, actions, expected_spec = workload.prepare_env(name, args, args.worker_num_envs, actions, expected_spec)
            try:
                workload.warmup(env, actions, args.warmup)
                connection.send({"kind": "ready"})
                window = connection.recv()
                measurement = measure_window(
                    partial(env.step, actions), args.worker_num_envs, window["start_ns"], window["end_ns"]
                )
                measurement.update(worker_index=args.worker_index, pid=os.getpid(), placement=placement)
                connection.send({"kind": "result", "worker": measurement})
            finally:
                workload.close_env(env)
            del env
        connection.recv()  # keep final worker alive until all results are collected
    except BaseException:
        try:
            connection.send({"kind": "error", "error": traceback.format_exc()})
        except (BrokenPipeError, OSError):
            pass
        raise
    finally:
        connection.close()
