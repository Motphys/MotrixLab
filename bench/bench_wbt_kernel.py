# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Microbenchmark the managed WBT fused kernel and its MotrixSim read plan."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from importlib.metadata import version
from typing import Any

import numba
import numpy as np

from motrix_env_core import registry
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.numba.kernel import clone_kernel_value, get_num_threads, get_threading_layer


def _launch_kernel(source: np.ndarray[Any, Any], target: np.ndarray[Any, Any]) -> None:
    """Small parallel loop used to expose dispatch and barrier overhead."""
    for index in numba.prange(source.shape[0]):
        target[index] = source[index] + np.float32(1.0)


LAUNCH_KERNEL = numba.njit(cache=True, nogil=True, parallel=True)(_launch_kernel)


def _measure(
    call: Callable[[int], None],
    *,
    warmup: int,
    steps: int,
    setup: Callable[[int], None] | None = None,
    setup_delay_seconds: float = 0.0,
) -> np.ndarray[Any, Any]:
    for index in range(warmup):
        if setup is not None:
            setup(index)
            if setup_delay_seconds > 0.0:
                time.sleep(setup_delay_seconds)
        call(index)

    samples_ns = np.empty((steps,), dtype=np.int64)
    for index in range(steps):
        if setup is not None:
            setup(warmup + index)
            if setup_delay_seconds > 0.0:
                time.sleep(setup_delay_seconds)
        started_ns = time.perf_counter_ns()
        call(warmup + index)
        samples_ns[index] = time.perf_counter_ns() - started_ns
    return samples_ns


def _array_bytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, tuple):
        return sum(_array_bytes(item) for item in value)
    if is_dataclass(value):
        return sum(_array_bytes(getattr(value, field.name)) for field in fields(value))
    return 0


def _array_checksum(value: Any) -> float:
    if isinstance(value, np.ndarray):
        return float(np.where(np.isfinite(value), value, 0).sum(dtype=np.float64))
    if isinstance(value, tuple):
        return sum(_array_checksum(item) for item in value)
    if is_dataclass(value):
        return sum(_array_checksum(getattr(value, field.name)) for field in fields(value))
    return 0.0


def _array_nonfinite_count(value: Any) -> int:
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
        return int(np.count_nonzero(~np.isfinite(value)))
    if isinstance(value, tuple):
        return sum(_array_nonfinite_count(item) for item in value)
    if is_dataclass(value):
        return sum(_array_nonfinite_count(getattr(value, field.name)) for field in fields(value))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("main", "main-after-read", "read", "partial-reset-read", "read-main", "step", "launch"),
        required=True,
    )
    parser.add_argument("--env", default="g1-wbt-dance", help="Registered ManagerEnv name.")
    parser.add_argument("--mode", default="play")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reset-size", type=int, default=256)
    parser.add_argument(
        "--setup-delay-ms",
        type=float,
        default=0.0,
        help="Untimed delay between a stage setup operation and the measured call.",
    )
    parser.add_argument(
        "--input-copies",
        type=int,
        default=1,
        help="Rotate this many cloned real input snapshots to control cache residency.",
    )
    parser.add_argument(
        "--parallel-chunksize",
        type=int,
        default=0,
        help="Numba parallel chunksize; zero keeps the default static schedule.",
    )
    args = parser.parse_args()
    if args.input_copies < 1:
        parser.error("--input-copies must be at least 1")
    if args.setup_delay_ms < 0.0:
        parser.error("--setup-delay-ms must not be negative")
    if not 1 <= args.reset_size <= args.num_envs:
        parser.error("--reset-size must be between 1 and --num-envs")

    numba.set_parallel_chunksize(args.parallel_chunksize)
    env = registry.make(args.env, mode=args.mode, num_envs=args.num_envs)
    assert isinstance(env, ManagerEnv)
    env.init_state()
    env.warmup()
    env._refresh_sim_reads()
    inputs = env._kernel_inputs
    input_snapshots = (inputs,) + tuple(clone_kernel_value(inputs) for _ in range(args.input_copies - 1))
    assert env._task_program is not None
    assert env._kernel_buffers is not None
    assert env._kernel_outputs is not None

    task_kernel = env._task_program.evaluate_kernel
    reward_weights = env._task_program.reward_weights
    buffers = env._kernel_buffers
    outputs = env._kernel_outputs
    latest_inputs = [inputs]
    launch_source = np.arange(args.num_envs, dtype=np.float32)
    launch_target = np.zeros_like(launch_source)
    reset_env_ids = np.arange(args.reset_size, dtype=np.int64)
    actions = np.zeros((args.num_envs, *env.action_space.shape), dtype=np.float32)

    def call_main(index: int) -> None:
        task_kernel(input_snapshots[index % args.input_copies], reward_weights, buffers, outputs)

    def call_main_after_read(_index: int) -> None:
        task_kernel(latest_inputs[0], reward_weights, buffers, outputs)

    def setup_read(_index: int) -> None:
        env._refresh_sim_reads()
        latest_inputs[0] = env._kernel_inputs

    def call_read(index: int) -> None:
        setup_read(index)

    def call_partial_reset_read(_index: int) -> None:
        env.sim_data.execute(reset_env_ids)

    def call_read_main(index: int) -> None:
        setup_read(index)
        call_main_after_read(index)

    def call_launch(_index: int) -> None:
        LAUNCH_KERNEL(launch_source, launch_target)

    def call_step(_index: int) -> None:
        env.step(actions)

    call = {
        "main": call_main,
        "main-after-read": call_main_after_read,
        "read": call_read,
        "partial-reset-read": call_partial_reset_read,
        "read-main": call_read_main,
        "step": call_step,
        "launch": call_launch,
    }[args.stage]
    setup = {"main-after-read": setup_read}.get(args.stage)

    samples_ns = _measure(
        call,
        warmup=args.warmup,
        steps=args.steps,
        setup=setup,
        setup_delay_seconds=args.setup_delay_ms / 1e3,
    )
    percentiles_ns = np.percentile(samples_ns, (10, 50, 90, 99))
    sim_data = env.sim_data
    kernel_values = (outputs, buffers) if args.stage in {"main", "main-after-read", "read-main", "step"} else ()
    checksum_values = (*(sim_data[key] for key in sim_data.keys), *kernel_values, launch_target)
    result = {
        "stage": args.stage,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "warmup": args.warmup,
        "numba_threads": get_num_threads(),
        "numba_version": numba.__version__,
        "motrixsim_version": version("motrixsim"),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--short"], text=True).strip()),
        "parallel_chunksize": numba.get_parallel_chunksize(),
        "threading_layer": get_threading_layer(),
        "rayon_threads": os.environ.get("RAYON_NUM_THREADS", "default"),
        "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY", "default"),
        "gomp_spincount": os.environ.get("GOMP_SPINCOUNT", "default"),
        "affinity_cpus": sorted(os.sched_getaffinity(0)),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "input_bytes": _array_bytes(inputs),
        "input_copies": args.input_copies,
        "input_working_set_bytes": _array_bytes(input_snapshots),
        "reset_size": args.reset_size,
        "sim_queries": len(sim_data.keys),
        "sim_query_layouts": [
            {
                "key": key,
                "query_type": type(sim_data.query(key)).__name__,
                "shape": sim_data[key].shape,
                "strides": sim_data[key].strides,
            }
            for key in sim_data.keys
        ],
        "arena_bytes": sim_data.arena_bytes,
        "alias_count": 0,
        "setup_delay_ms": args.setup_delay_ms,
        "reward_weights_bytes": _array_bytes(reward_weights),
        "buffer_bytes": _array_bytes(buffers),
        "output_bytes": _array_bytes(outputs),
        "min_ms": float(samples_ns.min() / 1e6),
        "p10_ms": float(percentiles_ns[0] / 1e6),
        "median_ms": float(percentiles_ns[1] / 1e6),
        "p90_ms": float(percentiles_ns[2] / 1e6),
        "p99_ms": float(percentiles_ns[3] / 1e6),
        "max_ms": float(samples_ns.max() / 1e6),
        "checksum": _array_checksum(checksum_values),
        "nonfinite_count": _array_nonfinite_count(checksum_values),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
