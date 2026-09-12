# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the public ``env.step()`` contract for one or two environments.

The reported CPU utilization is process CPU time divided by wall-clock time
over the timed ``env.step()`` calls. It is an aggregate across the process's
threads and can therefore exceed 100% when multiple CPU cores are busy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "0")

import numpy as np
from report import render_table

import motrix_envs  # noqa: F401  registers built-in environments
from motrix_env_core import registry
from motrix_env_core.perf import Perf, PerfNode


@dataclass(frozen=True, slots=True)
class _ProfileResult:
    elapsed_seconds: float
    root: PerfNode


def _profile_calls(
    perf: Perf,
    function: Callable[[], object],
    count: int,
    root_name: str,
) -> _ProfileResult:
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
    return _ProfileResult(elapsed_seconds=elapsed_seconds, root=root)


def _print_perf_tree(root: PerfNode) -> None:
    print("Perf values are aggregate means over production calls, not per-call medians.")
    print(f"{'scope':<38} | {'mean ms':>10} | {'self ms':>10} | {'share':>7} | {'calls/root':>10}")

    def print_node(node: PerfNode, depth: int) -> None:
        mean_ms = node.total_ns / root.count / 1e6
        self_ms = node.self_ns / root.count / 1e6
        share = node.total_ns / root.total_ns * 100
        calls_per_root = node.count / root.count
        name = f"{'  ' * depth}{node.name}"
        print(f"{name:<38} | {mean_ms:>10.4f} | {self_ms:>10.4f} | {share:>6.2f}% | {calls_per_root:>10.2f}")
        for child in node.children:
            print_node(child, depth + 1)

    print_node(root, 0)


def _make_env(name: str, args: argparse.Namespace, num_envs: int):
    kwargs = {
        "mode": args.mode,
        "num_envs": num_envs,
    }
    if args.sim_backend is not None:
        kwargs["sim_backend"] = args.sim_backend
    return registry.make(name, **kwargs)


def _action_spec(env) -> tuple[tuple[int, ...], np.dtype, np.ndarray, np.ndarray]:
    space = env.action_space
    return space.shape, space.dtype, np.asarray(space.low), np.asarray(space.high)


def _make_actions(env, *, seed: int, zero_actions: bool) -> np.ndarray:
    space = env.action_space
    shape = (env.num_envs, *space.shape)
    if zero_actions:
        return np.zeros(shape, dtype=space.dtype)

    low = np.where(np.isfinite(space.low), space.low, -1.0)
    high = np.where(np.isfinite(space.high), space.high, 1.0)
    return np.random.default_rng(seed).uniform(low, high, size=shape).astype(space.dtype)


def _measure_env(
    name: str,
    args: argparse.Namespace,
    num_envs: int,
    actions: np.ndarray | None,
    expected_action_spec: tuple[tuple[int, ...], np.dtype, np.ndarray, np.ndarray] | None,
) -> tuple[
    dict[str, object],
    np.ndarray,
    tuple[tuple[int, ...], np.dtype, np.ndarray, np.ndarray],
    _ProfileResult | None,
]:
    env = _make_env(name, args, num_envs)
    action_spec = _action_spec(env)
    if expected_action_spec is not None:
        expected_shape, expected_dtype, expected_low, expected_high = expected_action_spec
        shape, dtype, low, high = action_spec
        if shape != expected_shape or dtype != expected_dtype:
            raise ValueError(
                f"Cannot compare {name!r}: action space {(shape, dtype)} does not match "
                f"the reference {(expected_shape, expected_dtype)}."
            )
        if not np.array_equal(low, expected_low) or not np.array_equal(high, expected_high):
            raise ValueError(f"Cannot compare {name!r}: action bounds do not match the reference environment.")
    if actions is None:
        actions = _make_actions(env, seed=args.seed, zero_actions=args.zero_actions)

    for _ in range(args.warmup):
        env.step(actions)

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
    result: dict[str, object] = {
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

    profile = None
    if args.breakdown:
        profile = _profile_calls(env.perf, lambda: env.step(actions), args.steps, "step")

    return result, actions, action_spec, profile


_RESULT_HEADERS = [
    "environment",
    "num_envs",
    "p10 ms",
    "median ms",
    "p90 ms",
    "env steps/s",
    "CPU %",
    "speedup",
]


def _result_row(result: dict[str, object], speedup: float | None) -> list[str]:
    speedup_cell = "—" if speedup is None else f"{speedup:.2f}x"
    return [
        str(result["env"]),
        str(int(result["num_envs"])),
        f"{float(result['p10_ms']):.3f}",
        f"{float(result['median_ms']):.3f}",
        f"{float(result['p90_ms']):.3f}",
        f"{float(result['env_steps_per_s']):,.0f}",
        f"{float(result['cpu_util_percent']):.1f}%",
        speedup_cell,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="cartpole", help="Reference registered environment name.")
    parser.add_argument("--compare-env", help="Optional registered environment to compare with --env.")
    parser.add_argument("--mode", default="train", help="Environment mode (train/play).")
    parser.add_argument("--num-envs", type=int, nargs="+", default=[2048], help="Batch sizes to benchmark.")
    parser.add_argument("--steps", type=int, default=1000, help="Timed env.step() calls per measurement.")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed env.step() calls per measurement.")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--zero-actions",
        action="store_true",
        help="Use zero actions instead of seeded random actions.",
    )
    parser.add_argument("--sim-backend", help="Optional simulation backend override.")
    parser.add_argument("--breakdown", action="store_true", help="Print the environment Perf scope tree.")
    parser.add_argument("--json", action="store_true", help="Print one JSON object per result.")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("--steps must be positive and --warmup must not be negative")
    if any(num_envs < 1 for num_envs in args.num_envs):
        parser.error("--num-envs values must be positive")
    if args.json and args.breakdown:
        parser.error("--json and --breakdown cannot be used together")

    rows: list[list[str]] = []
    perf_sections: list[tuple[str, int, _ProfileResult]] = []
    for num_envs in args.num_envs:
        reference, actions, action_spec, reference_profile = _measure_env(
            args.env,
            args,
            num_envs,
            None,
            None,
        )
        results = [reference]
        profiles = [(args.env, reference_profile)]
        if args.compare_env is not None:
            compared, _, _, compared_profile = _measure_env(
                args.compare_env,
                args,
                num_envs,
                actions,
                action_spec,
            )
            results.append(compared)
            profiles.append((args.compare_env, compared_profile))

        reference_median = float(reference["median_ms"])
        for index, result in enumerate(results):
            speedup = None if index == 0 else reference_median / float(result["median_ms"])
            if speedup is not None:
                result["speedup_vs_reference"] = speedup
            if args.json:
                print(json.dumps(result, sort_keys=True))
            else:
                rows.append(_result_row(result, speedup))

        for name, profile in profiles:
            if profile is not None:
                perf_sections.append((name, num_envs, profile))

    if not args.json:
        print(render_table(_RESULT_HEADERS, rows))
        for name, num_envs, profile in perf_sections:
            print()
            print(f"{name}, num_envs={num_envs} step breakdown ({profile.elapsed_seconds:.3f}s wall time)")
            _print_perf_tree(profile.root)


if __name__ == "__main__":
    main()
