# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Interval-event manager overhead benchmark (design probe for interval-events).

Measures the manager-side per-step cost of ``IntervalEventManager.apply()`` in
isolation from the real simulator: per-event timer scan, Numba event-kernel
dispatch, and WriteProgram execution on the fake in-memory backend reused from
``motrix_env_core/tests``. Numbers therefore bound the manager overhead only;
the motrixsim-native write cost is excluded.

Scenarios (per config interval range, ctrl dt 0.02 s):

- ``hot``:       interval ~= one step, nearly every row fires every step (upper bound)
- ``staggered``: per-env staggered timers, a few rows fire per step (typical large scale)
- ``cold``:      nothing ever fires; pure timer scan cost

Also compares timer-scan strategies (per-event flatnonzero vs batched 2D/flat
nonzero) to keep the recorded negative result measurable on new hardware.

Examples:
    python scripts/bench_interval_events.py
    python scripts/bench_interval_events.py --num-envs 1024 4096 8192 --events 1 2 8
    python scripts/bench_interval_events.py --num-envs 4096 --events 8 --steps 1000
"""

from __future__ import annotations

import os

# Avoid OpenMP busy-wait spinning in BLAS/OpenMP pools that distorts μs-scale timing.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "0")

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "motrix_env_core" / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_interval_events import _FakeManagerCfg, _kick  # noqa: E402

from motrix_env_core.numba.manager.env import ManagerEnv  # noqa: E402

SCENARIOS = {
    "hot": (0.02, 0.04),
    "staggered": (0.1, 0.3),
    "cold": (1000.0, 2000.0),
}


def _make_env(num_envs: int, num_events: int, interval: tuple[float, float], seed: int) -> ManagerEnv:
    cfg = _FakeManagerCfg(interval_events={f"kick{i}": _kick(interval=interval) for i in range(num_events)})
    env = ManagerEnv(cfg, num_envs=num_envs, backend="fake-interval-events", seed=seed)
    env.sim._sim_dt = 0.02 / cfg.sim_substeps
    env.init_state()
    return env


def _median_apply_us(
    env: ManagerEnv, num_envs: int, interval: tuple[float, float], dt: float, steps: int, reps: int
) -> float:
    rng = np.random.default_rng(0)
    for event in env.interval_event_manager.events.values():
        event.remaining_s[:] = rng.uniform(interval[0], interval[1], num_envs).astype(np.float32)
    env.interval_event_manager.apply(dt)  # warm the njit specialization out of timing
    samples = []
    for _ in range(reps):
        env.sim.event_writes.clear()  # the fake backend appends per execute; keep memory flat
        t0 = time.perf_counter()
        for _ in range(steps):
            env.interval_event_manager.apply(dt)
        samples.append((time.perf_counter() - t0) / steps * 1e6)
    return statistics.median(samples)


def bench_apply(args: argparse.Namespace) -> None:
    print(
        f"apply() per-step cost  dt={args.dt}s steps={args.steps} reps={args.reps}"
        "  (fake write backend: manager overhead only)"
    )
    header = f"{'scenario':>9} {'events':>6} {'num_envs':>8} {'us/step':>9} {'us/event':>9}"
    print(header)
    for num_events in args.events:
        for num_envs in args.num_envs:
            env = _make_env(num_envs, num_events, SCENARIOS["staggered"], seed=args.seed)
            for name, interval in SCENARIOS.items():
                per_step = _median_apply_us(env, num_envs, interval, args.dt, args.steps, args.reps)
                print(f"{name:>9} {num_events:>6} {num_envs:>8} {per_step:>9.1f} {per_step / num_events:>9.1f}")
    print()


def _median_scan_us(
    timers: np.ndarray, variant: str, dt: np.float32, num_events: int, num_envs: int, reps: int
) -> float:
    def per_event(arr: np.ndarray) -> None:
        for row in range(num_events):
            arr[row] -= dt
            np.flatnonzero(arr[row] <= 0)

    def batched_2d(arr: np.ndarray) -> None:
        arr -= dt
        rows, env_ids = np.nonzero(arr <= 0.0)
        ends = np.cumsum(np.bincount(rows, minlength=num_events))
        start = 0
        for index in range(num_events):
            end = int(ends[index])
            if end > start:
                _ = env_ids[start:end]
            start = end

    def batched_flat(arr: np.ndarray) -> None:
        flat = arr.reshape(-1)
        flat -= dt
        idx = np.flatnonzero(flat <= 0)
        rows = idx // num_envs
        env_ids = idx - rows * num_envs
        ends = np.cumsum(np.bincount(rows, minlength=num_events))
        start = 0
        for index in range(num_events):
            end = int(ends[index])
            if end > start:
                _ = env_ids[start:end]
            start = end

    variants = {"per_event": per_event, "batched_2d": batched_2d, "batched_flat": batched_flat}
    samples = []
    for _ in range(reps):
        arr = timers.copy()
        t0 = time.perf_counter()
        variants[variant](arr)
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


def bench_scan(args: argparse.Namespace) -> None:
    print("timer-scan strategies  (numpy only; due density ~= staggered scale)")
    header = f"{'events':>6} {'num_envs':>8} {'per_event':>10} {'batched_2d':>11} {'batched_flat':>13}"
    print(header)
    rng = np.random.default_rng(0)
    dt = np.float32(args.dt)
    for num_events in args.events:
        for num_envs in args.num_envs:
            timers = rng.uniform(0.001, 10.0, (num_events, num_envs)).astype(np.float32)
            due_target = max(1, int(num_envs * args.dt / 7.5))
            for row in range(num_events):
                timers[row, rng.choice(num_envs, size=min(due_target, num_envs), replace=False)] = np.float32(1e-4)
            cells = [
                _median_scan_us(timers, name, dt, num_events, num_envs, args.reps * 10)
                for name in ("per_event", "batched_2d", "batched_flat")
            ]
            print(f"{num_events:>6} {num_envs:>8} {cells[0]:>9.1f}u {cells[1]:>10.1f}u {cells[2]:>12.1f}u")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--events", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-scan", action="store_true", help="skip the numpy timer-scan comparison")
    args = parser.parse_args(argv)

    bench_apply(args)
    if not args.skip_scan:
        bench_scan(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
