# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Real-backend interval-event benchmark (full pipeline, motrixsim).

Runs a registered Manager env end-to-end on the real simulator and measures the
complete event pipeline per step via the env's own ``interval_events`` perf
scope: timer scan, the compiled Numba event kernel, and the native MotrixSim
WriteProgram execution whose writes the following physics step actually
consumes. Unlike ``bench_interval_events.py`` (fake write backend, manager
overhead only), this measures the true end-to-end cost on the production path
- ``env.step`` fires the events exactly once per step through
``ManagerEnv.physics_step``.

A zero-event baseline of the same env separates event cost from physics cost.

Requires a machine with the real dependencies (motrixsim, motrix_envs assets).

Examples:
    python scripts/bench_interval_events_real.py
    python scripts/bench_interval_events_real.py --num-envs 4096 --events 1 2 8
    python scripts/bench_interval_events_real.py --interval 5.0 10.0
"""

from __future__ import annotations

import argparse
import os
import statistics
from time import perf_counter

# Avoid OpenMP busy-wait spinning that distorts microsecond-scale timing.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("GOMP_SPINCOUNT", "0")

import numpy as np


def _make_env(num_envs: int, num_events: int, interval: tuple[float, float], task: str):
    import motrix_envs  # noqa: F401  (registers env configs)
    from motrix_env_core import registry
    from motrix_env_core.mdp.events import RandomVelocityKickCfg

    cfg = registry.make_env_config(task, mode="play")
    if num_events:
        body = cfg.scene.objs.robot.resolved_base_link_name
        cfg.interval_events = {
            f"kick{i}": RandomVelocityKickCfg(
                interval_range_s=interval,
                body=body,
                velocity_delta_x=(-0.5, 0.5),
                velocity_delta_y=(-0.5, 0.5),
            )
            for i in range(num_events)
        }
    from motrix_env_core.numba.manager.env import ManagerEnv

    env = ManagerEnv(cfg, num_envs=num_envs)
    env.init_state()
    env.perf.enable()
    return env


def _find(node, name):
    if node.name == name:
        return node
    for child in node.children:
        found = _find(child, name)
        if found is not None:
            return found
    return None


def _interval_events_us(env) -> float:
    for root in env.perf.snapshot():
        node = _find(root, "interval_events")
        if node is not None:
            return node.total_ns / 1e3 / max(node.count, 1)
    raise RuntimeError("interval_events perf scope did not run; enable perf before stepping.")


def _bench(num_envs: int, num_events: int, interval: tuple[float, float], task: str, steps: int, reps: int) -> None:
    step_samples = []
    event_samples = []
    for _ in range(reps):
        env = _make_env(num_envs, num_events, interval, task)
        action = np.zeros((num_envs, env.action_space.shape[0]), dtype=np.float32)
        for _ in range(30):  # warmup: timers deplete, events fire, physics consumes writes
            env.step(action)
        env.perf.reset()
        t0 = perf_counter()
        for _ in range(steps):
            env.step(action)
        step_samples.append((perf_counter() - t0) / steps * 1e3)
        event_samples.append(_interval_events_us(env))
    per_event = statistics.median(event_samples) / num_events if num_events else 0.0
    print(
        f"envs={num_envs:>5} events={num_events}: full_step={statistics.median(step_samples):8.3f} ms  "
        f"events={statistics.median(event_samples):8.1f} us/step  ({per_event:6.1f} us/event)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--events", type=int, nargs="+", default=[0, 1, 2, 8])
    parser.add_argument("--interval", type=float, nargs=2, default=(0.1, 0.3), help="event interval range in seconds")
    parser.add_argument("--task", default="g1-wbt-dance")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args(argv)
    print(
        f"real-backend interval events  task={args.task} interval={tuple(args.interval)}s "
        f"steps={args.steps} reps={args.reps}"
    )
    for num_envs in args.num_envs:
        for num_events in args.events:
            _bench(num_envs, num_events, tuple(args.interval), args.task, args.steps, args.reps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
