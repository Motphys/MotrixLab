# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""``mxlab bench``: performance measurement entry point.

``bench env`` benchmarks the public ``env.step()`` contract for one or two
environments. The reported CPU utilization is process CPU time divided by
wall-clock time over the timed ``env.step()`` calls. It is an aggregate across
the process's threads and can therefore exceed 100% when multiple CPU cores
are busy.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import TYPE_CHECKING

from motrixlab.benchmark import runner
from motrixlab.benchmark.report import render_table

if TYPE_CHECKING:
    from motrix_env_core.perf import PerfNode


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mxlab bench", description=__doc__)
    parser.add_argument("--mode", choices=("env", "physics"), required=True, help="Benchmark target.")
    parser.add_argument("--env", default="cartpole", help="(env) Reference registered environment name.")
    parser.add_argument("--compare-env", help="(env) Optional registered environment to compare with --env.")
    parser.add_argument("--env-mode", default="train", help="(env) Environment mode (train/play).", dest="env_mode")
    parser.add_argument("--robot", default="g1-29dof", help="(physics) Registered robot name.")
    parser.add_argument(
        "--num-envs",
        type=int,
        nargs="+",
        default=[2048],
        help="Total environment batch sizes to benchmark.",
    )
    measurement = parser.add_mutually_exclusive_group()
    measurement.add_argument("--steps", type=int, help="(env) Timed calls; selects legacy single-process mode.")
    measurement.add_argument("--duration", type=float, help="(env) Positive finite measurement seconds (default: 10).")
    parser.add_argument(
        "--numa",
        choices=("single", "node", "shard", "_worker"),
        help="Placement (env: single/shard, default shard for duration and single for steps; "
        "physics: single/node/shard, default single).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        help="Untimed calls per measurement (default: 10 for env, 20 for physics).",
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--zero-actions",
        action="store_true",
        help="(env) Use zero actions instead of seeded random actions.",
    )
    parser.add_argument("--sim-backend", help="(env) Optional simulation backend override.")
    parser.add_argument("--breakdown", action="store_true", help="(env) Print the environment Perf scope tree.")
    parser.add_argument("--json", action="store_true", help="(env) Print one JSON object per result.")
    parser.add_argument("--dt", type=float, default=0.005, help="(physics) Simulation timestep in seconds.")
    parser.add_argument("--solver-iterations", type=int, default=3, help="(physics) Constraint solver iterations.")
    parser.add_argument("--nstep", type=int, default=1, help="(physics) Native substeps per step_n call.")
    parser.add_argument("--iters", type=int, default=200, help="(physics) Timed step_n calls per measurement.")
    parser.add_argument("--node", type=int, default=0, help="(physics, internal worker) which NUMA node.")
    parser.add_argument("--out-json", default="", help="(physics, internal worker) JSON output path.")
    return parser


def _run_env(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.numa not in (None, "single", "shard"):
        parser.error("--numa node/_worker only apply to --mode physics")
    if args.warmup is None:
        args.warmup = 10
    if args.steps is None and args.duration is None:
        args.duration = 10.0
    if args.numa is None:
        args.numa = "shard" if args.duration is not None else "single"
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive and finite")
    if args.numa == "shard" and args.duration is None:
        parser.error("--numa shard requires --duration")
    if args.duration is not None and args.breakdown:
        parser.error("--breakdown cannot be used with --duration")
    if (args.steps is not None and args.steps < 1) or args.warmup < 0:
        parser.error("--steps must be positive and --warmup must not be negative")
    if any(num_envs < 1 for num_envs in args.num_envs):
        parser.error("--num-envs values must be positive")
    if args.json and args.breakdown:
        parser.error("--json and --breakdown cannot be used together")

    if args.duration is not None:
        duration_rows = []
        for num_envs in args.num_envs:
            results = runner.run_duration(args, num_envs)
            reference_rate = float(results[0]["env_steps_per_s"])
            for index, result in enumerate(results):
                rate = float(result["env_steps_per_s"])
                speedup = None if index == 0 or reference_rate == 0 else rate / reference_rate
                if speedup is not None:
                    result["speedup_vs_reference"] = speedup
                if args.json:
                    print(json.dumps(result, sort_keys=True))
                else:
                    duration_rows.append(
                        [
                            str(result["env"]),
                            str(num_envs),
                            str(result["worker_count"]),
                            f"{args.duration:g}",
                            str(result["completed_env_steps"]),
                            f"{rate:,.0f}",
                            "—" if speedup is None else f"{speedup:.2f}x",
                        ]
                    )
                    for worker in result["workers"]:
                        print(
                            f"{result['env']} worker {worker['worker_index']}: "
                            f"batch={worker['num_envs']} calls={worker['completed_calls']} "
                            f"start_offset_ns={worker['actual_start_offset_ns']} "
                            f"placement={json.dumps(worker['placement'], sort_keys=True)}"
                        )
        if not args.json:
            print(
                render_table(
                    [
                        "environment",
                        "total num_envs",
                        "workers",
                        "seconds",
                        "completed env steps",
                        "env steps/s",
                        "speedup",
                    ],
                    duration_rows,
                )
            )
        return

    rows: list[list[str]] = []
    perf_sections: list[tuple[str, int, object]] = []
    for num_envs in args.num_envs:
        results, profiles = runner.run_steps(args, num_envs)
        reference = results[0]

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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.mode == "env":
        _run_env(parser, args)
    else:
        # Imported lazily so the env mode (and the parent process of shard
        # workers) never loads the simulation runtime.
        from motrixlab.benchmark import physics

        if args.numa is None:
            args.numa = "single"
        physics.main(parser, args)


if __name__ == "__main__":
    main()
