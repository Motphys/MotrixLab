# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, replace

from motrix_bench import BenchmarkConfig, BenchmarkResult, PreparedCase, run_case

from . import runner, workload
from .argtypes import nonnegative_int, positive_int
from .registry import BenchmarkSpec


@dataclass(frozen=True, slots=True)
class EnvStepConfig(BenchmarkConfig):
    # ``env_name``/``num_envs`` carry defaults only because the inherited
    # ``warmup`` already has one; ``build_config`` always sets them explicitly.
    env_name: str = ""
    num_envs: tuple[int, ...] = ()
    mode: str = "train"
    seed: int = 20260808
    zero_actions: bool = False
    sim_backend: str | None = None
    sim_device: str | None = None
    breakdown: bool = False

    def __post_init__(self) -> None:
        # Direct call: zero-arg super() resolves to the pre-slots class.
        BenchmarkConfig.__post_init__(self)
        if not self.num_envs:
            raise ValueError("num_envs requires at least one value")
        if any(num_envs < 1 for num_envs in self.num_envs):
            raise ValueError("num_envs values must be positive")


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", required=True, help="Registered environment name.")
    parser.add_argument("--env-mode", default="train", choices=("train", "play"))
    parser.add_argument("--num-envs", type=positive_int, nargs="+", default=[2048])
    parser.add_argument("--steps", type=positive_int, default=1000)
    parser.add_argument("--warmup", type=nonnegative_int, default=10)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--zero-actions", action="store_true")
    parser.add_argument("--sim-backend")
    parser.add_argument("--sim-device", help="Simulation device, e.g. cpu or cuda:0.")
    parser.add_argument("--breakdown", action="store_true")


def build_config(args: argparse.Namespace) -> EnvStepConfig:
    return EnvStepConfig(
        steps=args.steps,
        warmup=args.warmup,
        env_name=args.env,
        num_envs=tuple(args.num_envs),
        mode=args.env_mode,
        seed=args.seed,
        zero_actions=args.zero_actions,
        sim_backend=args.sim_backend,
        sim_device=args.sim_device,
        breakdown=args.breakdown,
    )


def prepare(config: EnvStepConfig) -> PreparedCase:
    single = replace(config, num_envs=(config.num_envs[0],))
    _, case = workload.prepare_case(single, single.num_envs[0])
    return case


def _run_batch(batch_config: EnvStepConfig, num_envs: int) -> BenchmarkResult:
    case = prepare(batch_config)
    started_cpu = time.process_time()
    result = run_case(
        case,
        BenchmarkConfig(batch_config.steps, batch_config.warmup),
        benchmark="env.step",
        label=f"{batch_config.env_name} (num_envs={num_envs})",
        config=asdict(batch_config),
    )
    elapsed_cpu = time.process_time() - started_cpu
    cpu_util_percent = float(elapsed_cpu / result.elapsed_seconds * 100.0)
    result = replace(result, metadata={"requested_num_envs": num_envs, "cpu_util_percent": cpu_util_percent})
    if batch_config.breakdown:
        profile_env, profile_case = workload.prepare_case(batch_config, num_envs)
        try:
            profile = runner._profile_calls(
                profile_env.perf, profile_case.step, batch_config.steps, "step", synchronize=profile_case.synchronize
            )
        finally:
            profile_case.close_once()
        print(
            f"env.step, num_envs={num_envs} step breakdown ({profile.elapsed_seconds:.3f}s wall time)",
            file=sys.stderr,
        )
        with redirect_stdout(sys.stderr):
            runner.print_profile_tree(profile.root)
    return result


def run(config: EnvStepConfig) -> Sequence[BenchmarkResult]:
    return [_run_batch(replace(config, num_envs=(num_envs,)), num_envs) for num_envs in config.num_envs]


def register() -> BenchmarkSpec:
    return BenchmarkSpec(
        "env.step", "Benchmark the public environment step contract.", configure, build_config, prepare, run
    )
