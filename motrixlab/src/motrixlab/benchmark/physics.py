# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixSim physics-only throughput benchmark with NUMA-aware modes.

Drives MotrixSim directly: builds a robot-on-flat-ground scene via the scene
compiler, allocates batched SceneData, and times ``model.step_n(data, nstep)``.
No task env, no observation/reward/termination -- the number is
MotrixSim rigid-body simulation alone.

State drift is eliminated by re-applying ``data.reset(model)`` before every
timed call (outside the timed region), so each measurement starts from the
same stable state regardless of how many iterations run.

NUMA modes (``--numa``):
  single  run in this process as-is (whatever binding the caller gave).
  node    re-exec the measurement under ``numactl --cpunodebind=0 --membind=0``.
  shard   spawn one worker per NUMA node, each pinned via numactl, each running
          ``--num-envs / num_nodes`` envs concurrently; aggregate throughput is
          the sum across workers. Reproduces the per-node process sharding
          validated on a many-core host.

Results are structured ``BenchmarkResult`` objects rendered by the dispatcher;
per-call latency (median + p10/p90) and env steps/s are carried by the
standard schema, and NUMA aggregation notes travel in result metadata.

This module imports the simulation runtime at import time and is therefore
loaded lazily by the named ``physics.step`` benchmark.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path

import motrixsim as mtx
import numpy as np
import torch

from motrix_bench import (
    BenchmarkConfig,
    BenchmarkResult,
    PreparedCase,
    dump_results,
    load_results,
    run_case,
)
from motrix_env_core import registry
from motrix_env_core.config.sim import SimCfg
from motrix_env_core.sim.registry import create_sim_backend
from motrix_env_motrixsim.compiler import MotrixSimSceneCompiler
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg


@dataclass(frozen=True, slots=True)
class PhysicsStepConfig(BenchmarkConfig):
    """Typed configuration for the ``physics.step`` benchmark."""

    robot: str = field(kw_only=True)
    num_envs: tuple[int, ...] = field(kw_only=True)
    numa: str = "single"
    node: int = 0
    dt: float = 0.005
    solver_iterations: int = 3
    nstep: int = 1
    out_json: str = ""
    sim_device: str = "cpu"

    def __post_init__(self) -> None:
        # Direct call: zero-arg super() resolves to the pre-slots class.
        BenchmarkConfig.__post_init__(self)
        if not self.num_envs:
            raise ValueError("num_envs requires at least one value")
        if any(num_envs < 1 for num_envs in self.num_envs):
            raise ValueError("num_envs values must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.solver_iterations < 1:
            raise ValueError("solver_iterations must be positive")
        if self.nstep < 1:
            raise ValueError("nstep must be positive")


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--robot", required=True, help="Registered robot name.")
    parser.add_argument("--sim-device", default="cpu", help="Simulation device, e.g. cpu or cuda:0.")
    parser.add_argument("--num-envs", type=int, nargs="+", default=[2048])
    parser.add_argument("--numa", choices=("single", "node", "shard", "_worker"), default="single")
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--solver-iterations", type=int, default=3)
    parser.add_argument("--nstep", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200, help="Timed iterations; was --iters.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out-json", default="")


def build_config(args: argparse.Namespace) -> PhysicsStepConfig:
    return PhysicsStepConfig(
        steps=args.steps,
        warmup=args.warmup,
        robot=args.robot,
        num_envs=tuple(args.num_envs),
        numa=args.numa,
        node=args.node,
        dt=args.dt,
        solver_iterations=args.solver_iterations,
        nstep=args.nstep,
        out_json=args.out_json,
        sim_device=args.sim_device,
    )


def _detect_numa_nodes() -> list[int]:
    """Return online NUMA node IDs, preserving sparse IDs (for example 0,2)."""
    try:
        text = Path("/sys/devices/system/node/online").read_text().strip()
        nodes: list[int] = []
        for part in text.split(","):
            if "-" in part:
                lo, hi = (int(value) for value in part.split("-", 1))
                nodes.extend(range(lo, hi + 1))
            elif part:
                nodes.append(int(part))
        return nodes or [0]
    except (OSError, ValueError):
        return [0]


def _current_binding() -> str:
    """Return a short string describing the process's current CPU/memory binding."""
    try:
        with open("/proc/self/status") as f:
            status = f.read()
        cpus = mems = "?"
        for line in status.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                cpus = line.split(":", 1)[1].strip()
            if line.startswith("Mems_allowed_list:"):
                mems = line.split(":", 1)[1].strip()
        return f"cpus={cpus} mems={mems}"
    except Exception:
        return "unknown"


def _device_context() -> dict:
    return {
        "num_cpus": os.cpu_count(),
        "num_numa_nodes": len(_detect_numa_nodes()),
        "binding": _current_binding(),
    }


def _prepare_with_info(config: PhysicsStepConfig) -> tuple[PreparedCase, dict]:
    """Prepare the case for ``config.num_envs[0]`` plus its scene metadata."""
    num_envs = config.num_envs[0]
    robot_cfg = registry.make_robot_config(config.robot)
    scene = StandardSceneCfg(objs=StandardSceneObjsCfg(robot=robot_cfg))
    sim = SimCfg(dt=config.dt, solver_iterations=config.solver_iterations, device=config.sim_device)
    sim.validate()
    if torch.device(sim.device).type == "cuda":
        backend = create_sim_backend("motrixsim", sim.device)(scene, sim, num_envs)
        reset = backend.compile_writes({}, reset=True)
        status = torch.empty(num_envs, dtype=torch.int32, device=backend.device)
        case = PreparedCase(
            step=partial(backend.step, config.nstep, result=status),
            before_step=reset.execute,
            synchronize=partial(torch.cuda.synchronize, backend.device),
            items_per_step=float(num_envs * config.nstep),
            unit="env steps",
        )
        return case, _metadata(config, backend.num_actuators, backend.model.num_links)
    model = MotrixSimSceneCompiler().compile(scene, sim)

    data = mtx.SceneData(model, batch=[num_envs])
    data.reset(model)
    model.forward_kinematic(data)

    num_actuators = int(model.num_actuators)
    num_links = int(model.num_links)
    # Constant ctrl held across all substeps in a call; zeros keep the robot
    # quiet. Reset before each call bounds any drift to a single step_n window.
    ctrl = np.zeros((num_envs, num_actuators), dtype=np.float32)

    def reset_state() -> None:
        data.reset(model)
        model.forward_kinematic(data)
        if num_actuators > 0:
            data.actuator_ctrls = np.ascontiguousarray(ctrl)

    case = PreparedCase(
        step=lambda: model.step_n(data, config.nstep),
        before_step=reset_state,
        items_per_step=float(num_envs * config.nstep),
        unit="env steps",
    )
    return case, _metadata(config, num_actuators, num_links)


def _metadata(config: PhysicsStepConfig, num_actuators: int, num_links: int) -> dict:
    return {
        "robot": config.robot,
        "dt": config.dt,
        "solver_iterations": config.solver_iterations,
        "nstep": config.nstep,
        "actuators": num_actuators,
        "links": num_links,
        "device": _device_context(),
    }


def prepare(config: PhysicsStepConfig) -> PreparedCase:
    case, _ = _prepare_with_info(config)
    return case


def _worker_command(config: PhysicsStepConfig, node: int, num_envs: int, out_json: str) -> list[str]:
    """Build a numactl-wrapped ``python -m`` command for one pinned worker."""
    numactl = shutil.which("numactl")
    if numactl is None:
        raise RuntimeError("numactl not found on PATH; install numactl for --numa node/shard")
    return [
        numactl,
        "--cpunodebind",
        str(node),
        "--membind",
        str(node),
        sys.executable,
        "-m",
        "motrixlab.commands.bench",
        "physics.step",
        "--numa",
        "_worker",
        "--node",
        str(node),
        "--num-envs",
        str(num_envs),
        "--robot",
        config.robot,
        "--sim-device",
        config.sim_device,
        "--dt",
        str(config.dt),
        "--solver-iterations",
        str(config.solver_iterations),
        "--nstep",
        str(config.nstep),
        "--warmup",
        str(config.warmup),
        "--steps",
        str(config.steps),
        "--out-json",
        out_json,
    ]


def _run_worker(config: PhysicsStepConfig) -> list[BenchmarkResult]:
    """Internal single-batch mode used by node/shard drivers via a subprocess."""
    if not config.out_json:
        raise ValueError("--numa _worker requires --out-json")
    batch = replace(config, num_envs=(config.num_envs[0],))
    case, metadata = _prepare_with_info(batch)
    result = run_case(
        case,
        batch,
        benchmark="physics.step",
        label=f"node {batch.node} num_envs={batch.num_envs[0]}",
        config=asdict(batch),
        metadata=metadata,
    )
    dump_results(config.out_json, [result])
    return []


def _run_single(config: PhysicsStepConfig) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for num_envs in config.num_envs:
        batch = replace(config, num_envs=(num_envs,))
        case, metadata = _prepare_with_info(batch)
        results.append(
            run_case(
                case,
                batch,
                benchmark="physics.step",
                label=f"single-process (numa=single) num_envs={num_envs}",
                config=asdict(batch),
                metadata=metadata,
            )
        )
    return results


def _run_node(config: PhysicsStepConfig) -> list[BenchmarkResult]:
    """Re-exec the measurement pinned to the requested NUMA node."""
    numa_nodes = _detect_numa_nodes()
    if config.node not in numa_nodes:
        raise ValueError(f"--node {config.node} is not an online NUMA node; available: {numa_nodes}")
    num_envs = config.num_envs[0]
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_json = f.name
    try:
        cmd = _worker_command(config, node=config.node, num_envs=num_envs, out_json=out_json)
        subprocess.run(cmd, check=True, env=os.environ.copy())
        results = load_results(out_json)
    finally:
        Path(out_json).unlink(missing_ok=True)
    label = f"node-pinned (cpunodebind={config.node}, membind={config.node}) num_envs={num_envs}"
    return [replace(result, label=label) for result in results]


def _run_shard(config: PhysicsStepConfig) -> list[BenchmarkResult]:
    """Spawn one pinned worker per NUMA node, each running num_envs/num_nodes envs."""
    numa_nodes = _detect_numa_nodes()
    num_nodes = len(numa_nodes)
    if num_nodes < 2:
        raise RuntimeError(f"--numa shard requires >=2 NUMA nodes; detected {num_nodes}. Use --numa single or node.")
    all_results: list[BenchmarkResult] = []
    for total_envs in config.num_envs:
        if total_envs % num_nodes != 0:
            raise ValueError(f"--num-envs ({total_envs}) must be divisible by num_nodes ({num_nodes})")
        per_node = total_envs // num_nodes
        tmp_paths: list[str] = []
        procs: list[tuple[int, subprocess.Popen[bytes]]] = []
        try:
            for node in numa_nodes:
                fd, tmp_path = tempfile.mkstemp(suffix=f"_node{node}.json")
                os.close(fd)
                tmp_paths.append(tmp_path)
                cmd = _worker_command(config, node=node, num_envs=per_node, out_json=tmp_path)
                procs.append((node, subprocess.Popen(cmd, env=os.environ.copy())))
            for node, p in procs:
                rc = p.wait()
                if rc != 0:
                    raise RuntimeError(f"node {node} worker exited with code {rc}")
            results = [loaded for path in tmp_paths for loaded in load_results(path)]
        finally:
            for _, p in procs:
                if p.poll() is None:
                    p.terminate()
            for _, p in procs:
                if p.poll() is None:
                    p.wait()
            for path in tmp_paths:
                Path(path).unlink(missing_ok=True)

        aggregate_env_steps = sum(result.sustained_items_per_second or 0.0 for result in results)
        slowest_median_ms = max(result.median_ms for result in results)
        notes = [
            f"sharded {total_envs} envs across {num_nodes} NUMA nodes -> {per_node} envs/worker",
            f"aggregate ({len(results)} workers x {per_node} envs = {total_envs} total): "
            f"sum env steps/s {aggregate_env_steps:,.0f}",
            f"slowest worker median {slowest_median_ms:.3f} ms -> "
            f"{total_envs / slowest_median_ms * 1000:,.0f} env steps/s at wall-clock",
        ]
        for result in results:
            metadata = dict(result.metadata)
            metadata["notes"] = list(notes)
            all_results.append(replace(result, metadata=metadata))
    return all_results


def run(config: PhysicsStepConfig) -> list[BenchmarkResult]:
    if config.numa == "_worker":
        return _run_worker(config)
    if config.numa == "single":
        return _run_single(config)
    if config.numa == "node":
        return _run_node(config)
    return _run_shard(config)


def register():
    from .registry import BenchmarkSpec

    return BenchmarkSpec(
        "physics.step",
        "Benchmark native MotrixSim physics throughput.",
        configure,
        build_config,
        prepare,
        run,
    )
