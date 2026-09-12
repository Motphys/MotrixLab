# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""MotrixSim physics-only throughput benchmark with NUMA-aware modes.

Drives MotrixSim directly: builds a robot-on-flat-ground scene via the scene
compiler, allocates batched SceneData, and times ``model.step_n(data, nstep)``.
No task env, no observation/reward/termination, no numba -- the number is
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
          validated in wiki/design/numba-server-perf.md section 6.3.

Reports per-call latency (median + p10/p90), batch steps/s (step_n calls per
second over all parallel envs in the process), and total env steps/s.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import motrixsim as mtx
import numpy as np
from report import render_table

import motrix_envs  # noqa: F401  registers robots and scene assets
from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_motrixsim.compiler import MotrixSimSceneCompiler
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg


def _detect_num_nodes() -> int:
    """Count online NUMA nodes from /sys; return 1 if unavailable / single-NUMA."""
    try:
        text = Path("/sys/devices/system/node/online").read_text().strip()
        # formats like "0" or "0-1" or "0,2"
        if "-" in text:
            lo, hi = text.split("-")
            return int(hi) - int(lo) + 1
        return len([x for x in text.split(",") if x])
    except Exception:
        return 1


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
    ctx = {
        "num_cpus": os.cpu_count(),
        "num_numa_nodes": _detect_num_nodes(),
        "binding": _current_binding(),
    }
    try:
        import numba as _numba

        ctx["numba_threads"] = _numba.get_num_threads()
        ctx["numba_layer"] = _numba.threading_layer()
    except Exception:
        pass
    return ctx


def measure(
    *,
    robot: str,
    num_envs: int,
    dt: float,
    solver_iterations: int,
    nstep: int,
    warmup: int,
    iters: int,
) -> dict:
    """Run the physics-only measurement in this process. Returns a metrics dict."""
    robot_cfg = registry.make_robot_config(robot)
    scene = StandardSceneCfg(objs=StandardSceneObjsCfg(robot=robot_cfg))
    sim = SimCfg(dt=dt, solver_iterations=solver_iterations)
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

    for _ in range(warmup):
        reset_state()
        model.step_n(data, nstep)

    samples = np.empty((iters,), dtype=np.float64)
    for i in range(iters):
        reset_state()
        start = time.perf_counter()
        model.step_n(data, nstep)
        samples[i] = time.perf_counter() - start

    median_s = float(np.median(samples))
    p10_s = float(np.percentile(samples, 10))
    p90_s = float(np.percentile(samples, 90))
    batch_steps_per_s = 1.0 / median_s
    # Each step_n call advances nstep native substeps across num_envs envs.
    total_env_steps_per_s = batch_steps_per_s * num_envs * nstep

    return {
        "robot": robot,
        "num_envs": num_envs,
        "dt": dt,
        "solver_iterations": solver_iterations,
        "nstep": nstep,
        "iters": iters,
        "actuators": num_actuators,
        "links": num_links,
        "median_ms": median_s * 1e3,
        "p10_ms": p10_s * 1e3,
        "p90_ms": p90_s * 1e3,
        "batch_steps_per_s": batch_steps_per_s,
        "total_env_steps_per_s": total_env_steps_per_s,
    }


_RESULT_HEADERS = [
    "mode",
    "robot",
    "num_envs",
    "median ms",
    "p10 ms",
    "p90 ms",
    "batch steps/s",
    "total env steps/s",
]


def _result_row(label: str, r: dict) -> list[str]:
    return [
        label,
        str(r["robot"]),
        str(r["num_envs"]),
        f"{r['median_ms']:.3f}",
        f"{r['p10_ms']:.3f}",
        f"{r['p90_ms']:.3f}",
        f"{r['batch_steps_per_s']:.1f}",
        f"{r['total_env_steps_per_s']:,.0f}",
    ]


def run_single(args) -> tuple[list[list[str]], list[str]]:
    r = measure(
        robot=args.robot,
        num_envs=args.num_envs,
        dt=args.dt,
        solver_iterations=args.solver_iterations,
        nstep=args.nstep,
        warmup=args.warmup,
        iters=args.iters,
    )
    return ([_result_row(f"single-process ({args.numa})", r)], [])


def _worker_entry(args) -> None:
    """Internal: run one measurement and write JSON to --out-json. Used by node/shard drivers."""
    r = measure(
        robot=args.robot,
        num_envs=args.num_envs,
        dt=args.dt,
        solver_iterations=args.solver_iterations,
        nstep=args.nstep,
        warmup=args.warmup,
        iters=args.iters,
    )
    r["node"] = args.node
    Path(args.out_json).write_text(json.dumps(r), encoding="utf-8")


def _numactl_wrap(node: int, py_args: list[str]) -> list[str]:
    numactl = shutil.which("numactl")
    if numactl is None:
        raise RuntimeError("numactl not found on PATH; install numactl for --numa node/shard")
    return [numactl, "--cpunodebind", str(node), "--membind", str(node)] + py_args


def _measure_via_subprocess(args, node: int, num_envs: int, out_json: str) -> dict:
    """Spawn a pinned worker that runs `measure` and writes JSON; block until done."""
    script = str(Path(sys.argv[0]).resolve())
    py_args = [
        sys.executable,
        script,
        "--numa",
        "_worker",
        "--node",
        str(node),
        "--num-envs",
        str(num_envs),
        "--robot",
        args.robot,
        "--dt",
        str(args.dt),
        "--solver-iterations",
        str(args.solver_iterations),
        "--nstep",
        str(args.nstep),
        "--warmup",
        str(args.warmup),
        "--iters",
        args.iters if isinstance(args.iters, str) else str(args.iters),
        "--out-json",
        out_json,
    ]
    cmd = _numactl_wrap(node, py_args)
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)
    return json.loads(Path(out_json).read_text())


def run_node(args) -> tuple[list[list[str]], list[str]]:
    """Re-exec the measurement pinned to NUMA node 0."""
    notes: list[str] = []
    num_nodes = _detect_num_nodes()
    if num_nodes < 2:
        notes.append(f"only {num_nodes} NUMA node(s) detected; --numa node is a no-op here")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_json = f.name
    try:
        r = _measure_via_subprocess(args, node=0, num_envs=args.num_envs, out_json=out_json)
        return ([_result_row("node-pinned (cpunodebind=0, membind=0)", r)], notes)
    finally:
        Path(out_json).unlink(missing_ok=True)


def run_shard(args) -> tuple[list[list[str]], list[str]]:
    """Spawn one pinned worker per NUMA node, each running num_envs/num_nodes envs."""
    num_nodes = _detect_num_nodes()
    if num_nodes < 2:
        raise RuntimeError(f"--numa shard requires >=2 NUMA nodes; detected {num_nodes}. Use --numa single or node.")
    if args.num_envs % num_nodes != 0:
        raise ValueError(f"--num-envs ({args.num_envs}) must be divisible by num_nodes ({num_nodes})")
    per_node = args.num_envs // num_nodes
    tmp_paths = []
    procs = []
    for node in range(num_nodes):
        fd, tmp_path = tempfile.mkstemp(suffix=f"_node{node}.json")
        os.close(fd)
        tmp_paths.append(tmp_path)
        script = str(Path(sys.argv[0]).resolve())
        py_args = [
            sys.executable,
            script,
            "--numa",
            "_worker",
            "--node",
            str(node),
            "--num-envs",
            str(per_node),
            "--robot",
            args.robot,
            "--dt",
            str(args.dt),
            "--solver-iterations",
            str(args.solver_iterations),
            "--nstep",
            str(args.nstep),
            "--warmup",
            str(args.warmup),
            "--iters",
            args.iters if isinstance(args.iters, str) else str(args.iters),
            "--out-json",
            tmp_path,
        ]
        cmd = _numactl_wrap(node, py_args)
        procs.append((node, subprocess.Popen(cmd, env=os.environ.copy())))
    for node, p in procs:
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"node {node} worker exited with code {rc}")

    results = [json.loads(Path(p).read_text()) for p in tmp_paths]
    for p in tmp_paths:
        Path(p).unlink(missing_ok=True)

    aggregate_env_steps = sum(r["total_env_steps_per_s"] for r in results)
    slowest_median = max(r["median_ms"] for r in results)
    notes = [
        f"sharded {args.num_envs} envs across {num_nodes} NUMA nodes -> {per_node} envs/worker",
        f"aggregate ({len(results)} workers x {per_node} envs = {args.num_envs} total): "
        f"sum total env steps/s {aggregate_env_steps:,.0f}",
        f"slowest worker median {slowest_median:.3f} ms -> "
        f"{args.num_envs / slowest_median * 1000:,.0f} env steps/s at wall-clock",
        "compare: single-process total env steps/s is the number to beat; see wiki/design/numba-server-perf.md 6.3",
    ]
    rows = [_result_row(f"shard node {r['node']}", r) for r in results]
    return rows, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="g1-29dof")
    parser.add_argument(
        "--num-envs",
        type=int,
        nargs="+",
        default=[2048],
        help="Batch sizes to benchmark. Multiple values run a sweep.",
    )
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--solver-iterations", type=int, default=3)
    parser.add_argument("--nstep", type=int, default=1, help="Native substeps per step_n call.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--numa",
        choices=["single", "node", "shard", "_worker"],
        default="single",
        help="single=in-process; node=re-exec pinned to node 0; shard=one worker per NUMA node.",
    )
    parser.add_argument("--node", type=int, default=0, help="(internal worker) which node.")
    parser.add_argument("--out-json", type=str, default="", help="(internal worker) JSON output path.")
    args = parser.parse_args()
    if any(num_envs < 1 for num_envs in args.num_envs):
        parser.error("--num-envs values must be positive")

    if args.numa == "_worker":
        if not args.out_json:
            raise SystemExit("--numa _worker requires --out-json")
        if len(args.num_envs) != 1:
            parser.error("internal worker mode requires exactly one --num-envs value")
        args.num_envs = args.num_envs[0]
        _worker_entry(args)
        return

    num_envs_values = args.num_envs
    rows: list[list[str]] = []
    notes: list[str] = []
    for num_envs in num_envs_values:
        args.num_envs = num_envs
        if args.numa == "single":
            batch_rows, batch_notes = run_single(args)
        elif args.numa == "node":
            batch_rows, batch_notes = run_node(args)
        else:
            batch_rows, batch_notes = run_shard(args)
        rows.extend(batch_rows)
        for note in batch_notes:
            if note not in notes:
                notes.append(note)

    context = _device_context()
    print(render_table(_RESULT_HEADERS, rows))
    print(
        f"sim: robot={args.robot} dt={args.dt}s solver_iterations={args.solver_iterations} "
        f"nstep={args.nstep} warmup={args.warmup} iters={args.iters}"
    )
    print(f"context: cpus={context['num_cpus']} numa_nodes={context['num_numa_nodes']} binding=[{context['binding']}]")
    for note in notes:
        print(note)


if __name__ == "__main__":
    main()
