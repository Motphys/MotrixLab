# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Benchmark what live metrics cost a training loop, healthy and when its outputs fail.

Every scenario runs the same synthetic training loop and logs through
``TrainingLogger``, the interface trainers use, with TensorBoard on in every
scenario: the difference to ``off`` (live metrics disabled) is the cost of live
metrics alone. Faulty scenarios break the jsonl disk (slow, stuck for good,
failing) or the ingest service (slow, never answering, refusing connections),
and one breaks both at once.

The report gives the wall time of each ``scalars`` call on the training
thread, the loop's throughput against ``off`` (``off, control`` repeats ``off``
and shows the noise floor), and what reached the disk and the service. Only the
training loop is measured: closing is not timed, and a faulty scenario closes
without waiting so it never sits out the close timeout (at most 5 s, once, at
the end of a real run; see wiki/design/live-metrics.md). The ingest service runs
in a separate process, like the real one, so its work does not compete for this
process's GIL. ``--workload python`` holds the GIL in the training loop itself,
the worst case for the sink threads. ``--max-overhead`` turns the run into a
check that exits non-zero when a scenario slows the loop down by more.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing
import socket
import statistics
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from report import render_table

from motrix_rl.metrics import JSONL_FILENAME, HttpSink, JsonlSink, MetricsLogger
from motrix_rl.training_logger import TrainingLogger

_HEADERS = [
    "scenario",
    "loop it/s",
    "overhead",
    "emit p50 us",
    "emit p99 us",
    "emit max us",
    "jsonl lines",
    "delivered",
]

# Disk faults, standing in for a slow, hung or broken run_dir.
_SLOW, _STUCK, _ERROR = "slow", "stuck", "error"
# Service faults: answers after a delay, accepts but never answers, refuses connections.
_OK, _SLOW_SERVICE, _SILENT, _DOWN = "ok", "slow", "silent", "down"


@dataclass(frozen=True, slots=True)
class _Scenario:
    name: str
    jsonl: str | None = None  # None: off; "ok" or a disk fault
    http: str | None = None  # None: off; a service mode

    @property
    def faulty(self) -> bool:
        return self.jsonl not in (None, _OK) or self.http not in (None, _OK)


@dataclass(slots=True)
class _Result:
    its_per_s: float
    emit_ns: list[int]
    jsonl_lines: int | None
    delivered: int | None


SCENARIOS = [
    _Scenario("off"),
    _Scenario("off, control"),  # the same as "off": its overhead is the noise floor
    _Scenario("jsonl", jsonl=_OK),
    _Scenario("jsonl, slow disk", jsonl=_SLOW),
    _Scenario("jsonl, stuck disk", jsonl=_STUCK),
    _Scenario("jsonl, disk error", jsonl=_ERROR),
    _Scenario("http", http=_OK),
    _Scenario("http, slow service", http=_SLOW_SERVICE),
    _Scenario("http, no response", http=_SILENT),
    _Scenario("http, service down", http=_DOWN),
    _Scenario("jsonl + http", jsonl=_OK, http=_OK),
    _Scenario("stuck disk + no response", jsonl=_STUCK, http=_SILENT),
]


class _FaultyDisk(JsonlSink):
    """A jsonl sink whose batch writes stall for *delay* seconds, or forever."""

    def __init__(self, path: Path, delay: float | None) -> None:
        self._delay = delay
        self._never = threading.Event()  # never set: a write that hangs for good
        super().__init__(path)

    def _write_batch(self, batch):
        if self._delay is None:
            self._never.wait()
        else:
            time.sleep(self._delay)
        super()._write_batch(batch)


def _serve(port_queue, delivered, delay: float | None) -> None:
    """Ingest service for the child process: counts records, answers 200 after *delay* (never if None)."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = gzip.decompress(self.rfile.read(int(self.headers["Content-Length"])))
            if delay is None:
                threading.Event().wait()  # hold the connection open without ever answering
            if delay:
                time.sleep(delay)
            with delivered.get_lock():
                delivered.value += len(json.loads(body)["records"])
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    port_queue.put(server.server_address[1])
    server.serve_forever()


class _Service:
    def __init__(self, ctx, delay: float | None) -> None:
        port_queue = ctx.Queue()
        self.delivered = ctx.Value("q", 0)
        self._process = ctx.Process(target=_serve, args=(port_queue, self.delivered, delay), daemon=True)
        self._process.start()
        self.url = f"http://127.0.0.1:{port_queue.get(timeout=30)}"

    def take_count(self) -> int:
        with self.delivered.get_lock():
            count, self.delivered.value = self.delivered.value, 0
        return count

    def stop(self) -> None:
        self._process.terminate()
        self._process.join()


def _unreachable_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"  # nothing listens once the socket is closed


def _numpy_workload(step_ms: float, size: int):
    """A numpy step of about *step_ms* milliseconds; BLAS releases the GIL, like most of a torch step."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((size, size), dtype=np.float32)
    b = rng.standard_normal((size, size), dtype=np.float32)
    for _ in range(50):  # warm up BLAS threads before timing
        a @ b

    def once():
        a @ b

    return _repeat_to(once, step_ms)


def _python_workload(step_ms: float):
    """A pure-Python step of about *step_ms* milliseconds that holds the GIL throughout."""

    def once():
        total = 0
        for i in range(1000):
            total += i * i
        return total

    return _repeat_to(once, step_ms)


def _repeat_to(once, step_ms: float):
    count = 0
    started = time.perf_counter()
    while time.perf_counter() - started < 0.5:
        once()
        count += 1
    repeats = max(1, round(step_ms / 1000 / ((time.perf_counter() - started) / count)))

    def step():
        for _ in range(repeats):
            once()

    return step


def _live(scenario: _Scenario, run_dir: Path, args, services: dict[str, _Service], down_url: str) -> MetricsLogger:
    jsonl = None
    path = run_dir / JSONL_FILENAME
    if scenario.jsonl == _OK:
        jsonl = JsonlSink(path)
    elif scenario.jsonl == _SLOW:
        jsonl = _FaultyDisk(path, args.disk_delay)
    elif scenario.jsonl == _STUCK:
        jsonl = _FaultyDisk(path, None)
    elif scenario.jsonl == _ERROR:
        (run_dir / "not-a-dir").write_text("")
        jsonl = JsonlSink(run_dir / "not-a-dir" / JSONL_FILENAME)  # every write fails with NotADirectoryError
    http = None
    if scenario.http is not None:
        http = HttpSink(down_url if scenario.http == _DOWN else services[scenario.http].url)
    return MetricsLogger("bench", jsonl=jsonl, http=http) if jsonl or http else MetricsLogger.disabled()


def _run(scenario: _Scenario, args, workload, services: dict[str, _Service], down_url: str) -> _Result:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        live = _live(scenario, run_dir, args, services, down_url)
        log = TrainingLogger(run_dir, live)
        values = {f"train/key_{index}": float(index) for index in range(args.keys)}
        sys_samples = {"sys/cpu_percent": 50.0, "sys/mem_used_gb": 8.0}
        log.scalars({}, 0, env_steps=0)  # opens TensorBoard (a one-off startup cost) before timing; sends nothing

        emit_ns = []
        started = time.perf_counter()
        for iteration in range(args.iters):
            workload()
            if iteration % args.log_every == 0:
                values["train/key_0"] = float(iteration)
                emit_started = time.perf_counter_ns()
                log.scalars(values, iteration, env_steps=iteration * 4096, live_extra=lambda: sys_samples)
                emit_ns.append(time.perf_counter_ns() - emit_started)
        elapsed = time.perf_counter() - started

        # Not timed. Healthy sinks flush in milliseconds; a faulty one would only
        # sit out the close timeout, so it is left behind instead.
        log.close()
        live.close()
        if not scenario.faulty:
            live.join(5)
        path = run_dir / JSONL_FILENAME
        lines = len(path.read_text().splitlines()) if scenario.jsonl and path.exists() else None
        if scenario.jsonl is not None and lines is None:
            lines = 0
    delivered = None
    if scenario.http in services:
        delivered = services[scenario.http].take_count()
    elif scenario.http == _DOWN:
        delivered = 0
    return _Result(args.iters / elapsed, emit_ns, lines, delivered)


def _percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] / 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iters", type=int, default=2000, help="Training-loop iterations per run.")
    parser.add_argument("--step-ms", type=float, default=2.0, help="Target duration of one workload step.")
    parser.add_argument("--workload", choices=["numpy", "python"], default="numpy", help="What a step spends time on.")
    parser.add_argument("--matrix", type=int, default=256, help="Side of the square matmul in the numpy workload.")
    parser.add_argument("--log-every", type=int, default=1, help="Iterations between scalars() calls (1 = worst case).")
    parser.add_argument("--keys", type=int, default=40, help="Scalars per record.")
    parser.add_argument("--repeats", type=int, default=5, help="Runs per scenario; throughput is the median.")
    parser.add_argument("--disk-delay", type=float, default=0.5, help="Stall per batch for the slow-disk scenario.")
    parser.add_argument("--service-delay", type=float, default=0.5, help="Reply delay for the slow-service scenario.")
    parser.add_argument(
        "--only", nargs="+", metavar="NAME", help="Run only scenarios whose name contains one of these."
    )
    parser.add_argument(
        "--max-overhead",
        type=float,
        default=None,
        help="Fail (exit 1) when a scenario's median throughput is this many percent below 'off'.",
    )
    args = parser.parse_args()
    # The faulty scenarios warn once per sink by design; keep the report readable.
    logging.getLogger("motrix_rl").setLevel(logging.ERROR)

    scenarios = SCENARIOS
    if args.only:
        scenarios = [s for s in SCENARIOS if s.name.startswith("off") or any(o in s.name for o in args.only)]
    ctx = multiprocessing.get_context("spawn")
    services = {_OK: _Service(ctx, 0.0), _SLOW_SERVICE: _Service(ctx, args.service_delay), _SILENT: _Service(ctx, None)}
    down_url = _unreachable_url()
    workload = (
        _numpy_workload(args.step_ms, args.matrix) if args.workload == "numpy" else _python_workload(args.step_ms)
    )
    results: dict[str, list[_Result]] = {scenario.name: [] for scenario in scenarios}
    try:
        for repeat in range(args.repeats):
            # Rotate the order so drift (thermal, other load) does not always hit the same scenario.
            shift = repeat % len(scenarios)
            for scenario in scenarios[shift:] + scenarios[:shift]:
                results[scenario.name].append(_run(scenario, args, workload, services, down_url))
    finally:
        for service in services.values():
            service.stop()

    baseline = statistics.median(result.its_per_s for result in results["off"])
    rows = []
    worst = 0.0
    for scenario in scenarios:
        runs = results[scenario.name]
        its = statistics.median(run.its_per_s for run in runs)
        overhead = (1 - its / baseline) * 100
        if not scenario.name.startswith("off"):
            worst = max(worst, overhead)
        emits = [ns for run in runs for ns in run.emit_ns]
        last = runs[-1]
        rows.append(
            [
                scenario.name,
                f"{its:,.1f}",
                "—" if scenario.name == "off" else f"{overhead:+.2f}%",
                f"{_percentile(emits, 0.50):.1f}",
                f"{_percentile(emits, 0.99):.1f}",
                f"{max(emits) / 1000:.1f}",
                "—" if last.jsonl_lines is None else str(last.jsonl_lines),
                "—" if last.delivered is None else str(last.delivered),
            ]
        )

    print(render_table(_HEADERS, rows))
    emitted = len(range(0, args.iters, args.log_every))
    print(
        f"loop: workload={args.workload} iters={args.iters} step~{args.step_ms}ms log_every={args.log_every} "
        f"keys={args.keys} repeats={args.repeats}; {emitted} records per run; TensorBoard on in every scenario"
    )
    print(f"slow disk: {args.disk_delay}s stall per batch; slow service: {args.service_delay}s per reply")
    print(
        "overhead: median loop throughput against 'off', read against the 'off, control' noise floor; "
        "jsonl lines / delivered: from the last run"
    )
    if args.max_overhead is not None and worst > args.max_overhead:
        print(f"FAIL: worst overhead {worst:.2f}% exceeds --max-overhead {args.max_overhead}%")
        sys.exit(1)


if __name__ == "__main__":
    main()
