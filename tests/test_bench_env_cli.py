# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from bench import bench_env
from bench.env_benchmark import runner, workload


@pytest.fixture
def benchmark():
    return bench_env


def test_default_is_whole_machine_duration_without_parent_runtime_import(benchmark, monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["bench_env.py", "--json"])
    monkeypatch.setattr(workload, "load_runtime", lambda: pytest.fail("parent must stay lightweight"))

    def run(args, total):
        calls.append((args.duration, args.numa, total))
        return [{"env_steps_per_s": 1.0}]

    monkeypatch.setattr(runner, "run_duration", run)
    benchmark.main()
    assert len(calls) == 1
    seconds, mode, total = calls[0]
    assert seconds > 0
    assert mode == "shard"
    assert total > 0


@pytest.mark.parametrize(
    "arguments",
    [
        ["--duration", "nan"],
        ["--duration", "inf"],
        ["--duration", "0"],
        ["--duration", "-1"],
        ["--duration", "1", "--steps", "1"],
        ["--duration", "1", "--breakdown"],
        ["--steps", "1", "--numa", "shard"],
    ],
)
def test_invalid_measurement_options_fail_before_runtime(benchmark, monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["bench_env.py", *arguments])
    monkeypatch.setattr(workload, "load_runtime", lambda: pytest.fail("must not load runtime"))
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
