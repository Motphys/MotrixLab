# Copyright Motphys Technology Co., Ltd. 2025, 2026

import sys

import pytest
from motrixlab.benchmark import collector
from motrixlab.commands import bench

from motrix_bench import BenchmarkResult


def _fake_result(benchmark: str) -> BenchmarkResult:
    return BenchmarkResult(
        schema_version=1,
        benchmark=benchmark,
        label="cartpole",
        steps=2,
        warmup=0,
        elapsed_seconds=0.001,
        samples_ms=(2.0, 2.0),
        p10_ms=2.0,
        median_ms=2.0,
        p90_ms=2.0,
        items_per_step=1.0,
        sustained_items_per_second=1000.0,
        equivalent_items_per_second=500.0,
    )


def test_collector_benchmark_dispatches_and_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(collector, "run", lambda config: [_fake_result("fastsac.collector")])
    monkeypatch.setattr(sys, "argv", ["mxlab bench", "fastsac.collector", "--env", "cartpole", "--json"])
    bench.main()
    assert '"benchmark": "fastsac.collector"' in capsys.readouterr().out


def test_collector_config_rejects_small_ring_capacity():
    with pytest.raises(ValueError):
        collector.CollectorConfig(steps=1, warmup=0, env_name="cartpole", num_envs=(1,), ring_capacity=1)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--steps", "0"],
        ["--ring-capacity", "1"],
        ["--num-envs", "0"],
    ],
)
def test_collector_invalid_options_fail_before_runtime(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["mxlab bench", "fastsac.collector", "--env", "cartpole", *arguments])
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2
