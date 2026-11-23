# Copyright Motphys Technology Co., Ltd. 2025, 2026

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from motrixlab.benchmark import env as env_benchmark
from motrixlab.commands import bench

from motrix_bench import SCHEMA_VERSION, BenchmarkResult, PreparedCase


@pytest.fixture
def benchmark():
    return bench


def _result(**overrides):
    keywords: dict = {
        "benchmark": "env.step",
        "label": "cartpole (num_envs=2048)",
        "steps": 1000,
        "warmup": 10,
        "elapsed_seconds": 1.0,
        "samples_ms": (1.0, 1.0, 1.0),
        "p10_ms": 1.0,
        "median_ms": 1.0,
        "p90_ms": 1.0,
        "items_per_step": 2048.0,
        "sustained_items_per_second": 2048000.0,
        "equivalent_items_per_second": 2048000.0,
    }
    keywords.update(overrides)
    return BenchmarkResult(SCHEMA_VERSION, **keywords)


def test_named_dispatch_renders_env_step_results(benchmark, monkeypatch, capsys):
    run_calls = []

    def run(config):
        run_calls.append(config)
        return [_result()]

    monkeypatch.setattr(env_benchmark, "run", run)
    monkeypatch.setattr(sys, "argv", ["mxlab bench", "env.step", "--env", "cartpole"])
    assert benchmark.main() == 0
    out = capsys.readouterr().out
    assert "env.step" in out
    assert "cartpole (num_envs=2048)" in out
    assert run_calls[0].env_name == "cartpole"
    assert run_calls[0].num_envs == (2048,)
    assert run_calls[0].steps == 1000
    assert run_calls[0].warmup == 10


def test_named_dispatch_env_step_json(benchmark, monkeypatch, capsys):
    monkeypatch.setattr(env_benchmark, "run", lambda config: [_result()])
    monkeypatch.setattr(sys, "argv", ["mxlab bench", "env.step", "--env", "cartpole", "--json"])
    assert benchmark.main() == 0
    out = capsys.readouterr().out.strip()
    assert '"benchmark": "env.step"' in out
    assert '"label": "cartpole (num_envs=2048)"' in out


def test_env_step_runs_each_requested_batch(monkeypatch):
    configs = []

    def fake_prepare_case(config, num_envs):
        configs.append(num_envs)
        return SimpleNamespace(perf=Mock()), PreparedCase(
            step=lambda: None,
            items_per_step=float(num_envs),
        )

    monkeypatch.setattr(
        "motrixlab.benchmark.workload.prepare_case",
        fake_prepare_case,
    )
    config = env_benchmark.EnvStepConfig(steps=3, warmup=1, env_name="cartpole", num_envs=(4, 8), zero_actions=True)
    results = env_benchmark.run(config)
    assert configs == [4, 8]
    assert [result.items_per_step for result in results] == [4.0, 8.0]
    assert [result.label for result in results] == ["cartpole (num_envs=4)", "cartpole (num_envs=8)"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--steps", "0"],
        ["--steps", "-1"],
        ["--warmup", "-1"],
        ["--num-envs", "0"],
    ],
)
def test_invalid_measurement_options_fail_before_runtime(benchmark, monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["mxlab bench", "env.step", "--env", "cartpole", *arguments])
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2
