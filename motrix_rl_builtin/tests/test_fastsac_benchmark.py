# Copyright Motphys Technology Co., Ltd. 2025, 2026

import argparse

import pytest
from motrix_rl_builtin.fastsac import benchmark


def _args(**overrides) -> argparse.Namespace:
    values = {
        "env": "cartpole",
        "num_envs": [2048],
        "steps": 1000,
        "warmup": 100,
        "ring_capacity": 8,
        "seed": 20260808,
        "sim_backend": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_spec_wires_definition_hooks():
    spec = benchmark.register()
    assert spec.name == benchmark._BENCH_NAME
    assert spec.prepare is not None
    assert spec.run is not None
    config = spec.build_config(_args(num_envs=[64, 256]))
    assert config.num_envs == (64, 256)
    assert config.steps == 1000


@pytest.mark.parametrize(
    "overrides",
    [
        {"steps": 0},
        {"warmup": -1},
        {"ring_capacity": 1},
        {"num_envs": [0]},
        {"num_envs": []},
    ],
)
def test_build_config_rejects_invalid_options(overrides):
    with pytest.raises(ValueError):
        benchmark.register().build_config(_args(**overrides))


def test_run_sweeps_requested_batches(monkeypatch):
    spec = benchmark.register()
    prepared = []

    def fake_prepare(config):
        prepared.append(config.num_envs)
        from motrix_bench import PreparedCase

        return PreparedCase(step=lambda: None, items_per_step=float(config.num_envs[0]))

    monkeypatch.setattr(benchmark, "prepare", fake_prepare)
    results = spec.run(spec.build_config(_args(num_envs=[16, 32], steps=2, warmup=0)))
    assert prepared == [(16,), (32,)]
    assert [result.items_per_step for result in results] == [16.0, 32.0]
    assert results[0].benchmark == benchmark._BENCH_NAME
    assert results[0].metadata["device"] == "cpu"
