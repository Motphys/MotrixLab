# Copyright Motphys Technology Co., Ltd. 2025, 2026

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from motrixlab.benchmark import workload
from motrixlab.benchmark.env import EnvStepConfig


def test_benchmark_modules_import_without_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import motrixlab.commands.bench; import motrixlab.benchmark.runner; "
            "import motrixlab.benchmark.workload; "
            "assert not {'numpy', 'torch', 'motrixsim'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_prepare_case_builds_standard_case_and_closes_environment(monkeypatch):
    space = SimpleNamespace(
        shape=(2,), dtype=np.dtype("float32"), low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0])
    )
    reference = SimpleNamespace(action_space=space, num_envs=3, step=Mock(), close=Mock())

    def make(name, *, mode, num_envs, sim=None):
        assert (name, mode, num_envs, sim) == ("reference", "train", 3, "test-backend")
        return reference

    registry = SimpleNamespace(make=make)
    monkeypatch.setattr(workload, "load_runtime", lambda: (np, registry, dict))
    config = EnvStepConfig(steps=2, warmup=0, env_name="reference", num_envs=(3,), seed=123, sim_backend="test-backend")
    env, case = workload.prepare_case(config, 3)
    assert env is reference
    assert case.items_per_step == 3
    case.step()
    reference.step.assert_called_once()
    case.close_once()
    reference.close.assert_called_once()


def test_prepare_case_rejects_env_that_changes_requested_batch(monkeypatch):
    space = SimpleNamespace(
        shape=(2,), dtype=np.dtype("float32"), low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0])
    )
    clamped = SimpleNamespace(action_space=space, num_envs=2, step=Mock(), close=Mock())
    registry = SimpleNamespace(make=lambda *args, **kwargs: clamped)
    monkeypatch.setattr(workload, "load_runtime", lambda: (np, registry, dict))
    config = EnvStepConfig(steps=2, warmup=0, env_name="reference", num_envs=(3,))
    try:
        workload.prepare_case(config, 3)
    except ValueError as error:
        assert "requested 3" in str(error)
    else:
        raise AssertionError("expected ValueError")
    clamped.close.assert_called_once()
