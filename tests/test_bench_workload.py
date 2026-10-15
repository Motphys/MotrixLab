# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from bench.env_benchmark import workload


def test_benchmark_modules_import_without_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import bench.bench_env; import bench.env_benchmark.runner; "
            "import bench.env_benchmark.workload; "
            "assert not {'numpy', 'torch', 'numba', 'motrixsim'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_comparison_reuses_actions_and_closes_incompatible_environment(monkeypatch):
    space = SimpleNamespace(
        shape=(2,), dtype=np.dtype("float32"), low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0])
    )
    reference = SimpleNamespace(action_space=space, num_envs=3, close=Mock())
    compared = SimpleNamespace(action_space=space, num_envs=3, close=Mock())
    incompatible = SimpleNamespace(
        action_space=SimpleNamespace(shape=(1,), dtype=space.dtype, low=np.array([-1.0]), high=np.array([1.0])),
        num_envs=3,
        close=Mock(),
    )
    environments = iter([reference, compared, incompatible])
    registry = SimpleNamespace(make=lambda *args, **kwargs: next(environments))
    monkeypatch.setattr(workload, "load_runtime", lambda: (np, registry))
    args = SimpleNamespace(mode="train", sim_backend=None, zero_actions=False, seed=123)
    env, actions, spec = workload.prepare_env("reference", args, 3)
    assert env is reference
    assert actions.shape == (3, 2)
    assert actions.dtype == space.dtype
    env, reused, _ = workload.prepare_env("compared", args, 3, actions, spec)
    assert env is compared
    assert reused is actions
    with pytest.raises(ValueError, match="action space"):
        workload.prepare_env("incompatible", args, 3, actions, spec)
    incompatible.close.assert_called_once()
