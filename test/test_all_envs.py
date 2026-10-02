# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""All registered environments must build and step for a few iterations.

Each environment is smoke-tested in its own child process: manager
environments own per-process simulation state, and hosting several live
manager models in one process is not a supported runtime contract, so the
contract under test is per-env build/step in isolation.

The first whole-body-tracking case runs serially as a warmup: on a cold
numba kernel cache it pays the one-time kernel compilation cost, so the
parallel pool only ever hits the warm cache.
"""

import concurrent.futures
import subprocess
import sys

import pytest

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry

_RUNNER = """
import sys

import numpy as np

import motrix_envs  # noqa: F401 registers built-in environments
from motrix_env_core import registry

env = registry.make(sys.argv[1], num_envs=int(sys.argv[2]))
action_space = env.action_space
action = np.zeros((env.num_envs, *action_space.shape), dtype=action_space.dtype)
for _ in range(10):
    env.step(action)
"""

# Fixed on purpose: the value only trades throughput for memory (four full
# interpreter imports), it must not change test behaviour across machines.
_MAX_WORKERS = 4
_CASE_TIMEOUT_SECONDS = 1500
_WARMUP_ENV = "g1-29dof-wbt-largebox"


def _smoke_in_subprocess(case: tuple[str, int]) -> tuple[str, str | None]:
    env_name, num_env = case
    label = f"{env_name} (num_envs={num_env})"
    try:
        result = subprocess.run(
            [sys.executable, "-c", _RUNNER, env_name, str(num_env)],
            capture_output=True,
            text=True,
            timeout=_CASE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        tail = "\n".join((exc.stderr or "").strip().splitlines()[-5:])
        return (label, f"timed out after {_CASE_TIMEOUT_SECONDS}s\n{tail}")
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        return (label, tail)
    return (label, None)


@pytest.mark.slow
@pytest.mark.integration
def test_all_demos():
    all_envs = sorted(registry.list_registered_envs())
    cases = [(env_name, num_env) for num_env in (1, 2) for env_name in all_envs]

    failures = []
    warmup = [case for case in cases if case[0] == _WARMUP_ENV]
    for case in warmup:
        label, error = _smoke_in_subprocess(case)
        if error is not None:
            failures.append(f"{label}:\n{error}")
    remaining = [case for case in cases if case not in warmup]

    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        for label, error in pool.map(_smoke_in_subprocess, remaining):
            if error is not None:
                failures.append(f"{label}:\n{error}")

    assert not failures, f"{len(failures)} env smoke cases failed:\n" + "\n".join(failures)
