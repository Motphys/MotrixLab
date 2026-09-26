# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""All registered environments must build and step for a few iterations.

Each environment is smoke-tested in its own child process: manager
environments own per-process simulation state, and hosting several live
manager models in one process is not a supported runtime contract, so the
contract under test is per-env build/step in isolation.
"""

import concurrent.futures
import subprocess
import sys

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


def _smoke_in_subprocess(case: tuple[str, int]) -> tuple[str, str | None]:
    env_name, num_env = case
    result = subprocess.run(
        [sys.executable, "-c", _RUNNER, env_name, str(num_env)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        return (f"{env_name} (num_envs={num_env})", tail)
    return (f"{env_name} (num_envs={num_env})", None)


def test_all_demos():
    all_envs = sorted(registry.list_registered_envs())
    cases = [(env_name, num_env) for num_env in (1, 2) for env_name in all_envs]

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for label, error in pool.map(_smoke_in_subprocess, cases):
            if error is not None:
                failures.append(f"{label}:\n{error}")

    assert not failures, f"{len(failures)} env smoke cases failed:\n" + "\n".join(failures)
