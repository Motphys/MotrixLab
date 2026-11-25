# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Environment preparation shared by benchmark measurements.

Runtime imports are deliberately lazy so workers can apply NUMA placement first.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from motrix_bench import PreparedCase

if TYPE_CHECKING:
    from .env import EnvStepConfig


def load_runtime():
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("GOMP_SPINCOUNT", "0")
    import numpy as np

    from motrix_env_core import registry
    from motrix_env_core.direct.env import DirectEnv
    from motrix_env_core.manager import ManagerEnv

    return np, registry, (DirectEnv, ManagerEnv)


def prepare_case(config: EnvStepConfig, num_envs: int) -> tuple[object, PreparedCase]:
    """Create one environment and its standard :class:`PreparedCase`."""
    np, registry, manager_env_type = load_runtime()
    name = config.env_name
    kwargs = {"mode": config.mode, "num_envs": num_envs}
    if config.sim_backend is not None:
        kwargs["sim"] = config.sim_backend
    env = registry.make(name, **kwargs)
    try:
        actual_num_envs = int(env.num_envs)
        if actual_num_envs != num_envs:
            raise ValueError(f"Environment {name!r} created {actual_num_envs} environments, requested {num_envs}.")
        space = env.action_space
        shape = (actual_num_envs, *space.shape)
        if config.zero_actions:
            actions = np.zeros(shape, dtype=space.dtype)
        else:
            low = np.where(np.isfinite(space.low), space.low, -1.0)
            high = np.where(np.isfinite(space.high), space.high, 1.0)
            actions = np.random.default_rng(config.seed).uniform(low, high, size=shape).astype(space.dtype)
        if isinstance(env, manager_env_type):
            import torch

            actions = torch.from_numpy(np.ascontiguousarray(actions))
        case = PreparedCase(
            step=lambda: env.step(actions),
            close=lambda: close_env(env),
            items_per_step=float(actual_num_envs),
            unit="env steps",
        )
        return env, case
    except BaseException:
        close_env(env)
        raise


def close_env(env) -> None:
    close = getattr(env, "close", None)
    if close is not None:
        close()
