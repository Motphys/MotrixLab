# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Environment preparation shared by latency and throughput measurements.

Runtime imports are deliberately lazy so workers can apply NUMA placement first.
"""

from __future__ import annotations

import argparse
import os


def load_runtime():
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("GOMP_SPINCOUNT", "0")
    import numpy as np

    from motrix_env_core import registry

    return np, registry


def prepare_env(
    name: str,
    args: argparse.Namespace,
    num_envs: int,
    actions=None,
    expected_action_spec=None,
):
    """Create an environment and compatible fixed actions, checking comparisons."""
    np, registry = load_runtime()
    kwargs = {"mode": args.mode, "num_envs": num_envs}
    if args.sim_backend is not None:
        kwargs["sim_backend"] = args.sim_backend
    env = registry.make(name, **kwargs)
    try:
        space = env.action_space
        action_spec = space.shape, space.dtype, np.asarray(space.low), np.asarray(space.high)
        if expected_action_spec is not None:
            expected_shape, expected_dtype, expected_low, expected_high = expected_action_spec
            shape, dtype, low, high = action_spec
            if shape != expected_shape or dtype != expected_dtype:
                raise ValueError(
                    f"Cannot compare {name!r}: action space {(shape, dtype)} does not match "
                    f"the reference {(expected_shape, expected_dtype)}."
                )
            if not np.array_equal(low, expected_low) or not np.array_equal(high, expected_high):
                raise ValueError(f"Cannot compare {name!r}: action bounds do not match the reference environment.")
        if actions is None:
            shape = (env.num_envs, *space.shape)
            if args.zero_actions:
                actions = np.zeros(shape, dtype=space.dtype)
            else:
                low = np.where(np.isfinite(space.low), space.low, -1.0)
                high = np.where(np.isfinite(space.high), space.high, 1.0)
                actions = np.random.default_rng(args.seed).uniform(low, high, size=shape).astype(space.dtype)
        return env, actions, action_spec
    except BaseException:
        close_env(env)
        raise


def warmup(env, actions, count: int) -> None:
    for _ in range(count):
        env.step(actions)


def close_env(env) -> None:
    close = getattr(env, "close", None)
    if close is not None:
        close()
