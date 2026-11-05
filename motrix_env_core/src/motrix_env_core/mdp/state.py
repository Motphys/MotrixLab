# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable states for manager-based environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

from motrix_env_core.manager.rand import RandValue, initialize_rand_states

if TYPE_CHECKING:
    from motrix_env_core.manager import ManagerEnv


def _create_rand_value(env: ManagerEnv) -> RandValue:
    """Create the framework-owned per-environment PRNG state."""
    return RandValue(initialize_rand_states(env.num_envs, env._rand_seed))


__all__ = [
    "RandValue",
]
