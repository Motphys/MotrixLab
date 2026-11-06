# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable states for manager-based environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from motrix_env_core.manager.rand import RandValue, initialize_rand_states

if TYPE_CHECKING:
    from motrix_env_core.manager import ManagerEnv


def _create_rand_value(env: ManagerEnv) -> RandValue:
    """Create the framework-owned per-environment PRNG state."""
    # View the uint64 bit buffer as signed int64 so Torch kernels can consume
    # it directly; the SplitMix64 bit pattern is preserved exactly.
    states = initialize_rand_states(env.num_envs, env._rand_seed)
    return RandValue(torch.from_numpy(states.view(np.int64)))


__all__ = [
    "RandValue",
]
