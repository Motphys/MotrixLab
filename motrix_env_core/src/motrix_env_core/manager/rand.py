# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SplitMix64 PRNG primitives for manager environments."""

from dataclasses import dataclass

import numpy as np
import torch

_GOLDEN_RATIO_64 = np.uint64(0x9E3779B97F4A7C15)
_ENV_SALT_64 = np.uint64(0xD2B74407B1CE6E93)
_MIX_MULTIPLIER_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_MULTIPLIER_2 = np.uint64(0x94D049BB133111EB)


@dataclass(frozen=True, slots=True)
class RandValue:
    """Mutable per-environment PRNG state shared by manager terms.

    ``state`` is a signed-int64 torch view over the per-lane uint64 bit
    buffer; the bit pattern is the SplitMix64 state.
    """

    state: torch.Tensor

    def next_uniform(self) -> np.float32:
        """Advance this environment's PRNG and return a value in ``[-1, 1)``."""
        return next_uniform(self.state.numpy().view(np.uint64))


def initialize_rand_states(num_envs: int, rand_seed: int) -> np.ndarray:
    """Derive one deterministic, independently mutable SplitMix64 state per environment."""

    if rand_seed < 0:
        raise ValueError(f"rand_seed must be non-negative, got {rand_seed}.")
    env_ids = np.arange(1, num_envs + 1, dtype=np.uint64)
    seeds = np.full((num_envs,), np.uint64(rand_seed), dtype=np.uint64)
    states = _mix_uint64_array(seeds ^ env_ids * _ENV_SALT_64)
    return states[:, None]


def _mix_uint64_array(value: np.ndarray) -> np.ndarray:
    value = (value ^ (value >> np.uint64(30))) * _MIX_MULTIPLIER_1
    value = (value ^ (value >> np.uint64(27))) * _MIX_MULTIPLIER_2
    return value ^ (value >> np.uint64(31))


def uniform_range(rng_state: np.ndarray, low: np.float32, high: np.float32) -> np.float32:
    """Uniform sample in ``[low, high)`` from the ``[-1, 1)`` PRNG primitive."""
    return low + (high - low) * np.float32(0.5) * (next_uniform(rng_state) + np.float32(1.0))


def next_uniform(rng_state: np.ndarray) -> np.float32:
    state = rng_state[0] + _GOLDEN_RATIO_64
    rng_state[0] = state
    value = (state ^ (state >> np.uint64(30))) * _MIX_MULTIPLIER_1
    value = (value ^ (value >> np.uint64(27))) * _MIX_MULTIPLIER_2
    value ^= value >> np.uint64(31)
    unit = np.float32(value >> np.uint64(40)) * np.float32(1.0 / (1 << 24))
    return unit * np.float32(2.0) - np.float32(1.0)


__all__ = [
    "RandValue",
    "initialize_rand_states",
    "next_uniform",
    "uniform_range",
]
