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

_TORCH_GOLDEN_RATIO_64 = -0x61C8864680B583EB  # 0x9E3779B97F4A7C15 as int64
_TORCH_MIX_MULTIPLIER_1 = -4658895280553007687  # 0xBF58476D1CE4E5B9 as int64
_TORCH_MIX_MULTIPLIER_2 = -7723592293110705685  # 0x94D049BB133111EB as int64

# Masks make an arithmetic signed right shift equivalent to a uint64 logical shift.
_MASK_SHIFT_30 = (1 << 34) - 1
_MASK_SHIFT_27 = (1 << 37) - 1
_MASK_SHIFT_31 = (1 << 33) - 1
_MASK_SHIFT_40 = (1 << 24) - 1


@dataclass(frozen=True, slots=True)
class RandValue:
    """Per-environment PRNG state shared by manager terms.

    ``state`` is a signed-int64 torch view over the per-lane uint64 bit
    buffer; the bit pattern is the SplitMix64 state. Terms draw whole-batch
    samples through :meth:`next_uniform`/:meth:`uniform_range`, which update
    the state tensor in place, so host and kernels share one stream.
    """

    state: torch.Tensor

    def next_uniform(self, count: int = 1) -> torch.Tensor:
        """Draw ``count`` values per lane in ``[-1, 1)``; shape ``(N, count)``."""
        samples, state = draw_uniform(self.state, count)
        self.state.copy_(state)
        return samples

    def uniform_range(self, count: int, low: float, high: float) -> torch.Tensor:
        """Draw ``count`` values per lane in ``[low, high)``; shape ``(N, count)``."""
        return low + (high - low) * np.float32(0.5) * (self.next_uniform(count) + np.float32(1.0))


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


def _logical_rshift(value: torch.Tensor, amount: int, mask: int) -> torch.Tensor:
    return (value >> amount) & mask


def _mix(value: torch.Tensor) -> torch.Tensor:
    value = (value ^ _logical_rshift(value, 30, _MASK_SHIFT_30)) * _TORCH_MIX_MULTIPLIER_1
    value = (value ^ _logical_rshift(value, 27, _MASK_SHIFT_27)) * _TORCH_MIX_MULTIPLIER_2
    return value ^ _logical_rshift(value, 31, _MASK_SHIFT_31)


def draw_uniform(state: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ``count`` SplitMix64 values per row, returning samples and updated state.

    ``state`` is an ``(N, 1)`` signed int64 tensor whose bits are interpreted as
    uint64.  Arithmetic intentionally wraps at 64 bits, and samples are float32
    values in ``[-1, 1)``.  The count is fixed for a call and may be zero.
    """
    if count == 0:
        return torch.empty((*state.shape[:-1], 0), device=state.device, dtype=torch.float32), state.clone()
    # Pre-wrap fixed offsets in Python: Inductor may otherwise use signed C++
    # loop-index multiplication, whose overflow is undefined rather than wrapping.
    offsets = tuple(
        ((index * _TORCH_GOLDEN_RATIO_64 + (1 << 63)) % (1 << 64)) - (1 << 63) for index in range(1, count + 1)
    )
    increments = torch.tensor(offsets, device=state.device, dtype=torch.int64)
    draw_states = state + increments
    mixed = _mix(draw_states)
    top = _logical_rshift(mixed, 40, _MASK_SHIFT_40)
    samples = top.to(torch.float32) * (1.0 / (1 << 23)) - 1.0
    return samples, draw_states[..., -1:]


__all__ = [
    "RandValue",
    "draw_uniform",
    "initialize_rand_states",
    "next_uniform",
    "uniform_range",
]
