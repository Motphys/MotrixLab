# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Explicit, compile-friendly access to the default Torch random stream."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RandomSource:
    """Stateless random capability, not an independently owned random stream.

    The training entry point seeds Torch. Instances share the default stream
    on each device; draws from other consumers can change subsequent results.
    """

    def uniform(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Draw unit-interval samples using Torch's default generator."""
        return torch.rand(shape, dtype=dtype, device=device)

    def uniform_like(self, value: torch.Tensor) -> torch.Tensor:
        return torch.rand_like(value)
