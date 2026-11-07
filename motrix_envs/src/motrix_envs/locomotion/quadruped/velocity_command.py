# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-specific random planar-velocity command binding for quadruped training."""

import torch
from numpy.typing import NDArray

from motrix_env_core.input import CommandBinding, PlanarVelocityCommand


class RandomPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Sample independent planar velocities and optional standing commands."""

    def __init__(
        self,
        lower: NDArray,
        upper: NDArray,
        *,
        rng: torch.Generator,
        standing_probability: float = 0.0,
    ) -> None:
        self._lower = torch.as_tensor(lower, dtype=torch.float32)
        self._upper = torch.as_tensor(upper, dtype=torch.float32)
        self._standing_probability = standing_probability
        self._rng = rng

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        u = torch.rand((batch_size, 3), generator=self._rng)
        values = self._lower + (self._upper - self._lower) * u
        if self._standing_probability > 0.0:
            standing = torch.rand((batch_size,), generator=self._rng) < self._standing_probability
            values[standing] = 0.0
        return PlanarVelocityCommand(values.numpy())


__all__ = ["RandomPlanarVelocityBinding"]
