# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-specific random planar-velocity command binding for quadruped training."""

import numpy as np
from numpy.typing import NDArray

from motrix_env_core.input import CommandBinding, PlanarVelocityCommand


class RandomPlanarVelocityBinding(CommandBinding[PlanarVelocityCommand]):
    """Sample independent planar velocities and optional standing commands."""

    def __init__(
        self,
        lower: NDArray[np.float32],
        upper: NDArray[np.float32],
        *,
        rng: np.random.Generator,
        standing_probability: float = 0.0,
    ) -> None:
        self._lower = lower
        self._upper = upper
        self._standing_probability = standing_probability
        self._rng = rng

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        values = self._rng.uniform(self._lower, self._upper, size=(batch_size, 3))
        if self._standing_probability > 0.0:
            standing = self._rng.random(batch_size) < self._standing_probability
            values[standing] = 0.0
        return PlanarVelocityCommand(values.astype(np.float32))


__all__ = ["RandomPlanarVelocityBinding"]
