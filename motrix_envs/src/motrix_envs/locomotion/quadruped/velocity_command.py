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
        single_axis_probability: float = 0.0,
        boundary_probability: float = 0.0,
    ) -> None:
        self._lower = lower
        self._upper = upper
        self._standing_probability = standing_probability
        self._single_axis_probability = single_axis_probability
        self._boundary_probability = boundary_probability
        self._rng = rng

    def read_command(self, *, batch_size: int = 1) -> PlanarVelocityCommand:
        values = self._rng.uniform(self._lower, self._upper, size=(batch_size, 3))
        single_axis = self._rng.random(batch_size) < self._single_axis_probability
        if np.any(single_axis):
            rows = np.flatnonzero(single_axis)
            values[rows] = 0.0
            axes = self._rng.integers(0, 3, size=len(rows))
            values[rows, axes] = self._rng.uniform(self._lower[axes], self._upper[axes])
        boundary = self._rng.random(batch_size) < self._boundary_probability
        if np.any(boundary):
            rows = np.flatnonzero(boundary)
            axes = self._rng.integers(0, 3, size=len(rows))
            signs = self._rng.integers(0, 2, size=len(rows))
            values[rows] = 0.0
            values[rows, axes] = np.where(signs, self._upper[axes], self._lower[axes])
        if self._standing_probability > 0.0:
            standing = self._rng.random(batch_size) < self._standing_probability
            values[standing] = 0.0
        return PlanarVelocityCommand(values.astype(np.float32))


__all__ = ["RandomPlanarVelocityBinding"]
