# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Common high-level command value types."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlanarVelocityCommand:
    """Batch-first body-frame ``[vx, vy, yaw_rate]`` command."""

    values: torch.Tensor

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.values, dtype=torch.float32).clone()
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 3:
            raise ValueError("values must have shape (batch_size, 3), with batch_size > 0")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("values must contain only finite numbers")
        object.__setattr__(self, "values", values)

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    @property
    def linear_velocity_x_mps(self) -> torch.Tensor:
        return self.values[:, 0]

    @property
    def linear_velocity_y_mps(self) -> torch.Tensor:
        return self.values[:, 1]

    @property
    def yaw_rate_rad_s(self) -> torch.Tensor:
        return self.values[:, 2]


__all__ = ["PlanarVelocityCommand"]
