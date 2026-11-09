# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Torch-only batched rotation primitives (rotation matrices act on column vectors).

Unlike :mod:`motrix_env_core.math.quaternion` (dual NumPy/Torch, ``xyzw``
quaternions), these helpers are torch-native so they stay differentiable
inside training graphs.
"""

from __future__ import annotations

import torch

__all__ = ["axis_angle_to_mat"]


def axis_angle_to_mat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues for a fixed unit axis ``(..., 3)`` against angle ``(...)``.

    The axis broadcasts against batched angles; the result is a row-major
    rotation matrix of shape ``broadcast(axis.batch, angle.batch) + (3, 3)``,
    differentiable in both inputs.
    """
    x, y, z = axis.unbind(-1)
    c, s = torch.cos(angle), torch.sin(angle)
    C = 1 - c
    return torch.stack(
        (
            c + x * x * C, x * y * C - z * s, x * z * C + y * s,
            y * x * C + z * s, c + y * y * C, y * z * C - x * s,
            z * x * C - y * s, z * y * C + x * s, c + z * z * C,
        ),
        dim=-1,
    ).reshape(torch.broadcast_shapes(axis.shape[:-1], angle.shape) + (3, 3))  # fmt: skip
