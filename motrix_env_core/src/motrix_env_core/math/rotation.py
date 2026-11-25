# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Torch-only batched rotation primitives (rotation matrices act on column vectors).

Unlike :mod:`motrix_env_core.math.quaternion` (dual NumPy/Torch, ``xyzw``
quaternions), these helpers are torch-native so they stay differentiable
inside training graphs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "axis_angle_to_mat",
    "mat_to_quat",
    "mm3",
    "mv3",
    "quat_to_mat",
    "quat_slerp",
    "rot_angle_sq",
    "so3_exp",
    "so3_log",
    "yaw_mat",
]

# Express 3x3 products as broadcast multiply+sum so torch.compile/inductor
# can fuse them instead of dispatching tiny batched GEMM/GEMV operations.


def mm3(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A @ B for (..., 3, 3) matrices (broadcasting)."""
    return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(-2)


def mv3(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A @ v for (..., 3, 3) matrices and (..., 3) vectors (broadcasting)."""
    return (A * v.unsqueeze(-2)).sum(-1)


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


def yaw_mat(R: torch.Tensor) -> torch.Tensor:
    """Heading (yaw-only) rotation of R, from the projected x axis."""
    fx, fy = R[..., 0, 0], R[..., 1, 0]
    n = torch.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / n, fy / n
    o, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, -s, o, s, c, o, o, o, one), -1).reshape(R.shape[:-2] + (3, 3))


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """``xyzw`` quaternion batch to rotation matrices (column-vector convention)."""
    x, y, z, w = q.unbind(-1)
    norm2 = (q * q).sum(-1)
    two_s = 2.0 / torch.where(norm2 > 0, norm2, torch.ones_like(norm2))
    return torch.stack(
        (
            1 - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w),
            two_s * (x * y + z * w), 1 - two_s * (x * x + z * z), two_s * (y * z - x * w),
            two_s * (x * z - w * y), two_s * (y * z + w * x), 1 - two_s * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))  # fmt: skip


def mat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrices to ``xyzw`` quaternions; robust, w >= 0."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    qw = torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=1e-10)) / 2
    qx = torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-10)) / 2
    qy = torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=1e-10)) / 2
    qz = torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=1e-10)) / 2
    cands = torch.stack(
        (
            # each row is (x, y, z, w); the selected branch fills its own
            # component with the sqrt estimate and the others from products
            torch.stack(((m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw), qw), -1),
            torch.stack((qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx), (m21 - m12) / (4 * qx)), -1),
            torch.stack(((m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy), (m02 - m20) / (4 * qy)), -1),
            torch.stack(((m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz, (m10 - m01) / (4 * qz)), -1),
        ),
        dim=-2,
    )
    idx = torch.stack((qw, qx, qy, qz), -1).argmax(-1)
    q = torch.gather(cands, -2, idx[..., None, None].expand(idx.shape + (1, 4))).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True)
    return q * torch.where(q[..., 3:] < 0, -1.0, 1.0)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix batch to rotation vector, via the quaternion."""
    q = mat_to_quat(R)
    v = q[..., :3]
    sin_half = v.norm(dim=-1, keepdim=True)
    half = torch.atan2(sin_half, q[..., 3:])
    scale = torch.where(sin_half < 1e-6, 2.0 / torch.clamp(q[..., 3:], min=1e-6), 2 * half / (sin_half + 1e-12))
    return v * scale


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    """Rotation vector batch to matrix, smooth at zero."""
    theta2 = (v * v).sum(-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / (theta2 + 1e-12))
    K = _skew(v)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand(K.shape)
    return eye + a[..., None] * K + b[..., None] * mm3(K, K)


def _skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(v.shape[:-1] + (3, 3))


def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Shortest-path spherical interpolation of unit ``xyzw`` quaternions.

    ``q0`` and ``q1`` have shape ``(..., 4)``; ``t`` has a broadcastable
    leading batch shape, without a trailing singleton coordinate dimension.
    The normalized result has shape ``broadcast(q0.batch, q1.batch, t) + (4,)``.
    Near-coincident rotations use linear weights to avoid a small sine divisor.
    """
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot.clamp(max=1 - torch.finfo(dot.dtype).eps))
    s = torch.sin(theta)
    t = t[..., None]
    w0 = torch.where(dot > 1 - 1e-6, 1 - t, torch.sin((1 - t) * theta) / (s + 1e-12))
    w1 = torch.where(dot > 1 - 1e-6, t, torch.sin(t * theta) / (s + 1e-12))
    return F.normalize(w0 * q0 + w1 * q1, dim=-1)


def rot_angle_sq(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """Squared angular distance (rad²) between broadcastable ``(..., 3, 3)`` rotations.

    The relative-angle cosine is clipped to ``[-1 + 1e-6, 1 - 1e-6]``
    before acos to keep endpoint gradients finite. Consequently identical
    rotations return a small positive floor, and angles near pi are capped.
    """
    tr = (Ra * Rb).sum((-1, -2))  # trace(Ra^T Rb)
    cos = torch.clamp((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos) ** 2
