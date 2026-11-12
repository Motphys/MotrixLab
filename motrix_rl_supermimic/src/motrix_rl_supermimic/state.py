# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
# Adapted from SuperMimic (supermimic_vibed), Copyright (c) 2026 ETH Zurich,
# Chong Zhang. MIT License.

"""Canonical state helpers shared by the networks, replay, and learner.

The canonical state is the flat ``(p, quat, v, w, q, qd)`` tuple (root
position, rotation as ``xyzw`` quaternion, root linear/angular velocity,
joint positions/velocities). ``StateSpec`` dimensions drive every derived
quantity; the layout itself is part of the TaskMDP ABI and this module is
the single place that knows the offsets.
"""

from __future__ import annotations

import torch

# Express 3x3 products as broadcast multiply+sum so torch.compile/inductor
# fuses them instead of dispatching cuBLAS batched GEMM/GEMV (the WM BPTT
# rollout is dominated by these small rotations; the rewrite agrees with
# the matmul form to 1.5e-6).
ELEMENTWISE = True


def mm3(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """A @ B for (..., 3, 3) matrices (broadcasting)."""
    if not ELEMENTWISE:
        return A @ B
    return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(-2)


def mv3(A: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """A @ v for (..., 3, 3) matrices and (..., 3) vectors (broadcasting)."""
    if not ELEMENTWISE:
        return (A @ v.unsqueeze(-1)).squeeze(-1)
    return (A * v.unsqueeze(-2)).sum(-1)

from motrix_rl_supermimic.task_mdp import StateSpec

__all__ = [
    "canonical_offsets",
    "error_floor",
    "flatten_state",
    "invariant_features",
    "inv_dim",
    "mat_to_quat",
    "num_joints",
    "quat_to_mat",
    "so3_exp",
    "so3_log",
    "unflatten_state",
    "yaw_mat",
]


def canonical_offsets(spec: StateSpec) -> tuple[int, ...]:
    """Flat canonical layout offsets ``(p0, q0, v0, w0, j0, j1)`` for the spec.

    Requires the spec's canonical groups ``(3, 4, 3, 3, nJ, nJ)`` in order.
    """
    widths = tuple(w for w, _ in spec.groups())
    if (
        len(widths) != 6
        or widths[0] != 3
        or widths[1] != 4
        or widths[2] != 3
        or widths[3] != 3
        or widths[4] != widths[5]
    ):
        raise ValueError(f"Canonical state layout requires groups (3, 4, 3, 3, nJ, nJ); got {widths}.")
    p0 = 0
    quat0 = 3
    v0 = 7
    w0 = 10
    j0 = 13
    return (p0, quat0, v0, w0, j0, j0 + widths[4])


def num_joints(spec: StateSpec) -> int:
    return (spec.state_dim - 13) // 2


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


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(spec: StateSpec, x: torch.Tensor):
    p0, quat0, v0, w0, j0, j1 = canonical_offsets(spec)
    return (
        x[..., p0 : p0 + 3],
        quat_to_mat(x[..., quat0 : quat0 + 4]),
        x[..., v0 : v0 + 3],
        x[..., w0 : w0 + 3],
        x[..., j0:j1],
        x[..., j1:],
    )


def invariant_features(spec: StateSpec, S) -> torch.Tensor:
    """Heading- and position-invariant state features for network inputs."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    tilt6 = mm3(H.transpose(-1, -2), R)[..., :, :2].reshape(R.shape[:-2] + (6,))
    return torch.cat(
        (
            p[..., 2:3],
            tilt6,
            mv3(Rt, v),
            mv3(Rt, w),
            q,
            qd,
        ),
        -1,
    )


def inv_dim(spec: StateSpec) -> int:
    return 1 + 6 + 3 + 3 + spec.state_dim - 13


def error_floor(spec: StateSpec, device) -> torch.Tensor:
    """Per-dimension WM error floors in the frame-local target layout.

    The target layout replaces the quaternion group with a rotation-vector
    increment: ``(dp, dtheta, dv, dw, dq, dqd)``; each keeps its canonical
    group's floor.
    """
    widths = tuple(w for w, _ in spec.groups())
    target_widths = (3, 3, 3, 3, widths[4], widths[5])
    return torch.cat([torch.full((w,), floor) for w, (_, floor) in zip(target_widths, spec.groups())]).to(device)
