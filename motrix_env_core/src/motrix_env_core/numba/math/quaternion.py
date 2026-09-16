# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Quaternion operations for scalar Numba kernels."""

import math

import numba
import numpy as np


@numba.njit(inline="always")
def from_euler(
    roll: float,
    pitch: float,
    yaw: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Convert roll/pitch/yaw angles to an ``[x, y, z, w]`` quaternion.

    The scalar implementation is intended for use inside fused Numba lanes.
    An optional caller-owned output avoids an allocation in reset and other
    hot paths.
    """
    if out is None:
        out = np.empty(4, dtype=np.float32)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    out[0] = sr * cp * cy - cr * sp * sy
    out[1] = cr * sp * cy + sr * cp * sy
    out[2] = cr * cp * sy - sr * sp * cy
    out[3] = cr * cp * cy + sr * sp * sy
    return out


@numba.njit(inline="always")
def inverse(
    quaternion: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Invert one ``[x, y, z, w]`` quaternion and return the output array."""
    if out is None:
        out = np.empty(4, dtype=quaternion.dtype)
    x, y, z, w = quaternion
    norm_sq = max(x * x + y * y + z * z + w * w, 1e-12)
    out[0] = -x / norm_sq
    out[1] = -y / norm_sq
    out[2] = -z / norm_sq
    out[3] = w / norm_sq
    return out


@numba.njit(inline="always")
def mul(
    lhs: np.ndarray,
    rhs: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Multiply two ``[x, y, z, w]`` quaternions and return the output array."""
    if out is None:
        out = np.empty(4, dtype=lhs.dtype)
    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    out[0] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    out[1] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    out[2] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    out[3] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return out


@numba.njit(inline="always")
def rotate_vector(
    quaternion: np.ndarray,
    vector: np.ndarray | tuple[float, float, float],
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate one 3D vector by the quaternion and return the output array."""
    if out is None:
        out = np.empty(3, dtype=quaternion.dtype)
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    out[0] = vx + w * tx + y * tz - z * ty
    out[1] = vy + w * ty + z * tx - x * tz
    out[2] = vz + w * tz + x * ty - y * tx
    return out


@numba.njit(inline="always")
def rotate_inverse(
    quaternion: np.ndarray,
    vector: np.ndarray | tuple[float, float, float],
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Rotate one 3D vector by the inverse quaternion and return the output array."""
    if out is None:
        out = np.empty(3, dtype=quaternion.dtype)
    x, y, z, w = quaternion
    vx, vy, vz = vector
    cx = y * vz - z * vy - w * vx
    cy = z * vx - x * vz - w * vy
    cz = x * vy - y * vx - w * vz
    out[0] = vx + 2.0 * (y * cz - z * cy)
    out[1] = vy + 2.0 * (z * cx - x * cz)
    out[2] = vz + 2.0 * (x * cy - y * cx)
    return out


@numba.njit(inline="always")
def rotation_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Compute the rotation distance between two quaternions in radians."""
    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    dx = -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2
    dy = -w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2
    dz = -w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2
    imaginary_norm = min(max(math.sqrt(dx * dx + dy * dy + dz * dz), 0.0), 1.0)
    return 2.0 * math.asin(imaginary_norm)


@numba.njit(inline="always")
def rotate_inverse_components(quaternion: np.ndarray, vector):
    """Rotate one 3D vector by the inverse ``[x, y, z, w]`` quaternion.

    Allocation-free variant of :func:`rotate_inverse` that returns the three
    components as a tuple, so callers inside fused kernels can destructure
    them without a scratch array.
    """
    x, y, z, w = quaternion
    vx, vy, vz = vector
    cx = y * vz - z * vy - w * vx
    cy = z * vx - x * vz - w * vy
    cz = x * vy - y * vx - w * vz
    return (
        vx + 2.0 * (y * cz - z * cy),
        vy + 2.0 * (z * cx - x * cz),
        vz + 2.0 * (x * cy - y * cx),
    )


@numba.njit(inline="always")
def to_matrix_first_two_rows(
    quaternion: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Write the first two rows of the quaternion rotation matrix in row-major order."""
    if out is None:
        out = np.empty(6, dtype=quaternion.dtype)
    x, y, z, w = quaternion
    two_s = 2.0 / (x * x + y * y + z * z + w * w)
    out[0] = 1.0 - two_s * (y * y + z * z)
    out[1] = two_s * (x * y - z * w)
    out[2] = two_s * (x * z + y * w)
    out[3] = two_s * (x * y + z * w)
    out[4] = 1.0 - two_s * (x * x + z * z)
    out[5] = two_s * (y * z - x * w)
    return out


__all__ = [
    "from_euler",
    "inverse",
    "mul",
    "rotate_inverse",
    "rotate_inverse_components",
    "rotate_vector",
    "rotation_distance",
    "to_matrix_first_two_rows",
]
