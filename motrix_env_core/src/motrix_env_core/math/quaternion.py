# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Quaternion helpers in ``[x, y, z, w]`` convention.

The batch operations accept NumPy arrays and Torch tensors through the
array API (a single ``isinstance`` check keeps Dynamo specialization
possible inside compiled graphs); the remaining helpers are NumPy-only
host utilities.
"""

import array_api_compat.numpy as _numpy_namespace
import array_api_compat.torch as _torch_namespace
import numpy as np
import torch as _torch  # noqa: F401  (namespace check only)

from . import utils


def _array_namespace(value, *others):
    # A single isinstance check keeps Dynamo specialization possible inside
    # compiled graphs; array_api_compat's own dispatcher is too dynamic here.
    return _torch_namespace if isinstance(value, _torch.Tensor) else _numpy_namespace


def inverse(q):
    """Inverse of ``(..., 4)`` quaternion(s); conjugate over squared norm."""
    xp = _array_namespace(q)
    norm = xp.maximum(xp.sum(q * q, axis=-1, keepdims=True), xp.full_like(q[..., :1], 1e-12))
    return xp.concat((-q[..., :3], q[..., 3:]), axis=-1) / norm


def mul(lhs, rhs):
    """Hamilton product of two ``(..., 4)`` quaternion batches."""
    xp = _array_namespace(lhs, rhs)
    x1, y1, z1, w1 = (lhs[..., i] for i in range(4))
    x2, y2, z2, w2 = (rhs[..., i] for i in range(4))
    return xp.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        axis=-1,
    )


def rotate_inverse(q, vector):
    """Rotate ``(..., 3)`` vectors by the inverse of ``(..., 4)`` quaternions."""
    xp = _array_namespace(q, vector)
    x, y, z, w = (q[..., i] for i in range(4))
    vx, vy, vz = (vector[..., i] for i in range(3))
    cx = y * vz - z * vy - w * vx
    cy = z * vx - x * vz - w * vy
    cz = x * vy - y * vx - w * vz
    return xp.stack((vx + 2 * (y * cz - z * cy), vy + 2 * (z * cx - x * cz), vz + 2 * (x * cy - y * cx)), axis=-1)


def matrix6d(q):
    """First two rows of the rotation matrix as a ``(..., 6)`` 6D representation."""
    xp = _array_namespace(q)
    x, y, z, w = (q[..., i] for i in range(4))
    s = 2 / xp.sum(q * q, axis=-1)
    return xp.stack(
        (
            1 - s * (y * y + z * z),
            s * (x * y - z * w),
            s * (x * z + y * w),
            s * (x * y + z * w),
            1 - s * (x * x + z * z),
            s * (y * z - x * w),
        ),
        axis=-1,
    )


def relative_orientation(reference, target):
    """6D orientation of ``target`` expressed in the ``reference`` frame."""
    return matrix6d(mul(inverse(reference), target))


def align_targets(robot_position, robot_quat, reference_position, reference_quat, body_position, body_quat):
    """Yaw-align corpus bodies, anchoring robot XY and corpus reference Z."""
    xp = _array_namespace(robot_position)
    delta = mul(robot_quat, inverse(reference_quat))
    x, y, z, w = (delta[..., i] for i in range(4))
    yaw = xp.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = xp.zeros_like(yaw)
    yaw_quat = xp.stack((zero, zero, xp.sin(yaw / 2), xp.cos(yaw / 2)), axis=-1)
    anchor = xp.concat((robot_position[..., :2], reference_position[..., 2:]), axis=-1)
    # Forward rotation uses the same unit-quaternion algebra as scalar helper.
    conjugate_quat = xp.concat((-yaw_quat[..., :3], yaw_quat[..., 3:]), axis=-1)
    aligned_position = rotate_inverse(conjugate_quat[:, None, :], body_position - reference_position[:, None, :])
    return aligned_position + anchor[:, None, :], mul(yaw_quat[:, None, :], body_quat)


def rotation_distance(lhs, rhs):
    """Geodesic angle between unit quaternions, without normalization."""
    xp = _array_namespace(lhs, rhs)
    x1, y1, z1, w1 = (lhs[..., i] for i in range(4))
    x2, y2, z2, w2 = (rhs[..., i] for i in range(4))
    dx = -w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2
    dy = -w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2
    dz = -w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2
    imaginary_norm = xp.sqrt(dx * dx + dy * dy + dz * dz)
    imaginary_norm = xp.where(imaginary_norm < 0.0, 0.0, xp.where(imaginary_norm > 1.0, 1.0, imaginary_norm))
    return 2.0 * xp.asin(imaginary_norm)


def from_euler(euler):
    """Quaternion (xyzw) from ``(..., 3)`` roll/pitch/yaw angles; trig keeps the input dtype."""
    xp = _array_namespace(euler)
    roll, pitch, yaw = (euler[..., i] for i in range(3))
    cy = xp.cos(yaw * 0.5)
    sy = xp.sin(yaw * 0.5)
    cp = xp.cos(pitch * 0.5)
    sp = xp.sin(pitch * 0.5)
    cr = xp.cos(roll * 0.5)
    sr = xp.sin(roll * 0.5)
    return xp.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * cy,
        ),
        axis=-1,
    )


def rotate_inverse_components(quaternion, vector):
    """Rotate vectors by the inverse of ``[x, y, z, w]`` quaternions, batched.

    ``quaternion`` is ``(..., 4)`` and ``vector`` is ``(..., 3)``; the rotated
    components are returned as three tensors of the leading shape.
    """
    x, y, z, w = (quaternion[..., i] for i in range(4))
    vx, vy, vz = (vector[..., i] for i in range(3))
    cx = y * vz - z * vy - w * vx
    cy = z * vx - x * vz - w * vy
    cz = x * vy - y * vx - w * vz
    return (
        vx + 2.0 * (y * cz - z * cy),
        vy + 2.0 * (z * cx - x * cz),
        vz + 2.0 * (x * cy - y * cx),
    )


def conjugate(q):
    """
    Compute the conjugate of a quaternion (NumPy).

    For q = (x, y, z, w), conjugate is (-x, -y, -z, w)

    Args:
        q: Input quaternion(s). Shape: (..., 4)

    Returns:
        Conjugate quaternion(s). Shape: (..., 4)
    """
    return q * np.array([-1, -1, -1, 1], dtype=q.dtype)


def from_angle_axis(angle, axis):
    """
    Create quaternion from angle-axis representation (NumPy).

    Args:
        angle: Rotation angle in radians. Shape: (batch,) or scalar
        axis: Rotation axis (will be normalized). Shape: (batch, 3) or (3,)

    Returns:
        Quaternion in (x, y, z, w) format. Shape: (batch, 4) or (4,)
    """
    # Ensure angle has proper shape for broadcasting
    if np.isscalar(angle):
        angle = np.array([angle])

    # Normalize axis
    axis = utils.normalize(axis)

    # Compute half angle
    half_angle = angle / 2.0

    # Compute quaternion components
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    xyz = axis * sin_half[..., np.newaxis]

    return utils.normalize(np.concatenate([xyz, cos_half[..., np.newaxis]], axis=-1))


def rotate_vector(quats: np.ndarray, v: np.ndarray):
    """Rotate vectors ``v`` by quaternions ``quats`` (``[x, y, z, w]`` format, NumPy).

    Both inputs broadcast on their leading dims:

    - ``quats``: shape ``(..., 4)``.
    - ``v``:     shape ``(..., 3)`` or ``(3,)``.

    Returns shape ``broadcast(quats.batch, v.batch) + (3,)``. So a single
    quaternion against a batch of vectors (or vice versa), and any
    higher-rank combination, all work without manual flatten/reshape.
    """
    batch = np.broadcast_shapes(quats.shape[:-1], v.shape[:-1])
    q = np.broadcast_to(quats, batch + (4,))
    vec = np.broadcast_to(v, batch + (3,))

    w = q[..., -1]
    im = q[..., :3]

    t = 2.0 * np.cross(im, vec)
    return vec + w[..., None] * t + np.cross(im, t)


def similarity(q_current, q_target):
    """
    Compute attitude alignment reward between two batches of quaternions (NumPy).

    Parameters:
        q_current (np.ndarray): Quaternion of current pose, shape (num_envs, 4).
        q_target (np.ndarray): Quaternion of target pose, shape (num_envs, 4) or (4,).
                                If (4,), it will be broadcast to all environments.

    Returns:
        np.ndarray: Reward value for each environment, shape (num_envs,). Reward value range is [-1, 1].
    """
    # If q_target is a single quaternion, broadcast to all environments
    if q_target.ndim == 1:
        q_target = np.tile(q_target, (q_current.shape[0], 1))

    # Relative quaternion q_rel = q_target * conjugate(q_current); its real
    # part gives the rotation angle, clamped for numerical stability.
    x1, y1, z1, w1 = q_target[..., 0], q_target[..., 1], q_target[..., 2], q_target[..., 3]
    q_current_conj = np.copy(q_current)
    q_current_conj[..., :3] *= -1
    x2, y2, z2, w2 = (
        q_current_conj[..., 0],
        q_current_conj[..., 1],
        q_current_conj[..., 2],
        q_current_conj[..., 3],
    )

    w_rel = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    w_rel_clamped = np.clip(w_rel, -1.0, 1.0)

    theta = 2.0 * np.arccos(w_rel_clamped)
    return np.cos(theta)


def get_euler_xyz(q: np.ndarray):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) (NumPy).

    Args:
        q: Quaternion(s) in (x, y, z, w) format. Shape: (..., 4)

    Returns:
        Tuple of (roll, pitch, yaw) in radians. Each has shape: (...)
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2.0, sinp), np.arcsin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def get_yaw(quat: np.ndarray) -> np.ndarray:
    _, _, yaw = get_euler_xyz(quat)
    return yaw


def from_yaw(yaw: np.ndarray) -> np.ndarray:
    """Build a yaw-only (z-axis) quaternion from a yaw angle (NumPy).

    Args:
        yaw: yaw angle(s) in radians, any shape (including scalar).

    Returns:
        Quaternion(s) of shape ``yaw.shape + (4,)`` in ``[x, y, z, w]`` format,
        with x=y=0 and (z, w) = (sin(yaw/2), cos(yaw/2)). Always unit-norm.
    """
    yaw = np.asarray(yaw, dtype=np.float32)
    half = yaw * 0.5
    out = np.zeros(yaw.shape + (4,), dtype=np.float32)
    out[..., 2] = np.sin(half)
    out[..., 3] = np.cos(half)
    return out


def to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) to row-major 3x3 rotation matrix(ices) (NumPy).

    Uses the Shepperd-style closed form with ``2/|q|^2`` normalization, so
    non-unit inputs are handled too. Output shape: ``q.shape[:-1] + (3, 3)``.
    ``R @ v`` rotates a column vector ``v`` by ``q``.
    """
    q = np.asarray(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / np.sum(q * q, axis=-1)
    return np.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def generate_random_shoemake(size):
    """
    Generate uniformly distributed random quaternions using Shoemake's method (NumPy).

    This ensures uniform distribution over SO(3) rotation space.

    Args:
        size: Number of quaternions to generate (int or tuple)

    Returns:
        Random quaternions in (x, y, z, w) format. Shape: (size, 4) or (*size, 4)

    Reference:
        K. Shoemake, "Uniform Random Rotations", Graphics Gems III, 1992
    """
    # Convert size to tuple if it's a scalar (handles numpy int64)
    if np.isscalar(size):
        size = (int(size),)
    elif not isinstance(size, tuple):
        size = tuple(size)

    # Generate three uniform random numbers
    u1, u2, u3 = np.random.uniform(0, 1, size=(3, *size)).astype(np.float32)

    # Shoemake's method
    sqrt1_u1 = np.sqrt(1 - u1)
    sqrtu1 = np.sqrt(u1)

    w = sqrt1_u1 * np.sin(2 * np.pi * u2)
    x = sqrt1_u1 * np.cos(2 * np.pi * u2)
    y = sqrtu1 * np.sin(2 * np.pi * u3)
    z = sqrtu1 * np.cos(2 * np.pi * u3)

    return np.stack([x, y, z, w], axis=-1)
