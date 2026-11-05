# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared NumPy/Torch batch quaternion operations, using xyzw convention."""

from motrix_env_core.math.namespace import array_namespace


def inverse(q):
    xp = array_namespace(q)
    norm = xp.maximum(xp.sum(q * q, axis=-1, keepdims=True), xp.full_like(q[..., :1], 1e-12))
    return xp.concat((-q[..., :3], q[..., 3:]), axis=-1) / norm


def mul(lhs, rhs):
    xp = array_namespace(lhs, rhs)
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
    xp = array_namespace(q, vector)
    x, y, z, w = (q[..., i] for i in range(4))
    vx, vy, vz = (vector[..., i] for i in range(3))
    cx = y * vz - z * vy - w * vx
    cy = z * vx - x * vz - w * vy
    cz = x * vy - y * vx - w * vz
    return xp.stack((vx + 2 * (y * cz - z * cy), vy + 2 * (z * cx - x * cz), vz + 2 * (x * cy - y * cx)), axis=-1)


def matrix6d(q):
    xp = array_namespace(q)
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
    return matrix6d(mul(inverse(reference), target))


def align_targets(robot_position, robot_quat, reference_position, reference_quat, body_position, body_quat):
    """Yaw-align corpus bodies, anchoring robot XY and corpus reference Z."""
    xp = array_namespace(robot_position)
    delta = mul(robot_quat, inverse(reference_quat))
    x, y, z, w = (delta[..., i] for i in range(4))
    yaw = xp.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    zero = xp.zeros_like(yaw)
    yaw_quat = xp.stack((zero, zero, xp.sin(yaw / 2), xp.cos(yaw / 2)), axis=-1)
    anchor = xp.concat((robot_position[..., :2], reference_position[..., 2:]), axis=-1)
    # Forward rotation uses the same unit-quaternion algebra as scalar helper.
    conjugate = xp.concat((-yaw_quat[..., :3], yaw_quat[..., 3:]), axis=-1)
    aligned_position = rotate_inverse(conjugate[:, None, :], body_position - reference_position[:, None, :])
    return aligned_position + anchor[:, None, :], mul(yaw_quat[:, None, :], body_quat)


def rotation_distance(lhs, rhs):
    """Geodesic angle between unit quaternions, without normalization."""
    xp = array_namespace(lhs, rhs)
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
    xp = array_namespace(euler)
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
        dim=-1,
    )


def rotate_inverse_components(quaternion, vector):
    """Rotate vectors by the inverse of ``[x, y, z, w]`` quaternions, batched.

    ``quaternion`` is ``(..., 4)`` and ``vector`` is ``(..., 3)``; the rotated
    components are returned as three tensors of the leading shape.
    """
    xp = array_namespace(quaternion, vector)
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
