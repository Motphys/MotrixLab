# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Quaternion helpers in ``[x, y, z, w]`` convention.

Every operation accepts NumPy arrays and Torch tensors (a single
``isinstance`` check keeps Dynamo specialization possible inside compiled
graphs) and returns the namespace of its inputs. The one exception is
:func:`generate_random_shoemake`, which samples from the host NumPy RNG
and returns NumPy by default (``device=`` returns a Torch copy).
"""

import array_api_compat.numpy as _numpy_namespace
import array_api_compat.torch as _torch_namespace
import numpy as np
import torch as _torch  # noqa: F401  (namespace check only)


def _array_namespace(value, *others):
    # A single isinstance check keeps Dynamo specialization possible inside
    # compiled graphs; array_api_compat's own dispatcher is too dynamic here.
    return _torch_namespace if isinstance(value, _torch.Tensor) else _numpy_namespace


def _cross3(xp, a, b):
    """Cross product of two ``(..., 3)`` batches along the last axis."""
    ax, ay, az = (a[..., i] for i in range(3))
    bx, by, bz = (b[..., i] for i in range(3))
    return xp.stack((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), axis=-1)


def _normalize(xp, x):
    """Normalize ``(..., n)`` batches along the last axis."""
    norm = xp.sqrt(xp.sum(x * x, axis=-1, keepdims=True))
    if bool(xp.any(norm == 0.0)):
        raise ValueError("Zero vector could not be normalized.")
    return x / norm


def inverse(q):
    """Inverse of ``(..., 4)`` quaternion(s); conjugate over squared norm."""
    xp = _array_namespace(q)
    norm = xp.maximum(xp.sum(q * q, axis=-1, keepdims=True), xp.full_like(q[..., :1], 1e-12))
    return xp.concat((-q[..., :3], q[..., 3:]), axis=-1) / norm


def conjugate(q):
    """Conjugate of ``(..., 4)`` quaternion(s): ``(-x, -y, -z, w)``."""
    xp = _array_namespace(q)
    return xp.concat((-q[..., :3], q[..., 3:]), axis=-1)


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


def rotate_vector(quats, v):
    """Rotate vectors ``v`` by quaternions ``quats`` (``[x, y, z, w]`` format).

    Both inputs broadcast on their leading dims:

    - ``quats``: shape ``(..., 4)``.
    - ``v``:     shape ``(..., 3)`` or ``(3,)``.

    Returns shape ``broadcast(quats.batch, v.batch) + (3,)``. So a single
    quaternion against a batch of vectors (or vice versa), and any
    higher-rank combination, all work without manual flatten/reshape.
    The closed form is exact for unit quaternions.
    """
    xp = _array_namespace(quats, v)
    batch = xp.broadcast_shapes(quats.shape[:-1], v.shape[:-1])
    q = xp.broadcast_to(quats, batch + (4,))
    vec = xp.broadcast_to(v, batch + (3,))

    w = q[..., -1]
    im = q[..., :3]

    t = 2.0 * _cross3(xp, im, vec)
    return vec + w[..., None] * t + _cross3(xp, im, t)


def rotate_inverse(q, vector):
    """Rotate ``(..., 3)`` vectors by the inverse of ``(..., 4)`` quaternions."""
    xp = _array_namespace(q, vector)
    x, y, z, w = (q[..., i] for i in range(4))
    vx, vy, vz = (vector[..., i] for i in range(3))
    cx = y * vz - z * vy - w * vx
    cy = z * vx - x * vz - w * vy
    cz = x * vy - y * vx - w * vz
    return xp.stack((vx + 2 * (y * cz - z * cy), vy + 2 * (z * cx - x * cz), vz + 2 * (x * cy - y * cx)), axis=-1)


def rotate_inverse_(quaternion, vector, out) -> None:
    """In-place :func:`rotate_inverse`: write the rotated components into ``out``.

    Used inside compiled graphs where per-call allocation must be avoided.
    """
    x, y, z, w = (quaternion[..., i] for i in range(4))
    vx, vy, vz = (vector[..., i] for i in range(3))
    cx = y * vz - z * vy - w * vx
    cy = z * vx - x * vz - w * vy
    cz = x * vy - y * vx - w * vz
    out[..., 0] = vx + 2.0 * (y * cz - z * cy)
    out[..., 1] = vy + 2.0 * (z * cx - x * cz)
    out[..., 2] = vz + 2.0 * (x * cy - y * cx)


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


def similarity(q_current, q_target):
    """Attitude alignment reward in ``[-1, 1]`` between two orientations.

    ``q_current`` is ``(num_envs, 4)``; ``q_target`` is ``(num_envs, 4)`` or a
    single ``(4,)`` quaternion broadcast to all rows.
    """
    xp = _array_namespace(q_current, q_target)
    if q_target.ndim == 1:
        q_target = xp.broadcast_to(q_target, q_current.shape)
    # Relative rotation q_target * conjugate(q_current): its real part gives
    # the rotation angle; cos(theta) = 2*w^2 - 1 with w clamped to [-1, 1].
    x1, y1, z1, w1 = (q_target[..., i] for i in range(4))
    x2, y2, z2, w2 = (conjugate(q_current)[..., i] for i in range(4))
    w_rel = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    w_rel = xp.where(w_rel > 1.0, 1.0, xp.where(w_rel < -1.0, -1.0, w_rel))
    return 2.0 * w_rel * w_rel - 1.0


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
            cr * cp * cy + sr * sp * sy,
        ),
        axis=-1,
    )


def get_euler_xyz(q):
    """
    Convert ``(..., 4)`` quaternion(s) to Euler angles (roll, pitch, yaw).

    Returns a tuple of three arrays of the leading shape, in radians.
    """
    xp = _array_namespace(q)
    x, y, z, w = (q[..., i] for i in range(4))

    roll = xp.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = xp.where(
        xp.abs(sinp) >= 1,
        xp.copysign(xp.full_like(sinp, xp.pi / 2.0), sinp),
        xp.asin(sinp),
    )
    yaw = xp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def get_yaw(quat):
    _, _, yaw = get_euler_xyz(quat)
    return yaw


def from_yaw(yaw):
    """Build a yaw-only (z-axis) quaternion from yaw angle(s).

    Accepts scalars or arrays of any shape and any namespace; returns shape
    ``yaw.shape + (4,)`` with x=y=0 and (z, w) = (sin(yaw/2), cos(yaw/2)),
    always unit-norm, in float32.
    """
    xp = _array_namespace(yaw)
    yaw = xp.asarray(yaw, dtype=xp.float32)
    half = yaw * 0.5
    zero = xp.zeros_like(yaw)
    return xp.stack((zero, zero, xp.sin(half), xp.cos(half)), axis=-1)


def from_angle_axis(angle, axis):
    """
    Create quaternion(s) from angle-axis representation.

    Args:
        angle: Rotation angle(s) in radians; scalar or any shape.
        axis: Rotation axis(es), normalized internally. Shape: ``(..., 3)``
            (a leading ``(3,)`` axis broadcasts against batched angles).

    Returns:
        Unit quaternion(s) in (x, y, z, w) format.
    """
    xp = _array_namespace(angle, axis)
    angle = xp.asarray(angle)
    if angle.ndim == 0:
        angle = angle[xp.newaxis]
    # Torch concat requires matching dtypes; angles and axes commonly differ.
    axis = xp.asarray(axis, dtype=angle.dtype)
    axis = _normalize(xp, axis)
    if axis.ndim == 1:
        axis = axis[xp.newaxis, :]
    half = angle / 2.0
    quat = xp.concat((axis * xp.sin(half)[..., xp.newaxis], xp.cos(half)[..., xp.newaxis]), axis=-1)
    return _normalize(xp, quat)


def to_matrix(q):
    """Convert ``(..., 4)`` quaternion(s) to row-major 3x3 rotation matrices.

    Uses the Shepperd-style closed form with ``2/|q|^2`` normalization, so
    non-unit inputs are handled too. Output shape: ``q.shape[:-1] + (3, 3)``.
    ``R @ v`` rotates a column vector ``v`` by ``q``.
    """
    xp = _array_namespace(q)
    x, y, z, w = (q[..., i] for i in range(4))
    two_s = 2.0 / xp.sum(q * q, axis=-1)
    return xp.stack(
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


def generate_random_shoemake(size, *, device=None):
    """
    Generate uniformly distributed random unit quaternions using Shoemake's method.

    Sampling always uses the host NumPy RNG (random streams are host-owned);
    pass ``device=`` to receive a Torch tensor on that device instead of the
    NumPy result.

    Args:
        size: Number of quaternions to generate (int or tuple)
        device: Optional Torch device for the returned tensor.

    Returns:
        Unit quaternions in (x, y, z, w) format. Shape: (size, 4) or (*size, 4)

    Reference:
        K. Shoemake, "Uniform Random Rotations", Graphics Gems III, 1992
    """
    if np.isscalar(size):
        size = (int(size),)
    elif not isinstance(size, tuple):
        size = tuple(size)

    u1, u2, u3 = np.random.uniform(0, 1, size=(3, *size)).astype(np.float32)

    sqrt1_u1 = np.sqrt(1 - u1)
    sqrtu1 = np.sqrt(u1)

    x = sqrt1_u1 * np.cos(2 * np.pi * u2)
    y = sqrtu1 * np.sin(2 * np.pi * u3)
    z = sqrtu1 * np.cos(2 * np.pi * u3)
    w = sqrt1_u1 * np.sin(2 * np.pi * u2)

    result = np.stack([x, y, z, w], axis=-1)
    if device is not None:
        return _torch.from_numpy(result).to(device)
    return result
