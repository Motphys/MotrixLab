# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import numba
import numpy as np

from motrix_env_core.math import quaternion
from motrix_env_core.numba.math.quaternion import (
    from_euler,
    inverse,
    mul,
    rotate_inverse,
    rotate_vector,
    rotation_distance,
    to_matrix_first_two_columns,
)


@numba.njit
def _from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return from_euler(roll, pitch, yaw)


@numba.njit
def _inverse(value: np.ndarray) -> np.ndarray:
    return inverse(value)


@numba.njit
def _mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return mul(lhs, rhs)


@numba.njit
def _rotate_inverse_array(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return rotate_inverse(rotation, vector)


@numba.njit
def _rotate_vector_array(rotation: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return rotate_vector(rotation, vector)


@numba.njit
def _rotate_inverse_tuple(rotation: np.ndarray, vector: tuple[float, float, float]) -> np.ndarray:
    buffer = np.full(5, np.nan, dtype=np.float32)
    rotate_inverse(rotation, vector, buffer[1:4])
    return buffer


@numba.njit
def _rotation_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return rotation_distance(lhs, rhs)


@numba.njit
def _to_matrix_first_two_columns(value: np.ndarray) -> np.ndarray:
    return to_matrix_first_two_columns(value)


def test_numba_quaternion_from_euler_matches_numpy_and_supports_caller_owned_out() -> None:
    roll, pitch, yaw = np.float32(0.2), np.float32(-0.3), np.float32(0.4)
    expected = quaternion.from_euler(roll, pitch, yaw)

    np.testing.assert_allclose(_from_euler(roll, pitch, yaw), expected, atol=1e-6)
    out = np.empty((4,), dtype=np.float32)
    returned = from_euler(roll, pitch, yaw, out)
    assert returned is out
    np.testing.assert_allclose(returned, expected, atol=1e-6)


def test_numba_quaternion_inverse_matches_numpy_and_supports_aliased_out() -> None:
    value = np.asarray([0.1, -0.2, 0.3, 0.9], dtype=np.float32)
    expected = quaternion.inverse(value)

    np.testing.assert_allclose(_inverse(value), expected, atol=1e-6)
    out = value.copy()
    returned = inverse(out, out)
    assert returned is out
    np.testing.assert_allclose(returned, expected, atol=1e-6)


def test_numba_rotate_vector_matches_numpy_and_supports_aliased_out() -> None:
    rotation = np.asarray([0.1, -0.2, 0.3, 0.9], dtype=np.float32)
    rotation /= np.linalg.norm(rotation)
    vector = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    expected = quaternion.rotate_vector(rotation, vector)

    np.testing.assert_allclose(_rotate_vector_array(rotation, vector), expected, atol=1e-6)
    out = vector.copy()
    returned = rotate_vector(rotation, out, out)
    assert returned is out
    np.testing.assert_allclose(returned, expected, atol=1e-6)


def test_numba_rotate_inverse_matches_numpy_for_array_and_tuple_vectors() -> None:
    rotation = np.asarray([0.1, -0.2, 0.3, 0.9], dtype=np.float32)
    rotation /= np.linalg.norm(rotation)
    vector = np.asarray([1.0, -2.0, 0.5], dtype=np.float32)
    expected = quaternion.rotate_inverse(rotation, vector)

    np.testing.assert_allclose(_rotate_inverse_array(rotation, vector), expected, atol=1e-6)
    tuple_result = _rotate_inverse_tuple(rotation, tuple(vector))
    assert np.isnan(tuple_result[0]) and np.isnan(tuple_result[4])
    np.testing.assert_allclose(tuple_result[1:4], expected, atol=1e-6)

    out = np.empty(3, dtype=np.float32)
    returned = rotate_inverse(rotation, vector, out)
    assert returned is out
    np.testing.assert_allclose(returned, expected, atol=1e-6)


def test_numba_rotation_distance_matches_numpy() -> None:
    lhs = np.asarray([0.1, -0.2, 0.3, 0.9], dtype=np.float32)
    rhs = np.asarray([-0.3, 0.4, 0.2, 0.8], dtype=np.float32)
    lhs /= np.linalg.norm(lhs)
    rhs /= np.linalg.norm(rhs)

    expected = quaternion.rotation_distance(lhs, rhs)
    np.testing.assert_allclose(_rotation_distance(lhs, rhs), expected, atol=1e-6)


def test_numba_to_matrix_first_two_columns_matches_numpy_and_supports_overlapping_out() -> None:
    value = np.asarray([0.1, -0.2, 0.3, 0.9], dtype=np.float32)
    expected = quaternion.to_matrix(value)[:, :2].reshape(6)

    np.testing.assert_allclose(_to_matrix_first_two_columns(value), expected, atol=1e-6)
    out = np.full(6, np.nan, dtype=np.float32)
    out[:4] = value
    returned = to_matrix_first_two_columns(out[:4], out)
    assert returned is out
    np.testing.assert_allclose(returned, expected, atol=1e-6)


def test_numba_quaternion_mul_matches_numpy_and_supports_aliased_out() -> None:
    lhs = np.asarray([0.1, -0.2, 0.3, 0.9], dtype=np.float32)
    rhs = np.asarray([-0.3, 0.4, 0.2, 0.8], dtype=np.float32)
    expected = quaternion.mul(lhs, rhs)

    np.testing.assert_allclose(_mul(lhs, rhs), expected, atol=1e-6)
    out = lhs.copy()
    returned = mul(out, rhs, out)
    assert returned is out
    np.testing.assert_allclose(returned, expected, atol=1e-6)
