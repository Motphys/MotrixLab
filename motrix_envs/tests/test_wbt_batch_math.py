# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Batch quaternion math keeps NumPy/Torch semantics and stays compilable."""

import numpy as np
import torch

from motrix_env_core.math import quaternion


def test_quaternion_batch_matches_reference_and_compiles():
    rng = np.random.default_rng(8)
    q = rng.normal(size=(7, 4)).astype(np.float32)
    other = rng.normal(size=(7, 4)).astype(np.float32)
    v = rng.normal(size=(7, 3)).astype(np.float32)

    def compute(q, other, v, q_unit):
        return (
            quaternion.inverse(q),
            quaternion.mul(q, other),
            quaternion.rotate_inverse(q_unit, v),
            quaternion.matrix6d(q),
        )

    # References built from the NumPy host helpers.
    q_inv = quaternion.conjugate(q) / np.maximum(np.sum(q * q, axis=-1, keepdims=True), 1e-12)
    # mul via matrices: R(a * b) == R(a) @ R(b).
    mul_ref = np.stack([quaternion.to_matrix(a) @ quaternion.to_matrix(b) for a, b in zip(q, other, strict=True)])
    # rotate_vector/rotate_inverse closed forms are exact for unit quaternions
    # only; the simulator always delivers unit quaternions.
    q_unit = q / np.linalg.norm(q, axis=-1, keepdims=True)
    rotate_inv_ref = quaternion.rotate_vector(quaternion.conjugate(q_unit), v)
    # matrix6d: first two rows of R(q) as a (..., 6) vector.
    matrix6d_ref = quaternion.to_matrix(q)[:, :2, :].reshape(-1, 6)

    for actuals in (
        compute(q, other, v, q_unit),
        torch.compile(compute, fullgraph=True)(*(torch.from_numpy(a) for a in (q, other, v, q_unit))),
    ):
        for actual, reference in zip(actuals, (q_inv, None, rotate_inv_ref, matrix6d_ref), strict=True):
            if isinstance(actual, torch.Tensor):
                actual = actual.numpy()
            if reference is not None:
                np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
        mul_actual = actuals[1].numpy() if isinstance(actuals[1], torch.Tensor) else actuals[1]
        mul_ref_actual = np.stack([quaternion.to_matrix(m) for m in mul_actual])
        np.testing.assert_allclose(mul_ref_actual, mul_ref, rtol=2e-5, atol=2e-5)
