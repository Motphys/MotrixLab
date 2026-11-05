# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Batch quaternion math retains scalar helper semantics and broadcasting."""

import numpy as np
import torch

from motrix_env_core.math import quaternion as scalar
from motrix_env_core.math import quaternion_array as batch


def test_quaternion_batch_matches_scalar_and_compiles():
    rng = np.random.default_rng(8)
    q = rng.normal(size=(7, 4)).astype(np.float32)
    other = rng.normal(size=(7, 4)).astype(np.float32)
    v = rng.normal(size=(7, 3)).astype(np.float32)

    def compute(q, other, v):
        return batch.inverse(q), batch.mul(q, other), batch.rotate_inverse(q, v), batch.matrix6d(q)

    expected = tuple(
        np.stack([function(*args) for args in zip(*arrays, strict=True)])
        for function, arrays in (
            (scalar.inverse, (q,)),
            (scalar.mul, (q, other)),
            (scalar.rotate_inverse, (q, v)),
            # Rows of the rotation matrix: rotate_vector(q, e_i) yields
            # column i, so transpose the column stack and keep two rows.
            (
                lambda quat: np.stack(
                    [
                        np.stack(
                            [
                                scalar.rotate_vector(unit[None, :], np.eye(3, dtype=quat.dtype)[i : i + 1])[0]
                                for i in range(3)
                            ]
                        )
                        .T[:2]
                        .reshape(-1)
                        for unit in (quat / np.linalg.norm(quat, axis=-1, keepdims=True),)
                    ]
                )[0],
                (q,),
            ),
        )
    )
    for actuals in (
        compute(q, other, v),
        torch.compile(compute, fullgraph=True)(*(torch.from_numpy(a) for a in (q, other, v))),
    ):
        for actual, reference in zip(actuals, expected, strict=True):
            if isinstance(actual, torch.Tensor):
                actual = actual.numpy()
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
