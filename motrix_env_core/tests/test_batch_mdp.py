# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared batch reductions preserve environment independence and native dtype."""

import numpy as np
import pytest
import torch

from motrix_env_core.array import mx
from motrix_env_core.manager.dispatch import dispatch


@dispatch
def _batch_pose(position, default, weights):
    delta = position - default
    return mx.sum(delta * delta * weights, axis=-1)


@pytest.mark.parametrize("shape", [(1, 1), (7, 29), (4, 0)])
def test_shared_batch_pose_preserves_lane_independence(shape):
    rng = np.random.default_rng(5)
    position = rng.normal(size=shape).astype(np.float32)
    default = rng.normal(size=shape[1]).astype(np.float32)
    weights = rng.uniform(size=shape[1]).astype(np.float32)
    expected = np.sum((position - default) ** 2 * weights, axis=-1)
    for native_args in (
        (position, default, weights),
        tuple(torch.from_numpy(value) for value in (position, default, weights)),
    ):
        result = _batch_pose(*native_args)
        assert np.asarray(result).shape == (shape[0],)
        assert np.asarray(result).dtype == np.float32
        np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-6)


def test_sum_supports_runtime_axes_and_strided_input():
    value = np.arange(48, dtype=np.float32).reshape(4, 12)[:, ::2]
    for axis in (-2, -1, 0, 1):
        np.testing.assert_allclose(mx.sum(value, axis=axis), np.sum(value, axis=axis))
    for axis in (-3, 2):
        with pytest.raises(ValueError, match="bounds"):
            mx.sum(np.asarray(value), axis=axis)


def test_sum_without_axis():
    assert mx.sum(np.ones((2, 3), dtype=np.float32)) == 6
