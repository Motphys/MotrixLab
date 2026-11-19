# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Batched torch rotation primitives round-trip under their guard branches."""

import pytest
import torch

from motrix_env_core.math.rotation import mat_to_quat, mm3, mv3, quat_to_mat, so3_exp


def test_quaternion_layout_roundtrip() -> None:
    """Regression: mat_to_quat branch-component ordering corrupted the xyzw
    layout for most rotations, silently corrupting downstream increments."""
    torch.manual_seed(5)
    rotations = torch.linalg.qr(torch.randn(512, 3, 3))[0]
    err = (rotations - quat_to_mat(mat_to_quat(rotations))).abs().amax(dim=(1, 2))
    assert err.max() < 1e-4
    v = torch.randn(256, 3)
    v = v / v.norm(dim=-1, keepdim=True) * torch.rand(256, 1) * 3.0
    err2 = (so3_exp(v) - quat_to_mat(mat_to_quat(so3_exp(v)))).abs().amax(dim=(1, 2))
    assert err2.max() < 1e-4


@pytest.mark.parametrize("zero_rows", [(), (0,), (3,), (1, 5)])
def test_elementwise_products_match_matmul(zero_rows: tuple[int, ...]) -> None:
    """mm3/mv3 elementwise form must agree with plain matmul broadcasting."""
    torch.manual_seed(7)
    A = torch.randn(6, 3, 3)
    B = torch.randn(6, 3, 3)
    v = torch.randn(6, 3)
    for row in zero_rows:
        A[row] = 0.0
    torch.testing.assert_close(mm3(A, B), A @ B)
    torch.testing.assert_close(mv3(A, v), (A @ v.unsqueeze(-1)).squeeze(-1))
