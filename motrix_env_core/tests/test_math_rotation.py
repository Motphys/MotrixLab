# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Batched torch rotation primitives round-trip under their guard branches."""

import pytest
import torch

from motrix_env_core.math.rotation import mat_to_quat, mm3, mv3, quat_slerp, quat_to_mat, rot_angle_sq, so3_exp


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


def test_elementwise_products_match_matmul() -> None:
    """mm3/mv3 elementwise form must agree with plain matmul broadcasting."""
    torch.manual_seed(7)
    A = torch.randn(6, 3, 3)
    B = torch.randn(6, 3, 3)
    v = torch.randn(6, 3)
    A[[1, 5]] = 0.0
    torch.testing.assert_close(mm3(A, B), A @ B)
    torch.testing.assert_close(mv3(A, v), (A @ v.unsqueeze(-1)).squeeze(-1))


def test_quat_slerp_broadcast_endpoints_and_shortest_path() -> None:
    dtype = torch.float32
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype)
    # The negative representative of +90 degrees must still use the short arc.
    target = -torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5], dtype=dtype)
    fractions = torch.tensor([0.0, 0.5, 1.0], dtype=dtype)
    actual = quat_slerp(identity, target, fractions)
    expected = so3_exp(torch.stack((fractions * 0, fractions * 0, fractions * torch.pi / 2), -1))
    torch.testing.assert_close(quat_to_mat(actual), expected)
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones_like(fractions))
    repeated = quat_slerp(identity.expand(2, 1, 4), target, fractions)
    assert repeated.shape == (2, 3, 4)
    torch.testing.assert_close(repeated, actual.expand_as(repeated))


@pytest.mark.parametrize("opposite_sign", [False, True])
def test_quat_slerp_coincident_rotations_have_finite_gradients(opposite_sign: bool) -> None:
    q0 = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)
    q1 = (q0.detach() * (-1 if opposite_sign else 1)).requires_grad_()
    fraction = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    result = quat_slerp(q0, q1, fraction)
    torch.testing.assert_close(quat_to_mat(result), torch.eye(3, dtype=result.dtype))
    for gradient in torch.autograd.grad(result.sum(), (q0, q1, fraction)):
        assert torch.isfinite(gradient).all()


def test_rot_angle_sq_broadcast_distance_symmetry_and_finite_endpoints() -> None:
    angles = torch.tensor([0.0, 0.3, 1.2, torch.pi], dtype=torch.float64, requires_grad=True)
    vectors = torch.stack((angles * 0, angles * 0, angles), -1)
    rotations = so3_exp(vectors)
    identity = torch.eye(3, dtype=angles.dtype)
    actual = rot_angle_sq(identity, rotations)
    torch.testing.assert_close(actual[1:3], angles[1:3].square())
    torch.testing.assert_close(actual, rot_angle_sq(rotations, identity))
    assert torch.isfinite(actual).all()
    assert (actual >= 0).all()
    assert actual[0] < actual[1] < actual[2] < actual[3] <= torch.pi**2
    gradient = torch.autograd.grad(actual.sum(), angles)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient[1:3], 2 * angles[1:3])
