# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Analytic-solution and differentiability contracts of the FK engine.

The tree under test is a planar 3-revolute arm (all axes z, unit links along
x) whose positions, rotations and velocities have closed forms, so the tests
pin observable behavior without a simulator.
"""

import math

import pytest
import torch

from motrix_env_core.math.kinematics import KinematicTree
from motrix_env_core.math.rotation import axis_angle_to_mat


def test_axis_angle_to_mat_rodrigues():
    axis = torch.tensor([0.0, 0.0, 1.0])
    angle = torch.tensor([math.pi / 4, -math.pi / 3, 0.0])
    R = axis_angle_to_mat(axis, angle)
    c, s = torch.cos(angle), torch.sin(angle)
    expected = torch.stack(
        (
            c,
            -s,
            torch.zeros_like(c),
            s,
            c,
            torch.zeros_like(c),
            torch.zeros_like(c),
            torch.zeros_like(c),
            torch.ones_like(c),
        ),
        -1,
    ).reshape(3, 3, 3)
    torch.testing.assert_close(R, expected)
    # orthogonal with unit determinant for arbitrary axes
    torch.manual_seed(7)
    axes = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    angles = torch.rand(5) * 2 * math.pi
    R = axis_angle_to_mat(axes, angles)
    torch.testing.assert_close(R @ R.transpose(-1, -2), torch.eye(3).expand(5, 3, 3))
    torch.testing.assert_close(torch.linalg.det(R), torch.ones(5))


def planar_arm(device="cpu") -> KinematicTree:
    """Root body plus two moving links: z-hinge, unit link along x, identity offsets."""
    n = 3
    axes = torch.zeros(n, 3, device=device)
    axes[:, 2] = 1.0
    off_p = torch.zeros(n, 3, device=device)
    off_p[1:, 0] = 1.0
    off_R = torch.eye(3, device=device).repeat(n, 1, 1)
    return KinematicTree(parent=[-1, 0, 1], joint_idx=[-1, 0, 1], axes=axes, off_p=off_p, off_R=off_R, device=device)


def root_pose(theta, device="cpu"):
    """Root at origin with yaw ``theta`` and unit yaw rate; batch shape (B,)."""
    theta = torch.as_tensor(theta, dtype=torch.float32, device=device)
    c, s = torch.cos(theta), torch.sin(theta)
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)
    R = torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(theta.shape + (3, 3))
    p = torch.stack((theta * 0.0, theta * 0.0, theta * 0.0), -1)
    v = torch.stack((zero, zero, zero), -1)
    w = torch.stack((zero, zero, torch.ones_like(theta)), -1)
    return p, R, v, w


@pytest.mark.parametrize("device", ["cpu"])
def test_positions_match_closed_form(device):
    torch.manual_seed(0)
    B = 5
    q = torch.rand(B, 2, device=device) * 2 - 1
    theta = torch.rand(B, device=device) * 2 * math.pi
    p, R, _, _ = root_pose(theta, device)
    pos = planar_arm(device).positions(p, R, q)

    th1 = theta + q[:, 0]
    # root body origin is p; body-i origin = parent origin + parent rotation @ off_p[i]
    zeros = torch.zeros_like(theta)
    expected = torch.stack(
        (
            torch.stack((zeros, zeros, zeros), -1),
            torch.stack((torch.cos(theta), torch.sin(theta), zeros), -1),
            torch.stack((torch.cos(theta) + torch.cos(th1), torch.sin(theta) + torch.sin(th1), zeros), -1),
        ),
        -2,
    )
    torch.testing.assert_close(pos, expected)


def test_forward_rotations_and_angular_velocity():
    torch.manual_seed(1)
    B = 4
    q = torch.rand(B, 2) * 2 - 1
    qd = torch.rand(B, 2) * 2 - 1
    theta = torch.rand(B) * 2 * math.pi
    p, R, v, w = root_pose(theta)
    P, Rs, V, W = planar_arm().forward(p, R, v, w, q, qd)

    def yaw_mat(angle):
        c, s = torch.cos(angle), torch.sin(angle)
        zero, one = torch.zeros_like(angle), torch.ones_like(angle)
        return torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(angle.shape + (3, 3))

    assert Rs.shape == (B, 3, 3, 3)
    torch.testing.assert_close(Rs[:, 0], yaw_mat(theta))
    torch.testing.assert_close(Rs[:, 1], yaw_mat(theta + q[:, 0]))
    torch.testing.assert_close(Rs[:, 2], yaw_mat(theta + q[:, 0] + q[:, 1]))
    # angular velocities compose: root yaw rate + upstream joint rates, all about z
    expected_w = torch.zeros(B, 3, 3)
    expected_w[..., 2] = torch.stack((w[:, 2], w[:, 2] + qd[:, 0], w[:, 2] + qd[:, 0] + qd[:, 1]), -1)
    torch.testing.assert_close(W, expected_w)


def test_link_origin_linear_velocity_closed_form():
    torch.manual_seed(2)
    B = 3
    q = torch.rand(B, 2) * 2 - 1
    qd = torch.rand(B, 2) * 2 - 1
    theta = torch.rand(B) * 2 * math.pi
    _, R, _, _ = root_pose(theta)
    zero3 = torch.zeros(B, 3)
    w = torch.stack((zero3[:, 0], zero3[:, 1], torch.ones(B)), -1)
    # root linear velocity v0: pick a nonzero one so cross terms are exercised
    v0 = torch.tensor([[0.3, -0.2, 0.1]], device=theta.device).repeat(B, 1)
    P, _, V, _ = planar_arm().forward(zero3, R, v0, w, q, qd)

    th1 = theta + q[:, 0]
    r1 = torch.stack((torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)), -1)
    r12 = torch.stack((torch.cos(th1), torch.sin(th1), torch.zeros_like(theta)), -1)
    w1 = w[:, 2] + qd[:, 0]
    # V(root)=v0; V(body1)=v0 + ω_root × r1; V(body2) adds ω_1 × (r2 - r1)
    v1 = v0 + torch.cross(w, r1, dim=-1)
    v2 = v1 + torch.stack((-w1 * r12[:, 1], w1 * r12[:, 0], torch.zeros_like(w1)), -1)
    torch.testing.assert_close(V[:, 1], v1)
    torch.testing.assert_close(V[:, 2], v2)


def test_positions_matches_forward():
    torch.manual_seed(3)
    q = torch.rand(7, 2)
    p, R, v, w = root_pose(torch.rand(7) * 2 * math.pi)
    tree = planar_arm()
    P, *_ = tree.forward(p, R, v, w, q, torch.zeros_like(q))
    torch.testing.assert_close(tree.positions(p, R, q), P)


def test_gradients_flow_to_q_and_root():
    tree = planar_arm()
    q = torch.rand(6, 2, requires_grad=True)
    theta = (torch.rand(6) * 2 - 1) * math.pi
    p, R, v, w = root_pose(theta)
    p = p.detach().requires_grad_(True)

    P, Rs, V, W = tree.forward(p, R, v, w, q, q)
    loss = (P.square().sum() + W.square().sum()) / P.numel()
    loss.backward()
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert p.grad is not None and p.grad.abs().sum() > 0

    # finite-difference check on the position head in float64
    tree64 = _double(planar_arm())
    q64 = q.detach().double().requires_grad_(True)
    p64 = p.detach().double().requires_grad_(True)
    R64 = R.double()
    out = tree64.positions(p64, R64, q64)
    out.sum().backward()
    with torch.no_grad():
        eps = 1e-6
        for b in range(2):
            for j in range(2):
                qp = q64.detach().clone()
                qp[b, j] += eps
                qm = q64.detach().clone()
                qm[b, j] -= eps
                fd = (tree64.positions(p64.detach(), R64, qp).sum() - tree64.positions(p64.detach(), R64, qm).sum()) / (
                    2 * eps
                )
                torch.testing.assert_close(q64.grad[b, j], fd, rtol=1e-5, atol=1e-5)


def test_quaternion_offsets_accepted():
    # x-axis rotation pins the xyzw convention: (sin, 0, 0, cos), unlike wxyz
    half = math.pi / 8
    axes = torch.zeros(2, 3)
    axes[1, 2] = 1.0
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0], [math.sin(half), 0.0, 0.0, math.cos(half)]])
    c, s = math.cos(2 * half), math.sin(2 * half)
    off_R_mats = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        ]
    )
    tree_q = KinematicTree(
        parent=[-1, 0],
        joint_idx=[-1, 0],
        axes=axes,
        off_p=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        off_R=quat,
    )
    tree_m = KinematicTree(
        parent=[-1, 0],
        joint_idx=[-1, 0],
        axes=axes,
        off_p=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        off_R=off_R_mats,
    )
    p = torch.zeros(1, 3)
    R = torch.eye(3).unsqueeze(0)
    q = torch.zeros(1, 1)
    torch.testing.assert_close(tree_q.off_R, off_R_mats)
    torch.testing.assert_close(tree_q.positions(p, R, q), tree_m.positions(p, R, q))


def _double(tree: KinematicTree) -> KinematicTree:
    child = KinematicTree.__new__(KinematicTree)
    child.__dict__.update(tree.__dict__)
    child.axes = tree.axes.double()
    child.off_p = tree.off_p.double()
    child.off_R = tree.off_R.double()
    return child
