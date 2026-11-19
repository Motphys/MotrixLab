# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
# Adapted from SuperMimic (supermimic_vibed), Copyright (c) 2026 ETH Zurich,
# Chong Zhang. MIT License.

"""Canonical free-root robot state representation for world-model dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from motrix_env_core.math.rotation import mat_to_quat, mm3, mv3, quat_to_mat, so3_exp, so3_log, yaw_mat
from motrix_rl_supermimic.world_state import StateRepresentation

_RobotState = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class CanonicalRobotState(StateRepresentation):
    """Flat ``(p, quat_xyzw, v, w, q, qd)`` with frame-local predictions.

    Position increments use the root heading frame; rotation, linear velocity,
    and angular velocity increments use its full orientation. Error scales follow
    those six groups, with three rotation-vector components replacing quaternion.
    """

    num_joints: int
    error_scales: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    # Four semi-implicit substeps with linearly changing velocity give (n+1)/(2n).
    DRIFT_W: ClassVar[float] = 0.625

    def __post_init__(self) -> None:
        if self.num_joints < 0:
            raise ValueError("Canonical robot state requires a nonnegative joint count")
        if len(self.error_scales) != 6:
            raise ValueError("Canonical robot state requires six group error scales")

    @property
    def size(self) -> int:
        return 13 + 2 * self.num_joints

    @property
    def feature_dim(self) -> int:
        return 13 + 2 * self.num_joints

    @property
    def prediction_dim(self) -> int:
        return 12 + 2 * self.num_joints

    def unpack(self, state: torch.Tensor) -> _RobotState:
        joint_end = 13 + self.num_joints
        return (
            state[..., :3],
            quat_to_mat(state[..., 3:7]),
            state[..., 7:10],
            state[..., 10:13],
            state[..., 13:joint_end],
            state[..., joint_end:],
        )

    @staticmethod
    def pack(state: _RobotState) -> torch.Tensor:
        p, rotation, v, w, q, qd = state
        return torch.cat((p, mat_to_quat(rotation), v, w, q, qd), -1)

    def features(self, state: torch.Tensor) -> torch.Tensor:
        p, rotation, v, w, q, qd = self.unpack(state)
        heading = yaw_mat(rotation)
        inverse = rotation.transpose(-1, -2)
        tilt = mm3(heading.transpose(-1, -2), rotation)[..., :, :2].reshape(rotation.shape[:-2] + (6,))
        return torch.cat((p[..., 2:3], tilt, mv3(inverse, v), mv3(inverse, w), q, qd), -1)

    def increments(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        p, rotation, v, w, q, qd = self.unpack(state)
        p1, rotation1, v1, w1, q1, qd1 = self.unpack(next_state)
        heading = yaw_mat(rotation)
        inverse = rotation.transpose(-1, -2)
        return torch.cat(
            (
                mv3(heading.transpose(-1, -2), p1 - p),
                so3_log(mm3(inverse, rotation1)),
                mv3(inverse, v1 - v),
                mv3(inverse, w1 - w),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def _drift(
        self, state: _RobotState, v1: torch.Tensor, w1: torch.Tensor, qd1: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, rotation, v, w, _, qd = state
        a, b = dt * (1 - self.DRIFT_W), dt * self.DRIFT_W
        return (
            mv3(yaw_mat(rotation).transpose(-1, -2), a * v + b * v1),
            mv3(rotation.transpose(-1, -2), a * w + b * w1),
            a * qd + b * qd1,
        )

    def targets(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        increments = self.increments(state, next_state)
        if not structured:
            return increments
        dp, dth, dv, dw, dq, dqd = increments.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
        next_parts = self.unpack(next_state)
        gp, gth, gq = self._drift(self.unpack(state), next_parts[2], next_parts[3], next_parts[5], dt)
        return torch.cat((dp - gp, dth - gth, dv, dw, dq - gq, dqd), -1)

    def integrate(
        self, state: torch.Tensor, prediction: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        parts = self.unpack(state)
        p, rotation, v, w, q, qd = parts
        dp, dth, dv, dw, dq, dqd = prediction.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
        v1 = v + mv3(rotation, dv)
        w1 = w + mv3(rotation, dw)
        qd1 = qd + dqd
        if structured:
            gp, gth, gq = self._drift(parts, v1, w1, qd1, dt)
            dp, dth, dq = dp + gp, dth + gth, dq + gq
        return self.pack((p + mv3(yaw_mat(rotation), dp), mm3(rotation, so3_exp(dth)), v1, w1, q + dq, qd1))

    def error(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return self.increments(truth, pred)

    def error_floor(self, device: str | torch.device) -> torch.Tensor:
        widths = (3, 3, 3, 3, self.num_joints, self.num_joints)
        return torch.cat(
            [torch.full((width,), scale, device=device) for width, scale in zip(widths, self.error_scales)], -1
        )
