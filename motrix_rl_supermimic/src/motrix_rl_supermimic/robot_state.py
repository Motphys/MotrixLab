# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
# Adapted from SuperMimic (supermimic_vibed), Copyright (c) 2026 ETH Zurich,
# Chong Zhang. MIT License.

"""Canonical free-root robot state representation for world-model dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import torch

from motrix_env_core.math.rotation import mat_to_quat, mm3, mv3, quat_to_mat, so3_exp, so3_log, yaw_mat
from motrix_rl_supermimic.world_state import StateComputation, StateStatistics, WorldStateTerm


class RobotState(NamedTuple):
    """Decomposed free-root state; root velocities are in the world frame.

    Rotation is a matrix ``(..., 3, 3)``; root vectors are ``(..., 3)``
    and joint vectors are ``(..., num_joints)``. Fields borrow their tensors.
    """

    position: torch.Tensor
    rotation: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor


@dataclass(frozen=True)
class CanonicalRobotState(WorldStateTerm):
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

    def unpack(self, state: torch.Tensor) -> RobotState:
        joint_end = 13 + self.num_joints
        return RobotState(
            position=state[..., :3],
            rotation=quat_to_mat(state[..., 3:7]),
            linear_velocity=state[..., 7:10],
            angular_velocity=state[..., 10:13],
            joint_position=state[..., 13:joint_end],
            joint_velocity=state[..., joint_end:],
        )

    @staticmethod
    def pack(state: RobotState) -> torch.Tensor:
        return torch.cat(
            (
                state.position,
                mat_to_quat(state.rotation),
                state.linear_velocity,
                state.angular_velocity,
                state.joint_position,
                state.joint_velocity,
            ),
            -1,
        )

    def prepare(self, state: torch.Tensor) -> StateComputation:
        return _RobotStateComputation(self.unpack(state), self)

    @staticmethod
    def _features(parts: RobotState) -> torch.Tensor:
        rotation = parts.rotation
        heading = yaw_mat(rotation)
        inverse = rotation.transpose(-1, -2)
        tilt = mm3(heading.transpose(-1, -2), rotation)[..., :, :2].reshape(rotation.shape[:-2] + (6,))
        return torch.cat(
            (
                parts.position[..., 2:3],
                tilt,
                mv3(inverse, parts.linear_velocity),
                mv3(inverse, parts.angular_velocity),
                parts.joint_position,
                parts.joint_velocity,
            ),
            -1,
        )

    def increments(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return self._increments(self.unpack(state), self.unpack(next_state))

    @staticmethod
    def _increments(current: RobotState, following: RobotState) -> torch.Tensor:
        heading = yaw_mat(current.rotation)
        inverse = current.rotation.transpose(-1, -2)
        return torch.cat(
            (
                mv3(heading.transpose(-1, -2), following.position - current.position),
                so3_log(mm3(inverse, following.rotation)),
                mv3(inverse, following.linear_velocity - current.linear_velocity),
                mv3(inverse, following.angular_velocity - current.angular_velocity),
                following.joint_position - current.joint_position,
                following.joint_velocity - current.joint_velocity,
            ),
            -1,
        )

    def _drift(
        self, state: RobotState, v1: torch.Tensor, w1: torch.Tensor, qd1: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b = dt * (1 - self.DRIFT_W), dt * self.DRIFT_W
        return (
            mv3(yaw_mat(state.rotation).transpose(-1, -2), a * state.linear_velocity + b * v1),
            mv3(state.rotation.transpose(-1, -2), a * state.angular_velocity + b * w1),
            a * state.joint_velocity + b * qd1,
        )

    def targets(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> torch.Tensor:
        current, following = self.unpack(state), self.unpack(next_state)
        return self._targets(current, following, self._increments(current, following), structured=structured, dt=dt)

    def statistics(
        self, state: torch.Tensor, next_state: torch.Tensor, *, structured: bool = False, dt: float = 0.02
    ) -> StateStatistics:
        current, following = self.unpack(state), self.unpack(next_state)
        increments = self._increments(current, following)
        return StateStatistics(
            self._features(current),
            self._targets(current, following, increments, structured=structured, dt=dt),
            increments,
        )

    def _targets(
        self,
        current: RobotState,
        following: RobotState,
        increments: torch.Tensor,
        *,
        structured: bool,
        dt: float,
    ) -> torch.Tensor:
        if not structured:
            return increments
        dp, dth, dv, dw, dq, dqd = increments.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
        gp, gth, gq = self._drift(
            current, following.linear_velocity, following.angular_velocity, following.joint_velocity, dt
        )
        return torch.cat((dp - gp, dth - gth, dv, dw, dq - gq, dqd), -1)

    def _integrate(self, parts: RobotState, prediction: torch.Tensor, *, structured: bool, dt: float) -> torch.Tensor:
        rotation = parts.rotation
        dp, dth, dv, dw, dq, dqd = prediction.split((3, 3, 3, 3, self.num_joints, self.num_joints), -1)
        v1 = parts.linear_velocity + mv3(rotation, dv)
        w1 = parts.angular_velocity + mv3(rotation, dw)
        qd1 = parts.joint_velocity + dqd
        if structured:
            gp, gth, gq = self._drift(parts, v1, w1, qd1, dt)
            dp, dth, dq = dp + gp, dth + gth, dq + gq
        return self.pack(
            RobotState(
                position=parts.position + mv3(yaw_mat(rotation), dp),
                rotation=mm3(rotation, so3_exp(dth)),
                linear_velocity=v1,
                angular_velocity=w1,
                joint_position=parts.joint_position + dq,
                joint_velocity=qd1,
            )
        )

    def error(self, pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        return self.increments(truth, pred)

    def error_floor(self, device: str | torch.device) -> torch.Tensor:
        widths = (3, 3, 3, 3, self.num_joints, self.num_joints)
        return torch.cat(
            [torch.full((width,), scale, device=device) for width, scale in zip(widths, self.error_scales)], -1
        )


@dataclass(frozen=True)
class _RobotStateComputation(StateComputation):
    """Borrow a decoded robot state and its codec for one numerical step."""

    state: RobotState
    codec: CanonicalRobotState

    def features(self) -> torch.Tensor:
        return self.codec._features(self.state)

    def integrate(self, prediction: torch.Tensor, *, structured: bool = False, dt: float = 0.02) -> torch.Tensor:
        return self.codec._integrate(self.state, prediction, structured=structured, dt=dt)
