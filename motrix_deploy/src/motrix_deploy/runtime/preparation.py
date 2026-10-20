# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Reusable measured-state preparation before a finite skill takes control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motrix_deploy.contracts import FloatArray, JointServoCommand, RobotSpec, RobotState, float32_array
from motrix_deploy.operator import OperatorCommand
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.lifecycle import ControllerStep


@dataclass(frozen=True)
class SkillPreparationSpec:
    """Servo target and measured readiness requirements for one skill module."""

    joint_position: FloatArray
    kp: FloatArray
    kd: FloatArray
    ramp_duration_s: float = 2.0
    settle_duration_s: float = 1.0
    timeout_s: float = 5.0
    max_joint_error_rad: float = 0.2
    max_joint_velocity_rad_s: float = 1.0
    max_base_tilt_rad: float = 0.5

    def __post_init__(self) -> None:
        nonnegative = (
            "ramp_duration_s",
            "settle_duration_s",
            "max_joint_error_rad",
            "max_joint_velocity_rad_s",
            "max_base_tilt_rad",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")


class JointServoPreparationState:
    """Prepare whichever skill is pending with one reusable top-level state."""

    name = "skill_preparation"

    def __init__(self, robot: RobotSpec) -> None:
        self.joint_names = robot.joint_names
        self._robot = robot
        self._spec: SkillPreparationSpec | None = None
        self._target = np.empty(robot.joint_count, dtype=np.float32)
        self._kp = np.empty(robot.joint_count, dtype=np.float32)
        self._kd = np.empty(robot.joint_count, dtype=np.float32)
        self._zeros = np.zeros(robot.joint_count, dtype=np.float32)
        self._initial_position: FloatArray | None = None
        self._ready_since: float | None = None
        self._reached_ready = False
        self._stop_requested = False

    @property
    def spec(self) -> SkillPreparationSpec | None:
        return self._spec

    def select(self, spec: SkillPreparationSpec) -> None:
        """Select the preparation contract for the next requested skill."""
        joint_count = self._robot.joint_count
        self._target = np.clip(
            float32_array(spec.joint_position, path="skill.preparation.joint_position", shape=(joint_count,)),
            self._robot.position_lower,
            self._robot.position_upper,
        ).astype(np.float32)
        self._kp = float32_array(spec.kp, path="skill.preparation.kp", shape=(joint_count,))
        self._kd = float32_array(spec.kd, path="skill.preparation.kd", shape=(joint_count,))
        self._spec = spec

    def reset(self, state: RobotState) -> None:
        if self._spec is None:
            raise RuntimeError("a skill preparation spec must be selected before entering preparation")
        self._initial_position = state.joint_position.copy()
        self._ready_since = None
        self._reached_ready = False
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def hold_command(self) -> JointServoCommand:
        """Hold the selected target after readiness has been reached."""
        return JointServoCommand(self._target, self._zeros, self._zeros, self._kp, self._kd)

    def step(self, state: RobotState, context: ControlContext[OperatorCommand]) -> ControllerStep:
        if self._stop_requested:
            return ControllerStep(complete=True)
        assert self._spec is not None and self._initial_position is not None
        spec = self._spec
        elapsed_time_s = context.elapsed_time_s
        x, y, _, _ = state.base_orientation_xyzw
        tilt = float(np.arccos(np.clip(1 - 2 * (float(x) ** 2 + float(y) ** 2), -1, 1)))
        metrics = {
            "max_joint_error_rad": float(np.max(np.abs(state.joint_position - self._target))),
            "max_joint_velocity_rad_s": float(np.max(np.abs(state.joint_velocity))),
            "base_tilt_rad": tilt,
        }
        if tilt > spec.max_base_tilt_rad:
            return ControllerStep(metrics=metrics, error=f"unsafe base tilt during preparation: {metrics}")
        ready = (
            elapsed_time_s + 1e-9 >= spec.ramp_duration_s
            and metrics["max_joint_error_rad"] <= spec.max_joint_error_rad
            and metrics["max_joint_velocity_rad_s"] <= spec.max_joint_velocity_rad_s
        )
        if ready:
            if self._ready_since is None:
                self._ready_since = elapsed_time_s
        else:
            self._ready_since = None
        if (
            self._ready_since is not None
            and elapsed_time_s - self._ready_since + 1e-9 >= spec.settle_duration_s
            and (self._reached_ready or elapsed_time_s <= spec.timeout_s + 1e-9)
        ):
            self._reached_ready = True
            return ControllerStep(complete=True, metrics=metrics)
        if not self._reached_ready and elapsed_time_s + 1e-9 >= spec.timeout_s:
            return ControllerStep(metrics=metrics, error=f"preparation timed out: {metrics}")
        fraction = (
            np.clip((elapsed_time_s + context.dt_s) / spec.ramp_duration_s, 0, 1) if spec.ramp_duration_s > 0 else 1
        )
        target = self._initial_position + np.float32(fraction) * (self._target - self._initial_position)
        target = np.clip(target, self._robot.position_lower, self._robot.position_upper).astype(np.float32)
        return ControllerStep(
            command=JointServoCommand(target, self._zeros, self._zeros, self._kp, self._kd),
            metrics=metrics,
        )


__all__ = ["JointServoPreparationState", "SkillPreparationSpec"]
