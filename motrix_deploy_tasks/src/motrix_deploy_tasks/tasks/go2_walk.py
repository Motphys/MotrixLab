# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed artifact specification and runtime for ``go2_walk/v1``."""

from typing import Annotated, ClassVar

import numpy as np
from pydantic import Field, model_validator

from motrix_deploy.artifact.schema import TaskSpec
from motrix_deploy.contracts import FloatArray, JointServoCommand, RobotSpec, RobotState, float32_array
from motrix_deploy.errors import ValidationError
from motrix_deploy.policy import Policy
from motrix_deploy.policy.processing import PolicyProcessor
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.control import PolicyController
from motrix_deploy.runtime.lifecycle import ControllerStep
from motrix_deploy.task import DeployTask
from motrix_env_core.input import PlanarVelocityCommand

FLOAT32_MAX = float(np.finfo(np.float32).max)
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Float32Float = Annotated[FiniteFloat, Field(ge=-FLOAT32_MAX, le=FLOAT32_MAX)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0)]
PositiveFloat = Annotated[FiniteFloat, Field(gt=0)]
JointVector = Annotated[list[Float32Float], Field(min_length=1)]
NonNegativeJointVector = Annotated[list[Annotated[Float32Float, Field(ge=0)]], Field(min_length=1)]
CommandVector = Annotated[list[Float32Float], Field(min_length=3, max_length=3)]
CommandScale = Annotated[list[Annotated[Float32Float, Field(ge=0, le=1)]], Field(min_length=3, max_length=3)]
FeetPhaseOffsets = Annotated[list[Float32Float], Field(min_length=4, max_length=4)]


class Go2WalkTaskSpec(TaskSpec):
    """Self-contained control, command, gait and termination parameters."""

    task_name: ClassVar[str] = "go2_walk/v1"

    action_scale: NonNegativeJointVector
    kp: NonNegativeJointVector
    kd: NonNegativeJointVector
    # Policy action bounds before scaling/default offset, not robot position limits.
    action_lower: JointVector
    action_upper: JointVector
    command_lower: CommandVector
    command_upper: CommandVector
    command_scale: CommandScale
    feet_phase_offsets: FeetPhaseOffsets
    gait_frequency_hz: PositiveFloat
    standing_threshold: NonNegativeFloat
    termination_min_up_z: Annotated[FiniteFloat, Field(ge=-1, le=1)]

    @model_validator(mode="after")
    def validate_joint_vectors_and_ranges(self) -> "Go2WalkTaskSpec":
        joint_count = len(self.action_scale)
        for key in ("kp", "kd", "action_lower", "action_upper"):
            if len(getattr(self, key)) != joint_count:
                raise ValueError(f"{key} must have the same length as action_scale ({joint_count})")
        if any(lower > upper for lower, upper in zip(self.action_lower, self.action_upper)):
            raise ValueError("action_lower must be less than or equal to action_upper")
        if any(lower > upper for lower, upper in zip(self.command_lower, self.command_upper)):
            raise ValueError("command_lower must be less than or equal to command_upper")
        return self


class Go2WalkPolicyProcessor(PolicyProcessor[PlanarVelocityCommand]):
    """Go2 walking policy I/O and artifact-owned fall termination.

    Orientation termination is supported by both simulated and real IMU state.
    """

    def __init__(self, spec: Go2WalkTaskSpec, robot: RobotSpec) -> None:
        if len(spec.action_scale) != robot.joint_count:
            raise ValidationError(
                "task.config.action_scale", f"{robot.joint_count} joint values", len(spec.action_scale)
            )
        self.spec = spec
        self.robot = robot
        self._action_scale = np.asarray(spec.action_scale, dtype=np.float32)
        self._kp = np.asarray(spec.kp, dtype=np.float32)
        self._kd = np.asarray(spec.kd, dtype=np.float32)
        self._action_lower = np.asarray(spec.action_lower, dtype=np.float32)
        self._action_upper = np.asarray(spec.action_upper, dtype=np.float32)
        self._command_lower = np.asarray(spec.command_lower, dtype=np.float32)
        self._command_upper = np.asarray(spec.command_upper, dtype=np.float32)
        self._command_scale = np.asarray(spec.command_scale, dtype=np.float32)
        self._feet_phase_offsets = np.asarray(spec.feet_phase_offsets, dtype=np.float32)
        self._gait_frequency_hz = spec.gait_frequency_hz
        self._standing_threshold = spec.standing_threshold
        self._termination_min_up_z = spec.termination_min_up_z
        self._previous_action: FloatArray = np.zeros(self.robot.joint_count, dtype=np.float32)

    @property
    def command_lower(self) -> FloatArray:
        return self._command_lower * self._command_scale

    @property
    def command_upper(self) -> FloatArray:
        return self._command_upper * self._command_scale

    def check_termination(self, state: RobotState) -> str | None:
        x, y, _, _ = state.base_orientation_xyzw
        up_z = 1.0 - 2.0 * (x * x + y * y)
        if up_z <= self._termination_min_up_z:
            return "fall_orientation"
        return None

    def reset(self, state: RobotState, context: ControlContext[PlanarVelocityCommand]) -> None:
        """Reset action history; callers provide validated state and command."""
        self._previous_action.fill(0)

    def build_observation(self, state: RobotState, context: ControlContext[PlanarVelocityCommand]) -> FloatArray:
        """Consume runtime-validated state and PolicyController-validated input commands."""
        velocity = context.command.values[0, :]
        phase = np.zeros(4, dtype=np.float32)
        if context.step > 0 and np.linalg.norm(velocity) >= self._standing_threshold:
            phase = np.mod(
                context.elapsed_time_s * self._gait_frequency_hz + self._feet_phase_offsets,
                1.0,
            ).astype(np.float32)
        observation = np.concatenate(
            (
                state.base_angular_velocity,
                _projected_gravity(state.base_orientation_xyzw),
                state.joint_position - self.robot.default_joint_position,
                state.joint_velocity,
                self._previous_action,
                velocity,
                phase,
            )
        ).astype(np.float32)
        return observation

    def process_action(self, action: FloatArray) -> JointServoCommand:
        raw = float32_array(action, path="action.raw", shape=(self.robot.joint_count,))
        executed = np.clip(raw, self._action_lower, self._action_upper).astype(np.float32)
        target = self.robot.default_joint_position + self._action_scale * executed
        target = np.clip(target, self.robot.position_lower, self.robot.position_upper).astype(np.float32)
        self._previous_action = executed
        zeros = np.zeros(self.robot.joint_count, dtype=np.float32)
        return JointServoCommand(
            joint_position=target,
            joint_velocity=zeros,
            feedforward_torque=zeros,
            kp=self._kp,
            kd=self._kd,
        )

    def validate_command(self, command: PlanarVelocityCommand | None) -> None:
        if not isinstance(command, PlanarVelocityCommand):
            raise ValidationError("command.type", "PlanarVelocityCommand", type(command).__name__)
        if command.batch_size != 1:
            raise ValidationError("command.batch_size", "1", command.batch_size)
        velocity = command.values[0, :]
        if np.any(velocity < self._command_lower) or np.any(velocity > self._command_upper):
            raise ValidationError(
                "command.velocity",
                f"inside [{self._command_lower.tolist()}, {self._command_upper.tolist()}]",
                velocity.tolist(),
            )


def _projected_gravity(orientation_xyzw: FloatArray) -> FloatArray:
    x, y, z, w = orientation_xyzw
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return rotation @ np.array([0.0, 0.0, -1.0], dtype=np.float32)


class Go2WalkDeployTask(DeployTask[PlanarVelocityCommand]):
    """Complete Go2 walking task; its single policy phase handles normal stopping."""

    spec_type = Go2WalkTaskSpec

    def __init__(
        self, spec: Go2WalkTaskSpec, robot: RobotSpec, policy: Policy, steps: int | None = None, artifact=None
    ):
        self.spec = spec
        self.robot = robot
        self.policy_io = Go2WalkPolicyProcessor(spec, robot)
        self.policy_controller = PolicyController(self.policy_io, policy, steps=steps)

    @property
    def command_lower(self) -> FloatArray:
        return self.policy_io.command_lower

    @property
    def command_upper(self) -> FloatArray:
        return self.policy_io.command_upper

    def validate_command(self, command: PlanarVelocityCommand | None) -> None:
        self.policy_io.validate_command(command)

    def reset(self, state: RobotState) -> None:
        self.policy_controller.reset(state)

    def step(self, state: RobotState, context: ControlContext[PlanarVelocityCommand]) -> ControllerStep:
        return self.policy_controller.step(state, context)

    def request_stop(self) -> None:
        self.policy_controller.request_stop()
