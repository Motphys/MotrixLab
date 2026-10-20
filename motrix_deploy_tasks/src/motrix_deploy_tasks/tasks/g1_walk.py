# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed artifact specification and runtime skeleton for ``g1_walk/v1``."""

import math
from typing import Annotated, ClassVar

import numpy as np
from pydantic import Field, model_validator

from motrix_deploy.artifact.schema import TaskSpec
from motrix_deploy.contracts import FloatArray, JointServoCommand, RobotSpec, RobotState, float32_array
from motrix_deploy.errors import ValidationError
from motrix_deploy.operator import OperatorProfileCfg
from motrix_deploy.policy import Policy
from motrix_deploy.policy.processing import PolicyProcessor
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.control import PolicyController
from motrix_deploy.runtime.lifecycle import ControllerStep
from motrix_deploy.task import DeployTask
from motrix_env_core.input import (
    BoundedGamePadPlanarVelocityBinding,
    CommandBinding,
    ConstantPlanarVelocityBinding,
    GamePadDevice,
    InputDevice,
    KeyboardDevice,
    KeyboardPlanarVelocityBinding,
    PlanarVelocityCommand,
)

FLOAT32_MAX = float(np.finfo(np.float32).max)
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Float32Float = Annotated[FiniteFloat, Field(ge=-FLOAT32_MAX, le=FLOAT32_MAX)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0)]
PositiveFloat = Annotated[FiniteFloat, Field(gt=0)]
JointVector = Annotated[list[Float32Float], Field(min_length=1)]
NonNegativeJointVector = Annotated[list[Annotated[Float32Float, Field(ge=0)]], Field(min_length=1)]
CommandVector = Annotated[list[Float32Float], Field(min_length=3, max_length=3)]
GaitPhaseOffsets = Annotated[list[Float32Float], Field(min_length=2, max_length=2)]


class G1WalkTaskSpec(TaskSpec):
    """Self-contained G1 locomotion policy I/O and gait parameters.

    Joint vectors use the canonical order stored in the artifact's
    ``RobotSpec.joint_names``.  Action bounds apply to raw policy output before
    multiplication by ``action_scale`` and addition of the robot default pose.
    Command vectors use ``(vx, vy, wz)`` order in SI units.
    """

    task_name: ClassVar[str] = "g1_walk/v1"

    action_scale: NonNegativeJointVector
    kp: NonNegativeJointVector
    kd: NonNegativeJointVector
    action_lower: JointVector
    action_upper: JointVector

    command_lower: CommandVector
    command_upper: CommandVector

    base_angular_velocity_scale: PositiveFloat
    joint_position_scale: PositiveFloat
    joint_velocity_scale: PositiveFloat

    gait_period_s: PositiveFloat
    gait_phase_offsets_rad: GaitPhaseOffsets
    standing_command_threshold: NonNegativeFloat

    @model_validator(mode="after")
    def validate_vector_lengths_and_ranges(self) -> "G1WalkTaskSpec":
        joint_count = len(self.action_scale)
        for key in ("kp", "kd", "action_lower", "action_upper"):
            if len(getattr(self, key)) != joint_count:
                raise ValueError(f"{key} must have the same length as action_scale ({joint_count})")

        if any(lower > upper for lower, upper in zip(self.action_lower, self.action_upper, strict=True)):
            raise ValueError("action_lower must be less than or equal to action_upper")
        if any(lower > 0.0 or upper < 0.0 for lower, upper in zip(self.action_lower, self.action_upper, strict=True)):
            raise ValueError("raw action range must contain zero for every joint")

        if any(lower > upper for lower, upper in zip(self.command_lower, self.command_upper, strict=True)):
            raise ValueError("command_lower must be less than or equal to command_upper")
        if any(lower > 0.0 or upper < 0.0 for lower, upper in zip(self.command_lower, self.command_upper, strict=True)):
            raise ValueError("velocity command range must contain zero on every axis")

        left, right = self.gait_phase_offsets_rad
        phase_difference = (right - left) % (2.0 * math.pi)
        if not math.isclose(phase_difference, math.pi, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("gait phase lanes must be separated by pi radians")
        return self


class G1WalkPolicyProcessor(PolicyProcessor[PlanarVelocityCommand]):
    """Reproduce the G1 walk policy observation and joint-servo contracts."""

    def __init__(self, spec: G1WalkTaskSpec, robot: RobotSpec) -> None:
        joint_count = robot.joint_count
        if len(spec.action_scale) != joint_count:
            raise ValidationError("task.config.action_scale", f"{joint_count} joint values", len(spec.action_scale))
        self.spec = spec
        self.robot = robot
        shape = (joint_count,)
        self._action_scale = _spec_array(spec.action_scale, path="task.config.action_scale", shape=shape)
        self._kp = _spec_array(spec.kp, path="task.config.kp", shape=shape)
        self._kd = _spec_array(spec.kd, path="task.config.kd", shape=shape)
        self._action_lower = _spec_array(spec.action_lower, path="task.config.action_lower", shape=shape)
        self._action_upper = _spec_array(spec.action_upper, path="task.config.action_upper", shape=shape)
        self._command_lower = _spec_array(spec.command_lower, path="task.config.command_lower", shape=(3,))
        self._command_upper = _spec_array(spec.command_upper, path="task.config.command_upper", shape=(3,))
        self._gait_phase_offsets = _spec_array(
            spec.gait_phase_offsets_rad,
            path="task.config.gait_phase_offsets_rad",
            shape=(2,),
        )
        self._base_angular_velocity_scale = np.float32(spec.base_angular_velocity_scale)
        self._joint_position_scale = np.float32(spec.joint_position_scale)
        self._joint_velocity_scale = np.float32(spec.joint_velocity_scale)
        self._gait_period_s = np.float32(spec.gait_period_s)
        self._standing_command_threshold = np.float32(spec.standing_command_threshold)
        self._previous_action = np.zeros(joint_count, dtype=np.float32)
        self._phase = np.float32(0.0)
        self._last_step = 0

    @property
    def command_lower(self) -> FloatArray:
        return self._command_lower

    @property
    def command_upper(self) -> FloatArray:
        return self._command_upper

    def reset(self, state: RobotState, context: ControlContext[PlanarVelocityCommand]) -> None:
        """Clear raw-action history and restart the deterministic gait clock."""

        self._validate_state_joint_count(state)
        self._previous_action.fill(0.0)
        self._phase = np.float32(0.0)
        self._last_step = 0

    def build_observation(
        self,
        state: RobotState,
        context: ControlContext[PlanarVelocityCommand],
    ) -> FloatArray:
        """Build the exact 100-element actor input used by ``g1-walk-flat``."""

        self._validate_state_joint_count(state)
        velocity = self._validated_velocity(context.command)
        self._advance_phase(context)

        linear_speed = float(np.linalg.norm(velocity[:2]))
        standing = linear_speed < float(self._standing_command_threshold) and abs(float(velocity[2])) < float(
            self._standing_command_threshold
        )
        if standing:
            phase = np.full(2, np.float32(math.pi), dtype=np.float32)
        else:
            phase = self._phase + self._gait_phase_offsets
        sin_phase = np.sin(phase).astype(np.float32)
        cos_phase = np.cos(phase).astype(np.float32)

        observation = np.concatenate(
            (
                state.base_angular_velocity * self._base_angular_velocity_scale,
                _projected_gravity_body(state.base_orientation_xyzw),
                velocity,
                (state.joint_position - self.robot.default_joint_position) * self._joint_position_scale,
                state.joint_velocity * self._joint_velocity_scale,
                self._previous_action,
                sin_phase,
                cos_phase,
            )
        ).astype(np.float32)
        expected_shape = (9 + 3 * self.robot.joint_count + 4,)
        if observation.shape != expected_shape or not np.all(np.isfinite(observation)):
            raise ValidationError("observation", f"finite float32 values with shape {expected_shape}", observation)
        return observation

    def process_action(self, action: FloatArray) -> JointServoCommand:
        """Clip raw policy output and convert it to canonical joint servos."""

        shape = (self.robot.joint_count,)
        raw = float32_array(action, path="action.raw", shape=shape)
        executed = np.clip(raw, self._action_lower, self._action_upper).astype(np.float32)
        target = self.robot.default_joint_position + self._action_scale * executed
        target = np.clip(target, self.robot.position_lower, self.robot.position_upper).astype(np.float32)
        self._previous_action = executed.copy()
        zeros = np.zeros(self.robot.joint_count, dtype=np.float32)
        return JointServoCommand(
            joint_position=target,
            joint_velocity=zeros,
            feedforward_torque=zeros.copy(),
            kp=np.array(self._kp, dtype=np.float32, copy=True),
            kd=np.array(self._kd, dtype=np.float32, copy=True),
        )

    def validate_command(self, command: PlanarVelocityCommand | None) -> None:
        self._validated_velocity(command)

    def _validated_velocity(self, command: PlanarVelocityCommand | None) -> FloatArray:
        if not isinstance(command, PlanarVelocityCommand):
            raise ValidationError("command.type", "PlanarVelocityCommand", type(command).__name__)
        if command.batch_size != 1:
            raise ValidationError("command.batch_size", "1", command.batch_size)
        velocity = command.values[0]
        if np.any(velocity < self._command_lower) or np.any(velocity > self._command_upper):
            raise ValidationError(
                "command.velocity",
                f"inside [{self._command_lower.tolist()}, {self._command_upper.tolist()}]",
                velocity.tolist(),
            )
        return velocity

    def _validate_state_joint_count(self, state: RobotState) -> None:
        expected = (self.robot.joint_count,)
        if state.joint_position.shape != expected or state.joint_velocity.shape != expected:
            raise ValidationError(
                "state.joints.shape",
                str(expected),
                (state.joint_position.shape, state.joint_velocity.shape),
            )

    def _advance_phase(self, context: ControlContext[PlanarVelocityCommand]) -> None:
        if context.step < self._last_step:
            raise ValidationError("context.step", f">= {self._last_step}", context.step)
        if not np.isfinite(context.dt_s) or context.dt_s <= 0.0:
            raise ValidationError("context.dt_s", "a positive finite value", context.dt_s)
        elapsed_steps = context.step - self._last_step
        if elapsed_steps:
            increment = 2.0 * math.pi * elapsed_steps * context.dt_s / float(self._gait_period_s)
            self._phase = np.float32((float(self._phase) + increment) % (2.0 * math.pi))
            self._last_step = context.step


def _projected_gravity_body(orientation_xyzw: FloatArray) -> FloatArray:
    """Rotate world ``-Z`` into body coordinates for a body-to-world quaternion."""

    x, y, z, w = orientation_xyzw
    return np.asarray(
        (
            2.0 * (y * w - x * z),
            -2.0 * (x * w + y * z),
            2.0 * (x * x + y * y) - 1.0,
        ),
        dtype=np.float32,
    )


def _spec_array(value: object, *, path: str, shape: tuple[int, ...]) -> FloatArray:
    """Materialize one validated Pydantic numeric vector as immutable float32."""

    return float32_array(np.asarray(value, dtype=np.float32), path=path, shape=shape)


class G1WalkDeployTask(DeployTask[PlanarVelocityCommand]):
    """Assemble G1 locomotion policy I/O with an operator command source."""

    spec_type = G1WalkTaskSpec

    def __init__(
        self,
        spec: G1WalkTaskSpec,
        robot: RobotSpec,
        policy: Policy,
        steps: int | None = None,
        artifact: object | None = None,
    ) -> None:
        del artifact
        self.spec = spec
        self.robot = robot
        self.policy_io = G1WalkPolicyProcessor(spec, robot)
        self.policy_controller = PolicyController(self.policy_io, policy, steps=steps)
        self._keyboard_attached = False
        self._gamepad_attached = False
        self._operator_profile: OperatorProfileCfg | None = None

    def configure_operator(self, profile: OperatorProfileCfg | None) -> None:
        """Store task-owned gamepad command wiring from the selected recipe."""
        self._operator_profile = profile

    def prepare(
        self,
        *,
        simulation: bool,
        device: InputDevice | None = None,
        spawn_pose: str | None = None,
    ) -> CommandBinding[PlanarVelocityCommand]:
        """Bind viewer keys to velocity commands, or stand still without a keyboard."""

        del simulation, spawn_pose
        command_lower, command_upper = self.command_bounds
        if isinstance(device, KeyboardDevice):
            self._keyboard_attached = True
            self._gamepad_attached = False
            return KeyboardPlanarVelocityBinding(
                device,
                command_lower=command_lower,
                command_upper=command_upper,
            )
        if isinstance(device, GamePadDevice) and self._operator_profile is not None:
            velocity = self._operator_profile.planar_velocity
            if velocity is not None:
                self._keyboard_attached = False
                self._gamepad_attached = True
                return BoundedGamePadPlanarVelocityBinding(
                    device,
                    linear_x_axis=velocity.linear_x_axis,
                    linear_y_axis=velocity.linear_y_axis,
                    yaw_axis=velocity.yaw_axis,
                    command_lower=command_lower,
                    command_upper=command_upper,
                    deadzone=velocity.deadzone,
                    range_scale=velocity.range_scale,
                    invert_linear_x=velocity.invert_linear_x,
                    invert_linear_y=velocity.invert_linear_y,
                    invert_yaw=velocity.invert_yaw,
                    deadman_button=velocity.deadman_button,
                )
        self._keyboard_attached = False
        self._gamepad_attached = False
        binding = ConstantPlanarVelocityBinding((0.0, 0.0, 0.0))
        self.validate_command(binding.read_command())
        return binding

    @property
    def command_bounds(self) -> tuple[FloatArray, FloatArray]:
        return self.policy_io.command_lower, self.policy_io.command_upper

    @property
    def operator_hint(self) -> str | None:
        if self._keyboard_attached:
            return "w/s forward/back  a/d left/right  q/e turn left/right"
        if self._gamepad_attached:
            return "G1 remote: LT+Up stand  RB+X enable policy  LT+B passive  select emergency stop"
        return None

    def reset(self, state: RobotState) -> None:
        self.policy_controller.reset(state)

    def step(
        self,
        state: RobotState,
        context: ControlContext[PlanarVelocityCommand],
    ) -> ControllerStep:
        return self.policy_controller.step(state, context)

    def request_stop(self) -> None:
        self.policy_controller.request_stop()

    def validate_command(self, command: PlanarVelocityCommand | None) -> None:
        self.policy_io.validate_command(command)


__all__ = ["G1WalkDeployTask", "G1WalkPolicyProcessor", "G1WalkTaskSpec"]
