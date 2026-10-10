# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed artifact specification and runtime for ``go2_walk/v1``."""

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar

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
        self._phase = np.float32(0.0)

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
        """Reset action and gait history; callers provide validated state and command."""
        self._previous_action.fill(0)
        self._phase = np.float32(0.0)

    def build_observation(self, state: RobotState, context: ControlContext[PlanarVelocityCommand]) -> FloatArray:
        """Consume runtime-validated state and PolicyController-validated input commands."""
        velocity = context.command.values[0, :]
        phase = np.zeros(4, dtype=np.float32)
        if np.linalg.norm(velocity) < self._standing_threshold:
            self._phase = np.float32(0.0)
        elif context.step > 0:
            self._phase = np.float32((self._phase + context.dt_s * self._gait_frequency_hz) % 1.0)
            phase = np.mod(self._phase + self._feet_phase_offsets, 1.0).astype(np.float32)
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


@dataclass(frozen=True)
class Go2WalkPreparationSettings:
    """Simulation-only standing transition; keep it separate from policy gains."""

    ramp_duration_s: float = 2.0
    settle_duration_s: float = 0.5
    timeout_s: float = 6.0
    max_joint_error_rad: float = 0.2
    max_joint_velocity_rad_s: float = 1.0
    max_base_tilt_rad: float = 0.5
    kp: float = 50.0
    kd: float = 1.0
    damping_duration_s: float = 1.0
    stop_damping_duration_s: float = 10.0
    damping_kd: float = 8.0


class Go2WalkDeployTask(DeployTask[PlanarVelocityCommand]):
    """Go2 walking task, optionally with a simulated measured-state preparation flow."""

    spec_type = Go2WalkTaskSpec

    def __init__(
        self, spec: Go2WalkTaskSpec, robot: RobotSpec, policy: Policy, steps: int | None = None, artifact=None
    ):
        self.spec = spec
        self.robot = robot
        self.policy_io = Go2WalkPolicyProcessor(spec, robot)
        self.policy_controller = PolicyController(self.policy_io, policy, steps=steps)
        self._preparation: Go2WalkPreparationSettings | None = None
        self._stage = "policy"
        self._stage_start_s = 0.0
        self._stop_requested = False
        self._start_position: FloatArray | None = None
        self._ready_since_s: float | None = None
        self._policy_start_s = 0.0
        self._keyboard = None
        self._poll_keyboard = True
        self._automatic_start = True
        self._stand_requested = False
        self._policy_start_requested = False
        self._preparation_ready = False
        self._zeros = np.zeros(robot.joint_count, dtype=np.float32)

    def enable_simulation_preparation(self, settings: Go2WalkPreparationSettings | None = None) -> None:
        """Use measured crouch-to-stand and damping stages in opt-in simulation."""
        self._preparation = settings if settings is not None else Go2WalkPreparationSettings()

    def configure_run(self, options: dict[str, Any], *, simulation: bool) -> None:
        """Keep the crouch preparation choice inside Go2."""
        if not options:
            return
        if options != {"preparation": "lie_down"} or not simulation:
            raise ValueError("Go2 task_options require simulation and preparation: lie_down")
        self.enable_simulation_preparation()

    def attach_operator(self, keyboard: Any, *, polled_by_command: bool) -> str | None:
        if self._preparation is None:
            return None
        self.attach_keyboard(keyboard, poll=not polled_by_command)
        return self.operator_hint

    def attach_keyboard(self, keyboard, *, poll: bool = True) -> None:
        """Require Start and A analogues; a shared velocity binding polls the device first."""
        if keyboard is None:
            raise ValidationError("operator.keyboard", "a viewer keyboard device", None)
        self._keyboard = keyboard
        self._poll_keyboard = poll
        self._automatic_start = False

    def request_stand(self) -> None:
        """Latch the first operator confirmation without commanding from an input thread."""
        self._stand_requested = True

    def request_policy_start(self) -> None:
        """Latch takeover intent; policy still waits for measured standing readiness."""
        self._policy_start_requested = True

    @property
    def operator_hint(self) -> str:
        return "r stand up (Start)  p enable policy (A)  o damp for 10s then close"

    def _process_operator_input(self) -> None:
        if self._keyboard is not None:
            if self._poll_keyboard:
                self._keyboard.poll()
            if self._keyboard.is_key_down("r"):
                self.request_stand()
            if self._stage == "wait_policy" and self._keyboard.is_key_down("p"):
                self.request_policy_start()
            if self._keyboard.is_key_down("o"):
                self.request_stop()
        elif self._automatic_start:
            self.request_stand()
            if self._preparation_ready:
                self.request_policy_start()

    @property
    def phase(self) -> str:
        return self._stage

    @property
    def command_lower(self) -> FloatArray:
        return self.policy_io.command_lower

    @property
    def command_upper(self) -> FloatArray:
        return self.policy_io.command_upper

    def validate_command(self, command: PlanarVelocityCommand | None) -> None:
        self.policy_io.validate_command(command)

    def reset(self, state: RobotState) -> None:
        self._stage = "wait_stand" if self._preparation is not None else "policy"
        self._stage_start_s = 0.0
        self._stop_requested = False
        self._start_position = state.joint_position.copy()
        self._ready_since_s = None
        self._policy_start_s = 0.0
        self._stand_requested = False
        self._policy_start_requested = False
        self._preparation_ready = False
        if self._preparation is None:
            self.policy_controller.reset(state)

    def _servo(self, position: FloatArray, kp: float, kd: float) -> JointServoCommand:
        return JointServoCommand(
            np.clip(position, self.robot.position_lower, self.robot.position_upper).astype(np.float32),
            self._zeros,
            self._zeros,
            np.full(self.robot.joint_count, kp, dtype=np.float32),
            np.full(self.robot.joint_count, kd, dtype=np.float32),
        )

    def step(self, state: RobotState, context: ControlContext[PlanarVelocityCommand]) -> ControllerStep:
        settings = self._preparation
        if settings is None:
            return self.policy_controller.step(state, context)
        self._process_operator_input()
        if self._stop_requested and self._stage not in {"damping", "complete"}:
            self._stage = "damping"
            self._stage_start_s = context.elapsed_time_s
        if self._stage == "wait_stand":
            if not self._stand_requested:
                return ControllerStep(command=self._servo(state.joint_position, 0.0, settings.damping_kd))
            self._stage = "prepare"
            self._stage_start_s = context.elapsed_time_s
            self._start_position = state.joint_position.copy()
        if self._stage == "prepare":
            elapsed = context.elapsed_time_s - self._stage_start_s
            x, y, _, _ = state.base_orientation_xyzw
            tilt = float(np.arccos(np.clip(1 - 2 * (float(x) ** 2 + float(y) ** 2), -1, 1)))
            if tilt > settings.max_base_tilt_rad:
                return ControllerStep(error=f"unsafe base tilt during Go2 preparation: {tilt:.3f} rad")
            ready = (
                elapsed >= settings.ramp_duration_s
                and np.max(np.abs(state.joint_position - self.robot.default_joint_position))
                <= settings.max_joint_error_rad
                and np.max(np.abs(state.joint_velocity)) <= settings.max_joint_velocity_rad_s
            )
            if not ready:
                self._ready_since_s = None
            elif self._ready_since_s is None:
                self._ready_since_s = elapsed
            if self._ready_since_s is not None and elapsed - self._ready_since_s + 1e-9 >= settings.settle_duration_s:
                self._preparation_ready = True
                self._stage = "wait_policy"
            elif elapsed + 1e-9 >= settings.timeout_s:
                return ControllerStep(error="Go2 standing preparation timed out")
            else:
                assert self._start_position is not None
                fraction = np.float32(min((elapsed + context.dt_s) / settings.ramp_duration_s, 1.0))
                target = self._start_position + fraction * (self.robot.default_joint_position - self._start_position)
                return ControllerStep(command=self._servo(target, settings.kp, settings.kd))
        if self._stage == "wait_policy":
            if not self._policy_start_requested:
                return ControllerStep(command=self._servo(self.robot.default_joint_position, settings.kp, settings.kd))
            self.policy_controller.reset(state)
            self._policy_start_s = context.elapsed_time_s
            self._stage = "policy"
        if self._stage == "policy":
            local = ControlContext(
                context.step,
                context.elapsed_time_s - self._policy_start_s,
                context.command,
                context.dt_s,
            )
            decision = self.policy_controller.step(state, local)
            if not decision.complete:
                return decision
            self._stage = "damping"
            self._stage_start_s = context.elapsed_time_s
        if self._stage == "damping":
            damping_duration_s = (
                settings.stop_damping_duration_s if self._stop_requested else settings.damping_duration_s
            )
            if context.elapsed_time_s - self._stage_start_s + 1e-9 >= damping_duration_s:
                self._stage = "complete"
                return ControllerStep(complete=True, success=not self._stop_requested)
            return ControllerStep(
                command=self._servo(state.joint_position, 0.0, settings.damping_kd),
                success=not self._stop_requested,
            )
        return ControllerStep(complete=True, success=not self._stop_requested)

    def request_stop(self) -> None:
        if self._preparation is None:
            self.policy_controller.request_stop()
        else:
            self._stop_requested = True
