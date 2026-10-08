# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""G1 WBT task data, policy I/O and measured-state deployment controllers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, ClassVar

import numpy as np
from pydantic import Field

if TYPE_CHECKING:
    from motrix_deploy.artifact.io import Artifact

from motrix_deploy.artifact.schema import TaskSpec
from motrix_deploy.contracts import (
    FloatArray,
    JointServoCommand,
    RobotSpec,
    RobotState,
    float32_array,
)
from motrix_deploy.errors import ValidationError
from motrix_deploy.policy import Policy
from motrix_deploy.policy.processing import PolicyProcessor
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.control import PolicyController
from motrix_deploy.runtime.lifecycle import ControllerStep
from motrix_deploy.task import DeployTask

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0)]
JointVector = Annotated[list[FiniteFloat], Field(min_length=29, max_length=29)]
GainVector = Annotated[list[NonNegativeFloat], Field(min_length=29, max_length=29)]
Quaternion = Annotated[list[FiniteFloat], Field(min_length=4, max_length=4)]


class G1WbtTaskSpec(TaskSpec):
    """One motion clip in canonical robot order, inline or as an artifact NPZ payload."""

    task_name: ClassVar[str] = "g1_wbt/v1"
    period_s: Annotated[FiniteFloat, Field(gt=0)] = 0.02
    action_scale: GainVector
    kp: GainVector
    kd: GainVector
    preparation_joint_position: JointVector | None = None
    preparation_kp: GainVector | None = None
    preparation_kd: GainVector | None = None
    preparation_ramp_duration_s: NonNegativeFloat = 2.0
    preparation_settle_duration_s: NonNegativeFloat = 1.0
    preparation_timeout_s: Annotated[FiniteFloat, Field(gt=0)] = 5.0
    # The motion clip travels as the artifact's NPZ payload (joint_pos/joint_vel/
    # reference_quaternion_xyzw arrays); the task loads and validates it.
    motion_frames: Annotated[int, Field(ge=1)]
    reference_body_name: str = "torso_link"
    termination_ref_orientation_threshold: NonNegativeFloat
    termination_joint_position_threshold: NonNegativeFloat
    termination_joint_velocity_threshold: NonNegativeFloat


@dataclass(frozen=True)
class G1WbtMotion:
    """One validated motion clip in canonical robot order."""

    joint_pos: FloatArray
    joint_vel: FloatArray
    reference_quaternion_xyzw: FloatArray

    def __post_init__(self) -> None:
        joint_pos = np.asarray(self.joint_pos)
        joint_vel = np.asarray(self.joint_vel)
        quaternions = np.asarray(self.reference_quaternion_xyzw)
        if joint_pos.ndim != 2 or joint_pos.shape[0] < 1 or joint_pos.shape[1] < 1:
            raise ValidationError("motion.joint_pos", "a non-empty frame matrix", joint_pos.shape)
        if joint_vel.shape != joint_pos.shape:
            raise ValidationError("motion.joint_vel", f"shape {joint_pos.shape}", joint_vel.shape)
        if quaternions.shape != (joint_pos.shape[0], 4):
            raise ValidationError("motion.reference_quaternion_xyzw", f"({joint_pos.shape[0]}, 4)", quaternions.shape)
        if not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-3, rtol=0):
            raise ValidationError("motion.reference_quaternion_xyzw", "unit xyzw quaternions", None)

    def __len__(self) -> int:
        return int(np.asarray(self.joint_pos).shape[0])


def _load_motion(spec: G1WbtTaskSpec, artifact: Artifact | None) -> G1WbtMotion:
    """Load and validate the motion clip from the artifact's single NPZ payload."""
    if artifact is None:
        raise ValidationError("task.payloads", "the deployment artifact", None)
    if len(artifact.manifest.payloads) != 1:
        raise ValidationError("task.payloads", "exactly one motion payload entry", len(artifact.manifest.payloads))
    with np.load(artifact.root / artifact.manifest.payloads[0].path, allow_pickle=False) as arrays:
        try:
            motion = G1WbtMotion(arrays["joint_pos"], arrays["joint_vel"], arrays["reference_quaternion_xyzw"])
        except KeyError as error:
            raise ValidationError(
                f"task.payloads.{error.args[0]}", "a motion channel in the NPZ payload", None
            ) from error
    if len(motion) != spec.motion_frames:
        raise ValidationError("task.motion_frames", f"{len(motion)} motion frames", spec.motion_frames)
    return motion


class G1WbtPolicyProcessor(PolicyProcessor[None]):
    """Measured-state G1 policy I/O with raw history and servo safety limits.

    The embedded spec is already in canonical robot order (the profile compiler
    performs name-based motion remapping at export), so the actor math below
    indexes motion channels directly. The actor has 154 inputs: reference joint
    positions (29), reference joint velocities (29), relative torso orientation
    (6), body-frame base angular velocity (3), measured joint positions minus
    default (29), measured joint velocities (29), and the last raw policy action
    (29). ``action_scale`` is the *resolved* scale in radians per unit action:
    training WBT uses ``0.25 * effort_limit / kp`` (zero for zero kp), not a
    uniform 0.25-radian scale. No noise, clipping or frame advancement happens
    in the observation; ``previous_action`` history stores the raw action.
    """

    observation_size = 154
    action_size = 29

    def __init__(self, spec: G1WbtTaskSpec, robot: RobotSpec, motion: G1WbtMotion) -> None:
        if robot.joint_count != 29:
            raise ValidationError("robot.joint_count", 29, robot.joint_count)
        if robot.base_link_name != "pelvis" or spec.reference_body_name != "torso_link":
            raise ValidationError("g1_wbt.kinematics", "pelvis base and torso_link reference", spec.reference_body_name)
        self.spec = spec
        self.robot = robot
        self.motion = motion
        # G1 model: aligned pelvis IMU, identity fixed rotations, serial Z-X-Y waist.
        # Inertial-frame quaternions are NOT link-frame mounting transforms.
        try:
            self._waist_indices = np.asarray(
                [robot.joint_names.index(f"waist_{axis}_joint") for axis in ("yaw", "roll", "pitch")]
            )
        except ValueError as error:
            raise ValidationError("robot.joint_names", "G1 yaw/roll/pitch waist joints", robot.joint_names) from error
        self._frames = len(motion)
        self._motion_joint_position = np.asarray(motion.joint_pos, dtype=np.float32)
        self._motion_joint_velocity = np.asarray(motion.joint_vel, dtype=np.float32)
        self._reference_quaternion = np.asarray(motion.reference_quaternion_xyzw, dtype=np.float32)
        self._default_joint_position = robot.default_joint_position.astype(np.float32, copy=True)
        self._action_scale = np.asarray(spec.action_scale, dtype=np.float32)
        self._previous_action = np.zeros(self.action_size, dtype=np.float32)
        self._kp = np.asarray(spec.kp, dtype=np.float32)
        self._kd = np.asarray(spec.kd, dtype=np.float32)
        self._latest_frame = 0
        self._reference_frame: int | None = None

    def reset_reference(self) -> None:
        """Restore the initial reference when starting a new task execution."""
        self._reference_frame = 0
        self._latest_frame = 0

    def set_reference_frame(self, frame: int) -> None:
        """Select a reference independently of the continuously running policy."""
        if not 0 <= frame < self._frames:
            raise ValidationError("reference_frame", "a frame inside the single motion clip", frame)
        self._reference_frame = frame

    def validate_command(self, command: None) -> None:
        if command is not None:
            raise ValidationError("command", "None (motion is artifact-owned)", type(command).__name__)

    def reset(self, state: RobotState, context: ControlContext[None]) -> None:
        self._previous_action.fill(0)
        self._latest_frame = 0

    @property
    def previous_action(self) -> np.ndarray:
        """Last processed raw action in policy order, independently owned."""
        return self._previous_action.copy()

    def actor_observation(
        self,
        frame: int,
        torso_quaternion_xyzw: np.ndarray,
        base_angular_velocity_body: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        previous_action: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the raw actor vector for a caller-selected motion frame.

        Torso and motion quaternions are world-frame xyzw. The relative rotation
        is ``R(torso).T @ R(reference)``; its first TWO ROWS are flattened in
        row-major order (not the commonly used first-two-columns encoding).
        Base angular velocity is already expressed in the floating-base frame,
        not the torso frame. Explicit ``previous_action`` overrides the history
        for this observation only, without mutating it. Otherwise the most recent
        ``process_action`` input is echoed, as training observes the current raw
        action after stepping and thus the previous action at the next inference.
        """
        relative_rotation = _rotation_matrix(torso_quaternion_xyzw).T @ _rotation_matrix(
            self._reference_quaternion[frame]
        )
        raw_history = self._previous_action if previous_action is None else previous_action
        return np.concatenate(
            (
                self._motion_joint_position[frame],
                self._motion_joint_velocity[frame],
                relative_rotation[:2].reshape(-1),
                base_angular_velocity_body,
                np.asarray(joint_positions) - self._default_joint_position,
                joint_velocities,
                raw_history,
            )
        ).astype(np.float32)

    def torso_orientation(self, state: RobotState) -> FloatArray:
        """World XYZW torso attitude from measured pelvis attitude and waist FK.

        This fixed G1-29DOF chain matches the deployed model, not arbitrary robots.
        Hardware must calibrate its IMU into the pelvis frame and map serial joint
        coordinates before constructing RobotState. No simulator body query,
        root position, desired joint position or motion quaternion is used.
        """
        orientation = state.base_orientation_xyzw.astype(np.float64)
        yaw, roll, pitch = state.joint_position[self._waist_indices]
        for angle, axis in zip((yaw, roll, pitch), (2, 0, 1)):
            rotation = np.zeros(4, dtype=np.float64)
            rotation[axis] = np.sin(float(angle) / 2)
            rotation[3] = np.cos(float(angle) / 2)
            v, w = orientation[:3], orientation[3]
            r, s = rotation[:3], rotation[3]
            orientation = np.concatenate((w * r + s * v + np.cross(v, r), [w * s - np.dot(v, r)]))
        return (orientation / np.linalg.norm(orientation)).astype(np.float32)

    def build_observation(self, state: RobotState, context: ControlContext[None]) -> FloatArray:
        frame = context.step if self._reference_frame is None else self._reference_frame
        if not 0 <= frame < self._frames:
            raise ValidationError("context.step", "a frame inside the single motion clip", frame)
        self._latest_frame = frame
        return self.actor_observation(
            frame,
            self.torso_orientation(state),
            state.base_angular_velocity,
            state.joint_position,
            state.joint_velocity,
        )

    def process_action(self, action: FloatArray) -> JointServoCommand:
        raw = float32_array(action, path="action.raw", shape=(29,))
        self._previous_action = raw.copy()  # Boundary arrays are read-only; history must stay writable.
        # Neither the raw action nor the scaled target is clipped before the
        # robot limit clamp below: this matches the training joint-position term.
        target = self._default_joint_position + self._action_scale * raw
        target = np.clip(target, self.robot.position_lower, self.robot.position_upper).astype(np.float32)
        zeros = np.zeros(29, dtype=np.float32)
        return JointServoCommand(target, zeros, zeros, self._kp, self._kd)

    def check_termination(self, state: RobotState) -> str | None:
        frame = self._latest_frame
        torso = self.torso_orientation(state)
        reference = self._reference_quaternion[frame]
        gravity_error = abs(2 * (reference[0] ** 2 + reference[1] ** 2 - torso[0] ** 2 - torso[1] ** 2))
        if gravity_error > self.spec.termination_ref_orientation_threshold:
            return "bad_ref_ori"
        violation = np.maximum(self.robot.position_lower - state.joint_position, 0) + np.maximum(
            state.joint_position - self.robot.position_upper, 0
        )
        if np.max(violation) > self.spec.termination_joint_position_threshold:
            return "bad_dof_pos"
        if np.any(np.abs(state.joint_velocity) > self.spec.termination_joint_velocity_threshold):
            return "bad_dof_vel"
        return None


def _rotation_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion matrix with the same norm correction as training's encoder."""
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    s = 2.0 / (x * x + y * y + z * z + w * w)
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


@dataclass
class StageResult:
    """G1 controller-local readiness or damping diagnostics."""

    name: str
    steps: int = 0
    time_s: float = 0.0
    complete: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class G1PreparationSettings:
    """Measured readiness limits; preparation never injects a reference root state."""

    ramp_duration_s: float = 2.0
    settle_duration_s: float = 1.0
    timeout_s: float = 5.0
    max_joint_error_rad: float = 0.2
    max_joint_velocity_rad_s: float = 1.0
    max_base_tilt_rad: float = 0.5


class G1ReadyController:
    """Ramp from measured joints, then require an uninterrupted ready interval."""

    name = "prepare"

    def __init__(
        self,
        target: FloatArray,
        kp: FloatArray,
        kd: FloatArray,
        robot: RobotSpec,
        settings: G1PreparationSettings,
    ) -> None:
        self._initial_position: FloatArray | None = None
        self._target = np.clip(target, robot.position_lower, robot.position_upper).astype(np.float32)
        self._kp = kp
        self._kd = kd
        self._robot = robot
        self._settings = settings
        self._stop_requested = False
        self._ready_since: float | None = None
        self._reached_ready = False
        self._zeros = np.zeros(robot.joint_count, dtype=np.float32)

    def reset(self, state: RobotState) -> None:
        self._initial_position = state.joint_position.copy()
        self._ready_since = None
        self._reached_ready = False
        self._stop_requested = False

    def hold_command(self) -> JointServoCommand:
        """Keep the independent standing posture while waiting for a start request."""
        return JointServoCommand(self._target, self._zeros, self._zeros, self._kp, self._kd)

    def request_stop(self) -> None:
        self._stop_requested = True

    def step(self, state: RobotState, context: ControlContext[None]) -> ControllerStep:
        if self._stop_requested:
            return ControllerStep(complete=True)
        elapsed_time_s = context.elapsed_time_s
        assert self._initial_position is not None
        settings = self._settings
        x, y, _, _ = state.base_orientation_xyzw
        tilt = float(np.arccos(np.clip(1 - 2 * (float(x) ** 2 + float(y) ** 2), -1, 1)))
        metrics = {
            "max_joint_error_rad": float(np.max(np.abs(state.joint_position - self._target))),
            "max_joint_velocity_rad_s": float(np.max(np.abs(state.joint_velocity))),
            "base_tilt_rad": tilt,
        }
        if tilt > settings.max_base_tilt_rad:
            return ControllerStep(metrics=metrics, error=f"unsafe base tilt during preparation: {metrics}")
        ready = (
            elapsed_time_s + 1e-9 >= settings.ramp_duration_s
            and metrics["max_joint_error_rad"] <= settings.max_joint_error_rad
            and metrics["max_joint_velocity_rad_s"] <= settings.max_joint_velocity_rad_s
        )
        if ready:
            if self._ready_since is None:
                self._ready_since = elapsed_time_s
        else:
            self._ready_since = None
        if (
            self._ready_since is not None
            and elapsed_time_s - self._ready_since + 1e-9 >= settings.settle_duration_s
            and (self._reached_ready or elapsed_time_s <= settings.timeout_s + 1e-9)
        ):
            self._reached_ready = True
            return ControllerStep(complete=True, metrics=metrics)
        if not self._reached_ready and elapsed_time_s + 1e-9 >= settings.timeout_s:
            return ControllerStep(metrics=metrics, error=f"preparation timed out: {metrics}")
        fraction = (
            np.clip((elapsed_time_s + context.dt_s) / settings.ramp_duration_s, 0, 1)
            if settings.ramp_duration_s > 0
            else 1
        )
        target = self._initial_position + np.float32(fraction) * (self._target - self._initial_position)
        target = np.clip(target, self._robot.position_lower, self._robot.position_upper).astype(np.float32)
        return ControllerStep(
            command=JointServoCommand(target, self._zeros, self._zeros, self._kp, self._kd), metrics=metrics
        )


class G1DampingController:
    """Damp measured joint velocity briefly, without holding any reference pose."""

    name = "stop"

    def __init__(self, kd: FloatArray, robot: RobotSpec) -> None:
        self._kd = kd
        self._robot = robot
        self._zeros = np.zeros_like(kd)
        self._stop_requested = False

    def reset(self, state: RobotState) -> None:
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def step(self, state: RobotState, context: ControlContext[None]) -> ControllerStep:
        elapsed_time_s = context.elapsed_time_s
        if self._stop_requested or elapsed_time_s + 1e-9 >= 0.3:
            return ControllerStep(complete=True)
        # The backend validates position targets even with kp=0. Clipping keeps
        # the command valid but cannot add a position-holding torque.
        target = np.clip(state.joint_position, self._robot.position_lower, self._robot.position_upper).astype(
            np.float32
        )
        return ControllerStep(command=JointServoCommand(target, self._zeros, self._zeros, self._zeros, self._kd))


class G1WbtDeployTask(DeployTask[None]):
    """Own standing preparation, policy hold, motion playback and damping stop."""

    spec_type = G1WbtTaskSpec

    def __init__(
        self,
        spec: G1WbtTaskSpec,
        robot: RobotSpec,
        policy: Policy,
        steps: int | None = None,
        *,
        automatic_start: bool = True,
        motion: G1WbtMotion | None = None,
        artifact: Artifact | None = None,
    ):
        if motion is None:
            motion = _load_motion(spec, artifact)
        budget = len(motion) if steps is None else steps
        if not 0 < budget <= len(motion):
            raise ValidationError("steps", "a positive budget inside the single motion clip", budget)
        self.spec = spec
        self.robot = robot
        self.motion = motion
        self.policy_io = G1WbtPolicyProcessor(spec, robot, motion)
        self.policy_controller = PolicyController(self.policy_io, policy)
        self._playback_budget = budget
        self._standing_position = np.asarray(
            robot.default_joint_position
            if spec.preparation_joint_position is None
            else spec.preparation_joint_position,
            dtype=np.float32,
        ).copy()
        self._preparation_kp = np.asarray(
            spec.kp if spec.preparation_kp is None else spec.preparation_kp, dtype=np.float32
        )
        self._preparation_kd = np.asarray(
            spec.kd if spec.preparation_kd is None else spec.preparation_kd, dtype=np.float32
        )
        self._preparation_settings = G1PreparationSettings(
            ramp_duration_s=spec.preparation_ramp_duration_s,
            settle_duration_s=spec.preparation_settle_duration_s,
            timeout_s=spec.preparation_timeout_s,
        )
        self.ready = G1ReadyController(
            self._standing_position, self._preparation_kp, self._preparation_kd, robot, self._preparation_settings
        )
        self.damping = G1DampingController(self.policy_io._kd, robot)
        self.transitions: list[StageResult] = []
        self._phase = "prepare"
        self._record: StageResult | None = None
        self._stop_requested = False
        self._policy_start_requested = False
        self._motion_start_requested = False
        self._preparation_ready = False
        self._playback_steps = 0
        self._stage_origin_s: float | None = None
        self._keyboard = None
        # Default takeover mode: after measured readiness the policy starts by
        # itself and playback follows the task's single mandatory frozen-reference
        # interval. Programmatic applications pass automatic_start=False (manual
        # request methods only) or attach a keyboard for operator staging.
        self._automatic_hold_ticks = 0 if automatic_start else None
        self._held_ticks = 0

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def preparation_ready(self) -> bool:
        return self._preparation_ready

    @property
    def playback_steps(self) -> int:
        return self._playback_steps

    @classmethod
    def default_duration_s(cls, spec: G1WbtTaskSpec) -> float:
        return spec.motion_frames * spec.period_s

    def validate_command(self, command: None) -> None:
        self.policy_io.validate_command(command)

    def _enter_phase(self, phase: str, *, completed: bool = True) -> None:
        if self._record is not None:
            self._record.complete = completed
        self._phase = phase
        self._stage_origin_s = None
        self._record = StageResult(name=phase)
        self.transitions.append(self._record)

    def reset(self, state: RobotState) -> None:
        self._stop_requested = False
        self._policy_start_requested = False
        self._motion_start_requested = False
        self._preparation_ready = False
        self._playback_steps = 0
        self._held_ticks = 0
        self.policy_controller.steps = None
        self.policy_io.reset_reference()
        self.transitions.clear()
        self._record = None
        self._enter_phase("prepare")
        self.ready.reset(state)

    def request_policy_start(self) -> None:
        """Latch operator intent; takeover still requires measured preparation readiness."""
        self._policy_start_requested = True

    def request_motion_start(self) -> None:
        """Latch playback intent without changing control state from the input thread."""
        self._motion_start_requested = True

    def request_stop(self) -> None:
        self._stop_requested = True

    def attach_keyboard(self, keyboard, *, hold_ticks: int = 0) -> None:
        """Let the task translate viewer keyboard press edges into phase requests."""
        if keyboard is None:
            raise ValidationError("operator.keyboard", "an event-framed keyboard device", None)
        self._keyboard = keyboard
        self._automatic_hold_ticks = hold_ticks

    def enable_automatic_start(self, hold_ticks: int = 0) -> None:
        """Let the task script its own phase requests after measured readiness."""
        self._keyboard = None
        self._automatic_hold_ticks = hold_ticks

    @property
    def operator_hint(self) -> str:
        """One-line key reference printed by the CLI when keyboard input is attached."""
        return "p start policy  m start motion  o stop"

    def _process_operator_inputs(self) -> None:
        if self._keyboard is not None:
            self._keyboard.poll()
            # Letters only: MotrixSim's viewer input supports letters, function,
            # special and arrow keys, but not punctuation such as ']'.
            if self._keyboard.is_key_down("p"):
                self._policy_start_requested = True
            if self._keyboard.is_key_down("m"):
                self._motion_start_requested = True
            if self._keyboard.is_key_down("o"):
                self._stop_requested = True
        elif self._automatic_hold_ticks is None:
            return
        elif self._phase == "prepare":
            if self._preparation_ready:
                self._policy_start_requested = True
            return
        elif self._phase != "policy_hold":
            return
        # Once the policy runs, playback follows the configured hold target; an
        # explicit `m` press merely starts it earlier.
        self._held_ticks += 1
        if self._held_ticks >= self._automatic_hold_ticks:
            self._motion_start_requested = True

    def _begin_damping(self, state: RobotState) -> None:
        self._enter_phase("damping", completed=not self._stop_requested)
        self.damping.reset(state)

    def _record_decision(self, decision: ControllerStep, context: ControlContext[None]) -> None:
        assert self._record is not None
        if self._stage_origin_s is None:
            self._stage_origin_s = context.elapsed_time_s
        record = self._record
        record.time_s = context.elapsed_time_s - self._stage_origin_s
        record.metrics = dict(decision.metrics)
        if decision.error:
            record.error = decision.error
        elif decision.command is not None:
            record.steps += 1
            record.time_s += context.dt_s

    def _deterministic_step(self, state: RobotState, context: ControlContext[None]) -> ControllerStep:
        assert self._record is not None
        if self._stage_origin_s is None:
            self._stage_origin_s = context.elapsed_time_s
        local = ControlContext(
            self._record.steps, context.elapsed_time_s - self._stage_origin_s, context.command, context.dt_s
        )
        controller = self.ready if self._phase == "prepare" else self.damping
        decision = controller.step(state, local)
        if self._phase == "prepare":
            self._preparation_ready = decision.complete and decision.error is None
            if decision.complete and not self._policy_start_requested:
                decision = replace(decision, complete=False, command=self.ready.hold_command())
        self._record_decision(decision, context)
        return replace(decision, success=not self._stop_requested)

    def step(self, state: RobotState, context: ControlContext[None]) -> ControllerStep:
        self._process_operator_inputs()
        if self._phase == "complete":
            return ControllerStep(complete=True, success=not self._stop_requested)
        if self._stop_requested and self._phase != "damping":
            self._begin_damping(state)
        if self._phase == "prepare":
            decision = self._deterministic_step(state, context)
            if decision.error is not None or not (self._preparation_ready and self._policy_start_requested):
                return decision
            self.policy_controller.reset(state)
            self._enter_phase("policy_hold")
        if self._phase == "policy_hold":
            # Policy takeover always gets at least one held-reference interval before playback.
            assert self._record is not None
            if self._motion_start_requested and self._record.steps > 0:
                self._enter_phase("playback")
                # Preserve inference history; only the remaining playback budget becomes finite.
                self.policy_controller.steps = self.policy_controller.completed_steps + self._playback_budget
        if self._phase in {"policy_hold", "playback"}:
            frame = self._playback_steps if self._phase == "playback" else 0
            self.policy_io.set_reference_frame(min(frame, len(self.motion) - 1))
            decision = self.policy_controller.step(state, context)
            self._record_decision(decision, context)
            if not decision.complete:
                if self._phase == "playback" and decision.policy_tick:
                    self._playback_steps += 1
                return decision
            self._begin_damping(state)
        decision = self._deterministic_step(state, context)
        if decision.complete:
            assert self._record is not None
            self._record.complete = True
            self._phase = "complete"
        return decision
