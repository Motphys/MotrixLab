# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral control executor, independent of scheduling and physics."""

import hashlib
import time
from typing import TYPE_CHECKING, Generic, TypeVar

from motrix_deploy.contracts import JointServoCommand, RobotCommand, RobotState
from motrix_deploy.errors import EmergencyStopError, LieDownRequestedError, ValidationError
from motrix_deploy.policy import PolicyRuntime
from motrix_deploy.robot.interface import RobotInterface
from motrix_deploy.runtime.context import PolicyContext
from motrix_deploy.runtime.result import LatencyRecorder, RolloutResult
from motrix_env_core.input import CommandBinding

if TYPE_CHECKING:
    from motrix_deploy.task import DeployTask

CommandT = TypeVar("CommandT")


class _LoopFailure(RuntimeError):
    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


class ControlSession(Generic[CommandT]):
    """Own one control lifecycle; callers own scheduling and simulation advancement.

    A completed tick means a healthy command write, not a completed physics step.
    Errors are recorded by start/tick/fail; only stop performs backend cleanup.
    Sessions are single-use, and repeated start/stop calls are idempotent.
    Each command selects its mode, which must be supported by the robot backend.
    """

    def __init__(
        self,
        *,
        robot: RobotInterface,
        task: "DeployTask[CommandT]",
        policy: PolicyRuntime,
        command_binding: CommandBinding[CommandT],
        period_s: float,
        state_timeout_s: float,
    ) -> None:
        if period_s <= 0 or state_timeout_s <= 0:
            raise ValidationError("control.timing", "positive period and timeout", (period_s, state_timeout_s))
        self.robot = robot
        self.task = task
        self.policy = policy
        self.command_binding = command_binding
        self.period_s = period_s
        self.state_timeout_s = state_timeout_s
        self._started = False
        self._stopped = False
        self._completed_steps = 0
        self._initial_state: RobotState | None = None
        self._previous_sample_time: int | None = None
        self._recorder = LatencyRecorder()
        self._trace = hashlib.sha256()
        self._start_ns: int | None = None
        self._stop_ns: int | None = None
        self._exit_reason = "completed"
        self._error_message: str | None = None

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    @property
    def active(self) -> bool:
        return self._started and not self._stopped and self._exit_reason == "completed"

    def start(self) -> bool:
        """Open, validate and optionally enable the backend without scheduling."""
        if self._started or self._stopped or self._exit_reason != "completed":
            return self.active
        self._start_ns = time.monotonic_ns()
        try:
            self.robot.open()
            self._validate_capabilities()
            initial_state = self._read_and_validate_state(None)
            self._previous_sample_time = initial_state.sample_time_ns
            self._check_task_termination(initial_state)
            self.policy.reset()
            if self.robot.capabilities.requires_enable:
                initial_context = self._read_context(0, 0.0)
                self.task.reset(initial_state, initial_context)
                try:
                    self.robot.enable()
                except (EmergencyStopError, LieDownRequestedError, ValidationError):
                    raise
                except Exception as error:
                    raise _LoopFailure("backend_enable_error", str(error)) from error
                initial_state = self._read_and_validate_state(self._previous_sample_time)
                self._previous_sample_time = initial_state.sample_time_ns
                self._check_task_termination(initial_state)
                self.task.reset(initial_state, initial_context)
                self.policy.reset()
            self._initial_state = initial_state
            self._started = True
        except (Exception, KeyboardInterrupt) as error:
            self.fail(error)
        return self.active

    def tick(self, *, elapsed_time_s: float) -> bool:
        """Execute one control tick using caller-supplied relative elapsed time."""
        if not self.active:
            return False
        step = self.completed_steps
        loop_start = time.monotonic_ns()
        try:
            phase_start = time.monotonic_ns()
            context = self._read_context(step, elapsed_time_s)
            self._recorder.add("input", time.monotonic_ns() - phase_start)
            if step == 0:
                self.task.reset(self._initial_state, context)
            phase_start = time.monotonic_ns()
            state = self._read_and_validate_state(self._previous_sample_time)
            self._recorder.add("read", time.monotonic_ns() - phase_start)
            self._previous_sample_time = state.sample_time_ns
            self._check_task_termination(state)

            phase_start = time.monotonic_ns()
            try:
                observation = self.task.build_observation(state, context)
            except Exception as error:
                raise _LoopFailure("invalid_observation", str(error)) from error
            self._recorder.add("observation", time.monotonic_ns() - phase_start)

            phase_start = time.monotonic_ns()
            try:
                raw_action = self.policy.infer(observation)
            except Exception as error:
                raise _LoopFailure("policy_error", str(error)) from error
            self._recorder.add("inference", time.monotonic_ns() - phase_start)

            phase_start = time.monotonic_ns()
            try:
                command = self.task.process_action(raw_action)
                self._validate_command_mode(command)
            except Exception as error:
                raise _LoopFailure("invalid_action", str(error)) from error
            self._recorder.add("action", time.monotonic_ns() - phase_start)

            phase_start = time.monotonic_ns()
            try:
                self.robot.write_command(command)
            except (EmergencyStopError, LieDownRequestedError):
                raise
            except Exception as error:
                raise _LoopFailure("backend_write_error", str(error)) from error
            self._recorder.add("write", time.monotonic_ns() - phase_start)
            health = self.robot.health()
            if not health.healthy:
                raise _LoopFailure("backend_unhealthy", health.reason)
            self._trace.update(step.to_bytes(8, byteorder="little", signed=False))
            for value in (
                state.joint_position,
                state.joint_velocity,
                state.base_orientation_xyzw,
                observation,
                raw_action,
                command.joint_position if isinstance(command, JointServoCommand) else command.torque,
            ):
                self._trace.update(value.tobytes(order="C"))
            self._completed_steps += 1
            self._recorder.add("loop", time.monotonic_ns() - loop_start)
        except (Exception, KeyboardInterrupt) as error:
            self.fail(error)
            return False
        return True

    def fail(self, error: BaseException) -> None:
        """Record the first failure, including errors from an external driver."""
        if not isinstance(error, (Exception, KeyboardInterrupt)):
            raise error
        if self._exit_reason != "completed":
            return
        if isinstance(error, KeyboardInterrupt):
            self._exit_reason = "interrupted"
            self._error_message = "rollout interrupted by user"
            return
        if isinstance(error, EmergencyStopError):
            self._exit_reason = "emergency_stop"
        elif isinstance(error, LieDownRequestedError):
            self._exit_reason = "lie_down"
        elif isinstance(error, _LoopFailure):
            self._exit_reason = error.reason
        elif isinstance(error, ValidationError):
            self._exit_reason = "validation_error"
        else:
            self._exit_reason = "backend_error"
        self._error_message = str(error)

    def stop(self) -> None:
        """Stop and close once, preserving any earlier failure over cleanup errors."""
        if self._stopped:
            return
        self._stopped = True
        cleanup_error: BaseException | None = None
        for cleanup in (self.robot.stop, self.robot.close):
            try:
                cleanup()
            except (Exception, KeyboardInterrupt) as error:
                cleanup_error = cleanup_error or error
        if cleanup_error is not None and self._exit_reason == "completed":
            self._exit_reason = "cleanup_error"
            self._error_message = str(cleanup_error)
        self._stop_ns = time.monotonic_ns()

    def result(self, *, steps: int | None, overrun_count: int = 0) -> RolloutResult:
        """Summarize completed control ticks; finite success requires the target count."""
        end_ns = self._stop_ns if self._stop_ns is not None else time.monotonic_ns()
        wall_time_s = (end_ns - self._start_ns) / 1e9 if self._start_ns is not None else 0.0
        simulation_time_s = self.completed_steps * self.period_s
        return RolloutResult(
            success=self._exit_reason == "completed" and steps is not None and self.completed_steps == steps,
            exit_reason=self._exit_reason,
            completed_steps=self.completed_steps,
            simulation_time_s=simulation_time_s,
            wall_time_s=wall_time_s,
            real_time_factor=simulation_time_s / wall_time_s if wall_time_s > 0 else 0.0,
            overrun_count=overrun_count,
            trace_sha256=self._trace.hexdigest(),
            error=self._error_message,
            latency=self._recorder.summarize(),
        )

    def _check_task_termination(self, state: RobotState) -> None:
        try:
            reason = self.task.check_termination(state)
        except Exception as error:
            raise _LoopFailure("task_termination_error", str(error)) from error
        if reason is not None:
            raise _LoopFailure("task_terminated", reason)

    def _read_context(self, step: int, elapsed_time_s: float) -> PolicyContext[CommandT]:
        try:
            command = self.command_binding.read_command(batch_size=1)
            self.task.validate_command(command)
            return PolicyContext(step=step, elapsed_time_s=elapsed_time_s, command=command)
        except Exception as error:
            raise _LoopFailure("input_error", str(error)) from error

    def _validate_command_mode(self, command: RobotCommand) -> None:
        modes = self.robot.capabilities.control_modes
        if command.mode not in modes:
            raise ValidationError("backend.control_modes", command.mode.value, modes)

    def _validate_capabilities(self) -> None:
        capabilities = self.robot.capabilities
        if capabilities.max_command_rate_hz is not None:
            requested_rate = 1.0 / self.period_s
            if requested_rate > capabilities.max_command_rate_hz + 1e-9:
                raise ValidationError(
                    "control.command_rate_hz", f"<= {capabilities.max_command_rate_hz}", requested_rate
                )

    def _read_and_validate_state(self, previous_sample_time: int | None) -> RobotState:
        try:
            state = self.robot.read_state(self.state_timeout_s)
        except TimeoutError as error:
            raise _LoopFailure("state_timeout", str(error)) from error
        except (EmergencyStopError, LieDownRequestedError):
            raise
        except Exception as error:
            raise _LoopFailure("backend_read_error", str(error)) from error
        try:
            self._validate_state(state, previous_sample_time)
        except _LoopFailure:
            raise
        except Exception as error:
            raise _LoopFailure("invalid_state", str(error)) from error
        health = self.robot.health()
        if not health.healthy:
            raise _LoopFailure("backend_unhealthy", health.reason)
        return state

    def _validate_state(self, state: RobotState, previous_sample_time: int | None) -> None:
        expected = (self.robot.spec.joint_count,)
        if state.joint_position.shape != expected or state.joint_velocity.shape != expected:
            raise _LoopFailure("invalid_state", f"joint shape must be {expected}")
        if previous_sample_time is not None and state.sample_time_ns < previous_sample_time:
            raise _LoopFailure(
                "invalid_state",
                f"state sample timestamp moved backwards: {state.sample_time_ns} < {previous_sample_time}",
            )
        age_s = (time.monotonic_ns() - state.receive_time_ns) / 1e9
        if age_s > self.state_timeout_s:
            raise _LoopFailure("state_timeout", f"state age {age_s:.6f}s exceeds {self.state_timeout_s:.6f}s")
