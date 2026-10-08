# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Control sessions and policy-driven robot command generation."""

import hashlib
import time
from typing import Generic, TypeVar

from motrix_deploy.contracts import JointServoCommand, RobotState
from motrix_deploy.errors import ControlFailure, EmergencyStopError, LieDownRequestedError, ValidationError
from motrix_deploy.policy import Policy
from motrix_deploy.policy.processing import PolicyProcessor
from motrix_deploy.robot.interface import RobotInterface
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.lifecycle import Controller, ControllerStep
from motrix_deploy.runtime.result import LatencyRecorder, RolloutResult
from motrix_env_core.input import CommandBinding

CommandT = TypeVar("CommandT")


class PolicyController(Generic[CommandT]):
    """Map state and command input to robot commands through an observation-to-action policy.

    Policy I/O preprocessing and action conversion surround pure policy inference; no
    robot handles, I/O or physical timing are owned here.
    """

    def __init__(self, policy_io: PolicyProcessor[CommandT], policy: Policy, steps: int | None = None):
        self.policy_io = policy_io
        self.policy = policy
        self.steps = steps
        self.completed_steps = 0
        self._initial_state: RobotState | None = None
        self._initialized = False
        self._stop_requested = False
        self._time_origin_s: float | None = None

    def _check_termination(self, state: RobotState) -> None:
        try:
            reason = self.policy_io.check_termination(state)
        except Exception as error:
            raise ControlFailure("task_termination_error", str(error)) from error
        if reason is not None:
            raise ControlFailure("task_terminated", reason)

    def reset(self, state: RobotState) -> None:
        self._check_termination(state)
        self.policy.reset()
        self._initial_state = state
        self._initialized = False
        self._stop_requested = False
        self.completed_steps = 0
        self._time_origin_s = None

    def request_stop(self) -> None:
        self._stop_requested = True

    def step(self, state: RobotState, context: ControlContext[CommandT]) -> ControllerStep:
        # Check the measured result of the last interval before finite completion.
        self._check_termination(state)
        if self._stop_requested or (self.steps is not None and self.completed_steps >= self.steps):
            return ControllerStep(
                complete=True,
                success=not self._stop_requested or self.steps is not None and self.completed_steps >= self.steps,
            )
        try:
            self.policy_io.validate_command(context.command)
        except Exception as error:
            raise ControlFailure("input_error", str(error)) from error
        if self._time_origin_s is None:
            self._time_origin_s = context.elapsed_time_s
        local = ControlContext(
            self.completed_steps, context.elapsed_time_s - self._time_origin_s, context.command, context.dt_s
        )
        if not self._initialized:
            assert self._initial_state is not None
            self.policy_io.reset(self._initial_state, local)
            self._initialized = True
        latency_ns = {}
        phase_start = time.monotonic_ns()
        try:
            observation = self.policy_io.build_observation(state, local)
        except Exception as error:
            raise ControlFailure("invalid_observation", str(error)) from error
        latency_ns["observation"] = time.monotonic_ns() - phase_start
        phase_start = time.monotonic_ns()
        try:
            raw_action = self.policy.infer(observation)
        except Exception as error:
            raise ControlFailure("policy_error", str(error)) from error
        latency_ns["inference"] = time.monotonic_ns() - phase_start
        phase_start = time.monotonic_ns()
        try:
            command = self.policy_io.process_action(raw_action)
        except Exception as error:
            raise ControlFailure("invalid_action", str(error)) from error
        latency_ns["action"] = time.monotonic_ns() - phase_start
        trace = self.completed_steps.to_bytes(8, byteorder="little", signed=False)
        for value in (
            state.joint_position,
            state.joint_velocity,
            state.base_orientation_xyzw,
            observation,
            raw_action,
            command.joint_position if isinstance(command, JointServoCommand) else command.torque,
        ):
            trace += value.tobytes(order="C")
        self.completed_steps += 1
        return ControllerStep(command=command, policy_tick=True, trace_bytes=trace, latency_ns=latency_ns)


class ControlSession(Generic[CommandT]):
    """Execute a bound robot controller without interpreting its internal states."""

    def __init__(
        self,
        *,
        robot: RobotInterface,
        controller: Controller[CommandT],
        command_binding: CommandBinding[CommandT] | None = None,
        period_s: float,
        state_timeout_s: float,
    ) -> None:
        if period_s <= 0 or state_timeout_s <= 0:
            raise ValidationError("session.timing", "positive period and timeout", (period_s, state_timeout_s))
        self.robot = robot
        self.command_binding = command_binding
        self.period_s = period_s
        self.state_timeout_s = state_timeout_s
        self.controller = controller
        self._started = False
        self._stopped = False
        self._finished = False
        self._success = False
        self._previous: int | None = None
        self._interval_step = 0
        self.completed_steps = 0
        self.policy_tick = False
        self._trace = hashlib.sha256()
        self._latency = LatencyRecorder()
        self._start_ns: int | None = None
        self._stop_ns: int | None = None
        self._reason = "completed"
        self._message: str | None = None

    @property
    def active(self) -> bool:
        return self._started and not self._stopped and not self._finished and self._reason == "completed"

    def start(self) -> bool:
        if self._started or self._stopped or self._reason != "completed":
            return self.active
        self._start_ns = time.monotonic_ns()
        try:
            self.robot.open()
            rate = self.robot.capabilities.max_command_rate_hz
            if rate is not None and 1 / self.period_s > rate + 1e-9:
                raise ValidationError("session.command_rate_hz", f"<= {rate}", 1 / self.period_s)
            state = self._read_state(None)
            self._previous = state.sample_time_ns
            self.controller.reset(state)
            if self.robot.capabilities.requires_enable:
                try:
                    self.robot.enable()
                except (EmergencyStopError, LieDownRequestedError, ValidationError):
                    raise
                except Exception as error:
                    raise ControlFailure("backend_enable_error", str(error)) from error
                state = self._read_state(self._previous)
                self._previous = state.sample_time_ns
                self.controller.reset(state)
            self._started = True
        except (Exception, KeyboardInterrupt) as error:
            self.fail(error)
        return self.active

    def _read_state(self, previous: int | None) -> "RobotState":
        try:
            state = self.robot.read_state(self.state_timeout_s)
        except TimeoutError as error:
            raise ControlFailure("state_timeout", str(error)) from error
        except (EmergencyStopError, LieDownRequestedError):
            raise
        except Exception as error:
            raise ControlFailure("backend_read_error", str(error)) from error
        try:
            expected = (self.robot.spec.joint_count,)
            if state.joint_position.shape != expected or state.joint_velocity.shape != expected:
                raise ControlFailure("invalid_state", f"joint shape must be {expected}")
            if previous is not None and state.sample_time_ns < previous:
                raise ControlFailure("invalid_state", "state sample timestamp moved backwards")
            age_s = (time.monotonic_ns() - state.receive_time_ns) / 1e9
            if age_s > self.state_timeout_s:
                raise ControlFailure("state_timeout", f"state age {age_s:.6f}s exceeds {self.state_timeout_s:.6f}s")
        except ControlFailure:
            raise
        except Exception as error:
            raise ControlFailure("invalid_state", str(error)) from error
        self._check_health()
        return state

    def _check_health(self) -> None:
        health = self.robot.health()
        if not health.healthy:
            raise ControlFailure("backend_unhealthy", health.reason)

    def tick(self, *, elapsed_time_s: float) -> bool:
        if not self.active:
            return False
        self.policy_tick = False
        loop_start = time.monotonic_ns()
        try:
            phase = time.monotonic_ns()
            state = self._read_state(self._previous)
            self._previous = state.sample_time_ns
            self._latency.add("read", time.monotonic_ns() - phase)
            phase = time.monotonic_ns()
            try:
                command = self.command_binding.read_command(batch_size=1) if self.command_binding is not None else None
            except Exception as error:
                raise ControlFailure("input_error", str(error)) from error
            context = ControlContext(self._interval_step, elapsed_time_s, command, self.period_s)
            self._latency.add("input", time.monotonic_ns() - phase)
            decision = self.controller.step(state, context)
            if decision.error is not None:
                raise ControlFailure("controller_error", decision.error)
            if decision.complete:
                self._finished = True
                self._success = decision.success
                return False
            if decision.command is not None:
                if decision.command.mode not in self.robot.capabilities.control_modes:
                    raise ControlFailure("invalid_action", f"unsupported control mode {decision.command.mode}")
                phase = time.monotonic_ns()
                try:
                    self.robot.write_command(decision.command)
                except (EmergencyStopError, LieDownRequestedError):
                    raise
                except Exception as error:
                    raise ControlFailure("backend_write_error", str(error)) from error
                self._latency.add("write", time.monotonic_ns() - phase)
            self._check_health()
            self.policy_tick = decision.policy_tick
            if self.policy_tick:
                self.completed_steps += 1
            self._trace.update(decision.trace_bytes)
            for name, value in decision.latency_ns.items():
                self._latency.add(name, value)
            self._interval_step += 1
            self._latency.add("loop", time.monotonic_ns() - loop_start)
        except (Exception, KeyboardInterrupt) as error:
            self.fail(error)
            return False
        return True

    def request_stop(self) -> None:
        self.controller.request_stop()

    def fail(self, error: BaseException) -> None:
        if self._reason != "completed":
            return
        if isinstance(error, KeyboardInterrupt):
            self._reason, self._message = "interrupted", "rollout interrupted by user"
        elif isinstance(error, EmergencyStopError):
            self._reason, self._message = "emergency_stop", str(error)
        elif isinstance(error, LieDownRequestedError):
            self._reason, self._message = "lie_down", str(error)
        elif isinstance(error, ControlFailure):
            self._reason, self._message = error.reason, str(error)
        elif isinstance(error, ValidationError):
            self._reason, self._message = "validation_error", str(error)
        else:
            self._reason, self._message = "backend_error", str(error)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for cleanup in (self.robot.stop, self.robot.close):
            try:
                cleanup()
            except (Exception, KeyboardInterrupt) as error:
                if self._reason == "completed":
                    self._reason, self._message = "cleanup_error", str(error)
        self._stop_ns = time.monotonic_ns()

    def result(self, *, overrun_count: int = 0) -> RolloutResult:
        end_ns = self._stop_ns if self._stop_ns is not None else time.monotonic_ns()
        wall_s = (end_ns - self._start_ns) / 1e9 if self._start_ns is not None else 0.0
        return RolloutResult(
            success=self._reason == "completed" and self._finished and self._success,
            exit_reason=self._reason,
            completed_steps=self.completed_steps,
            wall_time_s=wall_s,
            overrun_count=overrun_count,
            trace_sha256=self._trace.hexdigest(),
            error=self._message,
            latency=self._latency.summarize(),
        )
