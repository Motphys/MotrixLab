# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Artifact-free Go2 joint diagnostic controller; no SDK or robot handles."""

import numpy as np

from motrix_deploy.contracts import JointServoCommand, RobotSpec, RobotState
from motrix_deploy.errors import ControlFailure
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.lifecycle import ControllerStep
from motrix_deploy_unitree.config.go2 import Go2JointSettings, Go2StartupLimits
from motrix_env_core.input import GamePadDevice


class Go2JointController:
    """Start -> full-body hold pose -> A -> move/hold/return; B requests return.

    All twelve joints are commanded. The hardware backend independently owns
    Select emergency stop and fault damping; this controller never polls SDK state.
    Preparation, hold and test targets stay within model position bounds.
    """

    def __init__(
        self,
        robot: RobotSpec,
        joint_name: str,
        target: float,
        settings: Go2JointSettings,
        *,
        gamepad: GamePadDevice,
        startup_limits: Go2StartupLimits,
    ):
        if not isinstance(gamepad, GamePadDevice):
            raise ValueError("Go2 joint diagnostics require a gamepad with Start/A/B buttons")
        if joint_name not in robot.joint_names:
            raise ValueError(f"unknown joint {joint_name!r}; expected one of {list(robot.joint_names)}")
        index = robot.joint_names.index(joint_name)
        if not np.isfinite(target) or not robot.position_lower[index] <= target <= robot.position_upper[index]:
            raise ValueError(f"target for {joint_name} must be finite and inside its hardware position range")
        startup_limits.validate()
        for name in ("kp", "kd"):
            gain = getattr(settings, name)
            if not np.isfinite(gain) or not 0 <= gain <= getattr(startup_limits, name):
                raise ValueError(f"diagnostic {name} must be finite, non-negative and within startup gain ceiling")
        self._startup_limits = startup_limits
        self._transition_duration_s = settings.transition_duration_s
        self._startup_progress_s = 0.0
        self._startup_last_now = 0.0
        self.robot = robot
        self.settings = settings
        self._reference = robot.default_joint_position.copy()
        self._target = self._reference.copy()
        self._target[index] = np.float32(target)
        self._kp = np.full(robot.joint_count, settings.kp, dtype=np.float32)
        self._kd = np.full(robot.joint_count, settings.kd, dtype=np.float32)
        self._zeros = np.zeros(robot.joint_count, dtype=np.float32)
        self._device = gamepad
        self.phase = "wait_start"
        self._phase_time = 0.0
        self._start = self._reference.copy()
        self._stop_requested = False

    def reset(self, state: RobotState) -> None:
        self.phase = "wait_start"
        self._phase_time = 0.0
        self._start = state.joint_position.copy()
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _servo(self, position: np.ndarray, *, enabled: bool = True) -> ControllerStep:
        return ControllerStep(
            command=JointServoCommand(
                position,
                self._zeros,
                self._zeros,
                self._kp if enabled else self._zeros,
                self._kd if enabled else self._zeros,
            )
        )

    def _ready(self, state: RobotState) -> bool:
        return bool(
            np.max(np.abs(state.joint_position - self._reference)) <= self.settings.max_joint_error_rad
            and np.max(np.abs(state.joint_velocity)) <= self.settings.max_joint_velocity_rad_s
        )

    def _enter(self, phase: str, time_s: float, state: RobotState) -> None:
        lower = self.robot.position_lower
        upper = self.robot.position_upper
        outside = (state.joint_position < lower) | (state.joint_position > upper)
        if phase in {"transition", "return"} and np.any(outside):
            details = "; ".join(
                f"{self.robot.joint_names[i]}={state.joint_position[i]:.6f} allowed=[{lower[i]:.6f},{upper[i]:.6f}] rad"
                for i in np.flatnonzero(outside)
            )
            raise ControlFailure(
                "initial_state_out_of_range",
                f"measured joints outside model position bounds; reposition safely before retrying: {details}",
            )
        self.phase = phase
        self._phase_time = time_s
        self._start = state.joint_position.copy()

    def step(self, state: RobotState, context: ControlContext[None]) -> ControllerStep:
        self._device.poll()
        now = context.elapsed_time_s
        if self._device.is_button_down("B"):
            self.request_stop()
        if self._stop_requested and self.phase not in {"return", "complete"}:
            if self.phase in {"wait_start", "transition"}:
                self.phase = "complete"
            else:
                self._enter("return", now, state)
        if self.phase == "wait_start":
            if not self._device.is_button_down("start"):
                return self._servo(self._reference, enabled=False)
            self._enter("transition", now, state)
            limits = self._startup_limits
            distance = float(np.max(np.abs(self._reference - self._start)))
            self._transition_duration_s = max(
                self.settings.transition_duration_s,
                distance / limits.max_speed_rad_s,
                limits.gain_ramp_s + context.dt_s,
            )
            if self._transition_duration_s + 2 * context.dt_s >= limits.timeout_s:
                raise ControlFailure(
                    "startup_duration_exceeds_timeout", "startup trajectory cannot meet configured timeout"
                )
            self._startup_progress_s = 0.0
            self._startup_last_now = now
            return ControllerStep(
                command=JointServoCommand(self._start.copy(), self._zeros, self._zeros, self._zeros, self._zeros)
            )
        if self.phase == "transition":
            limits = self._startup_limits
            if now - self._phase_time >= limits.timeout_s:
                raise ControlFailure("startup_timeout", "bounded startup timed out")
            elapsed = max(0.0, min(context.dt_s, now - self._startup_last_now))
            self._startup_last_now = now
            self._startup_progress_s += elapsed
            fraction = min(self._startup_progress_s / self._transition_duration_s, 1.0)
            # Ramp from the first measured zero-gain command without catch-up
            # jumps when control ticks are delayed.
            gain_fraction = min(max(self._startup_progress_s - context.dt_s, 0.0) / limits.gain_ramp_s, 1.0)
            position = (self._start + np.float32(fraction) * (self._reference - self._start)).astype(np.float32)
            command = JointServoCommand(
                position, self._zeros, self._zeros, self._kp * gain_fraction, self._kd * gain_fraction
            )
            if fraction >= 1.0:
                self._enter("wait_enable", now + context.dt_s, state)
            return ControllerStep(command=command)
        if self.phase == "wait_enable":
            if now - self._phase_time > self.settings.readiness_timeout_s and not self._ready(state):
                raise ControlFailure("preparation_timeout", "joint diagnostic hold pose did not become ready")
            if not self._device.is_button_down("A"):
                return self._servo(self._reference)
            if not self._ready(state):
                raise ControlFailure("preparation_not_ready", "measured joint error or velocity prevents test motion")
            self._enter("move", now, state)
            self._start = self._reference.copy()
        if self.phase == "move":
            fraction = min((now - self._phase_time + context.dt_s) / self.settings.move_duration_s, 1.0)
            position = (self._reference + np.float32(fraction) * (self._target - self._reference)).astype(np.float32)
            if fraction >= 1:
                self._enter("hold", now + context.dt_s, state)
            return self._servo(position)
        if self.phase == "hold":
            if now - self._phase_time + 1e-9 < self.settings.hold_duration_s:
                return self._servo(self._target)
            self._enter("return", now, state)
            self._start = self._target.copy()
        if self.phase == "return":
            duration = self.settings.return_duration_s
            if duration == 0:
                self.phase = "complete"
            else:
                fraction = min((now - self._phase_time + context.dt_s) / duration, 1.0)
                position = (self._start + np.float32(fraction) * (self._reference - self._start)).astype(np.float32)
                if fraction >= 1:
                    self.phase = "complete"
                return self._servo(position)
        return ControllerStep(complete=True, success=not self._stop_requested)
