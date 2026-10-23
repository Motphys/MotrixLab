# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Task-owned, tick-driven G1 startup shared by Walk and WBT."""

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from motrix_deploy.contracts import JointServoCommand, RobotSpec, RobotState, float32_array
from motrix_deploy.errors import ControlFailure, ValidationError
from motrix_deploy.operator import OperatorIntentEvaluator, OperatorProfileCfg
from motrix_deploy.robot.g1_standing import resolve_g1_standing_gains
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.runtime.lifecycle import ControllerStep
from motrix_env_core.input import GamePadDevice, KeyboardDevice


@dataclass
class G1StartupSettings:
    """Task startup settings, independent of hardware transport.

    Attributes:
        default_transition_duration_s: Measured-to-default interpolation duration in seconds.
        kp: Startup position gains: scalar, canonical vector, named mapping, or None for FixStand defaults.
        kd: Startup velocity gains, accepting the same forms as kp.
        wait_for_remote_buttons: Require separate stand and policy confirmations when an input
            device is present. Headless runs automatically request the same transitions.
        start_button: Gamepad stand confirmation when no operator profile is supplied.
            Viewer keyboards use R for the equivalent request.
        enable_button: Gamepad policy confirmation when no operator profile is supplied.
            Viewer keyboards use P; O cancels startup.
    """

    default_transition_duration_s: float = 3.0
    kp: Any = None
    kd: Any = None
    wait_for_remote_buttons: bool = True
    start_button: str = "start"
    enable_button: str = "A"

    def __post_init__(self) -> None:
        if not math.isfinite(self.default_transition_duration_s) or self.default_transition_duration_s <= 0:
            raise ValueError("preparation.default_transition_duration_s must be positive and finite")
        if not isinstance(self.wait_for_remote_buttons, bool):
            raise ValueError("preparation.wait_for_remote_buttons must be a boolean")
        for name in ("start_button", "enable_button"):
            button = getattr(self, name)
            if not isinstance(button, str) or not button.strip():
                raise ValueError(f"preparation.{name} must be a non-empty button name")
        if self.start_button == self.enable_button:
            raise ValueError("preparation start and enable buttons must be distinct")


class G1StartupController:
    """Generate startup commands without robot handles, sleeps or policy inference."""

    def __init__(
        self,
        robot: RobotSpec,
        settings: G1StartupSettings,
        device: GamePadDevice | KeyboardDevice | None,
        profile: OperatorProfileCfg | None,
        *,
        poll_device: bool = True,
        simulation: bool = False,
    ) -> None:
        self.robot = robot
        self.settings = settings
        self.device = device
        self.poll_device = poll_device
        self._simulation = simulation
        self._evaluator = None if profile is None else OperatorIntentEvaluator(profile.intents)
        self._buttons = set()
        if profile is not None:
            if settings.wait_for_remote_buttons and not {"request_stand", "enable_policy"} <= profile.intents.keys():
                raise ValueError("G1 startup operator profile requires request_stand and enable_policy intents")
            for gesture in profile.intents.values():
                self._buttons.update(gesture.held)
                self._buttons.update(gesture.held_for_s)
                self._buttons.update(button for button in (gesture.pressed, gesture.released) if button is not None)
        defaults = (
            resolve_g1_standing_gains(robot.joint_names) if settings.kp is None or settings.kd is None else (None, None)
        )
        gains = []
        for name, default in zip(("kp", "kd"), defaults, strict=True):
            value = getattr(settings, name)
            if value is None:
                gain = default
            else:
                if isinstance(value, dict):
                    if set(value) != set(robot.joint_names):
                        raise ValueError(f"preparation.{name} must cover exactly the robot joints")
                    value = [value[joint] for joint in robot.joint_names]
                array = np.asarray(value, dtype=np.float32)
                if array.shape == ():
                    array = np.full(robot.joint_count, array.item(), dtype=np.float32)
                gain = float32_array(array, path=f"preparation.{name}", shape=(robot.joint_count,))
                if np.any(gain < 0):
                    raise ValueError(f"preparation.{name} must be non-negative")
            gains.append(gain)
        self._kp, self._kd = gains
        self._zeros = np.zeros(robot.joint_count, dtype=np.float32)
        self._stop_requested = False
        self.phase = "fix_stand" if self._simulation else "wait_stand"
        self._start = robot.default_joint_position.copy()
        self._started_s = 0.0

    def reset(self, state: RobotState) -> None:
        self._stop_requested = False
        self.phase = "fix_stand" if self._simulation else "wait_stand"
        self._start = state.joint_position.copy()
        self._started_s = 0.0
        if self._evaluator is not None:
            self._evaluator.reset()

    def request_stop(self) -> None:
        self._stop_requested = True

    def step(self, state: RobotState, context: ControlContext) -> ControllerStep:
        if self._stop_requested:
            return ControllerStep(complete=True, success=False)
        if self.poll_device and self.device is not None:
            self.device.poll()
        intents = frozenset()
        if self._evaluator is not None and self.device is not None:
            held = (
                self.device.is_pressing if isinstance(self.device, KeyboardDevice) else self.device.is_button_pressing
            )
            pressed = frozenset(button for button in self._buttons if held(button))
            intents = self._evaluator.update(pressed, time.monotonic_ns())
        if "enter_passive" in intents:
            raise ControlFailure("passive_requested", "G1 startup cancelled by operator")
        if "emergency_stop" in intents:
            raise ControlFailure("emergency_stop", "G1 startup emergency stop requested")
        if self.phase == "complete":
            return ControllerStep(complete=True)
        if self.phase == "fix_stand":
            # Simulation starts at its configured standing pose. Apply PD before
            # accepting takeover, without a passive interval or stand-up ramp.
            self.phase = "wait_policy"
            return ControllerStep(
                command=JointServoCommand(
                    self.robot.default_joint_position.copy(), self._zeros, self._zeros, self._kp, self._kd
                )
            )
        automatic = not self.settings.wait_for_remote_buttons or self.device is None
        if isinstance(self.device, KeyboardDevice):
            if self.device.is_key_down("o"):
                raise ControlFailure("passive_requested", "G1 startup cancelled by operator")
            confirmed_stand = automatic or self.device.is_key_down("r")
            confirmed_policy = automatic or self.device.is_key_down("p")
        else:
            confirmed_stand = automatic or (
                "request_stand" in intents
                if self._evaluator is not None
                else self.device.is_button_down(self.settings.start_button)
            )
            confirmed_policy = automatic or (
                "enable_policy" in intents
                if self._evaluator is not None
                else self.device.is_button_down(self.settings.enable_button)
            )
        if self.phase == "wait_stand":
            if not confirmed_stand:
                # Zero torque before operator authorization, independent of the measured pose.
                position = np.clip(state.joint_position, self.robot.position_lower, self.robot.position_upper)
                return ControllerStep(
                    command=JointServoCommand(position, self._zeros, self._zeros, self._zeros, self._zeros)
                )
            if np.any(state.joint_position < self.robot.position_lower) or np.any(
                state.joint_position > self.robot.position_upper
            ):
                raise ValidationError("preparation.initial_pose", "inside robot position bounds", state.joint_position)
            self._start = state.joint_position.copy()
            self._started_s = context.elapsed_time_s
            self.phase = "stand"
        if self.phase == "stand":
            fraction = min(
                (context.elapsed_time_s - self._started_s + context.dt_s) / self.settings.default_transition_duration_s,
                1.0,
            )
            position = (self._start + np.float32(fraction) * (self.robot.default_joint_position - self._start)).astype(
                np.float32
            )
            if fraction >= 1.0:
                self.phase = "wait_policy"
            return ControllerStep(command=JointServoCommand(position, self._zeros, self._zeros, self._kp, self._kd))
        if confirmed_policy:
            self.phase = "complete"
            return ControllerStep(complete=True)
        return ControllerStep(
            command=JointServoCommand(
                self.robot.default_joint_position.copy(), self._zeros, self._zeros, self._kp, self._kd
            )
        )
