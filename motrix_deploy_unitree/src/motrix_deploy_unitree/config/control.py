# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared Unitree control schema and model-specific YAML recipes."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from motrix_deploy.errors import ValidationError
from motrix_deploy_unitree.remote import BUTTON_NAMES


@dataclass
class UnitreeControlConfig:
    """Backend publishing and safety settings; preparation belongs to tasks.

    Attributes:
        command_publish_period_s: Period in seconds for repeating the latest
            LowCmd, independent of the upper-level control loop period. Must be
            positive for Go2. None disables the G1 Python periodic writer; the
            G1 native backend instead falls back to 0.001 seconds.
        command_timeout_s: Maximum age in seconds of the latest command submitted
            by the controller before watchdog damping. Repeated publication of
            an old command does not refresh this deadline.
        damping_duration_s: Duration in seconds for publishing damping commands
            during stop/close. Zero removes the additional damping wait period.
        damping_kd: Velocity feedback gain in N m s/rad for damping protection,
            with position feedback gain Kp set to zero.
        emergency_stop_button: Remote button that triggers backend emergency
            protection, distinct from a normal task stop request.
        startup_mode: G1 operator safety mode. "one_shot" treats enter_passive
            as a request to stop the session; "state_machine" leaves passive
            transitions to the persistent controller. Neither mode moves the robot
            during backend enable.
    """

    command_publish_period_s: float | None = 0.002
    command_timeout_s: float = 0.1
    damping_duration_s: float = 1.0
    damping_kd: float = 8.0
    emergency_stop_button: str = "select"
    startup_mode: str = "one_shot"

    def __post_init__(self) -> None:
        if self.startup_mode not in {"one_shot", "state_machine"}:
            raise ValidationError("runtime.control.startup_mode", "one_shot or state_machine", self.startup_mode)
        for name in ("damping_duration_s", "damping_kd"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValidationError(f"runtime.control.{name}", "a finite non-negative number", value)
        if self.command_publish_period_s is not None and (
            not np.isfinite(self.command_publish_period_s) or self.command_publish_period_s <= 0
        ):
            raise ValidationError(
                "runtime.control.command_publish_period_s",
                "a positive finite duration or null",
                self.command_publish_period_s,
            )
        if not np.isfinite(self.command_timeout_s) or self.command_timeout_s <= 0:
            raise ValidationError(
                "runtime.control.command_timeout_s", "a positive finite duration", self.command_timeout_s
            )
        if self.emergency_stop_button not in BUTTON_NAMES:
            raise ValidationError(
                "runtime.control.emergency_stop_button", f"one of {sorted(BUTTON_NAMES)}", self.emergency_stop_button
            )


def load_control_config(robot: str) -> UnitreeControlConfig:
    """Load a typed model recipe, also used by non-Hydra entry points."""
    if robot not in {"go2", "g1"}:
        raise ValueError(f"Unsupported Unitree control recipe: {robot!r}")
    recipe = OmegaConf.load(Path(__file__).parent / robot / "control.yaml")
    return OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(UnitreeControlConfig), recipe))
