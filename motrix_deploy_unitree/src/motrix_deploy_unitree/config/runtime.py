# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared Unitree runtime assembly and DDS transport configuration."""

from dataclasses import dataclass, field

import numpy as np
from omegaconf import MISSING as OC_MISSING

from motrix_deploy.config import HardwareRuntimeCfg
from motrix_deploy.errors import ValidationError
from motrix_deploy.operator import OperatorProfileCfg
from motrix_deploy_unitree.config.control import UnitreeControlConfig
from motrix_deploy_unitree.remote import ACCEPTED_BUTTON_NAMES

UNITREE_LOWCMD_TOPIC = "rt/lowcmd"
UNITREE_LOWSTATE_TOPIC = "rt/lowstate"
UNITREE_SUBSCRIBER_QUEUE_DEPTH = 10


@dataclass
class UnitreeTransportConfig:
    """SDK implementation selection and DDS transport settings."""

    implementation: str = "python_sdk"
    network_interface: str = OC_MISSING
    domain_id: int = 0
    connect_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.implementation not in {"python_sdk", "native_cpp"}:
            raise ValidationError("runtime.transport.implementation", "python_sdk or native_cpp", self.implementation)
        if not isinstance(self.network_interface, str) or not self.network_interface:
            raise ValidationError(
                "runtime.transport.network_interface", "a non-empty interface name", self.network_interface
            )
        if (
            not isinstance(self.connect_timeout_s, (int, float))
            or isinstance(self.connect_timeout_s, bool)
            or not np.isfinite(self.connect_timeout_s)
            or self.connect_timeout_s <= 0
        ):
            raise ValidationError("runtime.transport.connect_timeout_s", "a positive duration", self.connect_timeout_s)
        if not isinstance(self.domain_id, int) or isinstance(self.domain_id, bool) or self.domain_id < 0:
            raise ValidationError("runtime.transport.domain_id", "a non-negative integer", self.domain_id)


@dataclass
class UnitreeRuntimeConfig(HardwareRuntimeCfg):
    """Select a supported robot and configure its transport and control."""

    backend: str = "unitree"
    robot: str = OC_MISSING
    transport: UnitreeTransportConfig = field(default_factory=UnitreeTransportConfig)
    control: UnitreeControlConfig = OC_MISSING
    operator: OperatorProfileCfg | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.control, UnitreeControlConfig):
            raise ValidationError("runtime.control", "UnitreeControlConfig", type(self.control).__name__)
        if self.robot == "go2":
            if self.control.command_publish_period_s is None:
                raise ValidationError("runtime.control.command_publish_period_s", "a positive duration for Go2", None)
            if self.transport.implementation != "python_sdk":
                raise ValidationError(
                    "runtime.transport.implementation", "python_sdk for Go2", self.transport.implementation
                )
        elif self.robot != "g1-29dof":
            raise ValidationError("runtime.robot", "go2 or g1-29dof", self.robot)
        if self.operator is not None:
            if not isinstance(self.operator, OperatorProfileCfg):
                raise ValidationError("runtime.operator", "an OperatorProfileCfg", type(self.operator).__name__)
            self._validate_operator_profile()

    def _validate_operator_profile(self) -> None:
        assert self.operator is not None
        for intent, gesture in self.operator.intents.items():
            buttons = [*gesture.held, *gesture.held_for_s]
            buttons.extend(name for name in (gesture.pressed, gesture.released) if name is not None)
            unknown = sorted(set(buttons) - set(ACCEPTED_BUTTON_NAMES))
            if unknown:
                raise ValidationError(
                    f"runtime.operator.intents.{intent}",
                    f"buttons in {sorted(ACCEPTED_BUTTON_NAMES)}",
                    unknown,
                )
        velocity = self.operator.planar_velocity
        if velocity is not None:
            axes = {velocity.linear_x_axis, velocity.linear_y_axis, velocity.yaw_axis}
            if not axes <= {"lx", "ly", "rx", "ry"}:
                raise ValidationError("runtime.operator.planar_velocity.axes", "Unitree joystick axes", sorted(axes))
            if velocity.deadman_button is not None and velocity.deadman_button not in ACCEPTED_BUTTON_NAMES:
                raise ValidationError(
                    "runtime.operator.planar_velocity.deadman_button",
                    f"one of {sorted(ACCEPTED_BUTTON_NAMES)}",
                    velocity.deadman_button,
                )
        if self.robot == "g1-29dof":
            required = (
                {"request_stand", "enable_policy", "enter_passive", "emergency_stop"}
                if self.control.startup_mode == "one_shot"
                else {"request_fixstand", "request_walk", "request_wbt", "request_passive", "request_emergency"}
            )
            missing = sorted(required - set(self.operator.intents))
            if missing:
                raise ValidationError("runtime.operator.intents", f"G1 safety intents {sorted(required)}", missing)
