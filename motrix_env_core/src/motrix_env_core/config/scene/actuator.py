# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Format-neutral joint actuator declarations for model instances."""

import math

from motrix_env_core.config import configclass
from motrix_env_core.config.scene._utils import Vec2, validate_range


@configclass(kw_only=True)
class ActuatorCfg:
    """Base declaration using undecorated names in the imported model."""

    joint_name: str
    ctrl_range: Vec2 | None = None
    force_range: Vec2 | None = None

    def validate(self) -> None:
        if not self.joint_name:
            raise ValueError("ActuatorCfg.joint_name must not be empty")
        validate_range("ActuatorCfg.ctrl_range", self.ctrl_range)
        validate_range("ActuatorCfg.force_range", self.force_range)


@configclass(kw_only=True)
class MotorActuatorCfg(ActuatorCfg):
    """Direct joint effort motor with unit transmission and gain."""


@configclass(kw_only=True)
class PositionActuatorCfg(ActuatorCfg):
    """Joint position servo with absolute damping coefficient kv."""

    kp: float
    kv: float = 0.0
    inherit_joint_range: bool = False

    def validate(self) -> None:
        super().validate()
        if self.inherit_joint_range and self.ctrl_range is not None:
            raise ValueError("PositionActuatorCfg.inherit_joint_range is mutually exclusive with ctrl_range")
        if not math.isfinite(self.kp) or self.kp <= 0.0:
            raise ValueError(f"PositionActuatorCfg.kp must be positive and finite, got {self.kp}")
        if not math.isfinite(self.kv) or self.kv < 0.0:
            raise ValueError(f"PositionActuatorCfg.kv must be non-negative and finite, got {self.kv}")
