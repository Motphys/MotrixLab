# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Robot-intrinsic actuator wiring metadata."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UnitreeMotorBinding:
    """Canonical joint to Unitree motor address and physical mode."""

    index: int
    enabled_mode: int = 0x0A

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise ValueError("motor index must be a non-negative integer")
        if (
            not isinstance(self.enabled_mode, int)
            or isinstance(self.enabled_mode, bool)
            or not 0 <= self.enabled_mode <= 0xFF
        ):
            raise ValueError("enabled motor mode must be an unsigned byte")


@dataclass(frozen=True)
class Actuation:
    """Canonical joint-to-motor mappings."""

    motors: dict[str, UnitreeMotorBinding] = field(default_factory=dict)

    def __post_init__(self) -> None:
        indices = [binding.index for binding in self.motors.values()]
        if len(indices) != len(set(indices)):
            raise ValueError("motor indices must be unique")
        if any(not name for name in self.motors):
            raise ValueError("actuator joint names must not be empty")


__all__ = ["Actuation", "UnitreeMotorBinding"]
