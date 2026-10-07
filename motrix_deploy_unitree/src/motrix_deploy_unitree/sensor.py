# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""SDK-neutral hardware sensor field bindings."""

from dataclasses import dataclass

from motrix_env_core.config import configclass


@dataclass(frozen=True)
class HardwareSensorBinding:
    """A declarative message field understood by a concrete hardware backend."""

    field: str
    quaternion_order: str | None = None

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("hardware sensor field must not be empty")
        if self.quaternion_order not in (None, "wxyz", "xyzw"):
            raise ValueError(f"unsupported quaternion order: {self.quaternion_order!r}")


@configclass
class HardwareSensorBindings:
    """Optional hardware message bindings for canonical base measurements."""

    base_orientation_xyzw: HardwareSensorBinding | None = None
    base_angular_velocity: HardwareSensorBinding | None = None
    base_linear_acceleration: HardwareSensorBinding | None = None


__all__ = ["HardwareSensorBinding", "HardwareSensorBindings"]
