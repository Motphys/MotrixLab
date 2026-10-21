# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared asset-free hardware configuration and motor wiring."""

from dataclasses import dataclass, field

import numpy as np

from motrix_deploy.contracts import RobotSpec
from motrix_env_core.config import configclass
from motrix_env_core.config.scene.base import KeyPoseCfg


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


@configclass(kw_only=True)
class HardwareRobotCfg:
    """SDK-neutral hardware wiring with asset-free canonical key poses.

    Enable sequences, transport and SDK sensor decoding remain backend-owned.
    """

    key_pose: KeyPoseCfg = KeyPoseCfg()
    actuation: Actuation = Actuation()
    base_link_name: str = ""
    position_lower: list[float] = []
    position_upper: list[float] = []
    torque_limit: list[float] = []

    def to_robot_spec(self, hold_pose: str = "default") -> RobotSpec:
        """Build a canonical control contract without loading models or policies.

        Limits bound accepted commands; they do not certify a collision-free pose
        or a safe operating torque on a supported real robot.
        """
        self.validate()
        if hold_pose not in self.key_pose.poses:
            raise ValueError(f"unknown hold pose {hold_pose!r}; expected one of {list(self.key_pose.poses)}")
        return RobotSpec(
            base_link_name=self.base_link_name,
            joint_names=tuple(self.key_pose.joint_names),
            default_joint_position=np.asarray(self.key_pose.poses[hold_pose], dtype=np.float32),
            position_lower=np.asarray(self.position_lower, dtype=np.float32),
            position_upper=np.asarray(self.position_upper, dtype=np.float32),
            torque_limit=np.asarray(self.torque_limit, dtype=np.float32),
        )

    def validate(self) -> None:
        self.key_pose.validate()
        unknown_joints = self.actuation.motors.keys() - set(self.key_pose.joint_names)
        if unknown_joints:
            raise ValueError(f"Actuation references unknown canonical joints: {sorted(unknown_joints)}")
