# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""G1 hardware, servo and runtime configuration."""

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from motrix_deploy.errors import ValidationError
from motrix_deploy_unitree.config.hardware import Actuation, HardwareRobotCfg, UnitreeMotorBinding
from motrix_env_core.config import configclass
from motrix_env_core.config.scene.base import KeyPoseCfg

G1_29DOF_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_G1_29DOF_DEFAULT_POSITION = (
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
)


@configclass(kw_only=True)
class UnitreeG129DOFHardwareCfg(HardwareRobotCfg):
    """G1 29-DOF wiring for the HG DDS protocol.

    Joint names are the stable semantic contract. Motor ids deliberately live
    here rather than in a policy-order permutation so exported artifacts may
    use any joint ordering without changing the physical wiring.
    """

    motor_slot_count: int = 35
    motor_mode: int | None = 1
    mode_pr: int | None = 0
    expected_mode_machine: int | None = 5

    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=list(G1_29DOF_JOINT_NAMES),
        poses={
            "default": list(_G1_29DOF_DEFAULT_POSITION),
            # G1 shutdown is damping-only. This pose only satisfies the shared
            # pose schema and is never selected by the G1 adapter.
            "lie_down": list(_G1_29DOF_DEFAULT_POSITION),
        },
    )
    actuation: Actuation = Actuation(
        motors={
            name: UnitreeMotorBinding(index=motor_id, enabled_mode=1)
            for motor_id, name in enumerate(G1_29DOF_JOINT_NAMES)
        }
    )

    def validate(self) -> None:
        super().validate()
        if (
            not isinstance(self.motor_slot_count, int)
            or isinstance(self.motor_slot_count, bool)
            or self.motor_slot_count <= 0
        ):
            raise ValidationError("hardware.motor_slot_count", "a positive integer", self.motor_slot_count)
        for name in ("motor_mode", "mode_pr", "expected_mode_machine"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFF):
                raise ValidationError(f"hardware.{name}", "an integer in [0, 255]", value)

    def motor_indices(self, joint_names: tuple[str, ...]) -> np.ndarray:
        mapping = {name: binding.index for name, binding in self.actuation.motors.items()}
        expected = set(joint_names)
        actual = set(mapping)
        if actual != expected:
            raise ValidationError(
                "hardware.actuation.motors",
                f"exactly joints {sorted(expected)}",
                f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}",
            )
        values = tuple(mapping[name] for name in joint_names)
        if any(not isinstance(index, int) or isinstance(index, bool) for index in values):
            raise ValidationError("hardware.motor_indices", "integer motor indices", values)
        if len(set(values)) != len(values) or any(not 0 <= index < self.motor_slot_count for index in values):
            raise ValidationError(
                "hardware.motor_indices", f"unique values in [0, {self.motor_slot_count - 1}]", values
            )
        return np.asarray(values, dtype=np.int64)


G1_29DOF_MOTOR_COUNT = 29
G1_HG_MOTOR_SLOT_COUNT = 35
G1_29DOF_JOINT_NAME_TO_MOTOR_ID: Mapping[str, int] = MappingProxyType(
    {name: binding.index for name, binding in UnitreeG129DOFHardwareCfg().actuation.motors.items()}
)
