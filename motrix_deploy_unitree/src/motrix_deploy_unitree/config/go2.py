# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Go2 hardware, DDS runtime and diagnostic configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from motrix_deploy_unitree.config.hardware import Actuation, HardwareRobotCfg, UnitreeMotorBinding
from motrix_env_core.config import configclass
from motrix_env_core.config.scene.base import KeyPoseCfg


@configclass(kw_only=True)
class UnitreeGo2HardwareCfg(HardwareRobotCfg):
    """Unitree Go2 hardware-owned key poses and SDK-neutral bindings."""

    base_link_name: str = "base"
    # Official Go2 URDF limits, in the canonical FL/FR/RL/RR order below:
    # https://github.com/unitreerobotics/unitree_ros/blob/
    # a3b70cae6fd4a82c0e1ece633d5c6f97e88c9d76/robots/go2_description/urdf/go2_description.urdf
    # Model bounds, not a certified bench-test envelope or continuous torque rating.
    position_lower: list[float] = [
        -1.0472,
        -1.5708,
        -2.7227,
        -1.0472,
        -1.5708,
        -2.7227,
        -1.0472,
        -0.5236,
        -2.7227,
        -1.0472,
        -0.5236,
        -2.7227,
    ]
    position_upper: list[float] = [
        1.0472,
        3.4907,
        -0.83776,
        1.0472,
        3.4907,
        -0.83776,
        1.0472,
        4.5379,
        -0.83776,
        1.0472,
        4.5379,
        -0.83776,
    ]
    torque_limit: list[float] = [23.7, 23.7, 45.43] * 4

    key_pose: KeyPoseCfg = KeyPoseCfg(
        joint_names=[
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
        ],
        poses={
            "default": [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5],
            "lie_down": [
                0.05175,
                1.238835,
                -2.74427,
                -0.0608,
                1.24118,
                -2.7375,
                0.31617,
                1.26637,
                -2.79547,
                -0.310495,
                1.266385,
                -2.80177,
            ],
        },
    )
    actuation: Actuation = Actuation(
        motors={
            "FL_hip_joint": UnitreeMotorBinding(index=3),
            "FL_thigh_joint": UnitreeMotorBinding(index=4),
            "FL_calf_joint": UnitreeMotorBinding(index=5),
            "FR_hip_joint": UnitreeMotorBinding(index=0),
            "FR_thigh_joint": UnitreeMotorBinding(index=1),
            "FR_calf_joint": UnitreeMotorBinding(index=2),
            "RL_hip_joint": UnitreeMotorBinding(index=9),
            "RL_thigh_joint": UnitreeMotorBinding(index=10),
            "RL_calf_joint": UnitreeMotorBinding(index=11),
            "RR_hip_joint": UnitreeMotorBinding(index=6),
            "RR_thigh_joint": UnitreeMotorBinding(index=7),
            "RR_calf_joint": UnitreeMotorBinding(index=8),
        }
    )


GO2_MOTOR_COUNT = 12
_DEFAULT_GO2_ROBOT = UnitreeGo2HardwareCfg()
GO2_JOINT_NAME_TO_MOTOR_INDEX: Mapping[str, int] = MappingProxyType(
    {name: binding.index for name, binding in _DEFAULT_GO2_ROBOT.actuation.motors.items()}
)


@dataclass(frozen=True)
class Go2StartupLimits:
    """Diagnostic-controller preparation speed, deadline and gain-ramp settings."""

    max_speed_rad_s: float
    timeout_s: float
    gain_ramp_s: float
    kp: float
    kd: float

    def validate(self) -> None:
        """Validate once when assembling the diagnostic controller."""
        for name in ("max_speed_rad_s", "timeout_s", "gain_ramp_s", "kp", "kd"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"startup.{name} must be finite")
            if name in {"kp", "kd"}:
                if value < 0 or value > np.finfo(np.float32).max:
                    raise ValueError(f"startup.{name} must be non-negative and representable as float32")
            elif value <= 0:
                raise ValueError(f"startup.{name} must be positive")


@dataclass
class Go2JointSettings:
    """Diagnostic servo gains and trajectory durations, in SI units."""

    kp: float
    kd: float
    transition_duration_s: float = 2.0
    move_duration_s: float = 2.0
    hold_duration_s: float = 1.0
    return_duration_s: float = 2.0
    readiness_timeout_s: float = 6.0
    max_joint_error_rad: float = 0.2
    max_joint_velocity_rad_s: float = 1.0
