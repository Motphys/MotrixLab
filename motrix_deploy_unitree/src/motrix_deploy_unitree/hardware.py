# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Asset-free hardware configuration and Unitree Go2 defaults."""

from motrix_deploy_unitree.actuation import Actuation, UnitreeMotorBinding
from motrix_deploy_unitree.sensor import HardwareSensorBinding, HardwareSensorBindings
from motrix_env_core.config import configclass
from motrix_env_core.config.scene.base import KeyPoseCfg


@configclass(kw_only=True)
class HardwareRobotCfg:
    """SDK-neutral hardware wiring with asset-free canonical key poses.

    Enable sequences and transport remain backend-owned. Sensor bindings name
    hardware message fields, not model resources or synthesized measurements.
    """

    key_pose: KeyPoseCfg = KeyPoseCfg()
    sensors: HardwareSensorBindings = HardwareSensorBindings()
    actuation: Actuation = Actuation()

    def validate(self) -> None:
        self.key_pose.validate()
        unknown_joints = self.actuation.motors.keys() - set(self.key_pose.joint_names)
        if unknown_joints:
            raise ValueError(f"Actuation references unknown canonical joints: {sorted(unknown_joints)}")


@configclass(kw_only=True)
class UnitreeGo2HardwareCfg(HardwareRobotCfg):
    """Unitree Go2 hardware-owned key poses and SDK-neutral bindings."""

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
    sensors: HardwareSensorBindings = HardwareSensorBindings(
        base_orientation_xyzw=HardwareSensorBinding(field="imu_state.quaternion", quaternion_order="wxyz"),
        base_angular_velocity=HardwareSensorBinding(field="imu_state.gyroscope"),
        base_linear_acceleration=HardwareSensorBinding(field="imu_state.accelerometer"),
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
