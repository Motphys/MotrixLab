# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Entry-point factory for the Unitree hardware runtime."""

from motrix_deploy.runtime.factory import RuntimeCreateContext
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_deploy_unitree.config import UnitreeBackendConfig, UnitreeG1_29DOFRobotCfg, UnitreeGo2RobotCfg
from motrix_deploy_unitree.interface import UnitreeG1_29DOFRobot, UnitreeGo2Robot
from motrix_deploy_unitree.native_interface import UnitreeG1NativeRobot


def create_runtime(
    config: UnitreeBackendConfig,
    context: RuntimeCreateContext,
) -> HardwareRuntime:
    """Construct the shared hardware runtime around the selected Unitree robot."""

    if not isinstance(config, UnitreeBackendConfig):
        raise TypeError(f"Unitree runtime requires UnitreeBackendConfig, got {type(config).__name__}")
    if isinstance(config.robot, UnitreeG1_29DOFRobotCfg):
        robot_type = UnitreeG1NativeRobot if config.transport == "native_cpp" else UnitreeG1_29DOFRobot
    elif isinstance(config.robot, UnitreeGo2RobotCfg):
        if config.transport == "native_cpp":
            raise ValueError("transport=native_cpp currently supports only the G1 29-DOF backend")
        robot_type = UnitreeGo2Robot
    else:
        raise TypeError(f"Unsupported Unitree robot config: {type(config.robot).__name__}")
    robot = robot_type(
        config,
        control_period_s=context.control.period_s,
        state_timeout_s=context.control.state_timeout_s,
        spec=context.robot,
    )
    return HardwareRuntime(robot, realtime=True)


__all__ = ["create_runtime"]
