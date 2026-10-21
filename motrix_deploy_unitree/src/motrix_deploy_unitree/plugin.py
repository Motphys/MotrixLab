# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Entry-point factory for the Unitree hardware runtime."""

from motrix_deploy.runtime.factory import RuntimeCreateContext
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_deploy_unitree.config.runtime import UnitreeRuntimeConfig
from motrix_deploy_unitree.g1.native_robot import UnitreeG1NativeRobot
from motrix_deploy_unitree.g1.python_robot import UnitreeG1_29DOFRobot
from motrix_deploy_unitree.go2.robot import UnitreeGo2Robot


def create_runtime(config: UnitreeRuntimeConfig, context: RuntimeCreateContext) -> HardwareRuntime:
    """Select a concrete adapter from the robot model and transport implementation."""
    if config.robot == "g1-29dof":
        robot_type = UnitreeG1NativeRobot if config.transport.implementation == "native_cpp" else UnitreeG1_29DOFRobot
        robot = robot_type(
            transport=config.transport,
            control=config.control,
            operator=config.operator,
            control_period_s=context.control.period_s,
            state_timeout_s=context.control.state_timeout_s,
            spec=context.robot,
        )
    elif config.robot == "go2":
        if config.transport.implementation != "python_sdk":
            raise ValueError("Go2 only supports transport.implementation=python_sdk")
        robot = UnitreeGo2Robot(
            transport=config.transport,
            control=config.control,
            control_period_s=context.control.period_s,
            state_timeout_s=context.control.state_timeout_s,
            spec=context.robot,
        )
    else:
        raise ValueError(f"Unsupported Unitree robot: {config.robot!r}")
    return HardwareRuntime(robot, realtime=True)


__all__ = ["create_runtime"]
