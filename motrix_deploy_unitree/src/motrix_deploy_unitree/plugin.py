# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Entry-point factory for the Unitree hardware runtime."""

from motrix_deploy.runtime.factory import RuntimeCreateContext
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_deploy_unitree.config import UnitreeBackendConfig, UnitreeGo2RobotCfg
from motrix_deploy_unitree.interface import UnitreeGo2Robot


def create_runtime(
    config: UnitreeBackendConfig,
    context: RuntimeCreateContext,
) -> HardwareRuntime:
    """Construct the shared hardware runtime around the selected Unitree robot."""

    if not isinstance(config, UnitreeBackendConfig):
        raise TypeError(f"Unitree runtime requires UnitreeBackendConfig, got {type(config).__name__}")
    if not isinstance(config.robot, UnitreeGo2RobotCfg):
        raise TypeError(f"Unsupported Unitree robot config: {type(config.robot).__name__}")
    robot = UnitreeGo2Robot(
        config,
        control_period_s=context.control.period_s,
        state_timeout_s=context.control.state_timeout_s,
        spec=context.robot,
    )
    return HardwareRuntime(robot, realtime=True)


__all__ = ["create_runtime"]
