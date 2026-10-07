# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Entry-point factory for the Unitree Go2 deployment runtime."""

from collections.abc import Mapping
from typing import Any

from motrix_deploy.runtime.factory import RuntimeCreateContext
from motrix_deploy.runtime.hardware import HardwareRuntime
from motrix_deploy_unitree.config import UnitreeGo2BackendConfig
from motrix_deploy_unitree.interface import UnitreeGo2RobotInterface


def create_runtime(
    config: Mapping[str, Any],
    context: RuntimeCreateContext,
) -> HardwareRuntime:
    """Construct a realtime runtime around the existing SDK2 hardware facade."""

    robot = UnitreeGo2RobotInterface(
        UnitreeGo2BackendConfig.from_mapping(config),
        control_period_s=context.control.period_s,
        state_timeout_s=context.control.state_timeout_s,
        spec=context.robot,
    )
    return HardwareRuntime(robot=robot, realtime=True)


__all__ = ["create_runtime"]
