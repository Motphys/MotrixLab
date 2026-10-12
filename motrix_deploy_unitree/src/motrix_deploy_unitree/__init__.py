# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree Go2 DDS hardware backend."""

__version__ = "0.3.0"

from motrix_deploy_unitree.config import (
    GO2_JOINT_NAME_TO_MOTOR_INDEX,
    GO2_MOTOR_COUNT,
    UnitreeBackendConfig,
    UnitreeGo2RobotCfg,
    UnitreeRobotCfg,
)
from motrix_deploy_unitree.direct import UnitreeGo2DirectInterface
from motrix_deploy_unitree.interface import UnitreeGo2Robot, UnitreeSdkBindings
from motrix_deploy_unitree.remote import (
    BUTTON_NAMES,
    UnitreeRemoteGamePadDevice,
    UnitreeRemoteState,
    decode_wireless_remote,
)

__all__ = [
    "BUTTON_NAMES",
    "GO2_JOINT_NAME_TO_MOTOR_INDEX",
    "GO2_MOTOR_COUNT",
    "UnitreeBackendConfig",
    "UnitreeGo2RobotCfg",
    "UnitreeRobotCfg",
    "UnitreeGo2DirectInterface",
    "UnitreeGo2Robot",
    "UnitreeSdkBindings",
    "UnitreeRemoteGamePadDevice",
    "UnitreeRemoteState",
    "decode_wireless_remote",
]
