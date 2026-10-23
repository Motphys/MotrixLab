# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree SDK2 DDS hardware backends."""

from importlib import metadata

__version__ = metadata.version("motrix-deploy-unitree")

from motrix_deploy_unitree.config.g1 import (
    G1_29DOF_JOINT_NAME_TO_MOTOR_ID,
    G1_29DOF_MOTOR_COUNT,
    G1_HG_MOTOR_SLOT_COUNT,
)
from motrix_deploy_unitree.config.go2 import (
    GO2_JOINT_NAME_TO_MOTOR_INDEX,
    GO2_MOTOR_COUNT,
)
from motrix_deploy_unitree.config.runtime import UnitreeRuntimeConfig, UnitreeTransportConfig
from motrix_deploy_unitree.remote import (
    BUTTON_NAMES,
    UnitreeRemoteGamePadDevice,
    UnitreeRemoteState,
    decode_wireless_remote,
)

__all__ = [
    "BUTTON_NAMES",
    "G1_29DOF_JOINT_NAME_TO_MOTOR_ID",
    "G1_29DOF_MOTOR_COUNT",
    "G1_HG_MOTOR_SLOT_COUNT",
    "GO2_JOINT_NAME_TO_MOTOR_INDEX",
    "GO2_MOTOR_COUNT",
    "UnitreeRuntimeConfig",
    "UnitreeTransportConfig",
    "UnitreeRemoteGamePadDevice",
    "UnitreeRemoteState",
    "decode_wireless_remote",
]
