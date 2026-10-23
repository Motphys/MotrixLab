# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Hydra schema registration for the supported Unitree robots."""

from hydra.core.config_store import ConfigStore

from motrix_deploy_unitree.config.control import load_control_config
from motrix_deploy_unitree.config.runtime import UnitreeRuntimeConfig


def install_hardware_runtime_schema() -> None:
    """Install shared runtime and model-specific control recipe schemas."""
    store = ConfigStore.instance()
    store.store(group="runtime", name="_unitree_schema", node=UnitreeRuntimeConfig)
    for robot in ("go2", "g1"):
        store.store(group="unitree_control", name=robot, node=load_control_config(robot))
