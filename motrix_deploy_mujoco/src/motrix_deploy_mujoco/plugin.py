# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Typed entry-point factory for the MuJoCo simulation runtime."""

from motrix_deploy.runtime.config import SimulationRuntimeConfig
from motrix_deploy_mujoco.runtime import MujocoRuntime


def create_runtime(config: SimulationRuntimeConfig) -> MujocoRuntime:
    """Create a simulation whose robot contract is derived from its scene."""
    return MujocoRuntime(config)


__all__ = ["create_runtime"]
