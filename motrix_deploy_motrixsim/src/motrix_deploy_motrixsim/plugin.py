# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Installed entry-point factory for native MotrixSim simulation owners."""

from motrix_deploy.runtime.config import SimulationRuntimeConfig
from motrix_deploy_motrixsim.runtime import MotrixSimRuntime


def create_runtime(config: SimulationRuntimeConfig) -> MotrixSimRuntime:
    """Create a simulation host from the shared, assembled scene recipe."""
    return MotrixSimRuntime(config)
