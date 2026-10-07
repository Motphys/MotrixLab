# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Shared explicit scene fixture for robot control integration tests."""

import numpy as np
import pytest
from scene_helpers import build_scene

from motrix_deploy.runtime.config import SensorBindings, SimulationRuntimeConfig
from motrix_env_core.config.sim import SimCfg


@pytest.fixture
def control_gains() -> tuple[np.ndarray, np.ndarray]:
    """Static test controller configuration, independent of model servo gains."""
    return np.ones(12, dtype=np.float32) * 30.0, np.ones(12, dtype=np.float32) * 0.8


@pytest.fixture
def control_scene_config() -> SimulationRuntimeConfig:
    return SimulationRuntimeConfig(
        scene=build_scene(),
        physics=SimCfg(dt=0.002, solver_iterations=100),
        sensor_bindings=SensorBindings(
            base_angular_velocity="gyro",
            base_linear_acceleration="accelerometer",
            base_linear_velocity="global_linvel",
        ),
    )
