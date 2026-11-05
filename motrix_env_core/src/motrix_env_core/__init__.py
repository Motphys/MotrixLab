# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Public API of the environment framework.

Importing this package never imports a simulator. Simulator backends live in
their own distributions (``motrix-env-motrixsim``, ``motrix-env-mujoco``) and
register through the ``motrix_env.sim_backends`` entry-point group; the
backend-neutral boundary is :mod:`motrix_env_core.sim` and its registry is
:mod:`motrix_env_core.sim.registry`. The torch-tensor frontend for MotrixSim
lives in :mod:`motrix_env_motrixsim.torch_env`, not here.
"""

from motrix_env_core import registry
from motrix_env_core.base import ABEnv, EnvCfg, ObsSpace, SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg
from motrix_env_core.input import (
    BoundedGamePadPlanarVelocityBinding,
    CommandBinding,
    ConstantPlanarVelocityBinding,
    GamePadDevice,
    GamePadPlanarVelocityBinding,
    InputDevice,
    KeyboardDevice,
    KeyboardPlanarVelocityBinding,
    PlanarVelocityCommand,
)
from motrix_env_core.perf import Perf, PerfNode, perf_root, perf_scope

__all__ = [
    "ABEnv",
    "BoundedGamePadPlanarVelocityBinding",
    "CommandBinding",
    "ConstantPlanarVelocityBinding",
    "EnvCfg",
    "GamePadDevice",
    "GamePadPlanarVelocityBinding",
    "InputDevice",
    "KeyboardDevice",
    "KeyboardPlanarVelocityBinding",
    "ObsSpace",
    "Perf",
    "PerfNode",
    "PlanarVelocityCommand",
    "SceneCfg",
    "SimCfg",
    "configclass",
    "perf_root",
    "perf_scope",
    "registry",
]
