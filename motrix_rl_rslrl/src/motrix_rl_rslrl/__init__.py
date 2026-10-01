# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""RSLRL integration module for MotrixLab.

This module provides configuration classes and utilities for using RSLRL
(ETH Zurich's RL library) with MotrixLab.

The configuration structure matches rsl_rl's flat format with separate
actor and critic configs at the top level.
"""

from motrix_rl_rslrl.cfg import (
    RslRlActorCfg,
    RslrlCfg,
    RslRlCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslrlRunnerCfg,
)

__all__ = [
    "RslRlActorCfg",
    "RslrlCfg",
    "RslRlCriticCfg",
    "RslRlPpoAlgorithmCfg",
    "RslrlRunnerCfg",
]
