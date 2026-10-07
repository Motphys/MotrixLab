# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Public simulation robot configuration types and model-owned assets.

Import model types from this public API or their defining leaf modules.
The core registry discovers builtin model IDs lazily through the installed
robot plugin entry point. Scene placement remains application-owned.
"""

from motrix_robots.anymal import AnymalC
from motrix_robots.booster import BoosterK1
from motrix_robots.dex_evt import DexEvt
from motrix_robots.humanoid import HumanoidRobotCfg
from motrix_robots.microduck import Microduck
from motrix_robots.quadruped import QuadrupedLegCfg, QuadrupedLegsCfg, QuadrupedRobotCfg
from motrix_robots.unitree import UnitreeG129Dof, UnitreeGo1Robot, UnitreeGo2Robot

__all__ = [
    "AnymalC",
    "BoosterK1",
    "DexEvt",
    "HumanoidRobotCfg",
    "Microduck",
    "QuadrupedLegCfg",
    "QuadrupedLegsCfg",
    "QuadrupedRobotCfg",
    "UnitreeG129Dof",
    "UnitreeGo1Robot",
    "UnitreeGo2Robot",
]
