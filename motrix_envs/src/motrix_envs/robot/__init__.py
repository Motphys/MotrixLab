# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from motrix_env_core import registry
from motrix_envs.robot.anymal import AnymalC
from motrix_envs.robot.booster import BoosterK1
from motrix_envs.robot.dex_evt import DexEvt
from motrix_envs.robot.humanoid import HumanoidRobotCfg
from motrix_envs.robot.microduck import Microduck
from motrix_envs.robot.quadruped import QuadrupedLegCfg, QuadrupedLegsCfg, QuadrupedRobotCfg
from motrix_envs.robot.unitree import (
    UnitreeG129Dof,
    UnitreeGo1Robot,
    UnitreeGo2Robot,
)

registry.robotcfg("anymal_c")(AnymalC)
registry.robotcfg("dex-evt")(DexEvt)
registry.robotcfg("g1-29dof")(UnitreeG129Dof)
registry.robotcfg("go1")(UnitreeGo1Robot)
registry.robotcfg("go2")(UnitreeGo2Robot)
registry.robotcfg("k1")(BoosterK1)
registry.robotcfg("microduck")(Microduck)

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
