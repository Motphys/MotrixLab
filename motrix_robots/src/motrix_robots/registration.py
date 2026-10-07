# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Builtin model registration callback for ``motrix_env_core.robots`` discovery.

Importing this module only loads model classes. The core registry invokes
:func:`register` lazily when an application first queries robots by ID.
"""

from motrix_env_core import registry
from motrix_robots.anymal import AnymalC
from motrix_robots.booster import BoosterK1
from motrix_robots.dex_evt import DexEvt
from motrix_robots.microduck import Microduck
from motrix_robots.unitree import (
    UnitreeG129Dof,
    UnitreeGo1Robot,
    UnitreeGo2Robot,
)


def register() -> None:
    """Register the seven established builtin simulation robot IDs."""
    registry.register_robot_config("anymal_c", AnymalC)
    registry.register_robot_config("dex-evt", DexEvt)
    registry.register_robot_config("g1-29dof", UnitreeG129Dof)
    registry.register_robot_config("go1", UnitreeGo1Robot)
    registry.register_robot_config("go2", UnitreeGo2Robot)
    registry.register_robot_config("k1", BoosterK1)
    registry.register_robot_config("microduck", Microduck)
