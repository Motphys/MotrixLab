# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from motrix_env_core.config import configclass
from motrix_env_core.config.scene import MjcfFileCfg, RobotCfg

PANDA_ASSET_DIR = Path(__file__).parent / "assets" / "franka_panda"


@configclass(kw_only=True)
class FrankaPanda(RobotCfg):
    """Fixed-base Franka Panda with the stock MJX mesh set."""

    model: MjcfFileCfg = MjcfFileCfg(file=PANDA_ASSET_DIR / "panda.xml")
    base_link_name: str = "link0"
