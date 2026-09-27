# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""G1 large-box carrying tracking task."""

from motrix_env_core import registry
from motrix_env_core.base import EnvCfg
from motrix_env_core.manager import ManagerEnv

from .common import MOTION_DIR, G1WbtEnvCfg


@registry.envcfg("g1-29dof-wbt-largebox")
def make_g129dof_wbt_largebox_cfg() -> EnvCfg:
    """Track a large-box carrying reference motion with Unitree G1.

    zh_CN: 让 Unitree G1 跟踪搬运大箱子的参考动作。
    """

    return G1WbtEnvCfg(motion_file=str(MOTION_DIR / "sub3_largebox_003.npz"))


registry.env("g1-29dof-wbt-largebox")(ManagerEnv)
