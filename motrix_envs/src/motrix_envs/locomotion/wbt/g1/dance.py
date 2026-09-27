# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""G1 dance tracking task."""

from motrix_env_core import registry
from motrix_env_core.base import EnvCfg
from motrix_env_core.manager import ManagerEnv

from .common import MOTION_DIR, G1WbtEnvCfg


@registry.envcfg("g1-wbt-dance")
def make_g129dof_wbt_dance_cfg() -> EnvCfg:
    """Track the bundled G1 dance motion with the manager-based environment.

    zh_CN: 让 Unitree G1 跟踪内置舞蹈参考动作。
    """

    return G1WbtEnvCfg(motion_file=str(MOTION_DIR / "dance1_subject2.npz"))


registry.env("g1-wbt-dance")(ManagerEnv)
