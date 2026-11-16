# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

import os

from motrix_robots.franka import FrankaPanda

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config import configclass
from motrix_env_core.config.scene import SceneCfg, SceneObjsCfg
from motrix_env_core.direct.env import DirectEnvCfg

model_file = os.path.dirname(__file__) + "/xmls/scene.xml"


@registry.envcfg("franka-open-cabinet")
@configclass
class FrankaOpenCabinetEnvCfg(DirectEnvCfg):
    """Control a Franka arm to grasp a handle and open a drawer.

    zh_CN: 控制 Franka 机械臂抓住把手并打开抽屉。
    """

    scene: SceneCfg = SceneCfg(file=model_file, objs=SceneObjsCfg(robot=FrankaPanda()))
    max_episode_seconds: float = 7.0
    sim: SimCfg = SimCfg(dt=0.01)
    ctrl_dt: float = 0.01
    render_spacing: float = 2.0
