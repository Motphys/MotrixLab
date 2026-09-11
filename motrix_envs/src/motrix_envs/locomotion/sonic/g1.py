# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import MjcfFileCfg
from motrix_env_core.manager import ManagerEnv
from motrix_envs.config.scene import StandardSceneCfg, StandardSceneObjsCfg
from motrix_envs.locomotion.sonic import mdp
from motrix_envs.locomotion.sonic.cfg import (
    SonicActionsCfg,
    SonicCommandsCfg,
    SonicManagerEnvCfg,
    SonicObservationsCfg,
    SonicPolicyObsCfg,
)
from motrix_envs.robot import UnitreeG129Dof
from motrix_envs.robot.unitree import UNITREE_G1_ASSET_DIR


def _make_g1_sonic_cfg(
    num_future_frames: int,
    *,
    default_packed_store: str = "data/sonic/lafan1-pack-smoke",
) -> SonicManagerEnvCfg:
    return SonicManagerEnvCfg(
        sim=SimCfg(dt=0.005, solver_iterations=3),
        scene=StandardSceneCfg(
            objs=StandardSceneObjsCfg(
                robot=UnitreeG129Dof(model=MjcfFileCfg(file=UNITREE_G1_ASSET_DIR / "g1_sonic.xml")),
            )
        ),
        commands=SonicCommandsCfg(
            motion=mdp.SonicMotionCommandCfg(
                motion_file=os.path.join(os.environ.get("SONIC_DATA_ROOT", "data/sonic"), "sonic.npz"),
                packed_store=os.environ.get("SONIC_PACKED_STORE", default_packed_store),
                num_future_frames=num_future_frames,
            )
        ),
        actions=SonicActionsCfg(
            joint_position=mdp.SonicJointPositionActionCfg(
                actuator_names=mdp.G1_SONIC_JOINTS,
            )
        ),
        observations=SonicObservationsCfg(
            policy=SonicPolicyObsCfg(
                obs=mdp.SonicActorObservationCfg(history_length=num_future_frames),
            )
        ),
    )


@registry.envcfg("g1-sonic")
def make_g1_sonic_cfg() -> SonicManagerEnvCfg:
    """Track SONIC motion on G1 with the release-capacity temporal profile.

    zh_CN: 使用发布容量的时序配置在 G1 上跟踪 SONIC 动作。
    """
    return _make_g1_sonic_cfg(10)


@registry.envcfg("g1-sonic-lafan")
def make_g1_sonic_lafan_cfg() -> SonicManagerEnvCfg:
    """Track a packed LAFAN motion corpus with the intermediate SONIC profile.

    zh_CN: 使用中等规模 SONIC 配置在 G1 上跟踪打包后的 LAFAN 动作集。
    """
    return _make_g1_sonic_cfg(4)


@registry.envcfg("g1-sonic-smoke")
def make_g1_sonic_smoke_cfg() -> SonicManagerEnvCfg:
    """Exercise the SONIC environment contract with the bundled smoke clip.

    zh_CN: 使用仓库内置的小型动作片段验证 SONIC 环境契约。
    """
    return _make_g1_sonic_cfg(1, default_packed_store="data/sonic/lafan1-pack-smoke")


registry.env("g1-sonic")(ManagerEnv)
registry.env("g1-sonic-lafan")(ManagerEnv)
registry.env("g1-sonic-smoke")(ManagerEnv)
