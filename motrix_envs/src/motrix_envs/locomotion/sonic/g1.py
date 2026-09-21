# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

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


def _sonic_num_future_frames() -> int:
    """Temporal depth from the single-source SONIC PolicyVariant config.

    The RL task recipe owns ``algo.variant.model``, so the environment
    observation layout and the actor input width always follow one edit. Paths
    are resolved from the working directory like the packed motion stores.
    """
    path = Path("configs/task/g1-sonic/motrix.fastsac.yaml")
    try:
        value = OmegaConf.load(path).algo.variant.model.num_future_frames
    except (OSError, OmegaConfBaseException) as exc:
        raise RuntimeError(
            f"Cannot read SONIC task config {path}: {exc}. Run from the repository root so configs/task is reachable."
        ) from exc
    num = int(value)
    if num < 1:
        raise ValueError(f"{path}: algo.variant.model.num_future_frames must be >= 1, got {num}")
    return num


def _make_g1_sonic_cfg() -> SonicManagerEnvCfg:
    num_future_frames = _sonic_num_future_frames()
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
                packed_store=os.environ.get("SONIC_PACKED_STORE", "data/sonic/lafan1-pack-smoke"),
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
    """Track SONIC motion on G1 with a 10-future-frame PolicyVariant.

    zh_CN: 使用 10 个未来帧的 SONIC PolicyVariant 在 G1 上跟踪动作。
    """
    return _make_g1_sonic_cfg()


registry.env("g1-sonic")(ManagerEnv)
