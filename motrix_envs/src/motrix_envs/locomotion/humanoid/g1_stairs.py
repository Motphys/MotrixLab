# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""Unitree G1 pyramid-stairs velocity-tracking task (mjlab stairs parity).

Reproduces the mjlab ``STAIRS_TERRAINS_CFG`` training in the MotrixLab
FastSAC stack: a flat/easy/moderate/challenging pyramid-stairs tile grid,
a yaw-aligned height-scan observation sampled from the static height field,
heading-random spawns bound to fixed tiles, and the shared walk reward and
termination scaffold.
"""

import math
from dataclasses import replace

import numpy as np

from motrix_env_core import registry
from motrix_env_core.base import SimCfg
from motrix_env_core.config.scene import (
    FlatTerrainGeneratorCfg,
    HFieldTerrainCfg,
    ProceduralHFieldAssetCfg,
    StairsTerrainGeneratorCfg,
    SystemCameraCfg,
    grid_terrain,
)
from motrix_env_core.manager import ManagerEnv
from motrix_env_core.mdp.terminations import BadDofVelocityTerminationCfg, CollidingTerminationCfg
from motrix_envs.config.scene import StandardSceneObjsCfg
from motrix_envs.locomotion.humanoid import cfg as humanoid_cfg
from motrix_envs.locomotion.humanoid.cfg import (
    HumanoidVelocityTrackingManagerEnvCfg,
    WalkCommandsCfg,
    WalkObservationsCfg,
    WalkTerminationsCfg,
)
from motrix_envs.locomotion.humanoid.g1 import (
    _G1_TERMINATION_GEOMS,
    _make_g1_rewards,
)
from motrix_envs.locomotion.humanoid.walk_manager_mdp.command import WalkCommandCfg
from motrix_envs.robot import UnitreeG129Dof

# mjlab STAIRS_TERRAINS_CFG proportions and step parameters.
_STAIRS_TIERS = (
    # (name, proportion, step_height range, step width, platform width)
    ("easy", 0.35, (0.02, 0.05), 0.40, 3.0),
    ("moderate", 0.25, (0.05, 0.08), 0.35, 2.5),
    ("challenging", 0.15, (0.08, 0.10), 0.30, 2.0),
)
_FLAT_PROPORTION = 0.25
_ROWS, _COLS = 10, 8
_TILE = 8.0
_BORDER = 20.0


def _make_stairs_terrain() -> ProceduralHFieldAssetCfg:
    """Build the mjlab-style flat/easy/moderate/challenging stairs mixture."""
    height_scale = 0.8
    rim_level = 0.5  # Flat tiles and stair rims share this normalized level.
    rng = np.random.default_rng(8)
    bounds = []
    cumulative = _FLAT_PROPORTION
    for _name, proportion, h_range, width, platform in _STAIRS_TIERS:
        cumulative += proportion
        bounds.append((cumulative, h_range, width, platform))
    cells = []
    for _ in range(_ROWS):
        row = []
        for _ in range(_COLS):
            pick = rng.random()
            if pick < _FLAT_PROPORTION:
                row.append(FlatTerrainGeneratorCfg(height=rim_level, height_scale=height_scale))
                continue
            for bound, h_range, width, platform in bounds:
                if pick < bound:
                    row.append(
                        StairsTerrainGeneratorCfg(
                            axis="radial",
                            profile="pyramid",
                            step_width=width,
                            step_height=float(rng.uniform(*h_range)),
                            platform_width=platform,
                            base_level=rim_level,
                            height_scale=height_scale,
                        )
                    )
                    break
        cells.append(row)
    size = (_COLS * _TILE + 2.0 * _BORDER, _ROWS * _TILE + 2.0 * _BORDER)
    shape = (int(size[1] / 0.1), int(size[0] / 0.1))
    return ProceduralHFieldAssetCfg(
        generator=grid_terrain(cells, size=size, border=(_BORDER, _BORDER)),
        size=size,
        shape=shape,
    )


@registry.envcfg("g1-walk-stairs")
def make_g129dof_walk_stairs_cfg() -> HumanoidVelocityTrackingManagerEnvCfg:
    """Track G1 walking commands over a pyramid-stairs tile grid."""
    robot = UnitreeG129Dof()
    return HumanoidVelocityTrackingManagerEnvCfg(
        scene=humanoid_cfg.HumanoidWalkSceneCfg(
            system_camera=SystemCameraCfg(distance=6.0, elevation=-20.0, azimuth=180.0),
            assets=humanoid_cfg.TerrainSceneAssetsCfg(terrain=_make_stairs_terrain()),
            objs=StandardSceneObjsCfg(
                floor=HFieldTerrainCfg(hfield="terrain", material="mat_ground"),
                robot=robot,
            ),
        ),
        rewards=_make_g1_rewards(robot),
        terminations=WalkTerminationsCfg(
            colliding=CollidingTerminationCfg(
                termination_geoms=_G1_TERMINATION_GEOMS,
                ground_geom="floor",
            ),
            bad_dof_velocity=BadDofVelocityTerminationCfg(threshold=100.0),
        ),
        # mjlab-style command space: yaw-rate range matches the velocity task
        # defaults and heading-random spawns face arbitrary stair directions.
        commands=WalkCommandsCfg(
            walk=WalkCommandCfg(
                vel_limit=((-1.0, -1.0, -0.5), (1.0, 1.0, 0.5)),
                stand_prob=0.1,
                resampling_time=6.0,
                gait_period_randomization_width=0.2,
            )
        ),
        observations=WalkObservationsCfg(
            policy=replace(
                WalkObservationsCfg.PolicyCfg(),
                height_scan=humanoid_cfg.HeightScanObsCfg(),
            ),
            value=replace(
                WalkObservationsCfg.ValueCfg(),
                height_scan=humanoid_cfg.HeightScanObsCfg(noise=0.0),
            ),
        ),
        sim_reset=humanoid_cfg.WalkResetCfg(
            humanoid_state=humanoid_cfg.WalkStateResetCfg(
                tile_spawn=True,
                spawn_tiles=(_ROWS, _COLS),
                spawn_border=(_BORDER, _BORDER),
                spawn_yaw_range=math.pi,
            )
        ),
        sim=SimCfg(dt=0.005, solver_iterations=8, solver_tolerance=1e-4),
        render_spacing=0.0,
    )


registry.env("g1-walk-stairs")(ManagerEnv)
